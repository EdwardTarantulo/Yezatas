"""The Yezatas web app: the setup screen, the SEO dashboard, the content table, video pages and
the edit, review and push screens, plus the AI suggestions and the agent log.

Pages only read from YouTube through the catalog pull. The one write is `POST /changes/push`,
which sends drafts the creator approved and ticked a confirmation box for. A suggestion only ever
makes a draft. Background jobs open their own Store, because a SQLite connection must not cross
threads.
"""

from __future__ import annotations

import secrets
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Sequence
from urllib.parse import quote, urlencode

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from jinja2 import StrictUndefined
from starlette.templating import Jinja2Templates

from .. import agent, auth, edits, push, subtitle_translate
from .. import chapters as chapters_mod
from .. import transcribe as transcribe_mod
from .. import translate
from ..catalog import NoChannelError, PullResult, pull_catalog
from ..config import Config, ensure_dirs
from ..quota import QuotaGuard
from ..seo import score_catalog
from ..store import Store
from ..subtitles import to_srt
from ..youtube import GoogleYouTubeReader, GoogleYouTubeWriter, YouTubeReader, YouTubeWriter
from .clientfile import (MAX_CLIENT_FILE_BYTES, ClientFileError, ClientInfo,
                         parse_client_file, read_saved_client, save_client_file)
from .guidance import NeedsSignIn, map_api_error, map_sign_in_error
from .jobs import Job, JobSnapshot, Report
from .security import CSRF_FIELD, Sessions, install_security, load_session_key, verify_csrf
from .steps import CheckState, compute_steps, estimate_pull_units
from ..validate import DESCRIPTION_BYTE_LIMIT, TITLE_LIMIT
from .views import (Entry, RunView, SuggestBatch, SuggestItem, build_listing, build_subtitles_view,
                    build_transcript_view, build_translate_view, change_cards, chapters_blocker,
                    dashboard_model, format_cost, format_count, format_date, group_checks, history_rows,
                    lang_name, make_entries, plan_lines, push_results, run_view, stops_batch, thumb_url)

DEFAULT_PORT = 8765
SIGN_IN_TIMEOUT_SECONDS = 300.0
SIGN_IN_TIMEOUT_MESSAGE = (
    "The sign-in did not finish within 5 minutes. If Google showed \"Access blocked\", your "
    "account is probably not on the Test users list: add it in step 4. Otherwise just start "
    "the sign-in again."
)
API_PAGE_URL = "https://console.cloud.google.com/apis/library/youtube.googleapis.com"
UPLOAD_BODY_LIMIT = MAX_CLIENT_FILE_BYTES + 16 * 1024  # file plus multipart overhead
# Generous caps on the edit form. YouTube's own limits are far lower; the validator reports those.
FORM_LIMITS = {"title": 1000, "description": 50_000, "tags": 5000}
HISTORY_LIMIT = 100
AGENT_LOG_LIMIT = 100
SUGGEST_TIMEOUT_SECONDS = 15 * 60.0
SUGGEST_TIMEOUT_MESSAGE = (
    "The suggestions took more than 15 minutes, so Yezatas stopped waiting. The Agent log shows "
    "what finished."
)
TRANSLATE_TIMEOUT_SECONDS = 30 * 60.0
TRANSLATE_TIMEOUT_MESSAGE = (
    "The translation took more than 30 minutes, so Yezatas stopped waiting. Translations that "
    "finished are kept below."
)
TRANSCRIBE_TIMEOUT_SECONDS = 60 * 60.0
TRANSCRIBE_TIMEOUT_MESSAGE = (
    "The transcription took more than 60 minutes, so Yezatas stopped waiting. Try again, or use a "
    "shorter file."
)
CHAPTERS_TIMEOUT_SECONDS = 10 * 60.0
CHAPTERS_TIMEOUT_MESSAGE = (
    "Making the chapters took more than 10 minutes, so Yezatas stopped waiting. The Agent log "
    "shows what finished."
)
SUBTITLES_TIMEOUT_SECONDS = 60 * 60.0
SUBTITLES_TIMEOUT_MESSAGE = (
    "The subtitles took more than 60 minutes, so Yezatas stopped waiting. Languages that "
    "finished are kept below."
)
MAX_PATH_CHARS = 1000
MAX_KEYWORDS = 20
MAX_KEYWORD_CHARS = 40
FIX_NEXT_COUNTS = (1, 3, 5, 10)
MAX_KEY_CHARS = 500

HERE = Path(__file__).parent

ReaderFactory = Callable[[Any], YouTubeReader]
WriterFactory = Callable[[Any], YouTubeWriter]
CredentialsLoader = Callable[[Config], Any]
SuggestFn = Callable[[Store, str, Report], "agent.SuggestResult"]
TranslateTools = Callable[[Config], "tuple[translate.Translator | None, translate.Verifier]"]
# (store, video id, media path, language, report); `keywords=` is added only when the creator typed some.
TranscribeFn = Callable[..., "transcribe_mod.TranscribeResult"]
ChaptersFn = Callable[[Store, str, Report], "chapters_mod.ChaptersResult"]


def default_translate_tools(config: Config) -> "tuple[translate.Translator | None, translate.Verifier]":
    """Spark Open (or Spark 1.3 as the fallback) to translate, Glimmer on this computer to check."""
    return translate.translator_from_config(config), translate.verifier_from_config(config)


class ProgressReader:
    """Wraps a reader and reports what the catalog pull is doing."""

    def __init__(self, inner: YouTubeReader, report: Report) -> None:
        self._inner = inner
        self._report = report
        self._pages = 0
        self._batches = 0
        self._listed = 0

    def channels_mine(self, part: str) -> dict[str, Any]:
        self._report("Reading your channel…")
        return self._inner.channels_mine(part)

    def playlist_items(self, playlist_id: str, page_token: str | None) -> dict[str, Any]:
        self._pages += 1
        self._report(f"Listing your uploads (page {self._pages}, {self._listed} found so far)…")
        page = self._inner.playlist_items(playlist_id, page_token)
        self._listed += len(page.get("items") or [])
        return page

    def videos(self, ids: Sequence[str], part: str) -> dict[str, Any]:
        self._batches += 1
        self._report(f"Fetching video details (batch {self._batches}, up to 50 videos each)…")
        return self._inner.videos(ids, part)


def _latest_pull(store: Store) -> tuple[int, str] | None:
    """(video_count, finished_at) of the newest finished pull, or None."""
    row = store.latest_pull()
    return (int(row["video_count"] or 0), str(row["finished_at"])) if row else None


def _local_time(iso_utc: str) -> str:
    try:
        return datetime.fromisoformat(iso_utc).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso_utc


def sign_in_with_write_access(config: Config) -> Any:
    """The default write sign-in: the same browser flow as setup, asking to edit videos."""
    return auth.sign_in(config, scopes=(auth.WRITE,))


def load_write_credentials(config: Config) -> Any:
    return auth.load_credentials(config, scopes=(auth.WRITE,))


def map_suggest_error(exc: BaseException) -> tuple[str, str]:
    return "suggest", f"The suggestions stopped unexpectedly ({exc.__class__.__name__})."


def map_translate_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, translate.TranslateError):
        return exc.kind, str(exc)
    if type(exc) is ValueError:  # translate_video's own plain sentences
        return "translate", str(exc)
    return "translate", f"The translation stopped unexpectedly ({exc.__class__.__name__})."


def map_subtitles_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, translate.TranslateError):
        return exc.kind, str(exc)
    if type(exc) is ValueError:  # translate_subtitles' own plain sentences
        return "subtitles", str(exc)
    return "subtitles", f"The subtitles stopped unexpectedly ({exc.__class__.__name__})."


def map_transcribe_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, transcribe_mod.TranscribeError):
        return exc.kind, str(exc)
    if type(exc) is ValueError:  # transcribe_video's own plain sentences
        return "transcribe", str(exc)
    return "transcribe", f"The transcription stopped unexpectedly ({exc.__class__.__name__})."


def map_chapters_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, (agent.AgentError, agent.ModelError)):
        return getattr(exc, "kind", "chapters"), str(exc)
    if type(exc) is ValueError:
        return "chapters", str(exc)
    return "chapters", f"Making the chapters stopped unexpectedly ({exc.__class__.__name__})."


def make_default_transcribe(config: Config) -> TranscribeFn:
    """The real transcribe function: Muse Voice Transcribe through Meta's Model API."""

    def run(store: Store, video_id: str, media_path: str, language: str, report: Report,
            keywords: Sequence[str] = ()) -> "transcribe_mod.TranscribeResult":
        key = agent.load_api_key(config)
        if not key:
            raise transcribe_mod.TranscribeError(
                "auth", "No Meta API key is set. Add one in Setup. Nothing was sent.")
        client = transcribe_mod.MetaAsrClient(key)
        return transcribe_mod.transcribe_video(store, client, video_id, media_path, language,
                                               keywords=keywords, report=report)

    return run


def make_default_chapters(config: Config) -> ChaptersFn:
    """The real chapters function: Muse Spark picks transcript segments, code makes the times."""

    def run(store: Store, video_id: str, report: Report) -> "chapters_mod.ChaptersResult":
        key = agent.load_api_key(config)
        if not key:
            raise agent.AgentError("No Meta API key is set. Add one in Setup.")
        from ..modelclient import MetaModelClient

        client = MetaModelClient(key, config.model)
        return chapters_mod.suggest_chapters(store, client, video_id, report=report)

    return run


def hide_path(exc: BaseException, media: Path) -> BaseException:
    """The same error with the file's folder taken out of its sentence (only the name stays)."""
    text = str(exc)
    shown = text
    for variant in (str(media), media.as_posix(), str(media).replace("\\", "\\\\")):
        shown = shown.replace(variant, media.name)
    for variant in (str(media.parent), media.parent.as_posix()):
        if len(variant) > 1:
            shown = shown.replace(variant, "...")
    if shown == text:
        return exc
    if isinstance(exc, transcribe_mod.TranscribeError):
        return transcribe_mod.TranscribeError(exc.kind, shown)
    if type(exc) is ValueError:
        return ValueError(shown)
    return exc


def parse_keywords(raw: str) -> tuple[list[str], str]:
    """(names to recognise, "") from a comma separated line, or ([], the problem in a sentence)."""
    words: list[str] = []
    for part in raw.replace("\n", ",").split(","):
        word = " ".join(part.split())
        if word and word.casefold() not in {w.casefold() for w in words}:
            words.append(word)
    if len(words) > MAX_KEYWORDS:
        return [], f"Give at most {MAX_KEYWORDS} names to recognise. Nothing was sent."
    if any(len(w) > MAX_KEYWORD_CHARS for w in words):
        return [], (f"Each name to recognise can be at most {MAX_KEYWORD_CHARS} characters. "
                    "Nothing was sent.")
    return words, ""


def make_default_suggest(config: Config) -> SuggestFn:
    """The real suggest function: Muse Spark through Meta's Model API, one client per run.

    The model client and the agent loop are looked up when a run starts, not at import.
    """

    def run(store: Store, video_id: str, report: Report) -> "agent.SuggestResult":
        key = agent.load_api_key(config)
        if not key:
            raise agent.AgentError("No Meta API key is set. Add one in Setup.")
        from ..modelclient import MetaModelClient

        client = MetaModelClient(key, config.model)
        return agent.suggest_for_video(store, client, video_id, report=report)

    return run


def check_title_map(entries: Sequence[Entry]) -> dict[str, str]:
    """check id -> title, read off the scorecards (the Agent log and Changes show titles)."""
    titles: dict[str, str] = {}
    for entry in entries:
        for check in entry.card.checks:
            titles.setdefault(check.check_id, check.title)
    return titles


def _job_json(snap: JobSnapshot) -> dict[str, Any]:
    return {"status": snap.status, "message": snap.message, "error_kind": snap.error_kind}


class WebState:
    """Everything the routes share: config, injected collaborators, jobs and the last check."""

    def __init__(
        self,
        config: Config,
        sign_in: Callable[[Config], Any],
        reader_factory: ReaderFactory,
        credentials_loader: CredentialsLoader,
        clock: Callable[[], float],
        sign_in_timeout: float,
        write_sign_in: Callable[[Config], Any],
        write_credentials_loader: CredentialsLoader,
        writer_factory: WriterFactory,
        suggest: SuggestFn,
        translate_tools: TranslateTools = default_translate_tools,
        transcribe_fn: TranscribeFn | None = None,
        chapters_fn: ChaptersFn | None = None,
    ) -> None:
        self.config = config
        self._sign_in = sign_in
        self._reader_factory = reader_factory
        self._credentials_loader = credentials_loader
        self._write_sign_in = write_sign_in
        self._write_credentials_loader = write_credentials_loader
        self._writer_factory = writer_factory
        self._suggest = suggest
        self._translate_tools = translate_tools
        self._transcribe = transcribe_fn or make_default_transcribe(config)
        self._chapters = chapters_fn or make_default_chapters(config)
        self.signin_job = Job(
            map_sign_in_error, sign_in_timeout, SIGN_IN_TIMEOUT_MESSAGE, clock
        )
        self.pull_job = Job(map_api_error)
        self.write_job = Job(map_sign_in_error, sign_in_timeout, SIGN_IN_TIMEOUT_MESSAGE, clock)
        self.push_job = Job(map_api_error)
        self._push_seen: JobSnapshot | None = None  # the finished push the creator dismissed
        self.suggest_job = Job(map_suggest_error, SUGGEST_TIMEOUT_SECONDS,
                               SUGGEST_TIMEOUT_MESSAGE, clock)
        self._suggest_seen: JobSnapshot | None = None  # the finished job the creator dismissed
        self._suggest_ids: tuple[str, ...] = ()
        self._suggest_batch = False
        self.translate_job = Job(map_translate_error, TRANSLATE_TIMEOUT_SECONDS,
                                 TRANSLATE_TIMEOUT_MESSAGE, clock)
        self._translate_seen: JobSnapshot | None = None  # the finished job the creator dismissed
        self._translate_video = ""
        self.subtitles_job = Job(map_subtitles_error, SUBTITLES_TIMEOUT_SECONDS,
                                 SUBTITLES_TIMEOUT_MESSAGE, clock)
        self._subtitles_seen: JobSnapshot | None = None  # the finished job the creator dismissed
        self._subtitles_video = ""
        self.transcribe_job = Job(map_transcribe_error, TRANSCRIBE_TIMEOUT_SECONDS,
                                  TRANSCRIBE_TIMEOUT_MESSAGE, clock)
        self._transcribe_seen: JobSnapshot | None = None  # the finished job the creator dismissed
        self._transcribe_video = ""
        self.chapters_job = Job(map_chapters_error, CHAPTERS_TIMEOUT_SECONDS,
                                CHAPTERS_TIMEOUT_MESSAGE, clock)
        self._chapters_seen: JobSnapshot | None = None
        self._chapters_video = ""
        self._lock = threading.Lock()
        self._check_lock = threading.Lock()
        self._check: CheckState | None = None
        self._flash: dict[str, list[tuple[str, str, str]]] = {}

    # --- flash messages (shown once on the next /setup) -------------------------------------
    def add_flash(self, sid: str, kind: str, text: str, step: str = "") -> None:
        with self._lock:
            self._flash.setdefault(sid, []).append((kind, text, step))

    def pop_flash(self, sid: str) -> list[dict[str, str]]:
        with self._lock:
            items = self._flash.pop(sid, [])
        return [{"kind": k, "text": t, "step": s} for k, t, s in items]

    # --- connection check -------------------------------------------------------------------
    @property
    def check(self) -> CheckState | None:
        with self._lock:
            return self._check

    def reset_check(self) -> None:
        with self._lock:
            self._check = None

    def run_check(self) -> CheckState:
        """Charge channels.list, call it, and remember the outcome. Never raises."""
        if not self._check_lock.acquire(blocking=False):
            return self.check or CheckState(False, "busy", "A check is already running.")
        try:
            try:
                state = self._do_check()
            except BaseException as exc:
                kind, message = map_api_error(exc)
                state = CheckState(False, kind, message)
        finally:
            self._check_lock.release()
        with self._lock:
            self._check = state
        return state

    def _do_check(self) -> CheckState:
        creds = self._credentials_loader(self.config)
        if creds is None:
            raise NeedsSignIn()
        with Store(self.config.catalog_db) as store:
            QuotaGuard(store, self.config.daily_quota_budget).charge(
                "channels.list", note="setup check"
            )
        reader = self._reader_factory(creds)
        response = reader.channels_mine("snippet,statistics,contentDetails")
        items = response.get("items") or []
        if not items:
            raise NoChannelError("this Google account has no YouTube channel")
        channel = items[0]
        uploads = ((channel.get("contentDetails") or {}).get("relatedPlaylists") or {}).get("uploads")
        try:
            video_count = int((channel.get("statistics") or {}).get("videoCount", 0))
        except (TypeError, ValueError):
            video_count = 0
        return CheckState(
            ok=True,
            channel_id=str(channel.get("id", "")),
            channel_title=str((channel.get("snippet") or {}).get("title", "")),
            video_count=video_count,
            has_uploads=bool(uploads),
        )

    # --- jobs -------------------------------------------------------------------------------
    def start_sign_in(self) -> bool:
        def work(report: Report) -> None:
            report("Waiting for you to finish in the browser tab…")
            self._sign_in(self.config)
            report("Signed in. Checking the connection…")
            self.run_check()

        return self.signin_job.start(work)

    def start_pull(self) -> bool:
        def work(report: Report) -> PullResult:
            creds = self._credentials_loader(self.config)
            if creds is None:
                raise NeedsSignIn()
            reader = ProgressReader(self._reader_factory(creds), report)
            with Store(self.config.catalog_db) as store:
                guard = QuotaGuard(store, self.config.daily_quota_budget)
                return pull_catalog(reader, store, guard)

        return self.pull_job.start(work)

    # --- write access and push --------------------------------------------------------------
    def write_credentials(self) -> Any:
        """Credentials that may edit videos, or None. Never raises."""
        try:
            return self._write_credentials_loader(self.config)
        except Exception:  # noqa: BLE001 - a broken token file just means "sign in again"
            return None

    def start_write_sign_in(self) -> bool:
        def work(report: Report) -> None:
            report("Waiting for you to finish in the browser tab…")
            self._write_sign_in(self.config)

        return self.write_job.start(work)

    def start_push(self, approved: dict[str, str], creds: Any) -> bool:
        def work(report: Report) -> push.PushReport:
            writer = self._writer_factory(creds)
            with Store(self.config.catalog_db) as store:
                guard = QuotaGuard(store, self.config.daily_quota_budget)
                return push.push_drafts(writer, store, guard, approved, report)

        return self.push_job.start(work)

    def push_snapshot(self) -> JobSnapshot:
        """The push job's state, or idle once the creator dismissed a finished result."""
        snap = self.push_job.snapshot()
        with self._lock:
            if snap is self._push_seen:
                return JobSnapshot("idle")
        return snap

    def dismiss_push_result(self) -> None:
        snap = self.push_job.snapshot()
        if not snap.running:
            with self._lock:
                self._push_seen = snap

    # --- AI suggestions ---------------------------------------------------------------------
    def start_suggest(self, video_ids: Sequence[str], batch: bool, sid: str) -> bool:
        """Suggest improvements for these videos, one after another, in the one suggestion job."""
        ids = tuple(video_ids)

        def work(report: Report) -> SuggestBatch:
            return self._run_suggest(ids, batch, sid, report)

        if not self.suggest_job.start(work):
            return False
        with self._lock:
            self._suggest_ids, self._suggest_batch = ids, batch
        return True

    def _run_suggest(self, ids: tuple[str, ...], batch: bool, sid: str,
                     report: Report) -> SuggestBatch:
        items: list[SuggestItem] = []
        stopped = ""
        with Store(self.config.catalog_db) as store:
            titles: dict[str, str] = {}
            for video_id in ids:
                video = store.get_video(video_id)
                titles[video_id] = video.title if video is not None else video_id
            for number, video_id in enumerate(ids, 1):
                title = titles[video_id]
                lead = f"{number} of {len(ids)}: {title}" if len(ids) > 1 else ""

                def say(text: str, lead: str = lead) -> None:
                    report(f"{lead} ({text})" if lead else text)

                if lead:
                    report(lead)
                try:
                    result = self._suggest(store, video_id, say)
                except agent.AgentError as exc:
                    items.append(SuggestItem(video_id, title, "skipped", str(exc)))
                    continue
                except Exception as exc:  # noqa: BLE001 - keep what finished, stop the rest
                    items.append(SuggestItem(video_id, title, "failed",
                                             f"Stopped unexpectedly ({exc.__class__.__name__})."))
                    stopped = "A suggestion stopped unexpectedly, so the rest were not tried."
                    break
                items.append(SuggestItem(video_id, title, result.status, result.message))
                if result.status == "failed" and stops_batch(store.get_agent_run(result.run_id)):
                    stopped = result.message
                    break
        done = {item.video_id for item in items}
        items.extend(SuggestItem(v, titles[v], "not_attempted") for v in ids if v not in done)
        if not batch and items:
            # One video: the result shows once, on the page the creator lands on after the reload.
            only = items[0]
            kind = ("ok" if only.status == "drafted"
                    else "info" if only.status == "no_change" else "error")
            self.add_flash(sid, kind, only.message or only.label)
        return SuggestBatch(tuple(items), stopped)

    def suggest_state(self) -> tuple[JobSnapshot, tuple[str, ...], bool]:
        """(job snapshot, video ids, is it a batch). A finished job the creator dismissed is idle."""
        snap = self.suggest_job.snapshot()
        with self._lock:
            if snap is self._suggest_seen:
                snap = JobSnapshot("idle")
            return snap, self._suggest_ids, self._suggest_batch

    def dismiss_suggest_result(self) -> None:
        snap = self.suggest_job.snapshot()
        if not snap.running:
            with self._lock:
                self._suggest_seen = snap

    # --- translations -----------------------------------------------------------------------
    def start_translate(self, video_id: str, source_language: str, targets: Sequence[str],
                        sid: str) -> bool:
        """Translate one video into these languages, in the one translation job."""
        wanted = tuple(targets)

        def work(report: Report) -> "translate.TranslateResult":
            return self._run_translate(video_id, source_language, wanted, sid, report)

        if not self.translate_job.start(work):
            return False
        with self._lock:
            self._translate_video = video_id
        return True

    def _run_translate(self, video_id: str, source_language: str, targets: tuple[str, ...],
                       sid: str, report: Report) -> "translate.TranslateResult":
        translator, verifier = self._translate_tools(self.config)
        if translator is None:
            raise translate.TranslateError(
                "auth", "There is no translation key. Add one in Setup. Nothing was sent.")
        with Store(self.config.catalog_db) as store:
            result = translate.translate_video(store, translator, verifier, video_id,
                                               source_language, list(targets), report=report)
        failed = all(r.status == "failed" for r in result.records)
        self.add_flash(
            sid, "error" if failed else "ok",
            f"Translation finished: {result.message}."
            + ("" if failed else " Read the results below, then add the ones you want to the "
                                 "draft. Nothing goes to YouTube until you approve and push."))
        return result

    def translate_state(self) -> tuple[JobSnapshot, str]:
        """(job snapshot, the video id it works on). A finished job the creator dismissed is idle."""
        snap = self.translate_job.snapshot()
        with self._lock:
            if snap is self._translate_seen:
                snap = JobSnapshot("idle")
            return snap, self._translate_video

    def dismiss_translate_result(self) -> None:
        snap = self.translate_job.snapshot()
        if not snap.running:
            with self._lock:
                self._translate_seen = snap

    # --- translated subtitles -----------------------------------------------------------------
    def start_subtitles(self, video_id: str, targets: Sequence[str], sid: str) -> bool:
        """Translate one video's transcript into these languages, in the one subtitles job."""
        wanted = tuple(targets)

        def work(report: Report) -> "subtitle_translate.SubtitleTranslateResult":
            return self._run_subtitles(video_id, wanted, sid, report)

        if not self.subtitles_job.start(work):
            return False
        with self._lock:
            self._subtitles_video = video_id
        return True

    def _run_subtitles(self, video_id: str, targets: tuple[str, ...], sid: str, report: Report
                       ) -> "subtitle_translate.SubtitleTranslateResult":
        translator, verifier = self._translate_tools(self.config)
        if translator is None:
            raise translate.TranslateError(
                "auth", "There is no translation key. Add one in Setup. Nothing was sent.")
        with Store(self.config.catalog_db) as store:
            return subtitle_translate.translate_subtitles(
                store, translator, verifier, video_id, list(targets), report=report)  # type: ignore[arg-type]

    def subtitles_state(self) -> tuple[JobSnapshot, str]:
        """(job snapshot, the video id it works on). A finished job the creator dismissed is idle."""
        snap = self.subtitles_job.snapshot()
        with self._lock:
            if snap is self._subtitles_seen:
                snap = JobSnapshot("idle")
            return snap, self._subtitles_video

    def dismiss_subtitles_result(self) -> None:
        snap = self.subtitles_job.snapshot()
        if not snap.running:
            with self._lock:
                self._subtitles_seen = snap

    # --- transcripts and chapters -------------------------------------------------------------
    def start_transcribe(self, video_id: str, media: Path, language: str,
                         keywords: Sequence[str]) -> bool:
        """Transcribe the video file at `media` (kept in memory only), in the one transcribe job."""
        words = tuple(keywords)

        def work(report: Report) -> "transcribe_mod.TranscribeResult":
            return self._run_transcribe(video_id, media, language, words, report)

        if not self.transcribe_job.start(work):
            return False
        with self._lock:
            self._transcribe_video = video_id
        return True

    def _run_transcribe(self, video_id: str, media: Path, language: str,
                        keywords: tuple[str, ...], report: Report
                        ) -> "transcribe_mod.TranscribeResult":
        with Store(self.config.catalog_db) as store:
            try:
                if keywords:
                    return self._transcribe(store, video_id, str(media), language, report,
                                            keywords=keywords)
                return self._transcribe(store, video_id, str(media), language, report)
            except Exception as exc:  # noqa: BLE001 - pages show the file name, never its folder
                hidden = hide_path(exc, media)
                if hidden is exc:
                    raise
                raise hidden from None

    def transcribe_state(self) -> tuple[JobSnapshot, str]:
        """(job snapshot, the video id it works on). A finished job the creator dismissed is idle."""
        snap = self.transcribe_job.snapshot()
        with self._lock:
            if snap is self._transcribe_seen:
                snap = JobSnapshot("idle")
            return snap, self._transcribe_video

    def dismiss_transcribe_result(self) -> None:
        snap = self.transcribe_job.snapshot()
        if not snap.running:
            with self._lock:
                self._transcribe_seen = snap

    def start_chapters(self, video_id: str) -> bool:
        """Make chapters for one video from its transcript, in the one chapters job."""
        def work(report: Report) -> "chapters_mod.ChaptersResult":
            with Store(self.config.catalog_db) as store:
                return self._chapters(store, video_id, report)

        if not self.chapters_job.start(work):
            return False
        with self._lock:
            self._chapters_video = video_id
        return True

    def chapters_state(self) -> tuple[JobSnapshot, str]:
        snap = self.chapters_job.snapshot()
        with self._lock:
            if snap is self._chapters_seen:
                snap = JobSnapshot("idle")
            return snap, self._chapters_video

    def dismiss_chapters_result(self) -> None:
        snap = self.chapters_job.snapshot()
        if not snap.running:
            with self._lock:
                self._chapters_seen = snap

    def sign_out(self) -> bool:
        removed = auth.sign_out(self.config)
        self.reset_check()
        return removed


def _api_page_url(client: ClientInfo | None) -> str:
    """The YouTube Data API page, opened on the client's own project when the file names one."""
    url = API_PAGE_URL
    if client is not None and client.project_id != "(not given)":
        url += "?" + urlencode({"project": client.project_id})
    return url


def create_app(
    config: Config,
    *,
    port: int = DEFAULT_PORT,
    token: str | None = None,
    sign_in: Callable[[Config], Any] = auth.sign_in,
    reader_factory: ReaderFactory = GoogleYouTubeReader,
    credentials_loader: CredentialsLoader = auth.load_credentials,
    open_browser: bool = True,
    sign_in_timeout: float = SIGN_IN_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    write_sign_in: Callable[[Config], Any] = sign_in_with_write_access,
    write_credentials_loader: CredentialsLoader = load_write_credentials,
    writer_factory: WriterFactory = GoogleYouTubeWriter,
    suggest: SuggestFn | None = None,
    translate_tools: TranslateTools | None = None,
    transcribe: TranscribeFn | None = None,
    chapters: ChaptersFn | None = None,
) -> FastAPI:
    """Build the app. `app.state.start_url` is the one-time link that signs the browser in."""
    ensure_dirs(config)
    token = token or secrets.token_urlsafe(32)
    start_url = f"http://127.0.0.1:{port}/?t={token}"
    tools = translate_tools or default_translate_tools
    state = WebState(config, sign_in, reader_factory, credentials_loader, clock, sign_in_timeout,
                     write_sign_in, write_credentials_loader, writer_factory,
                     suggest or make_default_suggest(config),
                     tools, transcribe, chapters)
    sessions = Sessions(token, port, load_session_key(config.credentials_dir / "session.key"))
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    # A missing value is a bug: fail loudly instead of rendering "" (an empty link, say).
    templates.env.undefined = StrictUndefined
    templates.env.filters["lang_name"] = lang_name

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if open_browser:
            threading.Timer(1.0, webbrowser.open, args=(start_url,)).start()
        yield

    app = FastAPI(title="Yezatas", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    app.state.web = state
    app.state.start_url = start_url
    app.state.token = token
    install_security(app, sessions, templates)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    def back_to_setup(step: str = "") -> RedirectResponse:
        return RedirectResponse("/setup" + (f"#step-{step}" if step else ""), status_code=303)

    async def read_form(request: Request) -> Any:
        form = await request.form()
        verify_csrf(request, str(form.get(CSRF_FIELD) or ""))
        return form

    def remaining_quota() -> int:
        with Store(config.catalog_db) as store:
            return QuotaGuard(store, config.daily_quota_budget).remaining()

    def page_data() -> dict[str, Any]:
        """Real state for the setup page (runs in a worker thread: it opens its own Store)."""
        client, client_problem = read_saved_client(config)
        signed_in = config.token_path.is_file()
        signin = state.signin_job.snapshot()
        pull = state.pull_job.snapshot()
        check = state.check
        with Store(config.catalog_db) as store:
            latest = _latest_pull(store)
            guard = QuotaGuard(store, config.daily_quota_budget)
            used, remaining = guard.used(), guard.remaining()
        steps = compute_steps(client, client_problem, signed_in, signin, check, pull,
                              latest is not None)
        # Steps 1 to 4 cannot be detected, so once the client file is valid the next action is
        # further down the list.
        candidates = steps[5:] if client else steps
        current = next((s.key for s in candidates if s.status != "done" and s.enabled), "")
        estimate = None
        if check is not None and check.ok:
            units = estimate_pull_units(check.video_count)
            estimate = {"units": units, "fits": units <= remaining}
        return {
            "steps": {s.key: s for s in steps}, "current": current,
            "client": client, "client_exists": config.client_secrets_path.exists(),
            "client_path": str(config.client_secrets_path),
            "api_url": _api_page_url(client),
            "signed_in": signed_in, "can_edit": signed_in and auth.can_edit(config), "signin": signin, "pull": pull, "check": check,
            "estimate": estimate,
            "latest_pull": ({"videos": latest[0], "when": _local_time(latest[1])}
                            if latest else None),
            "pull_result": pull.result if isinstance(pull.result, PullResult) else None,
            "quota": {"used": used, "remaining": remaining, "budget": config.daily_quota_budget},
            "home": str(config.home),
            "model": config.model, "model_key": agent.key_source(config),
            "key_env": agent.KEY_ENV,
            "poll": signin.running or pull.running,
            "max_kb": MAX_CLIENT_FILE_BYTES // 1024,
            "flash": [],
        }

    def catalog_data() -> dict[str, Any] | None:
        """Videos, scores and quota for the catalog pages; None before the first finished pull.

        Runs in a worker thread (it opens its own Store) and scores the catalog on every call.
        """
        with Store(config.catalog_db) as store:
            latest = _latest_pull(store)
            if latest is None:
                return None
            videos = store.list_videos()
            guard = QuotaGuard(store, config.daily_quota_budget)
            used, remaining = guard.used(), guard.remaining()
        entries = make_entries(videos, score_catalog(videos))
        return {
            "entries": entries,
            "latest_pull": {"videos": latest[0], "when": _local_time(latest[1])},
            "quota": {"used": used, "remaining": remaining, "budget": config.daily_quota_budget},
        }

    def draft_ids() -> frozenset[str]:
        with Store(config.catalog_db) as store:
            return frozenset(row.video_id for row in store.list_drafts())

    def render(request: Request, name: str, extra: dict[str, Any],
               status_code: int = 200) -> Response:
        context: dict[str, Any] = {"csrf": request.state.csrf, "draft_count": len(draft_ids()),
                                   **extra}
        # The setup page shows its own flash messages next to the step they belong to.
        if "flash" not in extra and "shell_flash" not in extra:
            context["shell_flash"] = state.pop_flash(request.state.session_id)
        return templates.TemplateResponse(request, name, context, status_code=status_code)

    def see_other(url: str) -> RedirectResponse:
        return RedirectResponse(url, status_code=303)

    def flash(request: Request, kind: str, text: str, step: str = "") -> None:
        state.add_flash(request.state.session_id, kind, text, step)

    def suggest_panel(video_id: str = "") -> dict[str, Any]:
        """What the suggestion job is doing, for the dashboard (no id) or one video page."""
        snap, ids, is_batch = state.suggest_state()
        mine = (video_id in ids) if video_id else is_batch
        batch = snap.result if (snap.status == "done" and is_batch and not video_id
                                and isinstance(snap.result, SuggestBatch)) else None
        return {
            "has_key": bool(agent.key_source(config)), "model": config.model,
            "running": snap.running and (mine or not video_id),
            "busy": snap.running and not mine and bool(video_id),
            "message": snap.message, "batch": batch,
            "failed": snap.message if snap.status == "failed" and mine else "",
            "counts": FIX_NEXT_COUNTS, "video_id": video_id,
        }

    def tools_or_none() -> "tuple[translate.Translator | None, translate.Verifier | None]":
        """The translator and the checker; a broken setup just means "not available"."""
        try:
            return tools(config)
        except Exception:  # noqa: BLE001
            return None, None

    down_until = [0.0]  # a refused connection takes about 2 seconds on Windows: remember "down"

    def verifier_ready(verifier: "translate.Verifier | None", fresh: bool = False) -> bool:
        """Is the checker up? The Setup page always asks (`fresh`); other pages reuse a slow "no"."""
        if verifier is None:
            return False
        if not fresh and clock() < down_until[0]:
            return False
        started = clock()
        try:
            ready = bool(verifier.available())
        except Exception:  # noqa: BLE001
            ready = False
        down_until[0] = 0.0 if ready or clock() - started < 0.5 else clock() + 30.0
        return ready

    def translate_panel(video_id: str) -> dict[str, Any]:
        """What the translation job and the tools are doing, for one video page (worker thread)."""
        snap, job_video = state.translate_state()
        mine = job_video == video_id
        running = snap.running and mine
        translator, verifier = tools_or_none()
        source = translate.translator_key_source(config)
        provider = "meta" if source == "meta" else "baseten" if source else "other"
        model = translator.model if translator is not None else (
            config.model if source == "meta" else config.translator_model)
        return {
            "running": running, "busy": snap.running and not mine,
            "message": snap.message if running else "",
            "failed": snap.message if snap.status == "failed" and mine else "",
            "has_translator": translator is not None, "model": model, "provider": provider,
            "verifier_model": verifier.model if verifier is not None else config.verifier_model,
            "verifier_ready": True if running else verifier_ready(verifier),
        }

    def job_panel(snap: JobSnapshot, job_video: str, video_id: str) -> dict[str, Any]:
        mine = job_video == video_id
        running = snap.running and mine
        return {
            "running": running, "busy": snap.running and not mine,
            "message": snap.message if running else "",
            "failed": snap.message if snap.status == "failed" and mine else "",
            "done": None,
        }

    def transcript_panel(video_id: str) -> dict[str, Any]:
        """What the transcribe and chapters jobs are doing, for one video page."""
        snap, job_video = state.transcribe_state()
        transcribing = job_panel(snap, job_video, video_id)
        if snap.status == "done" and job_video == video_id and isinstance(
                snap.result, transcribe_mod.TranscribeResult):
            transcribing["done"] = {
                "tone": "ok", "chapters": [], "drafted": False,
                "text": f"{snap.result.message} The cost was about {format_cost(snap.result.cost)}.",
            }
        snap, job_video = state.chapters_state()
        making = job_panel(snap, job_video, video_id)
        if snap.status == "done" and job_video == video_id and isinstance(
                snap.result, chapters_mod.ChaptersResult):
            result = snap.result
            making["done"] = {
                "tone": {"drafted": "ok", "no_change": "info"}.get(result.status, "error"),
                "text": result.message, "drafted": result.status == "drafted",
                "chapters": chapters_mod.chapter_lines(result.chapters),
            }
        return {"ffmpeg": transcribe_mod.find_ffmpeg() is not None,
                "has_key": bool(agent.key_source(config)),
                "transcribe": transcribing, "chapters": making}

    def subtitles_panel(video_id: str, tr: dict[str, Any]) -> dict[str, Any]:
        """What the subtitles job and the tools are doing, for one video page. `tr` is the
        translation panel, which already asked for the translator and the checker."""
        snap, job_video = state.subtitles_state()
        panel = job_panel(snap, job_video, video_id)
        if snap.status == "done" and job_video == video_id and isinstance(
                snap.result, subtitle_translate.SubtitleTranslateResult):
            result = snap.result
            failed = all(r.status == "failed" for r in result.records)
            panel["done"] = {
                "tone": "error" if failed else "ok",
                "text": f"Subtitles finished: {result.message}."
                        + ("" if failed else " Read the results below, then download the .srt "
                                             "files you want."),
            }
        panel.update({k: tr[k] for k in ("has_translator", "model", "provider", "verifier_model",
                                         "verifier_ready")})
        return panel

    def video_not_found(request: Request) -> Response:
        return render(request, "error.html", {
            "layout": "shell.html", "active": "content",
            "heading": "Video not found",
            "text": "There is no video with this id in your pulled catalog. It may have been "
                    "removed from your channel, or the catalog is out of date.",
            "link_url": "/videos", "link_text": "Back to Content",
        }, status_code=404)

    # --- pages ------------------------------------------------------------------------------
    @app.get("/")
    def index(request: Request) -> Response:
        data = catalog_data()
        if data is None:
            return back_to_setup()
        model = dashboard_model(data["entries"])
        return render(request, "dashboard.html", {
            "active": "dashboard", "latest_pull": data["latest_pull"], "quota": data["quota"],
            "sg": suggest_panel(), "poll": state.suggest_job.snapshot().running,
            **model,
        })

    @app.get("/videos")
    def videos_page(request: Request) -> Response:
        data = catalog_data()
        if data is None:
            return back_to_setup()
        listing = build_listing(data["entries"], dict(request.query_params))
        return render(request, "videos.html", {"active": "content", "listing": listing,
                                               "q": listing.q, "draft_ids": draft_ids()})

    @app.get("/videos/{video_id}")
    def video_page(request: Request, video_id: str) -> Response:
        data = catalog_data()
        if data is None:
            return back_to_setup()
        entry: Entry | None = next((e for e in data["entries"] if e.video.video_id == video_id),
                                   None)
        if entry is None:
            return video_not_found(request)
        video = entry.video
        with Store(config.catalog_db) as store:
            row = store.get_draft(video_id)
            runs = store.list_agent_runs(video_id, limit=1)
            records = store.latest_translations(video_id)
        card = change_cards([(row, video)])[0] if row is not None else None
        panel = suggest_panel(video_id)
        latest = (run_view(runs[0], {video_id: video.title}, check_title_map([entry]),
                           _local_time, interrupted=not panel["running"])
                  if runs else None)
        tr = translate_panel(video_id)
        ts = transcript_panel(video_id)
        with Store(config.catalog_db) as store:
            record = store.latest_transcript(video_id)
            subtitle_records = (store.latest_subtitle_translations(video_id, record.id)
                                if record is not None else {})
        tt = build_transcript_view(video, row, record,
                                   _local_time(record.created_at) if record else "")
        sv = build_subtitles_view(video_id, record, subtitle_records)
        sb = subtitles_panel(video_id, tr)
        text = ((row.title, row.description, row.tags) if row is not None
                else (video.title, video.description, tuple(video.tags)))
        return render(request, "video.html", {
            "change": card,
            "form": {"title": text[0], "description": text[1], "tags": ", ".join(text[2])},
            "title_limit": TITLE_LIMIT, "description_limit": DESCRIPTION_BYTE_LIMIT,
            "active": "content", "entry": entry, "video": video, "card": entry.card,
            "groups": group_checks(entry.card),
            "big_thumb": thumb_url(video, ("high", "medium", "default")),
            "views": format_count(video.view_count), "likes": format_count(video.like_count),
            "comments": format_count(video.comment_count),
            "published": format_date(video.published_at),
            "sg": panel, "latest_run": latest,
            "tr": tr, "tv": build_translate_view(video, row, records), "ts": ts, "tt": tt,
            "sv": sv, "sb": sb,
            "poll": (panel["running"] or panel["busy"] or tr["running"] or tr["busy"]
                     or ts["transcribe"]["running"] or ts["transcribe"]["busy"]
                     or ts["chapters"]["running"] or ts["chapters"]["busy"]
                     or sb["running"] or sb["busy"]),
        })

    @app.get("/setup")
    def setup(request: Request) -> Response:
        data = page_data()
        data["flash"] = state.pop_flash(request.state.session_id)
        _, verifier = tools_or_none()
        data.update({
            "translator_key": translate.translator_key_source(config),
            "translator_env": translate.TRANSLATOR_KEY_ENV,
            "translator_model": config.translator_model,
            "verifier_model": config.verifier_model, "verifier_url": config.verifier_url,
            "verifier_ready": verifier_ready(verifier, fresh=True),
            "ffmpeg": transcribe_mod.find_ffmpeg() is not None,
        })
        return render(request, "setup.html", {**data, "active": "setup"})

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        check = state.check
        return JSONResponse({
            "signin": _job_json(state.signin_job.snapshot()),
            "pull": _job_json(state.pull_job.snapshot()),
            "write": _job_json(state.write_job.snapshot()),
            "push": _job_json(state.push_job.snapshot()),
            "suggest": _job_json(state.suggest_job.snapshot()),
            "translate": _job_json(state.translate_job.snapshot()),
            "subtitles": _job_json(state.subtitles_job.snapshot()),
            "transcribe": _job_json(state.transcribe_job.snapshot()),
            "chapters": _job_json(state.chapters_job.snapshot()),
            "check": None if check is None else {"ok": check.ok, "kind": check.kind},
        })

    # --- actions (POST only, CSRF checked) ----------------------------------------------------
    @app.post("/setup/client")
    async def upload_client(request: Request) -> Response:
        sid = request.state.session_id
        try:
            declared = int(request.headers.get("content-length") or 0)
        except ValueError:
            declared = 0
        if declared > UPLOAD_BODY_LIMIT:
            # Not read, not acted on (so no CSRF needed): just explain and send them back.
            return templates.TemplateResponse(request, "error.html", {
                "heading": "That file is too big",
                "text": f"It is larger than {MAX_CLIENT_FILE_BYTES // 1024} KB, so it is not an "
                        "OAuth client file. Go back and choose the JSON you downloaded for your "
                        "Desktop app client.",
            }, status_code=413)
        form = await read_form(request)
        upload = form.get("client_file")
        if upload is None or not hasattr(upload, "read"):
            state.add_flash(sid, "error", "Choose the JSON file you downloaded first.", "client")
            return back_to_setup("client")
        data = await upload.read(MAX_CLIENT_FILE_BYTES + 1)
        try:
            info = parse_client_file(data)
        except ClientFileError as exc:
            state.add_flash(sid, "error", str(exc), "client")
            return back_to_setup("client")
        existing, _ = read_saved_client(config)
        if existing is not None and form.get("replace") != "yes":
            state.add_flash(sid, "error",
                            "A client file is already saved. Tick \"Replace the saved file\" "
                            "to confirm, then upload again.", "client")
            return back_to_setup("client")
        try:
            await run_in_threadpool(save_client_file, config, data)
        except OSError as exc:
            state.add_flash(sid, "error",
                            f"The file could not be saved ({exc.__class__.__name__}). Check that "
                            f"{config.credentials_dir} is writable.", "client")
            return back_to_setup("client")
        text = f"Saved the client file for project {info.project_id} (client {info.client_id_short})."
        if existing is not None:
            await run_in_threadpool(state.sign_out)
            text += " Because the client changed, you were signed out: sign in again in step 6."
        state.add_flash(sid, "ok", text, "client")
        return back_to_setup("signin" if existing is not None else "client")

    @app.post("/setup/signin")
    async def start_sign_in(request: Request) -> Response:
        await read_form(request)
        sid = request.state.session_id
        client, _ = read_saved_client(config)
        if client is None:
            state.add_flash(sid, "error", "Add the OAuth client file first (step 5).", "client")
            return back_to_setup("client")
        if config.token_path.is_file():
            state.add_flash(sid, "error", "You are already signed in. Sign out first to switch "
                            "accounts.", "signin")
            return back_to_setup("signin")
        if not state.start_sign_in():
            state.add_flash(sid, "error", "A sign-in is already waiting for you in the browser.",
                            "signin")
        return back_to_setup("signin")

    @app.post("/setup/signin/cancel")
    async def cancel_sign_in(request: Request) -> Response:
        await read_form(request)
        state.signin_job.cancel("Sign-in cancelled.")
        return back_to_setup("signin")

    @app.post("/setup/signout")
    async def sign_out(request: Request) -> Response:
        await read_form(request)
        removed = await run_in_threadpool(state.sign_out)
        state.add_flash(request.state.session_id, "ok",
                        "Signed out. The saved token was deleted." if removed
                        else "There was no saved sign-in.", "signin")
        return back_to_setup("signin")

    @app.post("/setup/check")
    async def check_connection(request: Request) -> Response:
        await read_form(request)
        sid = request.state.session_id
        if not config.token_path.is_file():
            state.add_flash(sid, "error", "Sign in first (step 6).", "signin")
            return back_to_setup("signin")
        await run_in_threadpool(state.run_check)
        return back_to_setup("check")

    @app.post("/setup/pull")
    async def start_pull(request: Request) -> Response:
        await read_form(request)
        sid = request.state.session_id
        check = state.check
        if check is None or not check.ok or not check.has_uploads:
            state.add_flash(sid, "error", "Check the connection first (step 7).", "check")
            return back_to_setup("check")
        remaining = await run_in_threadpool(remaining_quota)
        needed = estimate_pull_units(check.video_count)
        if needed > remaining:
            state.add_flash(sid, "error",
                            f"The pull needs about {needed} quota units but only {remaining} are "
                            "left today. Quota resets at midnight Pacific time.", "pull")
            return back_to_setup("pull")
        if not state.start_pull():
            state.add_flash(sid, "error", "A pull is already running.", "pull")
        return back_to_setup("pull")

    # --- edit, review, push, history -----------------------------------------------------------
    def why_runs(cards: Sequence[Any]) -> dict[str, RunView]:
        """For agent drafts: the latest run that made a draft of that video (worker thread)."""
        ids = [c.video_id for c in cards if c.row.source == "agent"]
        if not ids:
            return {}
        data = catalog_data()
        titles = check_title_map(data["entries"]) if data else {}
        found: dict[str, RunView] = {}
        with Store(config.catalog_db) as store:
            for video_id in ids:
                run = next((r for r in store.list_agent_runs(video_id)
                            if r.status == "drafted"), None)
                if run is not None:
                    found[video_id] = run_view(run, {}, titles, _local_time)
        return found

    def changes_data() -> list[Any]:
        """Every pending draft as a card, most viewed video first (worker thread)."""
        with Store(config.catalog_db) as store:
            return change_cards(edits.pending(store))

    def titles_of(ids: Sequence[str]) -> dict[str, str]:
        with Store(config.catalog_db) as store:
            found = {i: store.get_video(i) for i in ids}
        return {i: v.title for i, v in found.items() if v is not None}

    @app.get("/changes")
    def changes_page(request: Request) -> Response:
        cards = changes_data()
        return render(request, "changes.html", {
            "active": "changes", "cards": cards,
            "approved_count": sum(1 for c in cards if c.ready),
            "push_status": state.push_snapshot().status,
            "why": why_runs(cards),
        })

    async def change_step(request: Request, work: Callable[[Store], str], anchor: str = "") -> Response:
        """Run one draft step in a worker thread; a refused step becomes an error flash."""
        await read_form(request)

        def run() -> tuple[str, str]:
            with Store(config.catalog_db) as store:
                try:
                    return "ok", work(store)
                except edits.DraftError as exc:
                    return "error", str(exc)

        kind, message = await run_in_threadpool(run)
        flash(request, kind, message)
        return see_other("/changes" + (f"#d-{quote(anchor)}" if anchor else ""))

    @app.post("/changes/{video_id}/approve")
    async def approve_draft(request: Request, video_id: str) -> Response:
        def work(store: Store) -> str:
            edits.approve(store, video_id)
            return "Approved. It is sent to YouTube only after you review and push."

        return await change_step(request, work, video_id)

    @app.post("/changes/{video_id}/unapprove")
    async def unapprove_draft(request: Request, video_id: str) -> Response:
        def work(store: Store) -> str:
            edits.unapprove(store, video_id)
            return "Approval removed."

        return await change_step(request, work, video_id)

    @app.post("/changes/{video_id}/discard")
    async def discard_draft(request: Request, video_id: str) -> Response:
        def work(store: Store) -> str:
            return "Draft discarded." if edits.discard(store, video_id) else "There was no draft."

        return await change_step(request, work)

    @app.post("/changes/{video_id}/rebase")
    async def rebase_draft(request: Request, video_id: str) -> Response:
        def work(store: Store) -> str:
            if edits.rebase(store, video_id) is None:
                return ("The draft now matches the text on YouTube, so it was removed. "
                        "Nothing is left to push.")
            return ("The draft now builds on YouTube's current text. Review the changes and "
                    "approve again.")

        return await change_step(request, work, video_id)

    @app.post("/videos/{video_id}/draft")
    async def save_video_draft(request: Request, video_id: str) -> Response:
        form = await read_form(request)
        fields = {name: str(form.get(name) or "") for name in FORM_LIMITS}
        if any(len(fields[name]) > limit for name, limit in FORM_LIMITS.items()):
            flash(request, "error", "That text is far longer than YouTube allows, so it was not "
                  "saved. Shorten the title, description or tags.")
            return see_other(f"/videos/{quote(video_id)}#edit")

        def work() -> edits.SaveResult:
            with Store(config.catalog_db) as store:
                return edits.save_draft(store, video_id, fields["title"], fields["description"],
                                        fields["tags"].split(","))

        try:
            result = await run_in_threadpool(work)
        except edits.DraftError:
            return video_not_found(request)
        if result.discarded:
            flash(request, "info", "This text is the same as the one on YouTube, so no draft "
                  "was kept.")
            return see_other(f"/videos/{quote(video_id)}#edit")
        flash(request, "ok", "Draft saved. Nothing is sent to YouTube until you approve it "
              "and push.")
        return see_other(f"/changes#d-{quote(video_id)}")

    # --- push -------------------------------------------------------------------------------
    def push_page_data() -> dict[str, Any]:
        push_snap = state.push_snapshot()
        write_snap = state.write_job.snapshot()
        data: dict[str, Any] = {
            "active": "changes", "push": push_snap, "write": write_snap,
            "poll": push_snap.running or write_snap.running,
            "ready": [], "waiting": 0, "plan": None, "has_write": False, "results": [],
            "report": None, "sends_translations": False, "sends_language": False,
        }
        if push_snap.status == "done" and isinstance(push_snap.result, push.PushReport):
            report = push_snap.result
            data["report"] = report
            data["results"] = push_results(report.items,
                                           titles_of([i.video_id for i in report.items]))
            return data
        if push_snap.status != "idle":
            return data
        cards = changes_data()
        ready = [c for c in cards if c.ready]
        data["ready"] = ready
        data["sends_translations"] = any(c.sends_translations for c in ready)
        data["sends_language"] = any(c.sends_language for c in ready)
        data["waiting"] = len(cards) - len(ready)
        if ready:
            with Store(config.catalog_db) as store:
                preview = QuotaGuard(store, config.daily_quota_budget).preview(
                    push.push_units(len(ready)))
            data["plan"] = {"lines": plan_lines(preview.lines), "total": preview.total,
                            "used": preview.used, "remaining": preview.remaining,
                            "fits": preview.fits, "budget": config.daily_quota_budget}
            data["has_write"] = state.write_credentials() is not None
        return data

    @app.get("/changes/push")
    def push_page(request: Request) -> Response:
        return render(request, "push.html", push_page_data())

    @app.post("/changes/push")
    async def start_push(request: Request) -> Response:
        form = await read_form(request)
        back = see_other("/changes/push")
        if form.get("confirm") != "yes":
            flash(request, "error", "Tick the box to confirm that you approve publishing these "
                  "changes. Nothing was sent to YouTube.")
            return back
        approved = {str(k)[3:]: str(v) for k, v in form.multi_items()
                    if str(k).startswith("fp_") and len(str(k)) > 3}
        if not approved:
            flash(request, "error", "There is nothing to push. Approve a draft first.")
            return back
        creds = await run_in_threadpool(state.write_credentials)
        if creds is None:
            flash(request, "error", "Yezatas does not have permission to edit your videos yet. "
                  "Allow it first. Nothing was sent to YouTube.")
            return back
        if not state.start_push(approved, creds):
            flash(request, "error", "A push is already running.")
        return back

    @app.post("/changes/push/done")
    async def dismiss_push(request: Request) -> Response:
        await read_form(request)
        state.dismiss_push_result()
        return see_other("/changes")

    @app.post("/changes/allow-edits")
    async def allow_edits(request: Request) -> Response:
        await read_form(request)
        client, _ = read_saved_client(config)
        if client is None:
            flash(request, "error", "Add the OAuth client file first, in Setup (step 5).")
        elif not state.start_write_sign_in():
            flash(request, "error", "A sign-in is already waiting for you in the browser.")
        return see_other("/changes/push")

    @app.post("/changes/allow-edits/cancel")
    async def cancel_allow_edits(request: Request) -> Response:
        await read_form(request)
        state.write_job.cancel("Sign-in cancelled.")
        return see_other("/changes/push")

    # --- history ----------------------------------------------------------------------------
    @app.get("/history")
    def history_page(request: Request) -> Response:
        with Store(config.catalog_db) as store:
            records = store.list_pushes(limit=HISTORY_LIMIT)
            drafts = {row.video_id for row in store.list_drafts()}
            titles = {}
            for record in records:
                video = store.get_video(record.video_id)
                if video is not None:
                    titles[record.video_id] = video.title
        return render(request, "history.html", {
            "active": "history", "rows": history_rows(records, titles, _local_time, drafts),
            "limit": HISTORY_LIMIT,
        })

    @app.post("/history/{push_id}/restore")
    async def restore_text(request: Request, push_id: int) -> Response:
        await read_form(request)

        def work() -> tuple[str, str, str]:
            with Store(config.catalog_db) as store:
                record = store.get_push(push_id)
                try:
                    result = edits.restore_draft(store, push_id)
                except edits.DraftError as exc:
                    return "error", str(exc), "/history"
            if result.discarded:
                return "info", "That text is already on YouTube; nothing to restore.", "/history"
            anchor = f"#d-{quote(record.video_id)}" if record is not None else ""
            return ("ok", f"Restore draft made from push #{push_id}. Nothing changes on YouTube "
                    "until you approve and push it.", "/changes" + anchor)

        kind, message, target = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other(target)

    # --- AI suggestions ---------------------------------------------------------------------
    @app.post("/videos/{video_id}/suggest")
    async def suggest_for_video(request: Request, video_id: str) -> Response:
        await read_form(request)
        back = see_other(f"/videos/{quote(video_id)}")

        def check() -> str:
            with Store(config.catalog_db) as store:
                if store.get_video(video_id) is None:
                    return "missing"
                return "draft" if store.get_draft(video_id) is not None else ""

        problem = await run_in_threadpool(check)
        if problem == "missing":
            return video_not_found(request)
        if not agent.key_source(config):
            flash(request, "error", "Add a Meta API key first. Nothing was sent to Meta.", "ai")
            return back
        if problem == "draft":
            flash(request, "error", "This video already has a draft. Push or discard it first.")
            return back
        if not state.start_suggest([video_id], False, request.state.session_id):
            flash(request, "error", "A suggestion is already running. Wait for it to finish.")
        return back

    def fix_next_ids(count: int) -> list[str]:
        """The top `count` public videos to fix (the Fix next order) that have no draft."""
        data = catalog_data()
        if data is None:
            return []
        have = draft_ids()
        rows = [e for e in data["entries"]
                if e.video.privacy_status == "public" and e.card.failed
                and e.video.video_id not in have]
        rows.sort(key=lambda e: (-e.priority, -e.card.missing, e.video.video_id))
        return [e.video.video_id for e in rows[:count]]

    @app.post("/suggest/fix-next")
    async def suggest_fix_next(request: Request) -> Response:
        form = await read_form(request)
        back = see_other("/")
        raw = str(form.get("count") or "")
        count = int(raw) if raw.isdigit() and int(raw) in FIX_NEXT_COUNTS else 0
        if not count:
            flash(request, "error", "Choose 1, 3, 5 or 10 videos.")
            return back
        if not agent.key_source(config):
            flash(request, "error", "Add a Meta API key first. Nothing was sent to Meta.", "ai")
            return back
        ids = await run_in_threadpool(fix_next_ids, count)
        if not ids:
            flash(request, "info", "No public video needs a fix, or the ones that do already "
                  "have a draft.")
            return back
        if not state.start_suggest(ids, True, request.state.session_id):
            flash(request, "error", "A suggestion is already running. Wait for it to finish.")
        return back

    @app.post("/suggest/done")
    async def dismiss_suggest(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_suggest_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}" if video_id else "/")

    @app.get("/agent")
    def agent_page(request: Request) -> Response:
        with Store(config.catalog_db) as store:
            runs = store.list_agent_runs(limit=AGENT_LOG_LIMIT)
            titles = {}
            for run in runs:
                video = store.get_video(run.video_id)
                if video is not None:
                    titles[run.video_id] = video.title
        data = catalog_data()
        check_titles = check_title_map(data["entries"]) if data else {}
        snap = state.suggest_job.snapshot()
        return render(request, "agent.html", {
            "active": "agent", "limit": AGENT_LOG_LIMIT, "model": config.model,
            "rows": [run_view(r, titles, check_titles, _local_time, interrupted=not snap.running)
                     for r in runs],
        })

    # --- the Meta API key (optional, for the suggestions) --------------------------------------
    @app.post("/setup/model-key")
    async def save_model_key(request: Request) -> Response:
        form = await read_form(request)
        key = str(form.get("api_key") or "").strip()
        if agent.key_source(config) == "environment":
            flash(request, "error", f"The key comes from the {agent.KEY_ENV} environment "
                  "variable, so nothing was saved.", "ai")
        elif not key:
            flash(request, "error", "Paste your key first.", "ai")
        elif len(key) > MAX_KEY_CHARS or any(c.isspace() for c in key):
            flash(request, "error", "That does not look like a key: it must be one line with no "
                  "spaces. Nothing was saved.", "ai")
        else:
            try:
                await run_in_threadpool(agent.save_api_key, config, key)
            except (ValueError, OSError):
                flash(request, "error", "The key could not be saved. Check that "
                      f"{config.credentials_dir} is writable.", "ai")
            else:
                flash(request, "ok", "Saved the key on this computer. It is not shown again.", "ai")
        return back_to_setup("ai")

    @app.post("/setup/model-key/delete")
    async def delete_model_key(request: Request) -> Response:
        await read_form(request)
        removed = await run_in_threadpool(agent.delete_api_key, config)
        flash(request, "ok", "The saved key was deleted." if removed
              else "There was no saved key.", "ai")
        return back_to_setup("ai")

    # --- translations -----------------------------------------------------------------------
    @app.post("/videos/{video_id}/translate")
    async def translate_video_text(request: Request, video_id: str) -> Response:
        form = await read_form(request)
        back = see_other(f"/videos/{quote(video_id)}#translations")

        def exists() -> bool:
            with Store(config.catalog_db) as store:
                return store.get_video(video_id) is not None

        if not await run_in_threadpool(exists):
            return video_not_found(request)
        source = str(form.get("source_language") or "")
        picked = {str(v) for v in form.getlist("targets")}
        if not source:
            flash(request, "error", "Choose the language of this video first. Yezatas does not "
                  "guess it. Nothing was sent.")
        elif source not in translate.LANGUAGES:
            flash(request, "error", "Yezatas cannot translate from that language. Choose another "
                  "one. Nothing was sent.")
        elif not picked:
            flash(request, "error", "Pick at least one language to translate into. Nothing was "
                  "sent.")
        elif not picked <= set(translate.LANGUAGES):
            flash(request, "error", "One of the chosen languages is not available. Nothing was "
                  "sent.")
        elif source in picked:
            flash(request, "error", "A video cannot be translated into its own language. Untick "
                  f"{translate.LANGUAGES[source]} and try again. Nothing was sent.")
        else:
            translator, _ = await run_in_threadpool(tools_or_none)
            if translator is None:
                flash(request, "error", "There is no translation key yet. Add a Baseten key or a "
                      "Meta API key. Nothing was sent.", "translate")
            elif not state.start_translate(
                    video_id, source, [c for c in translate.LANGUAGES if c in picked],
                    request.state.session_id):
                flash(request, "error", "A translation is already running. Wait for it to finish.")
        return back

    @app.post("/translate/done")
    async def dismiss_translate(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_translate_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}#translations" if video_id else "/")

    @app.post("/videos/{video_id}/translations/add")
    async def add_video_translations(request: Request, video_id: str) -> Response:
        form = await read_form(request)
        back = see_other(f"/videos/{quote(video_id)}#translations")
        chosen = list(dict.fromkeys(str(v) for v in form.getlist("languages")))

        def work() -> tuple[str, str]:
            with Store(config.catalog_db) as store:
                video = store.get_video(video_id)
                if video is None:
                    return "missing", ""
                if not chosen:
                    return "error", "Tick at least one translation to add."
                row = store.get_draft(video_id)
                records = store.latest_translations(video_id)
                current = translate.source_hash(
                    *((row.title, row.description) if row is not None
                      else (video.title, video.description)))
                pool = list(row.localizations if row is not None and row.localizations is not None
                            else edits.plain_localizations(video.localizations))
                picked: dict[str, tuple[str, str]] = {}
                sources: set[str] = set()
                for language in chosen:
                    record = records.get(language)
                    name = translate.LANGUAGES.get(language, language)
                    if record is None:
                        return "error", f"There is no {name} translation to add."
                    if record.status == "failed":
                        return "error", f"The {name} translation failed, so it cannot be added."
                    if record.source_hash != current:
                        return "error", (f"The {name} translation was made from older text. "
                                         "Translate it again first.")
                    sources.add(record.source_language)
                    picked[translate.youtube_key(language, pool)] = (record.title, record.description)
                if len(sources) != 1:
                    return "error", ("These translations were made from different source "
                                     "languages. Add them one source language at a time.")
                try:
                    edits.add_translations(store, video_id, sources.pop(), picked)
                except (edits.DraftError, ValueError) as exc:
                    return "error", str(exc)
                count = len(picked)
                return "ok", (f"Added {count} translation{'' if count == 1 else 's'} to the "
                              "draft. Review them on the Changes page.")

        kind, message = await run_in_threadpool(work)
        if kind == "missing":
            return video_not_found(request)
        flash(request, kind, message)
        return back

    @app.post("/videos/{video_id}/translations/{language}/remove")
    async def remove_video_translation(request: Request, video_id: str, language: str) -> Response:
        await read_form(request)

        def work() -> tuple[str, str]:
            with Store(config.catalog_db) as store:
                try:
                    edits.remove_translation(store, video_id, language)
                except edits.DraftError as exc:
                    return "error", str(exc)
            return "ok", (f"Removed the {lang_name(language)} translation from the draft. "
                          "Nothing changes on YouTube until you approve and push.")

        kind, message = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other(f"/videos/{quote(video_id)}#translations")

    # --- transcripts, chapters and subtitles --------------------------------------------------
    @app.post("/videos/{video_id}/transcribe")
    async def transcribe_video_file(request: Request, video_id: str) -> Response:
        form = await read_form(request)
        back = see_other(f"/videos/{quote(video_id)}#transcript")

        def exists() -> bool:
            with Store(config.catalog_db) as store:
                return store.get_video(video_id) is not None

        if not await run_in_threadpool(exists):
            return video_not_found(request)
        raw_path = str(form.get("path") or "")
        language = str(form.get("language") or "")
        words, problem = parse_keywords(str(form.get("keywords") or ""))
        media: Path | None = None
        if len(raw_path) > MAX_PATH_CHARS:
            problem = "That path is too long to be a file path. Nothing was sent."
        elif not problem:
            try:
                media = await run_in_threadpool(transcribe_mod.check_media_path, raw_path)
            except transcribe_mod.TranscribeError as exc:
                problem = f"{exc} Nothing was sent."
        if problem:
            flash(request, "error", problem)
        elif language not in transcribe_mod.ASR_LANGUAGES:
            flash(request, "error", "Choose the language that is spoken in the video. Lithuanian "
                  "and some other languages are not supported. Nothing was sent.")
        elif await run_in_threadpool(transcribe_mod.find_ffmpeg) is None:
            flash(request, "error", transcribe_mod.NO_FFMPEG_MESSAGE)
        elif media is not None and not state.start_transcribe(video_id, media, language, words):
            flash(request, "error", "A transcription is already running. Wait for it to finish.")
        return back

    @app.post("/transcribe/done")
    async def dismiss_transcribe(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_transcribe_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}#transcript" if video_id else "/")

    @app.post("/videos/{video_id}/chapters")
    async def make_video_chapters(request: Request, video_id: str) -> Response:
        await read_form(request)
        back = see_other(f"/videos/{quote(video_id)}#transcript")

        def check() -> str | None:
            with Store(config.catalog_db) as store:
                video = store.get_video(video_id)
                if video is None:
                    return None
                return chapters_blocker(video, store.get_draft(video_id),
                                        store.latest_transcript(video_id))

        blocked = await run_in_threadpool(check)
        if blocked is None:
            return video_not_found(request)
        if blocked:
            flash(request, "error", blocked)
        elif not state.start_chapters(video_id):
            flash(request, "error", "Chapters are already being made. Wait for them to finish.")
        return back

    @app.post("/chapters/done")
    async def dismiss_chapters(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_chapters_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}#transcript" if video_id else "/")

    @app.post("/videos/{video_id}/subtitles")
    async def translate_video_subtitles(request: Request, video_id: str) -> Response:
        form = await read_form(request)
        back = see_other(f"/videos/{quote(video_id)}#subtitles")

        def lookup() -> tuple[bool, str]:
            """(the video exists, the language of its newest transcript or "")."""
            with Store(config.catalog_db) as store:
                if store.get_video(video_id) is None:
                    return False, ""
                transcript = store.latest_transcript(video_id)
                return True, (transcript.language or "") if transcript is not None else ""

        found, source = await run_in_threadpool(lookup)
        if not found:
            return video_not_found(request)
        picked = {str(v) for v in form.getlist("targets")}
        if not source:
            flash(request, "error", "This video has no transcript with a language yet. Transcribe "
                  "it first. Nothing was sent.")
        elif not picked:
            flash(request, "error", "Pick at least one language for the subtitles. Nothing was "
                  "sent.")
        elif not picked <= set(translate.LANGUAGES):
            flash(request, "error", "One of the chosen languages is not available. Nothing was "
                  "sent.")
        elif any(translate.same_language(code, source) for code in picked):
            flash(request, "error", "Subtitles cannot be translated into the language of the "
                  f"transcript ({subtitle_translate.language_name(source)}). Untick it and try "
                  "again. Nothing was sent.")
        else:
            translator, _ = await run_in_threadpool(tools_or_none)
            if translator is None:
                flash(request, "error", "There is no translation key yet. Add a Baseten key or a "
                      "Meta API key. Nothing was sent.", "translate")
            elif not state.start_subtitles(
                    video_id, [c for c in translate.LANGUAGES if c in picked],
                    request.state.session_id):
                flash(request, "error", "Subtitles are already being translated. Wait for them "
                      "to finish.")
        return back

    @app.post("/subtitles/done")
    async def dismiss_subtitles(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_subtitles_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}#subtitles" if video_id else "/")

    @app.get("/videos/{video_id}/subtitles/{language}.srt")
    def subtitles_srt(request: Request, video_id: str, language: str) -> Response:
        missing = Response("There are no subtitles in that language for this video.",
                           status_code=404, media_type="text/plain")
        if language not in translate.LANGUAGES:
            return missing
        with Store(config.catalog_db) as store:
            transcript = store.latest_transcript(video_id)
            record = (store.latest_subtitle_translations(video_id, transcript.id).get(language)
                      if transcript is not None else None)
        if record is None or record.status == "failed":
            return missing
        return Response(
            subtitle_translate.srt_for(record), media_type="application/x-subrip; charset=utf-8",
            headers={"Content-Disposition":
                     f'attachment; filename="{video_id}.{language}.srt"'})

    @app.get("/videos/{video_id}/transcript.srt")
    def transcript_srt(request: Request, video_id: str) -> Response:
        with Store(config.catalog_db) as store:
            record = store.latest_transcript(video_id)
        if record is None:
            return Response("There is no transcript for this video.", status_code=404,
                            media_type="text/plain")
        return Response(
            to_srt(record.segments), media_type="application/x-subrip",
            headers={"Content-Disposition": f'attachment; filename="{video_id}.srt"'})

    # --- the translation key (optional, Baseten; the Meta key is the fallback) -------------------
    @app.post("/setup/translator-key")
    async def save_translator_key(request: Request) -> Response:
        form = await read_form(request)
        key = str(form.get("api_key") or "").strip()
        if translate.translator_key_source(config) == "environment":
            flash(request, "error", f"The key comes from the {translate.TRANSLATOR_KEY_ENV} "
                  "environment variable, so nothing was saved.", "translate")
        elif not key:
            flash(request, "error", "Paste your key first.", "translate")
        elif len(key) > MAX_KEY_CHARS or any(c.isspace() for c in key):
            flash(request, "error", "That does not look like a key: it must be one line with no "
                  "spaces. Nothing was saved.", "translate")
        else:
            try:
                await run_in_threadpool(translate.save_translator_key, config, key)
            except (ValueError, OSError):
                flash(request, "error", "The key could not be saved. Check that "
                      f"{config.credentials_dir} is writable.", "translate")
            else:
                flash(request, "ok", "Saved the key on this computer. It is not shown again.",
                      "translate")
        return back_to_setup("translate")

    @app.post("/setup/translator-key/delete")
    async def delete_translator_key(request: Request) -> Response:
        await read_form(request)
        removed = await run_in_threadpool(translate.delete_translator_key, config)
        flash(request, "ok", "The saved key was deleted." if removed
              else "There was no saved key.", "translate")
        return back_to_setup("translate")

    return app
