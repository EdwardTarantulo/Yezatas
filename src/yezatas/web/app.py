"""The Yezatas web app: the setup screen, the SEO dashboard, the content table, video pages and
the edit, review and push screens.

Pages only read from YouTube through the catalog pull. The one write is `POST /changes/push`,
which sends drafts the creator approved and ticked a confirmation box for. Background jobs open
their own Store, because a SQLite connection must not cross threads.
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

from .. import auth, edits, push
from ..catalog import NoChannelError, PullResult, pull_catalog
from ..config import Config, ensure_dirs
from ..quota import QuotaGuard
from ..seo import score_catalog
from ..store import Store
from ..youtube import GoogleYouTubeReader, GoogleYouTubeWriter, YouTubeReader, YouTubeWriter
from .clientfile import (MAX_CLIENT_FILE_BYTES, ClientFileError, ClientInfo,
                         parse_client_file, read_saved_client, save_client_file)
from .guidance import NeedsSignIn, map_api_error, map_sign_in_error
from .jobs import Job, JobSnapshot, Report
from .security import CSRF_FIELD, Sessions, install_security, load_session_key, verify_csrf
from .steps import CheckState, compute_steps, estimate_pull_units
from ..validate import DESCRIPTION_BYTE_LIMIT, TITLE_LIMIT
from .views import (Entry, build_listing, change_cards, dashboard_model, format_count,
                    format_date, group_checks, history_rows, make_entries, plan_lines,
                    push_results, thumb_url)

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

HERE = Path(__file__).parent

ReaderFactory = Callable[[Any], YouTubeReader]
WriterFactory = Callable[[Any], YouTubeWriter]
CredentialsLoader = Callable[[Config], Any]


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
    ) -> None:
        self.config = config
        self._sign_in = sign_in
        self._reader_factory = reader_factory
        self._credentials_loader = credentials_loader
        self._write_sign_in = write_sign_in
        self._write_credentials_loader = write_credentials_loader
        self._writer_factory = writer_factory
        self.signin_job = Job(
            map_sign_in_error, sign_in_timeout, SIGN_IN_TIMEOUT_MESSAGE, clock
        )
        self.pull_job = Job(map_api_error)
        self.write_job = Job(map_sign_in_error, sign_in_timeout, SIGN_IN_TIMEOUT_MESSAGE, clock)
        self.push_job = Job(map_api_error)
        self._push_seen: JobSnapshot | None = None  # the finished push the creator dismissed
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
) -> FastAPI:
    """Build the app. `app.state.start_url` is the one-time link that signs the browser in."""
    ensure_dirs(config)
    token = token or secrets.token_urlsafe(32)
    start_url = f"http://127.0.0.1:{port}/?t={token}"
    state = WebState(config, sign_in, reader_factory, credentials_loader, clock, sign_in_timeout,
                     write_sign_in, write_credentials_loader, writer_factory)
    sessions = Sessions(token, port, load_session_key(config.credentials_dir / "session.key"))
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    # A missing value is a bug: fail loudly instead of rendering "" (an empty link, say).
    templates.env.undefined = StrictUndefined

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
            "signed_in": signed_in, "signin": signin, "pull": pull, "check": check,
            "estimate": estimate,
            "latest_pull": ({"videos": latest[0], "when": _local_time(latest[1])}
                            if latest else None),
            "pull_result": pull.result if isinstance(pull.result, PullResult) else None,
            "quota": {"used": used, "remaining": remaining, "budget": config.daily_quota_budget},
            "home": str(config.home),
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

    def flash(request: Request, kind: str, text: str) -> None:
        state.add_flash(request.state.session_id, kind, text)

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
        card = change_cards([(row, video)])[0] if row is not None else None
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
        })

    @app.get("/setup")
    def setup(request: Request) -> Response:
        data = page_data()
        data["flash"] = state.pop_flash(request.state.session_id)
        return render(request, "setup.html", {**data, "active": "setup"})

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        check = state.check
        return JSONResponse({
            "signin": _job_json(state.signin_job.snapshot()),
            "pull": _job_json(state.pull_job.snapshot()),
            "write": _job_json(state.write_job.snapshot()),
            "push": _job_json(state.push_job.snapshot()),
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
            "report": None,
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

    return app
