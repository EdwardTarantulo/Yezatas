"""The Yezatas web app: the setup screen, the SEO dashboard, the content table, video pages and
the edit, review and push screens, plus the AI suggestions and the agent log.

Pages only read from YouTube through the catalog pull. The one write is `POST /changes/push`,
which sends drafts the creator approved and ticked a confirmation box for. A suggestion only ever
makes a draft. A confirmed thumbnail and a confirmed comment reply are posted from their own
confirmation pages. Background jobs open their own Store, because a SQLite connection must not cross
threads.
"""

from __future__ import annotations

import importlib.util
import json
import os
import secrets
import shutil
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Sequence
from urllib.parse import quote, urlencode

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from googleapiclient.errors import HttpError
from jinja2 import StrictUndefined
from starlette.templating import Jinja2Templates

from .. import agent, auth, bulk, edits, push, subtitle_translate
from .. import chapters as chapters_mod
from .. import adcheck as adcheck_mod
from .. import shorts as shorts_mod
from .. import copilot as copilot_mod
from .. import strategy as strategy_mod
from .. import competitors as competitors_mod
from .. import searchgaps as searchgaps_mod
from .. import channelstats as channelstats_mod
from .. import videostats as videostats_mod
from .. import categorycheck as categorycheck_mod
from .. import pinnedcomment as pinnedcomment_mod
from .. import links as links_mod
from .. import outliers as outliers_mod
from .. import packaging as packaging_mod
from .. import thumbswap as thumbswap_mod
from .. import transcribe as transcribe_mod
from .. import captions as captions_mod
from .. import soundtags as soundtags_mod
from .. import primaryai, translate
from ..catalog import NoChannelError, PullResult, pull_catalog
from ..config import PRIMARY_AI_CHOICES, Config, ensure_dirs
from .. import affiliate as affiliate_mod
from .. import bulkagent as bulkagent_mod
from .. import comments as comments_mod
from .. import customchecks as customchecks_mod
from .. import glossary as glossary_mod
from .. import replytemplates as replytemplates_mod
from .. import postreply as postreply_mod
from ..quota import QuotaExceeded, QuotaGuard
from ..seo import TAGS_CHAR_LIMIT, CategoryJudgment, parse_duration, score_catalog, tags_char_count, video_category_id
from .videofacts import build_facts
from .views import stale_youtube_keys
from .video_analytics_view import build_video_analytics
from .category_view import build_category_view
from .pinned_view import build_pinned_view
from .nav_counts import build_nav_counts
from ..store import AdCheck, CategoryCheck, OutlierReport, PinnedComment, ShortsFind, Store, ThumbnailVariant
from ..subtitles import to_srt
from ..youtube import (CaptionReader, CommentReader, CommentWriter, GoogleCaptionReader, GoogleCommentReader,
                       GoogleCommentWriter, GoogleYouTubeReader, GoogleYouTubeWriter, YouTubeReader, YouTubeWriter,
                       api_error_info)
from . import upload as upload_mod
from .clientfile import (MAX_CLIENT_FILE_BYTES, ClientFileError, ClientInfo,
                         parse_client_file, read_saved_client, save_client_file)
from .guidance import NeedsSignIn, map_api_error, map_sign_in_error
from .competitors_view import build_competitor_cards, build_insight_view
from .searchgaps_view import build_gap_rows, build_ranking_view
from .analytics_view import build_analytics
from .today_view import build_today
from .thumbswap_view import build_swap_rows
from .jobs import Job, JobSnapshot, Report
from .security import CSRF_FIELD, Sessions, install_security, load_session_key, verify_csrf
from .steps import CheckState, compute_steps, estimate_pull_units
from ..validate import DESCRIPTION_BYTE_LIMIT, TITLE_LIMIT
from .views import (BULK_EXAMPLES, STARTER_QUESTIONS, build_agent_session, build_posted_rows, Entry, PreviewChoice, RunView, SuggestBatch, SuggestItem,
                    affiliate_apply_message, build_adcheck_view, build_shorts_view, build_affiliate_preview,
                    build_agent_review, build_bulk_preview, build_channel_comments_view, build_checks_view, build_comments_view, build_copilot_turns, build_glossary_rows, build_links_view, build_listing,
                    build_outlier_view, build_outdated_view, build_packaging_view, build_preview, build_strategy_view, build_subtitles_view,
                    build_transcript_view, build_translate_view, bulk_apply_message, change_cards,
                    chapters_blocker, dashboard_model, format_cost, format_count, format_date,
                    group_checks, history_groups, lang_name, links_check_line, make_entries,
                    outlier_count, outlier_line, parse_affiliate_rule_form, parse_bulk_form,
                    parse_glossary_form, parse_rule_proposal, plan_lines, preview_card,
                    pull_moment, push_results, run_restore_summary, run_view, stops_batch,
                    thumb_url, top_other_videos)

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
CHAPTERS_TIMEOUT_SECONDS = 20 * 60.0
CHAPTERS_TIMEOUT_MESSAGE = (
    "Making the chapters took more than 20 minutes, so Yezatas stopped waiting. The Agent log "
    "shows what finished."
)
SUBTITLES_TIMEOUT_SECONDS = 60 * 60.0
SUBTITLES_TIMEOUT_MESSAGE = (
    "The subtitles took more than 60 minutes, so Yezatas stopped waiting. Languages that "
    "finished are kept below."
)
PACKAGING_TIMEOUT_SECONDS = 20 * 60.0
PACKAGING_TIMEOUT_MESSAGE = (
    "The packaging check took more than 20 minutes, so Yezatas stopped waiting. Try again."
)
VARIANTS_TIMEOUT_SECONDS = 20 * 60.0
VARIANTS_TIMEOUT_MESSAGE = (
    "Making the variants took more than 20 minutes, so Yezatas stopped waiting. Variants that "
    "finished are kept on this page."
)
OUTLIERS_TIMEOUT_SECONDS = 20 * 60.0
OUTLIERS_TIMEOUT_MESSAGE = (
    "Explaining the outliers took more than 20 minutes, so Yezatas stopped waiting. Try again."
)
STRATEGY_TIMEOUT_SECONDS = 20 * 60.0
COMPETITORS_TIMEOUT_SECONDS = 20 * 60.0
COMPETITORS_TIMEOUT_MESSAGE = ("Reading or explaining the other channels took more than 20 minutes, so "
                               "Yezatas stopped waiting. Try again.")
SEARCHGAPS_TIMEOUT_SECONDS = 20 * 60.0
SEARCHGAPS_TIMEOUT_MESSAGE = ("Reading or ranking the search terms took more than 20 minutes, so "
                              "Yezatas stopped waiting. Try again.")
ANALYTICS_DISABLED = ("The YouTube Analytics API is not enabled in your Google Cloud project. Enable "
                      "it (APIs & Services, Library, YouTube Analytics API), wait a minute, then try again.")
ANALYTICS_SIGN_IN = ("Google did not accept the search-terms sign-in. Click Connect search terms and "
                     "sign in again.")
CHANNEL_STATS_TIMEOUT_SECONDS = 5 * 60.0
VIDEO_STATS_AUTO_HOURS = 12  # the video page reads its Analytics on opening when the stored read is older
CHANNEL_STATS_TIMEOUT_MESSAGE = ("Reading the channel's numbers took more than 5 minutes, so Yezatas "
                                 "stopped waiting. Try again.")
ANALYTICS_SIGN_IN_CHANNEL = ("Google did not accept the YouTube Analytics sign-in. Click Connect "
                             "Analytics and sign in again.")
STRATEGY_TIMEOUT_MESSAGE = (
    "The strategy report took more than 20 minutes, so Yezatas stopped waiting. Try again."
)
COPILOT_TIMEOUT_SECONDS = 5 * 60.0
COPILOT_TIMEOUT_MESSAGE = (
    "The answer took more than 5 minutes, so Yezatas stopped waiting. Try a shorter question."
)
BULK_AGENT_TIMEOUT_SECONDS = 15 * 60.0
BULK_AGENT_TIMEOUT_MESSAGE = (
    "The bulk edit agent took more than 15 minutes, so Yezatas stopped waiting. What it had staged is "
    "kept: reply to continue, or review it."
)
LINKS_TIMEOUT_SECONDS = 10 * 60.0
LINKS_TIMEOUT_MESSAGE = (
    "The link check took more than 10 minutes, so Yezatas stopped waiting. The links it "
    "finished are kept on the Links page."
)
SHORTS_TIMEOUT_SECONDS = 3 * 60.0
SHORTS_TIMEOUT_MESSAGE = (
    "Finding Shorts took more than 3 minutes, so Yezatas stopped waiting. Try again.")
NO_SHORTS_KEY = (
    "There is no Baseten or Meta API key yet, so the Shorts finder cannot run. Add one in Configuration.")
ADCHECK_TIMEOUT_SECONDS = 3 * 60.0
ADCHECK_TIMEOUT_MESSAGE = (
    "The advertiser-friendly check took more than 3 minutes, so Yezatas stopped waiting. Try again."
)
CATEGORY_TIMEOUT_SECONDS = 2 * 60.0
CATEGORY_TIMEOUT_MESSAGE = (
    "Checking the category took more than 2 minutes, so Yezatas stopped waiting. Try again.")
PINNED_TIMEOUT_SECONDS = 2 * 60.0
PINNED_TIMEOUT_MESSAGE = (
    "Writing the pinned comment took more than 2 minutes, so Yezatas stopped waiting. Try again.")
COMMENTS_TIMEOUT_SECONDS = 120.0
COMMENTS_TIMEOUT_MESSAGE = (
    "Reading the comments took more than 2 minutes, so Yezatas stopped waiting. Try again."
)
ASSIST_TIMEOUT_SECONDS = 180.0
ASSIST_TIMEOUT_MESSAGE = (
    "Sorting the comments took more than 3 minutes, so Yezatas stopped waiting. Try again."
)
CHANNEL_FETCH_TIMEOUT_SECONDS = 10 * 60.0
CHANNEL_FETCH_TIMEOUT_MESSAGE = (
    "Reading the comments took more than 10 minutes, so Yezatas stopped waiting. Try again."
)
CHANNEL_SORT_TIMEOUT_SECONDS = 30 * 60.0
CHANNEL_SORT_TIMEOUT_MESSAGE = (
    "Sorting the comments took more than 30 minutes, so Yezatas stopped waiting. Try again."
)
CHANNEL_COMMENTS_PATH = "/comments"
NO_PACKAGING_KEY = "There is no Meta API key. Add one in Configuration. Nothing was sent."
MAX_ID_DIGITS = 12  # a variant id in a URL; longer is never a real one
MAX_PATH_CHARS = 1000
UPLOADS_DIR = "tmp/uploads"  # under the Yezatas home: the video files the browser handed over
LOCAL_NO_KEY_MESSAGE = (
    "Transcribing speech from a file needs Meta's API. Primary AI is Local and no Meta key is "
    "saved. Add a key in Configuration, or use the YouTube captions.")
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
# (video id, media path, report) -> sound captions (an unstored record); free, runs on this computer
SoundTagsFn = Callable[[str, str, Report], "soundtags_mod.SoundTagResult"]
CaptionReaderFactory = Callable[[Any], CaptionReader]
ChaptersFn = Callable[[Store, str, Report], "chapters_mod.ChaptersResult"]
# (store, question, chat id, the time of the views, report) -> the id of the new copilot turn
CopilotFn = Callable[[Store, str, int, datetime, Report], int]
# (store, video id, the time now, report) -> the stored check (a failed one included)
AdcheckFn = Callable[[Store, str, datetime, Report], "AdCheck"]
# (store, video id, the time now, report) -> the stored category check (a failed one included)
CategoryFn = Callable[[Store, str, datetime, Report], "CategoryCheck"]
# (store, video id, the time now, report) -> the stored pinned comment with its newest suggestion
PinnedFn = Callable[[Store, str, datetime, Report], "PinnedComment"]
# (store, video id, the time now, report) -> the stored Shorts finder run (a failed one included)
ShortsFn = Callable[[Store, str, datetime, Report], "ShortsFind"]
# The three packaging clients (None without a Meta API key); the real one reads the key from the config.
PackagingToolsFactory = Callable[[Config], "packaging_mod.PackagingTools | None"]
CommentReaderFactory = Callable[[Any], CommentReader]
CommentWriterFactory = Callable[[Any], CommentWriter]
# (store, video id, the time now, report) -> what the sort did
CommentsAssistFn = Callable[[Store, str, datetime, Report], "comments_mod.AssistResult"]
# (the creator's words) -> the proposed fixed rule
RuleMakerFn = Callable[[str], "customchecks_mod.RuleDraft"]
# (store, home, the time of the views, report) -> the saved report
StrategyFn = Callable[[Store, Path, datetime, Report], "strategy_mod.StrategyReport"]
# (credentials) -> a reader of public channels
CompetitorReaderFactory = Callable[[Any], "competitors_mod.PublicChannelReader"]
# (the followed channels, report) -> the saved explanation
CompetitorExplainFn = Callable[[list["competitors_mod.CompetitorChannel"], Report],
                               "competitors_mod.CompetitorInsight"]
AnalyticsReaderFactory = Callable[[Any], "searchgaps_mod.AnalyticsReader"]
ChannelReaderFactory = Callable[[Any], "channelstats_mod.ChannelReader"]
GapRankFn = Callable[["searchgaps_mod.SearchTerms", list["searchgaps_mod.SearchGap"], Report],
                     "searchgaps_mod.GapRanking"]
# (store, the session, the creator's words, the time of the views, report) -> None: the staged changes
# stay in the session, in memory
BulkAgentFn = Callable[[Store, "bulkagent_mod.BulkSession", str, datetime, Report], None]


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


def _num(value: int) -> str:
    """A whole number with thousands separators: 9979 -> "9,979"."""
    return f"{value:,}"


def sign_in_with_write_access(config: Config) -> Any:
    """The default write sign-in: the same browser flow as setup, asking to edit videos."""
    scopes = (auth.WRITE, auth.ANALYTICS) if auth.can_read_analytics(config) else (auth.WRITE,)
    return auth.sign_in(config, scopes=scopes)


def load_write_credentials(config: Config) -> Any:
    return auth.load_credentials(config, scopes=(auth.WRITE,))


def sign_in_with_analytics_access(config: Config) -> Any:
    """The search-terms sign-in: the same browser flow, adding read access to YouTube Analytics and
    keeping the edit access already given."""
    return auth.sign_in(config, scopes=auth.analytics_scopes(config))


def load_analytics_credentials(config: Config) -> Any:
    return auth.load_credentials(config, scopes=(auth.ANALYTICS,))


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


def map_packaging_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, agent.ModelError):
        return exc.kind, str(exc)
    if isinstance(exc, packaging_mod.PackagingError):
        return "packaging", str(exc)
    return "packaging", f"The packaging check stopped unexpectedly ({exc.__class__.__name__})."


def has_meta_key(config: Config) -> bool:
    """Whether the text features can start: Primary AI is Local, or a Meta API key is set
    (environment or saved). Reads the key only; builds no client."""
    return primaryai.text_ai_ready(config)


def map_outliers_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, agent.ModelError):
        return exc.kind, str(exc)
    if type(exc) is ValueError:  # the plain sentences of start_outliers' own checks
        return "outliers", str(exc)
    return "outliers", f"Explaining the outliers stopped unexpectedly ({exc.__class__.__name__})."


def map_variants_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, agent.ModelError):
        return exc.kind, str(exc)
    if isinstance(exc, packaging_mod.PackagingError):
        return "variants", str(exc)
    return "variants", f"Making the variants stopped unexpectedly ({exc.__class__.__name__})."


def map_transcribe_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, captions_mod.CaptionsError):  # its sentences are plain English already
        return f"captions_{exc.kind}", str(exc)
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


def map_copilot_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, agent.ModelError):
        return exc.kind, str(exc)
    if isinstance(exc, copilot_mod.CopilotError) or type(exc) is ValueError:
        return "copilot", str(exc)  # the plain sentences of the copilot's own checks
    return "copilot", f"The answer stopped unexpectedly ({exc.__class__.__name__})."


def map_bulk_agent_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, agent.ModelError):
        return exc.kind, str(exc)
    if isinstance(exc, bulkagent_mod.BulkAgentError) or type(exc) is ValueError:
        return "bulk_agent", str(exc)
    return "bulk_agent", f"The bulk edit agent stopped unexpectedly ({exc.__class__.__name__})."


def map_links_error(exc: BaseException) -> tuple[str, str]:
    if type(exc) is ValueError:  # the plain sentence of _run_links' own check
        return "links", str(exc)
    return "links", f"The link check stopped unexpectedly ({exc.__class__.__name__})."


def map_strategy_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, agent.ModelError):
        return exc.kind, str(exc)
    if isinstance(exc, strategy_mod.StrategyError) or type(exc) is ValueError:
        return "strategy", str(exc)
    return "strategy", f"The strategy report stopped unexpectedly ({exc.__class__.__name__})."


def make_default_strategy(config: Config) -> StrategyFn:
    """The real strategy report: Muse Spark 1.3 through Meta's Model API, one client per report."""

    def run(store: Store, home: Path, as_of: datetime, report: Report) -> "strategy_mod.StrategyReport":
        client = primaryai.text_client(config, model=strategy_mod.STRATEGY_MODEL)
        made = strategy_mod.make_report(store, client, as_of=as_of, report=report)
        strategy_mod.save_report(home, made)
        return made

    return run


def map_thumb_error(exc: BaseException) -> tuple[str, str]:
    """Guidance for a failed thumbnail swap."""
    if isinstance(exc, thumbswap_mod.ThumbSwapError):
        return "thumb", str(exc)
    if isinstance(exc, packaging_mod.PackagingError):
        return (
            "thumb",
            "Yezatas could not save the current thumbnail, so nothing was changed on YouTube. "
            + str(exc),
        )
    if isinstance(exc, HttpError):
        reason = api_error_info(exc).reason
        if reason == "forbidden":
            return (
                "thumb",
                "YouTube refused to set this thumbnail. Custom thumbnails need a verified channel "
                "(youtube.com/verify), and YouTube can refuse them for some videos. Nothing was "
                "changed.",
            )
        if reason == "uploadRateLimitExceeded":
            return (
                "thumb",
                "YouTube says too many thumbnails were set recently. Try again later. Nothing was "
                "changed.",
            )
    return map_api_error(exc)


def map_competitors_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, competitors_mod.CompetitorError):
        return "competitors", str(exc)
    if isinstance(exc, agent.ModelError):
        return exc.kind, str(exc)
    return map_api_error(exc)


def make_default_competitor_explain(config: Config) -> CompetitorExplainFn:
    def run(
        channels: list["competitors_mod.CompetitorChannel"],
        report: Report,
    ) -> "competitors_mod.CompetitorInsight":
        client = primaryai.text_client(config)
        report("Asking the model")
        made = competitors_mod.explain(channels, client)
        competitors_mod.save_insight(config.home, made)
        return made

    return run


def map_searchgaps_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, searchgaps_mod.SearchGapsError):
        return ("search_gaps", str(exc))
    if isinstance(exc, agent.ModelError):
        return (exc.kind, str(exc))
    kind, text = map_api_error(exc)
    if kind == "api_disabled":
        return (kind, ANALYTICS_DISABLED)
    if kind == "auth":
        return (kind, ANALYTICS_SIGN_IN)
    return (kind, text)


def map_channel_stats_error(exc: BaseException) -> tuple[str, str]:
    kind, text = map_api_error(exc)
    if kind == "api_disabled":
        return (kind, ANALYTICS_DISABLED)
    if kind == "auth":
        return (kind, ANALYTICS_SIGN_IN_CHANNEL)
    return (kind, text)


def make_default_gap_rank(config: Config) -> GapRankFn:
    def run(
        terms: "searchgaps_mod.SearchTerms",
        gaps: list["searchgaps_mod.SearchGap"],
        report: Report,
    ) -> "searchgaps_mod.GapRanking":
        client = primaryai.text_client(config)
        report("Asking the model")
        made = searchgaps_mod.rank_gaps(terms, gaps, client)
        searchgaps_mod.save_ranking(config.home, made)
        return made

    return run


def make_default_copilot(config: Config) -> CopilotFn:
    """The real copilot function: Muse Spark through Meta's Model API, one client per question."""

    def run(store: Store, question: str, chat_id: int, as_of: datetime, report: Report) -> int:
        client = primaryai.text_client(config)
        return copilot_mod.ask(store, client, question, chat_id=chat_id, as_of=as_of,
                               report=report)

    return run


def map_adcheck_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, agent.ModelError):
        return exc.kind, str(exc)
    if type(exc) is ValueError:  # the plain sentence of the check's own guard
        return "adcheck", str(exc)
    return "adcheck", f"The advertiser-friendly check stopped unexpectedly ({exc.__class__.__name__})."


def map_shorts_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, translate.TranslateError):
        return exc.kind, str(exc)
    if type(exc) is ValueError:
        return "shorts", str(exc)
    return "shorts", f"The Shorts finder stopped unexpectedly ({exc.__class__.__name__})."


def make_default_shorts(config: Config) -> ShortsFn:
    def run(store: Store, video_id: str, now: datetime, report: Report) -> "ShortsFind":
        client = translate.translator_from_config(config)
        if client is None:
            raise translate.TranslateError("auth", NO_SHORTS_KEY)
        return shorts_mod.find_shorts(store, client, video_id, now=now, report=report)

    return run


def make_default_adcheck(config: Config) -> AdcheckFn:
    """The real check: Muse Spark through Meta's Model API, one client per check."""

    def run(store: Store, video_id: str, now: datetime, report: Report) -> "AdCheck":
        client = primaryai.text_client(config)
        return adcheck_mod.check_video(store, client, video_id, now=now, report=report)

    return run


def map_category_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, agent.ModelError):
        return exc.kind, str(exc)
    if type(exc) is ValueError:  # the plain sentence of the check's own guard
        return "category", str(exc)
    return "category", f"The category check stopped unexpectedly ({exc.__class__.__name__})."


def make_default_category(config: Config) -> CategoryFn:
    """The real check: Muse Spark through Meta's Model API, one client per check."""

    def run(store: Store, video_id: str, now: datetime, report: Report) -> "CategoryCheck":
        client = primaryai.text_client(config)
        return categorycheck_mod.judge_category(store, client, video_id, now=now, report=report)

    return run


def map_pinned_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, agent.ModelError):
        return exc.kind, str(exc)
    if type(exc) is ValueError:
        return "pinned", str(exc)
    return "pinned", f"The pinned comment suggestion stopped unexpectedly ({exc.__class__.__name__})."


def make_default_pinned(config: Config) -> PinnedFn:
    """The real suggestion: Muse Spark through Meta's Model API, one client per call."""

    def run(store: Store, video_id: str, now: datetime, report: Report) -> "PinnedComment":
        client = primaryai.text_client(config)
        return pinnedcomment_mod.suggest_pinned(store, client, video_id, now=now, report=report)

    return run


def map_comments_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, comments_mod.CommentsError):
        return "comments", str(exc)
    if isinstance(exc, QuotaExceeded):
        return "comments", str(exc)
    if isinstance(exc, agent.ModelError):
        return exc.kind, str(exc)
    if type(exc) is ValueError:
        return "comments", str(exc)
    return "comments", f"Reading the comments stopped unexpectedly ({exc.__class__.__name__})."


def map_assist_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, agent.ModelError):
        return exc.kind, str(exc)
    if isinstance(exc, comments_mod.CommentsError):
        return "comments", str(exc)
    if isinstance(exc, QuotaExceeded):
        return "comments", str(exc)
    if type(exc) is ValueError:
        return "assist", str(exc)
    return "assist", f"Sorting the comments stopped unexpectedly ({exc.__class__.__name__})."


def comments_back(next_path: str) -> str | None:
    """The /comments page (with its filters) a reply form asked to come back to, or None."""
    if next_path == CHANNEL_COMMENTS_PATH:
        return next_path
    if next_path.startswith(CHANNEL_COMMENTS_PATH + "?") and len(next_path) <= MAX_PATH_CHARS \
            and all(33 <= ord(ch) < 127 and ch != "\\" for ch in next_path):
        return next_path
    return None


def make_default_comments_assist(config: Config) -> CommentsAssistFn:
    """The real sort: Muse Spark through Meta's Model API, one client per sort."""

    def run(store: Store, video_id: str, now: datetime, report: Report) -> "comments_mod.AssistResult":
        client = primaryai.text_client(config)
        return comments_mod.assist(store, client, video_id, now=now, report=report)

    return run


def make_default_rule_maker(config: Config) -> RuleMakerFn:
    """The real rule maker: Muse Spark through Meta's Model API, one client per ask."""

    def run(request: str) -> "customchecks_mod.RuleDraft":
        client = primaryai.text_client(config)
        return customchecks_mod.rule_from_request(client, request)

    return run


def make_default_bulk_agent(config: Config) -> BulkAgentFn:
    """The real bulk edit agent: the Primary AI's text model (Muse Spark through Meta's Model API by
    default), one client per turn."""

    def run(store: Store, session: "bulkagent_mod.BulkSession", text: str, as_of: datetime,
            report: Report) -> None:
        client = primaryai.text_client(config)
        bulkagent_mod.run(store, client, session, text, as_of=as_of, report=report)

    return run


def make_default_transcribe(config: Config) -> TranscribeFn:
    """The real transcribe function: Muse Voice Transcribe through Meta's Model API."""

    def run(store: Store, video_id: str, media_path: str, language: str, report: Report,
            keywords: Sequence[str] = ()) -> "transcribe_mod.TranscribeResult":
        key = agent.load_api_key(config)
        if not key:
            raise transcribe_mod.TranscribeError(
                "auth", "No Meta API key is set. Add one in Configuration. Nothing was sent.")
        client = transcribe_mod.MetaAsrClient(key)
        return transcribe_mod.transcribe_video(store, client, video_id, media_path, language,
                                               keywords=keywords, report=report)

    return run


def make_default_soundtags(config: Config) -> SoundTagsFn:
    """The real sound tagger: PANNs on this computer. Nothing is sent anywhere."""

    def run(video_id: str, media_path: str, report: Report) -> "soundtags_mod.SoundTagResult":
        return soundtags_mod.tag_video(video_id, media_path, report=report)

    return run


def make_default_chapters(config: Config) -> ChaptersFn:
    """The real chapters function: Muse Spark picks transcript segments, code makes the times."""

    def run(store: Store, video_id: str, report: Report) -> "chapters_mod.ChaptersResult":
        if not primaryai.text_ai_ready(config):
            raise agent.AgentError("No Meta API key is set. Add one in Configuration.")
        client = primaryai.text_client(config)
        return chapters_mod.suggest_chapters(store, client, video_id, report=report)

    return run


def remove_folder(folder: Path | None) -> None:
    """Delete a folder of uploaded files (a no-op for None); never raises."""
    if folder is not None:
        shutil.rmtree(folder, ignore_errors=True)


class _ChargedCaptionReader:
    """A caption reader that writes each call into the quota log before it makes the call."""

    def __init__(self, reader: CaptionReader, guard: QuotaGuard) -> None:
        self._reader = reader
        self._guard = guard

    def caption_tracks(self, video_id: str) -> dict[str, Any]:
        self._guard.charge("captions.list", note="read captions")
        return self._reader.caption_tracks(video_id)

    def caption_srt(self, caption_id: str) -> bytes | str:
        self._guard.charge("captions.download", note="read captions")
        return self._reader.caption_srt(caption_id)


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
        if not primaryai.text_ai_ready(config):
            raise agent.AgentError("No Meta API key is set. Add one in Configuration.")
        client = primaryai.text_client(config)
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


# The jobs the top bar follows on every page: (name, job attribute, video attribute, while running, when done,
# when failed, the video page row it links to). Channel-wide jobs have no video and link to their own page.
ACTIVITY: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    ("pull", "pull_job", "", "Pulling your videos", "Videos pulled", "Pulling stopped", "/"),
    ("push", "push_job", "", "Pushing to YouTube", "Pushed to YouTube", "Push stopped", "/changes"),
    ("suggest", "suggest_job", "", "Writing suggestions", "Suggestions ready", "Suggestions stopped", "#tool-suggest"),
    ("translate", "translate_job", "_translate_video", "Translating", "Translations ready", "Translation stopped", "#tool-translations"),
    ("subtitles", "subtitles_job", "_subtitles_video", "Translating subtitles", "Subtitles ready", "Subtitles stopped", "#tool-subtitles"),
    ("transcribe", "transcribe_job", "_transcribe_video", "Transcribing", "Transcript ready", "Transcribing stopped", "#tool-chapters"),
    ("chapters", "chapters_job", "_chapters_video", "Making chapters", "Chapters ready", "Chapters stopped", "#tool-chapters"),
    ("packaging", "packaging_job", "_packaging_video", "Checking the thumbnail", "Thumbnail check ready", "Thumbnail check stopped", "#tool-thumbnail"),
    ("variants", "variants_job", "_variants_video", "Making thumbnail options", "Thumbnail options ready", "Thumbnail options stopped", "#tool-thumbnail"),
    ("adcheck", "adcheck_job", "_adcheck_video", "Checking for advertisers", "Advertiser check ready", "Advertiser check stopped", "#tool-ad-check"),
    ("category", "category_job", "_category_video", "Checking the category", "Category check ready", "Category check stopped", "#detail-category"),
    ("pinned", "pinned_job", "_pinned_video", "Writing a pinned comment", "Pinned comment ready", "Pinned comment stopped", "#detail-pinned"),
    ("shorts", "shorts_job", "_shorts_video", "Finding Shorts", "Shorts ideas ready", "Shorts finder stopped", "#tool-shorts"),
    ("comments", "comments_job", "_comments_video", "Reading comments", "Comments read", "Reading comments stopped", "#tool-comments"),
    ("assist", "assist_job", "_assist_video", "Sorting comments", "Comments sorted", "Sorting stopped", "#tool-comments"),
    ("bulk_agent", "bulk_agent_job", "", "Changing videos", "Changes ready to review", "Bulk edit stopped", "/bulk"),
)


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
        packaging_tools: PackagingToolsFactory = packaging_mod.packaging_tools_from_config,
        packaging_fetch: Callable[[str], bytes] = packaging_mod.fetch_thumbnail,
        packaging_key: Callable[[Config], bool] | None = None,
        copilot_fn: CopilotFn | None = None,
        link_fetch: links_mod.Fetch | None = None,
        adcheck_fn: AdcheckFn | None = None,
        comment_reader: CommentReaderFactory | None = None,
        comments_assist: CommentsAssistFn | None = None,
        rule_maker: RuleMakerFn | None = None, bulk_agent: BulkAgentFn | None = None,
        strategy_fn: StrategyFn | None = None,
        shorts_fn: ShortsFn | None = None,
        analytics_sign_in: Callable[[Config], Any] | None = None,
        analytics_credentials_loader: CredentialsLoader | None = None,
        analytics_reader: AnalyticsReaderFactory | None = None,
        gap_rank: GapRankFn | None = None,
        competitor_reader: CompetitorReaderFactory | None = None,
        competitor_explain: CompetitorExplainFn | None = None,
        channel_reader: ChannelReaderFactory | None = None,
        category_fn: CategoryFn | None = None,
        pinned_fn: PinnedFn | None = None,
        caption_reader: CaptionReaderFactory | None = None,
        soundtags_fn: SoundTagsFn | None = None,
        comment_writer: CommentWriterFactory | None = None,
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
        self._caption_reader_factory = caption_reader  # None: the real reader
        self._soundtags = soundtags_fn or make_default_soundtags(config)
        self._chapters = chapters_fn or make_default_chapters(config)
        self._copilot = copilot_fn or make_default_copilot(config)
        self._adcheck = adcheck_fn or make_default_adcheck(config)
        self._category = category_fn or make_default_category(config)
        self._pinned = pinned_fn or make_default_pinned(config)
        self._comment_reader_factory = comment_reader
        self._comment_writer_factory = comment_writer  # None: the real writer
        self._comments_assist = comments_assist or make_default_comments_assist(config)
        self._rule_maker = rule_maker or make_default_rule_maker(config)
        self._bulk_agent = bulk_agent or make_default_bulk_agent(config)
        self._strategy = strategy_fn or make_default_strategy(config)
        self._shorts = shorts_fn or make_default_shorts(config)
        self._competitor_reader = competitor_reader  # None: the real reader
        self._competitor_explain = competitor_explain or make_default_competitor_explain(config)
        self._analytics_sign_in = analytics_sign_in or sign_in_with_analytics_access
        self._analytics_credentials_loader = analytics_credentials_loader or load_analytics_credentials
        self._analytics_reader = analytics_reader  # None: the real YouTube Analytics reader
        self._channel_reader = channel_reader  # None: the real YouTube Analytics reader
        self._gap_rank = gap_rank or make_default_gap_rank(config)
        self._link_fetch = link_fetch  # None: the real visit, links_mod.fetch_url
        self.packaging_tools = packaging_tools
        # Whether a Meta key exists, asked on page loads without building any client. None: an
        # injected packaging_tools factory (tests) answers instead.
        self.packaging_key = packaging_key
        self._packaging_fetch = packaging_fetch
        self.images_dir = config.home / "thumbnails"  # review pictures and variants, by file name
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
        self.packaging_job = Job(map_packaging_error, PACKAGING_TIMEOUT_SECONDS,
                                 PACKAGING_TIMEOUT_MESSAGE, clock)
        self._packaging_seen: JobSnapshot | None = None
        self._packaging_video = ""
        self.outliers_job = Job(map_outliers_error, OUTLIERS_TIMEOUT_SECONDS,
                                OUTLIERS_TIMEOUT_MESSAGE, clock)
        self._outliers_seen: JobSnapshot | None = None
        self.strategy_job = Job(map_strategy_error, STRATEGY_TIMEOUT_SECONDS,
                                STRATEGY_TIMEOUT_MESSAGE, clock)
        self._strategy_seen: JobSnapshot | None = None
        self.competitors_job = Job(map_competitors_error, COMPETITORS_TIMEOUT_SECONDS,
                                   COMPETITORS_TIMEOUT_MESSAGE, clock)
        self.competitors_explain_job = Job(map_competitors_error, COMPETITORS_TIMEOUT_SECONDS,
                                           COMPETITORS_TIMEOUT_MESSAGE, clock)
        self._competitors_seen: JobSnapshot | None = None
        self._competitors_explain_seen: JobSnapshot | None = None
        self.searchgaps_signin_job = Job(
            map_sign_in_error, sign_in_timeout, SIGN_IN_TIMEOUT_MESSAGE, clock
        )
        self.searchgaps_job = Job(
            map_searchgaps_error, SEARCHGAPS_TIMEOUT_SECONDS, SEARCHGAPS_TIMEOUT_MESSAGE, clock
        )
        self.searchgaps_rank_job = Job(
            map_searchgaps_error, SEARCHGAPS_TIMEOUT_SECONDS, SEARCHGAPS_TIMEOUT_MESSAGE, clock
        )
        self._searchgaps_seen: JobSnapshot | None = None
        self._searchgaps_rank_seen: JobSnapshot | None = None
        self._searchgaps_signin_seen: JobSnapshot | None = None
        self.channel_stats_job = Job(map_channel_stats_error, CHANNEL_STATS_TIMEOUT_SECONDS,
                                     CHANNEL_STATS_TIMEOUT_MESSAGE, clock)
        self._channel_stats_seen: JobSnapshot | None = None
        self.video_stats_job = Job(map_channel_stats_error, CHANNEL_STATS_TIMEOUT_SECONDS,
                                   CHANNEL_STATS_TIMEOUT_MESSAGE, clock)
        self._video_stats_seen: JobSnapshot | None = None
        self.video_stats_video = ""  # the video whose numbers the job reads (one at a time)
        self.copilot_job = Job(map_copilot_error, COPILOT_TIMEOUT_SECONDS,
                               COPILOT_TIMEOUT_MESSAGE, clock)
        self._copilot_seen: JobSnapshot | None = None
        self._copilot_chat: int | None = None  # None: the chat of the newest turn
        self.bulk_agent_job = Job(map_bulk_agent_error, BULK_AGENT_TIMEOUT_SECONDS,
                                  BULK_AGENT_TIMEOUT_MESSAGE, clock)
        self._bulk_agent_seen: JobSnapshot | None = None
        self._bulk_session: bulkagent_mod.BulkSession | None = None  # in memory only, by design
        self.links_job = Job(map_links_error, LINKS_TIMEOUT_SECONDS, LINKS_TIMEOUT_MESSAGE, clock)
        self._links_seen: JobSnapshot | None = None
        self.adcheck_job = Job(map_adcheck_error, ADCHECK_TIMEOUT_SECONDS, ADCHECK_TIMEOUT_MESSAGE,
                               clock)
        self._adcheck_seen: JobSnapshot | None = None
        self._adcheck_video = ""
        self.category_job = Job(map_category_error, CATEGORY_TIMEOUT_SECONDS, CATEGORY_TIMEOUT_MESSAGE,
                                clock)
        self._category_seen: JobSnapshot | None = None
        self._category_video = ""
        self.pinned_job = Job(map_pinned_error, PINNED_TIMEOUT_SECONDS, PINNED_TIMEOUT_MESSAGE, clock)
        self._pinned_seen: JobSnapshot | None = None
        self._pinned_video = ""
        self.shorts_job = Job(map_shorts_error, SHORTS_TIMEOUT_SECONDS, SHORTS_TIMEOUT_MESSAGE, clock)
        self._shorts_seen: JobSnapshot | None = None
        self._shorts_video = ""
        self.variants_job = Job(map_variants_error, VARIANTS_TIMEOUT_SECONDS,
                                VARIANTS_TIMEOUT_MESSAGE, clock)
        self._variants_seen: JobSnapshot | None = None
        self._variants_video = ""
        self.comments_job = Job(map_comments_error, COMMENTS_TIMEOUT_SECONDS, COMMENTS_TIMEOUT_MESSAGE, clock)
        self._comments_seen: JobSnapshot | None = None
        self._comments_video = ""
        self.assist_job = Job(map_assist_error, ASSIST_TIMEOUT_SECONDS, ASSIST_TIMEOUT_MESSAGE, clock)
        self._assist_seen: JobSnapshot | None = None
        self._assist_video = ""
        self.channel_comments_job = Job(map_comments_error, CHANNEL_FETCH_TIMEOUT_SECONDS, CHANNEL_FETCH_TIMEOUT_MESSAGE, clock)
        self._channel_comments_seen: JobSnapshot | None = None
        self.channel_sort_job = Job(map_assist_error, CHANNEL_SORT_TIMEOUT_SECONDS, CHANNEL_SORT_TIMEOUT_MESSAGE, clock)
        self._channel_sort_seen: JobSnapshot | None = None
        self.thumb_job = Job(map_thumb_error)
        self._thumb_seen: JobSnapshot | None = None
        self._thumb_page = ""  # the page that started the swap
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

    def activity(self) -> list[dict[str, Any]]:
        """The jobs that ran since Yezatas started, for the top bar: what each does, its state and where it is."""
        now = time.monotonic()
        out: list[dict[str, Any]] = []
        for name, job_attr, video_attr, running, done, failed, anchor in ACTIVITY:
            snap: JobSnapshot = getattr(self, job_attr).snapshot()
            if snap.status == "idle":
                continue
            with self._lock:
                video_id = getattr(self, video_attr) if video_attr else ""
                if name == "suggest" and len(self._suggest_ids) == 1 and not self._suggest_batch:
                    video_id = self._suggest_ids[0]
            if anchor.startswith("#"):
                href = f"/videos/{quote(video_id)}{anchor}" if video_id else "/changes"
            else:
                href = anchor
            label = running if snap.running else done if snap.status == "done" else failed
            out.append({"job": name, "status": snap.status, "label": label, "message": snap.message,
                        "video_id": video_id, "href": href,
                        "seconds": int(now - snap.started_at) if snap.started_at else 0})
        return out

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
                "auth", "There is no translation key. Add one in Configuration. Nothing was sent.")
        try:
            glossary_terms = glossary_mod.load_glossary(
                self.config.home / glossary_mod.GLOSSARY_FILE
            )
            broken = False
        except glossary_mod.GlossaryError:
            glossary_terms = []
            broken = True
        with Store(self.config.catalog_db) as store:
            result = translate.translate_video(store, translator, verifier, video_id,
                                               source_language, list(targets), report=report,
                                               glossary=glossary_terms)
        if broken:
            result = replace(
                result,
                message=result.message + " (the glossary file could not be read, so it was not used)",
            )
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
                "auth", "There is no translation key. Add one in Configuration. Nothing was sent.")
        try:
            glossary_terms = glossary_mod.load_glossary(
                self.config.home / glossary_mod.GLOSSARY_FILE
            )
            broken = False
        except glossary_mod.GlossaryError:
            glossary_terms = []
            broken = True
        with Store(self.config.catalog_db) as store:
            result = subtitle_translate.translate_subtitles(
                store, translator, verifier, video_id, list(targets), report=report,
                glossary=glossary_terms)  # type: ignore[arg-type]
        if broken:
            result = replace(
                result,
                message=result.message + " (the glossary file could not be read, so it was not used)",
            )
        return result

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
                         keywords: Sequence[str], cleanup: Path | None = None) -> bool:
        """Transcribe the video file at `media` (kept in memory only), in the one transcribe job.
        `cleanup` is a folder to delete when the job ends, whatever way it ends."""
        words = tuple(keywords)

        def work(report: Report) -> "transcribe_mod.TranscribeResult":
            try:
                return self._run_transcribe(video_id, media, language, words, report)
            finally:
                remove_folder(cleanup)

        return self._start_source_job(video_id, work, cleanup)

    def start_sound_tags(self, video_id: str, media: Path, cleanup: Path | None = None) -> bool:
        """Tag the music and sounds of the file at `media` (on this computer, free), in the one
        transcribe job. The transcript is stored like a speech one."""
        def work(report: Report) -> "soundtags_mod.SoundTagResult":
            try:
                result = self._soundtags(video_id, str(media), report)
                with Store(self.config.catalog_db) as store:
                    store.add_transcript(result.record)
                return result
            except Exception as exc:  # noqa: BLE001 - pages show the file name, never its folder
                hidden = hide_path(exc, media)
                if hidden is exc:
                    raise
                raise hidden from None
            finally:
                remove_folder(cleanup)

        return self._start_source_job(video_id, work, cleanup)

    def start_captions(self, video_id: str, video_language: str, creds: Any) -> bool:
        """Read the captions that are already on YouTube, in the one transcribe job."""
        def work(report: Report) -> "captions_mod.CaptionsResult":
            return self._run_captions(video_id, video_language, creds, report)

        return self._start_source_job(video_id, work, None)

    def _start_source_job(self, video_id: str, work: Callable[[Report], Any],
                          cleanup: Path | None) -> bool:
        if not self.transcribe_job.start(work):
            remove_folder(cleanup)  # the job never ran, so it will not clean up
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

    def _run_captions(self, video_id: str, video_language: str, creds: Any,
                      report: Report) -> "captions_mod.CaptionsResult":
        report("Reading the captions on YouTube")
        factory = self._caption_reader_factory
        reader = GoogleCaptionReader(creds) if factory is None else factory(creds)
        needed = captions_mod.LIST_QUOTA + captions_mod.DOWNLOAD_QUOTA
        with Store(self.config.catalog_db) as store:
            guard = QuotaGuard(store, self.config.daily_quota_budget)
            if guard.remaining() < needed:
                raise captions_mod.CaptionsError(
                    "quota", f"Reading captions needs {needed} units of the daily YouTube quota "
                             f"budget, and {max(0, guard.remaining())} are left today. Try again "
                             "tomorrow.")
            result = captions_mod.fetch_captions(
                _ChargedCaptionReader(reader, guard), video_id, video_language)
            store.add_transcript(result.record)
        return result

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

    # --- packaging check and variant ideas ----------------------------------------------------
    def start_packaging(self, video_id: str) -> bool:
        """Check one video's title and thumbnail together, in the one packaging job."""
        def work(report: Report) -> "packaging_mod.PackagingResult":
            return self._run_packaging(video_id, report)

        if not self.packaging_job.start(work):
            return False
        with self._lock:
            self._packaging_video = video_id
        return True

    def _run_packaging(self, video_id: str, report: Report) -> "packaging_mod.PackagingResult":
        tools = self.packaging_tools(self.config)
        if tools is None:
            raise agent.ModelError("auth", NO_PACKAGING_KEY)
        with Store(self.config.catalog_db) as store:
            return packaging_mod.review_packaging(
                store, tools.vision, tools.segmenter, video_id, images_dir=self.images_dir,
                fetch=self._packaging_fetch, report=report)

    def packaging_state(self) -> tuple[JobSnapshot, str]:
        """(job snapshot, the video id it works on). A finished job the creator dismissed is idle."""
        snap = self.packaging_job.snapshot()
        with self._lock:
            if snap is self._packaging_seen:
                snap = JobSnapshot("idle")
            return snap, self._packaging_video

    def dismiss_packaging_result(self) -> None:
        snap = self.packaging_job.snapshot()
        if not snap.running:
            with self._lock:
                self._packaging_seen = snap

    def start_variants(self, video_id: str, review_id: int) -> bool:
        """Make the variant ideas of one stored check, in the one variants job."""
        def work(report: Report) -> "packaging_mod.VariantsResult":
            return self._run_variants(review_id, report)

        if not self.variants_job.start(work):
            return False
        with self._lock:
            self._variants_video = video_id
        return True

    def _run_variants(self, review_id: int, report: Report) -> "packaging_mod.VariantsResult":
        tools = self.packaging_tools(self.config)
        if tools is None:
            raise agent.ModelError("auth", NO_PACKAGING_KEY)
        with Store(self.config.catalog_db) as store:
            return packaging_mod.make_variants(
                store, tools.maker, tools.vision, tools.segmenter, review_id,
                images_dir=self.images_dir, report=report)

    def variants_state(self) -> tuple[JobSnapshot, str]:
        snap = self.variants_job.snapshot()
        with self._lock:
            if snap is self._variants_seen:
                snap = JobSnapshot("idle")
            return snap, self._variants_video

    def dismiss_variants_result(self) -> None:
        snap = self.variants_job.snapshot()
        if not snap.running:
            with self._lock:
                self._variants_seen = snap

    # --- thumbnail swap (F3): only after the creator confirmed it ----------------------------
    def start_thumb_swap(self, creds: Any, video_id: str, image: bytes, *, kind: str, new_file: str,
                         variant_id: int | None, page: str) -> bool:
        """Upload one confirmed thumbnail in the one thumbnail job; `page` is the confirmation page
        that started it."""
        def work(report: Report) -> thumbswap_mod.ThumbnailSwap:
            with Store(self.config.catalog_db) as store:
                guard = QuotaGuard(store, self.config.daily_quota_budget)
                return thumbswap_mod.swap_thumbnail(
                    self._writer_factory(creds), guard, self.images_dir, video_id, image,
                    kind=kind, new_file=new_file, variant_id=variant_id,
                    fetch=self._packaging_fetch, report=report)

        if not self.thumb_job.start(work):
            return False
        with self._lock:
            self._thumb_page = page
        return True

    def thumb_state(self) -> tuple[JobSnapshot, str]:
        snap = self.thumb_job.snapshot()
        with self._lock:
            if snap is self._thumb_seen:
                snap = JobSnapshot("idle")
            return snap, self._thumb_page

    def dismiss_thumb_result(self) -> None:
        snap = self.thumb_job.snapshot()
        if not snap.running:
            with self._lock:
                self._thumb_seen = snap

    # --- outliers: what the best videos share ---------------------------------------------------
    def start_outliers(self) -> bool:
        """Ask Muse Spark what the outliers share, in the one outliers job."""
        return self.outliers_job.start(self._run_outliers)

    def _run_outliers(self, report: Report) -> OutlierReport:
        tools = self.packaging_tools(self.config)
        if tools is None:
            raise agent.ModelError("auth", NO_PACKAGING_KEY)
        report("Asking Muse Spark what the outliers share…")
        with Store(self.config.catalog_db) as store:
            pull = store.latest_pull()
            if pull is None:
                raise ValueError("There is no pulled catalog yet. Pull it in Configuration first.")
            return outliers_mod.explain_outliers(
                store, tools.vision, as_of=pull_moment(pull), fetch=self._packaging_fetch)

    def outliers_state(self) -> JobSnapshot:
        """The job snapshot. A finished job the creator dismissed is idle."""
        snap = self.outliers_job.snapshot()
        with self._lock:
            return JobSnapshot("idle") if snap is self._outliers_seen else snap

    def dismiss_outliers_result(self) -> None:
        snap = self.outliers_job.snapshot()
        if not snap.running:
            with self._lock:
                self._outliers_seen = snap

    # --- strategy: what works, what is missing and what to make next ---------------------------
    def start_strategy(self) -> bool:
        """Make the strategy report in the one strategy job."""
        return self.strategy_job.start(lambda report: self._run_strategy(report))

    def _run_strategy(self, report: Report) -> "strategy_mod.StrategyReport":
        report("Starting…")
        with Store(self.config.catalog_db) as store:
            pull = store.latest_pull()
            if pull is None:
                raise ValueError("There is no pulled catalog yet. Pull it in Configuration first.")
            return self._strategy(store, self.config.home, pull_moment(pull), report)

    def strategy_state(self) -> JobSnapshot:
        """The job snapshot. A finished job the creator dismissed is idle."""
        snap = self.strategy_job.snapshot()
        with self._lock:
            return JobSnapshot("idle") if snap is self._strategy_seen else snap

    def dismiss_strategy_result(self) -> None:
        snap = self.strategy_job.snapshot()
        if not snap.running:
            with self._lock:
                self._strategy_seen = snap

    # --- competitors: outliers on other public channels (C8) -----------------------------------------
    def read_credentials(self) -> Any:
        """Read-only credentials (the setup sign-in), or None. Never raises."""
        try:
            return self._credentials_loader(self.config)
        except Exception:  # noqa: BLE001 - a broken token file just means "sign in again"
            return None

    def start_competitor_fetch(self, ref: Any, creds: Any) -> bool:
        return self.competitors_job.start(lambda report: self._run_competitor_fetch(ref, creds, report))

    def _run_competitor_fetch(self, ref: Any, creds: Any, report: Report) -> Any:
        report("Starting…")
        if self._competitor_reader is None:
            reader: Any = competitors_mod.GooglePublicChannelReader(creds)
        else:
            reader = self._competitor_reader(creds)
        with Store(self.config.catalog_db) as store:
            guard = QuotaGuard(store, self.config.daily_quota_budget)
            channel = competitors_mod.fetch_channel(reader, ref, guard, report=report)
        competitors_mod.save_channel(self.config.home, channel)
        return channel

    def start_competitor_explain(self) -> bool:
        return self.competitors_explain_job.start(
            lambda report: self._competitor_explain(competitors_mod.load_channels(self.config.home), report)
        )

    def competitors_state(self) -> tuple[JobSnapshot, JobSnapshot]:
        fetch_snap = self.competitors_job.snapshot()
        explain_snap = self.competitors_explain_job.snapshot()
        with self._lock:
            if fetch_snap is self._competitors_seen:
                fetch_snap = JobSnapshot("idle")
            if explain_snap is self._competitors_explain_seen:
                explain_snap = JobSnapshot("idle")
        return fetch_snap, explain_snap

    def dismiss_competitors_results(self) -> None:
        fetch_snap = self.competitors_job.snapshot()
        explain_snap = self.competitors_explain_job.snapshot()
        with self._lock:
            if not fetch_snap.running:
                self._competitors_seen = fetch_snap
            if not explain_snap.running:
                self._competitors_explain_seen = explain_snap

    # --- search gaps: real search terms missing from titles (C5) ---
    def start_analytics_sign_in(self) -> bool:
        def work(report: Report) -> None:
            report("Waiting for you to finish in the browser tab…")
            self._analytics_sign_in(self.config)

        return self.searchgaps_signin_job.start(work)

    def analytics_credentials(self) -> Any:
        """Credentials that may read search terms, or None. Never raises."""
        try:
            return self._analytics_credentials_loader(self.config)
        except Exception:  # noqa: BLE001 - a broken token file just means "sign in again"
            return None

    def start_search_terms(self, creds: Any) -> bool:
        return self.searchgaps_job.start(lambda report: self._run_search_terms(creds, report))

    def _run_search_terms(self, creds: Any, report: Report) -> "searchgaps_mod.SearchTerms":
        report("Starting…")
        if self._analytics_reader is None:
            reader: Any = searchgaps_mod.GoogleAnalyticsReader(creds)
        else:
            reader = self._analytics_reader(creds)
        with Store(self.config.catalog_db) as store:
            if store.latest_pull() is None:
                raise searchgaps_mod.SearchGapsError("There is no pulled catalog yet. Pull it in Configuration first.")
            videos = store.list_videos("video_id")
        terms = searchgaps_mod.fetch_terms(reader, videos, report=report)
        searchgaps_mod.save_terms(self.config.home, terms)
        return terms

    def start_gap_rank(self) -> bool:
        return self.searchgaps_rank_job.start(self._run_gap_rank)

    def _run_gap_rank(self, report: Report) -> "searchgaps_mod.GapRanking":
        terms = searchgaps_mod.load_terms(self.config.home)
        if terms is None:
            raise searchgaps_mod.SearchGapsError("Read the search terms first.")
        with Store(self.config.catalog_db) as store:
            videos = store.list_videos("video_id")
        gaps = searchgaps_mod.find_gaps(terms, videos)
        return self._gap_rank(terms, gaps, report)

    def searchgaps_state(self) -> tuple[JobSnapshot, JobSnapshot, JobSnapshot]:
        signin_snap = self.searchgaps_signin_job.snapshot()
        read_snap = self.searchgaps_job.snapshot()
        rank_snap = self.searchgaps_rank_job.snapshot()
        with self._lock:
            if signin_snap is self._searchgaps_signin_seen:
                signin_snap = JobSnapshot("idle")
            if read_snap is self._searchgaps_seen:
                read_snap = JobSnapshot("idle")
            if rank_snap is self._searchgaps_rank_seen:
                rank_snap = JobSnapshot("idle")
        return signin_snap, read_snap, rank_snap

    def dismiss_searchgaps_results(self) -> None:
        signin_snap = self.searchgaps_signin_job.snapshot()
        read_snap = self.searchgaps_job.snapshot()
        rank_snap = self.searchgaps_rank_job.snapshot()
        with self._lock:
            if not signin_snap.running:
                self._searchgaps_signin_seen = signin_snap
            if not read_snap.running:
                self._searchgaps_seen = read_snap
            if not rank_snap.running:
                self._searchgaps_rank_seen = rank_snap

    # --- the channel's numbers for Today (YouTube Analytics, read-only) ---
    def start_channel_stats(self, creds: Any) -> bool:
        return self.channel_stats_job.start(lambda report: self._run_channel_stats(creds, report))

    def _run_channel_stats(self, creds: Any, report: Report) -> "channelstats_mod.ChannelStats":
        report("Starting…")
        if self._channel_reader is None:
            reader: Any = channelstats_mod.GoogleChannelReader(creds)
        else:
            reader = self._channel_reader(creds)
        with Store(self.config.catalog_db) as store:
            videos = store.list_videos("video_id") if store.latest_pull() is not None else []
        stats = channelstats_mod.fetch_stats(reader, videos, report=report)
        channelstats_mod.save_stats(self.config.home, stats)
        return stats

    def channel_stats_state(self) -> JobSnapshot:
        snap = self.channel_stats_job.snapshot()
        with self._lock:
            if snap is self._channel_stats_seen:
                snap = JobSnapshot("idle")
        return snap

    def dismiss_channel_stats(self) -> None:
        snap = self.channel_stats_job.snapshot()
        with self._lock:
            if not snap.running:
                self._channel_stats_seen = snap

    # --- one video's numbers for its page (YouTube Analytics, read-only) ---
    def start_video_stats(self, creds: Any, video_id: str) -> bool:
        with self._lock:
            if self.video_stats_job.snapshot().running:
                return False
            self.video_stats_video = video_id
        return self.video_stats_job.start(lambda report: self._run_video_stats(creds, video_id, report))

    def _run_video_stats(self, creds: Any, video_id: str, report: Report) -> "videostats_mod.VideoStats":
        report("Starting…")
        if self._channel_reader is None:
            reader: Any = channelstats_mod.GoogleChannelReader(creds)
        else:
            reader = self._channel_reader(creds)
        stats = videostats_mod.fetch_video_stats(reader, video_id, report=report)
        videostats_mod.save_video_stats(self.config.home, stats)
        return stats

    def video_stats_state(self) -> JobSnapshot:
        snap = self.video_stats_job.snapshot()
        with self._lock:
            if snap is self._video_stats_seen:
                snap = JobSnapshot("idle")
        return snap

    def dismiss_video_stats(self) -> None:
        snap = self.video_stats_job.snapshot()
        with self._lock:
            if not snap.running:
                self._video_stats_seen = snap

    # --- copilot: questions about the channel -----------------------------------------------------
    def copilot_chat(self, store: Store) -> int | None:
        """The chat on screen: the one picked with "New chat", else the newest (None: no turns)."""
        with self._lock:
            chosen = self._copilot_chat
        return chosen if chosen is not None else store.latest_copilot_chat_id()

    def new_copilot_chat(self, store: Store) -> None:
        with self._lock:
            self._copilot_chat = store.next_copilot_chat_id()

    def start_copilot(self, question: str, chat_id: int) -> bool:
        """Answer one question in the one copilot job."""
        return self.copilot_job.start(lambda report: self._run_copilot(question, chat_id, report))

    def _run_copilot(self, question: str, chat_id: int, report: Report) -> int:
        report("Starting…")
        with Store(self.config.catalog_db) as store:
            pull = store.latest_pull()
            if pull is None:
                raise ValueError("There is no pulled catalog yet. Pull it in Configuration first.")
            return self._copilot(store, question, chat_id, pull_moment(pull), report)

    def copilot_state(self) -> JobSnapshot:
        """The job snapshot. A finished job the page already showed is idle."""
        snap = self.copilot_job.snapshot()
        with self._lock:
            return JobSnapshot("idle") if snap is self._copilot_seen else snap

    def dismiss_copilot_result(self) -> None:
        snap = self.copilot_job.snapshot()
        if not snap.running:
            with self._lock:
                self._copilot_seen = snap

    # --- bulk edit agent: plain words in, staged changes out (in memory) --------------------------
    def bulk_session(self) -> "bulkagent_mod.BulkSession | None":
        """The session on screen, if the creator started one that has not been discarded."""
        with self._lock:
            session = self._bulk_session
        return session if session is not None and session.started else None

    def start_bulk_agent(self, text: str, *, new: bool) -> str:
        """Run one turn in the one bulk agent job. "" when it started, else a plain sentence."""
        busy = "The bulk edit agent is still working. Wait for it to finish, or press Stop."
        if self.bulk_agent_job.snapshot().running:
            return busy
        if new:
            session = bulkagent_mod.BulkSession()
        else:
            found = self.bulk_session()
            if found is None:
                return "There is no session to continue. Start with your words."
            session = found
        session.pending = text.strip()
        if not self.bulk_agent_job.start(lambda report: self._run_bulk_agent(session, text, report)):
            session.pending = ""
            return busy
        with self._lock:
            self._bulk_session = session
        return ""

    def _run_bulk_agent(self, session: "bulkagent_mod.BulkSession", text: str, report: Report) -> None:
        report("Starting…")
        with Store(self.config.catalog_db) as store:
            pull = store.latest_pull()
            if pull is None:
                raise ValueError("There is no pulled catalog yet. Pull it in Configuration first.")
            try:
                self._bulk_agent(store, session, text, pull_moment(pull), report)
            finally:
                session.pending = ""  # a turn that never started leaves no half-sent words

    def stop_bulk_agent(self) -> bool:
        """Ask the running turn to stop after its current step. What is staged is kept."""
        session = self.bulk_session()
        if session is None or not self.bulk_agent_job.snapshot().running:
            return False
        session.stop.set()
        return True

    def discard_bulk_agent(self) -> bool:
        """Forget the session and everything staged in it. Refused while a turn is running."""
        if self.bulk_agent_job.snapshot().running:
            return False
        with self._lock:
            self._bulk_session = None
        return True

    def bulk_agent_state(self) -> JobSnapshot:
        """The job snapshot. A finished job the page already showed is idle. A turn that timed out is
        told to stop, so its abandoned thread does not keep working on the session."""
        snap = self.bulk_agent_job.snapshot()
        if snap.error_kind == "timeout":
            session = self.bulk_session()
            if session is not None:
                session.stop.set()
        with self._lock:
            return JobSnapshot("idle") if snap is self._bulk_agent_seen else snap

    def dismiss_bulk_agent_result(self) -> None:
        snap = self.bulk_agent_job.snapshot()
        if not snap.running:
            with self._lock:
                self._bulk_agent_seen = snap

    # --- links: visit each external link once ----------------------------------------------------
    def start_links(self) -> bool:
        """Check the links in the one links job."""
        return self.links_job.start(self._run_links)

    def _run_links(self, report: Report) -> links_mod.LinkCheckSummary:
        report("Starting…")
        with Store(self.config.catalog_db) as store:
            if store.latest_pull() is None:
                raise ValueError("There is no pulled catalog yet. Pull it in Configuration first.")
            entries = links_mod.inventory(store.list_videos(), store.latest_link_checks())
            return links_mod.check_links(store, entries, fetch=self._link_fetch, report=report)

    def links_state(self) -> JobSnapshot:
        """The job snapshot. A finished job the page already showed is idle."""
        snap = self.links_job.snapshot()
        with self._lock:
            return JobSnapshot("idle") if snap is self._links_seen else snap

    def dismiss_links_result(self) -> None:
        snap = self.links_job.snapshot()
        if not snap.running:
            with self._lock:
                self._links_seen = snap

    # --- Shorts finder (D5) ---
    def start_shorts(self, video_id: str) -> bool:
        if not self.shorts_job.start(lambda report: self._run_shorts(video_id, report)):
            return False
        with self._lock:
            self._shorts_video = video_id
        return True

    def _run_shorts(self, video_id: str, report: Report) -> "ShortsFind":
        report("Starting…")
        with Store(self.config.catalog_db) as store:
            return self._shorts(store, video_id, datetime.now(timezone.utc), report)

    def shorts_state(self) -> tuple[JobSnapshot, str]:
        snap = self.shorts_job.snapshot()
        with self._lock:
            if snap is self._shorts_seen:
                snap = JobSnapshot("idle")
            return snap, self._shorts_video

    def dismiss_shorts_result(self) -> None:
        snap = self.shorts_job.snapshot()
        if not snap.running:
            with self._lock:
                self._shorts_seen = snap

    # --- advertiser-friendly check of one video --------------------------------------------------
    def start_adcheck(self, video_id: str) -> bool:
        """Check one video in the one adcheck job; the result is stored by the check itself."""
        if not self.adcheck_job.start(lambda report: self._run_adcheck(video_id, report)):
            return False
        with self._lock:
            self._adcheck_video = video_id
        return True

    def _run_adcheck(self, video_id: str, report: Report) -> "AdCheck":
        report("Starting…")
        with Store(self.config.catalog_db) as store:
            return self._adcheck(store, video_id, datetime.now(timezone.utc), report)

    def adcheck_state(self) -> tuple[JobSnapshot, str]:
        """(job snapshot, the video id it works on). A finished job the creator dismissed is idle."""
        snap = self.adcheck_job.snapshot()
        with self._lock:
            if snap is self._adcheck_seen:
                snap = JobSnapshot("idle")
            return snap, self._adcheck_video

    def dismiss_adcheck_result(self) -> None:
        snap = self.adcheck_job.snapshot()
        if not snap.running:
            with self._lock:
                self._adcheck_seen = snap

    # --- the category check of one video (V7) --------------------------------------------------------
    def start_category(self, video_id: str) -> bool:
        """Check one video's category in the one category job; the check stores its verdict itself."""
        if not self.category_job.start(lambda report: self._run_category(video_id, report)):
            return False
        with self._lock:
            self._category_video = video_id
        return True

    def _run_category(self, video_id: str, report: Report) -> "CategoryCheck":
        report("Starting…")
        with Store(self.config.catalog_db) as store:
            return self._category(store, video_id, datetime.now(timezone.utc), report)

    def category_state(self) -> tuple[JobSnapshot, str]:
        """(job snapshot, the video id it works on). A finished job the creator dismissed is idle."""
        snap = self.category_job.snapshot()
        with self._lock:
            if snap is self._category_seen:
                snap = JobSnapshot("idle")
            return snap, self._category_video

    def dismiss_category_result(self) -> None:
        snap = self.category_job.snapshot()
        if not snap.running:
            with self._lock:
                self._category_seen = snap

    # --- the pinned comment suggestion of one video (V8) ---------------------------------------------
    def start_pinned(self, video_id: str) -> bool:
        """Ask for one video's pinned comment in the one pinned job; the suggestion is stored by itself."""
        if not self.pinned_job.start(lambda report: self._run_pinned(video_id, report)):
            return False
        with self._lock:
            self._pinned_video = video_id
        return True

    def _run_pinned(self, video_id: str, report: Report) -> "PinnedComment":
        report("Starting…")
        with Store(self.config.catalog_db) as store:
            return self._pinned(store, video_id, datetime.now(timezone.utc), report)

    def pinned_state(self) -> tuple[JobSnapshot, str]:
        """(job snapshot, the video id it works on). A finished job the creator dismissed is idle."""
        snap = self.pinned_job.snapshot()
        with self._lock:
            if snap is self._pinned_seen:
                snap = JobSnapshot("idle")
            return snap, self._pinned_video

    def dismiss_pinned_result(self) -> None:
        snap = self.pinned_job.snapshot()
        if not snap.running:
            with self._lock:
                self._pinned_seen = snap

    # --- comments on one video (G4, G5) -----------------------------------------------------------
    def _make_comment_reader(self, creds: Any) -> CommentReader:
        factory = self._comment_reader_factory
        if factory is None:
            return GoogleCommentReader(creds)
        return factory(creds)

    def make_comment_writer(self, creds: Any) -> CommentWriter:
        """The poster of confirmed comments and replies (a fake in tests)."""
        factory = self._comment_writer_factory
        return GoogleCommentWriter(creds) if factory is None else factory(creds)

    def start_comments(self, video_id: str, creds: Any) -> bool:
        """Read one video's comments in the one comments job."""
        if not self.comments_job.start(lambda report: self._run_comments(video_id, creds, report)):
            return False
        with self._lock:
            self._comments_video = video_id
        return True

    def _run_comments(self, video_id: str, creds: Any, report: Report) -> int:
        report("Starting…")
        reader = self._make_comment_reader(creds)
        with Store(self.config.catalog_db) as store:
            guard = QuotaGuard(store, self.config.daily_quota_budget)
            return comments_mod.fetch_comments(store, reader, video_id, guard=guard, report=report)

    def comments_state(self) -> tuple[JobSnapshot, str]:
        snap = self.comments_job.snapshot()
        with self._lock:
            if snap is self._comments_seen:
                snap = JobSnapshot("idle")
            return snap, self._comments_video

    def dismiss_comments_result(self) -> None:
        snap = self.comments_job.snapshot()
        if not snap.running:
            with self._lock:
                self._comments_seen = snap

    def start_assist(self, video_id: str) -> bool:
        """Sort one video's comments and draft replies in the one assist job."""
        if not self.assist_job.start(lambda report: self._run_assist(video_id, report)):
            return False
        with self._lock:
            self._assist_video = video_id
        return True

    def _run_assist(self, video_id: str, report: Report) -> "comments_mod.AssistResult":
        report("Starting…")
        with Store(self.config.catalog_db) as store:
            return self._comments_assist(store, video_id, datetime.now(timezone.utc), report)

    def assist_state(self) -> tuple[JobSnapshot, str]:
        snap = self.assist_job.snapshot()
        with self._lock:
            if snap is self._assist_seen:
                snap = JobSnapshot("idle")
            return snap, self._assist_video

    def dismiss_assist_result(self) -> None:
        snap = self.assist_job.snapshot()
        if not snap.running:
            with self._lock:
                self._assist_seen = snap

    # --- comments of the whole channel (/comments) ---------------------------------------------
    def start_channel_comments(self, video_ids: Sequence[str], creds: Any) -> bool:
        """Read the comments of several videos in the one channel fetch job."""
        ids = list(video_ids)
        return self.channel_comments_job.start(lambda report: self._run_channel_comments(ids, creds, report))

    def _run_channel_comments(self, video_ids: list[str], creds: Any, report: Report) -> "comments_mod.ChannelFetch":
        report("Starting…")
        reader = self._make_comment_reader(creds)
        with Store(self.config.catalog_db) as store:
            guard = QuotaGuard(store, self.config.daily_quota_budget)
            return comments_mod.fetch_channel_comments(store, reader, video_ids, guard=guard, report=report)

    def channel_comments_state(self) -> JobSnapshot:
        snap = self.channel_comments_job.snapshot()
        with self._lock:
            if snap is self._channel_comments_seen:
                snap = JobSnapshot("idle")
            return snap

    def dismiss_channel_comments_result(self) -> None:
        snap = self.channel_comments_job.snapshot()
        if not snap.running:
            with self._lock:
                self._channel_comments_seen = snap

    def start_channel_sort(self, video_ids: Sequence[str]) -> bool:
        """Sort the comments of several videos in the one channel sort job."""
        ids = list(video_ids)
        return self.channel_sort_job.start(lambda report: self._run_channel_sort(ids, report))

    def _run_channel_sort(self, video_ids: list[str], report: Report) -> "comments_mod.ChannelSort":
        report("Starting…")
        with Store(self.config.catalog_db) as store:
            now = datetime.now(timezone.utc)
            return comments_mod.sort_channel_comments(lambda vid: self._comments_assist(store, vid, now, lambda line: None), video_ids, report=report)

    def channel_sort_state(self) -> JobSnapshot:
        snap = self.channel_sort_job.snapshot()
        with self._lock:
            if snap is self._channel_sort_seen:
                snap = JobSnapshot("idle")
            return snap

    def dismiss_channel_sort_result(self) -> None:
        snap = self.channel_sort_job.snapshot()
        if not snap.running:
            with self._lock:
                self._channel_sort_seen = snap

    # --- custom checks: turn words into one fixed rule (G7) --------------------------------------
    def propose_rule(self, request: str) -> "customchecks_mod.RuleDraft":
        """Turn the creator's words into a proposed rule. Short: callers use a worker thread."""
        return self._rule_maker(request)

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
    packaging_tools: PackagingToolsFactory | None = None,
    packaging_fetch: Callable[[str], bytes] | None = None,
    packaging_key: Callable[[Config], bool] | None = None,
    copilot: CopilotFn | None = None,
    link_fetch: links_mod.Fetch | None = None,
    adcheck: AdcheckFn | None = None,
    comment_reader: CommentReaderFactory | None = None,
    comments_assist: CommentsAssistFn | None = None,
    rule_maker: RuleMakerFn | None = None, bulk_agent: BulkAgentFn | None = None,
    strategy: StrategyFn | None = None,
    shorts: ShortsFn | None = None, shorts_key: Callable[[Config], bool] | None = None,
    analytics_sign_in: Callable[[Config], Any] | None = None,
    analytics_credentials_loader: CredentialsLoader | None = None,
    analytics_reader: AnalyticsReaderFactory | None = None,
    gap_rank: GapRankFn | None = None,
    competitor_reader: CompetitorReaderFactory | None = None,
    competitor_explain: CompetitorExplainFn | None = None,
    channel_reader: ChannelReaderFactory | None = None,
    category: CategoryFn | None = None,
    pinned: PinnedFn | None = None,
    caption_reader: CaptionReaderFactory | None = None,
    soundtags: SoundTagsFn | None = None,
    comment_writer: CommentWriterFactory | None = None,
) -> FastAPI:
    """Build the app. `app.state.start_url` is the one-time link that signs the browser in."""
    ensure_dirs(config)
    token = token or secrets.token_urlsafe(32)
    start_url = f"http://127.0.0.1:{port}/?t={token}"
    tools = translate_tools or default_translate_tools
    state = WebState(config, sign_in, reader_factory, credentials_loader, clock, sign_in_timeout,
                     write_sign_in, write_credentials_loader, writer_factory,
                     suggest or make_default_suggest(config),
                     tools, transcribe, chapters,
                     packaging_tools or packaging_mod.packaging_tools_from_config,
                     packaging_fetch or packaging_mod.fetch_thumbnail,
                     packaging_key or (None if packaging_tools else has_meta_key), copilot,
                     link_fetch, adcheck, comment_reader, comments_assist, rule_maker, bulk_agent,
                     strategy_fn=strategy, shorts_fn=shorts,
                     competitor_reader=competitor_reader, competitor_explain=competitor_explain,
                     analytics_sign_in=analytics_sign_in, analytics_credentials_loader=analytics_credentials_loader,
                     analytics_reader=analytics_reader, gap_rank=gap_rank, channel_reader=channel_reader,
                     category_fn=category, pinned_fn=pinned, caption_reader=caption_reader,
                     soundtags_fn=soundtags, comment_writer=comment_writer)
    sessions = Sessions(token, port, load_session_key(config.credentials_dir / "session.key"))
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    # A missing value is a bug: fail loudly instead of rendering "" (an empty link, say).
    templates.env.undefined = StrictUndefined
    templates.env.filters["lang_name"] = lang_name
    templates.env.filters["num"] = _num
    # The Primary AI, read on every render: Local hides the prices and names Glimmer.
    templates.env.globals["ai_local"] = lambda: primaryai.is_local(config)
    templates.env.globals["ai_name"] = lambda: "Muse Glimmer" if primaryai.is_local(config) else "Muse Spark"

    def text_model_name() -> str:
        """The text model's name for the current Primary AI, for the labels next to the buttons."""
        return config.verifier_model if primaryai.is_local(config) else config.model

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
        return RedirectResponse("/configuration" + (f"#step-{step}" if step else ""), status_code=303)

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
            # Translations on YouTube made from an older title or description do not count for the score.
            stale = {v.video_id: keys for v in videos if v.localizations
                     and (keys := stale_youtube_keys(v, store.latest_translations(v.video_id)))}
            # Muse Spark's finished category checks; a video without one is not judged.
            judged = {vid: CategoryJudgment(c.current_id, c.suggested_id, bool(c.confirmed_at), c.reason)
                      for vid, c in store.category_checks().items() if c.status == "done"}
            # A comment from the channel among the fetched ones, or the user's "I pinned one already".
            pinned_states = store.pinned_states()
            guard = QuotaGuard(store, config.daily_quota_budget)
            used, remaining = guard.used(), guard.remaining()
            pull = store.latest_pull()
        entries = make_entries(videos, score_catalog(videos, stale, judged, pinned_states))
        return {
            "entries": entries, "as_of": pull_moment(pull) if pull is not None else None, "stale": stale,
            "latest_pull": {"videos": latest[0], "when": _local_time(latest[1])},
            "quota": {"used": used, "remaining": remaining, "budget": config.daily_quota_budget},
        }

    def outlier_scores(data: dict[str, Any]) -> list[outliers_mod.OutlierScore]:
        """Every video scored against its peers, with the views as of the latest pull."""
        return outliers_mod.score_videos([e.video for e in data["entries"]], data["as_of"])

    def draft_ids() -> frozenset[str]:
        with Store(config.catalog_db) as store:
            return frozenset(row.video_id for row in store.list_drafts())

    nav_cache: dict[str, Any] = {"key": None, "numbers": (0, 0, 0, 0)}

    def nav_numbers() -> tuple[int, int, int, int]:
        """(questions to answer, videos scoring under 50, translations out of date, broken links) for the
        sidebar. Scoring the catalog is the slow part, so the numbers are kept until the database changes."""
        try:
            info = os.stat(config.catalog_db)
            key: Any = (info.st_size, info.st_mtime_ns)
        except OSError:
            key = None
        if key is not None and nav_cache["key"] == key:
            return nav_cache["numbers"]
        with Store(config.catalog_db) as store:
            questions = len(store.comments_to_answer(limit=100))
            entries = links_mod.inventory(store.list_videos(), store.latest_link_checks())
        broken = sum(1 for e in entries if e.check is not None and e.check.status == "broken")
        data = catalog_data()
        poor = stale = 0
        if data is not None:
            poor = sum(1 for e in data["entries"] if e.band == "poor")
            stale = sum(len(keys) for keys in data["stale"].values())
        # The key read before counting: a write during the count makes the next page count again.
        nav_cache["key"], nav_cache["numbers"] = key, (questions, poor, stale, broken)
        return nav_cache["numbers"]

    def render(request: Request, name: str, extra: dict[str, Any],
               status_code: int = 200) -> Response:
        drafts = len(draft_ids())
        questions, poor, stale, broken = nav_numbers()
        context: dict[str, Any] = {"csrf": request.state.csrf, "draft_count": drafts, "nav_counts": build_nav_counts(
            questions=questions, poor_videos=poor, stale_translations=stale, broken_links=broken, drafts=drafts),
                                   **extra}
        # The setup page shows its own flash messages next to the step they belong to.
        if "flash" not in extra and "shell_flash" not in extra:
            context["shell_flash"] = state.pop_flash(request.state.session_id)
        return templates.TemplateResponse(request, name, context, status_code=status_code)

    def see_other(url: str) -> RedirectResponse:
        return RedirectResponse(url, status_code=303)

    @app.middleware("http")
    async def stay_in_place(request: Request, call_next):  # type: ignore[no-untyped-def]
        """A form the page's script marked ?stay=1 comes back without the #card anchor (which makes the browser jump
        before the script can restore the reader's place); the card goes in ?go= so the script can still show it."""
        response = await call_next(request)
        location = response.headers.get("location", "")
        if (request.method == "POST" and request.query_params.get("stay") == "1"
                and response.status_code == 303 and "#" in location):
            base, frag = location.split("#", 1)
            response.headers["location"] = f"{base}{'&' if '?' in base else '?'}go={quote(frag)}"
        return response

    def flash(request: Request, kind: str, text: str, step: str = "") -> None:
        state.add_flash(request.state.session_id, kind, text, step)

    def suggest_panel(video_id: str = "") -> dict[str, Any]:
        """What the suggestion job is doing, for the dashboard (no id) or one video page."""
        snap, ids, is_batch = state.suggest_state()
        mine = (video_id in ids) if video_id else is_batch
        batch = snap.result if (snap.status == "done" and is_batch and not video_id
                                and isinstance(snap.result, SuggestBatch)) else None
        return {
            "has_key": primaryai.text_ai_ready(config), "model": text_model_name(),
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
        """Is the checker up? The Configuration page always asks (`fresh`); other pages reuse a slow "no"."""
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
        provider = (source if source in ("meta", "modal", "local") else "baseten" if source
                    else "other")
        model = translator.model if translator is not None else (
            config.model if source == "meta" else config.modal_model if source == "modal"
            else config.verifier_model if source == "local" else config.translator_model)
        return {
            "running": running, "busy": snap.running and not mine,
            "message": snap.message if running else "",
            "failed": snap.message if snap.status == "failed" and mine else "",
            "has_translator": translator is not None, "model": model, "provider": provider,
            "verifier_model": verifier.model if verifier is not None else config.verifier_model,
            "verifier_ready": True if running else verifier_ready(verifier),
            "same_model": verifier is not None and model == verifier.model,
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

    def transcript_done_sentence(result: Any) -> str:
        """What a finished transcript job says: its result and what it cost, in a sentence."""
        if isinstance(result, transcribe_mod.TranscribeResult):
            return f"{result.message} The cost was about {format_cost(result.cost)}."
        if isinstance(result, captions_mod.CaptionsResult):
            return (f"{result.message} They were read with your sign-in: no cost, and it used "
                    f"{result.quota_units} units of your daily YouTube quota.")
        if isinstance(result, soundtags_mod.SoundTagResult):
            return f"{result.message} This ran on this computer: no cost, and nothing was sent."
        return ""

    def transcript_panel(video_id: str) -> dict[str, Any]:
        """What the transcribe and chapters jobs are doing, for one video page."""
        snap, job_video = state.transcribe_state()
        transcribing = job_panel(snap, job_video, video_id)
        transcribing["failed_kind"] = snap.error_kind if transcribing["failed"] else ""
        if snap.status == "done" and job_video == video_id:
            sentence = transcript_done_sentence(snap.result)
            if sentence:
                transcribing["done"] = {"tone": "ok", "chapters": [], "drafted": False,
                                        "text": sentence}
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
        has_key = bool(agent.key_source(config))
        local = primaryai.is_local(config)
        return {"ffmpeg": transcribe_mod.find_ffmpeg() is not None,
                "has_key": has_key, "local_no_key": local and not has_key,
                "has_write": state.write_credentials() is not None,
                "sounds_ready": soundtags_mod.available(),
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
                                         "verifier_ready", "same_model")})
        return panel

    def packaging_tools_or_none() -> "packaging_mod.PackagingTools | None":
        """The packaging clients; a broken setup just means "not available"."""
        try:
            return state.packaging_tools(config)
        except Exception:  # noqa: BLE001
            return None

    def has_packaging_key() -> bool:
        """Page loads only need to know that a key exists: no SAM, image or vision client is built."""
        if state.packaging_key is not None:
            return bool(state.packaging_key(config))
        return packaging_tools_or_none() is not None  # a factory injected by a test decides

    def packaging_panel(video_id: str) -> dict[str, Any]:
        """What the packaging and variants jobs are doing, for one video page (worker thread)."""
        snap, job_video = state.packaging_state()
        checking = job_panel(snap, job_video, video_id)
        if snap.status == "done" and job_video == video_id and isinstance(
                snap.result, packaging_mod.PackagingResult):
            checking["done"] = {"tone": "ok" if snap.result.status == "done" else "error",
                                "text": snap.result.message}
        snap, job_video = state.variants_state()
        making = job_panel(snap, job_video, video_id)
        if snap.status == "done" and job_video == video_id and isinstance(
                snap.result, packaging_mod.VariantsResult):
            making["done"] = {"tone": "ok" if snap.result.status == "done" else "error",
                              "text": snap.result.message}
        return {"has_tools": has_packaging_key(), "meta_blocked": primaryai.meta_only_blocked(config),
                "vision": text_model_name(), "segmenter": packaging_mod.SAM_MODEL,
                "maker": packaging_mod.IMAGE_MODEL, "check": checking, "variants": making}

    def baseten_key_source() -> str:
        """Where the Baseten key comes from ("environment", "saved" or ""), apart from which translator is used."""
        if os.environ.get(translate.TRANSLATOR_KEY_ENV, "").strip():
            return "environment"
        return "saved" if config.translator_key_path is not None and config.translator_key_path.is_file() else ""

    def soundtags_installed() -> bool:
        """Is the sound tagger (PANNs) installed? Looked up without importing it."""
        try:
            return importlib.util.find_spec("panns_inference") is not None
        except (ImportError, ValueError):
            return False

    def has_shorts_key() -> bool:
        if shorts_key is not None:
            return bool(shorts_key(config))
        return translate.translator_key_source(config) != ""

    def shorts_panel(video_id: str, model: str, provider: str, has_transcript: bool) -> dict[str, Any]:
        snap, job_video = state.shorts_state()
        panel = job_panel(snap, job_video, video_id)
        panel.update({"has_key": has_shorts_key(), "model": model, "provider": provider, "has_transcript": has_transcript})
        return panel

    def adcheck_panel(video_id: str, has_key: bool) -> dict[str, Any]:
        """What the advertiser-friendly check job is doing, for one video page."""
        snap, job_video = state.adcheck_state()
        panel = job_panel(snap, job_video, video_id)
        panel.update({"has_key": has_key, "model": text_model_name()})
        return panel

    def category_panel(video_id: str, has_key: bool) -> dict[str, Any]:
        """What the category check job is doing, for one video page."""
        snap, job_video = state.category_state()
        panel = job_panel(snap, job_video, video_id)
        panel["has_key"] = has_key
        return panel

    def pinned_panel(video_id: str, has_key: bool) -> dict[str, Any]:
        """What the pinned comment job is doing, for one video page."""
        snap, job_video = state.pinned_state()
        panel = job_panel(snap, job_video, video_id)
        panel["has_key"] = has_key
        return panel

    def comments_panel(video_id: str, has_write: bool) -> dict[str, Any]:
        snap, job_video = state.comments_state()
        panel = job_panel(snap, job_video, video_id)
        panel.update({"has_write": has_write, "model": text_model_name()})
        return panel

    def assist_panel(video_id: str, has_key: bool, has_comments: bool) -> dict[str, Any]:
        snap, job_video = state.assist_state()
        panel = job_panel(snap, job_video, video_id)
        result = None
        if snap.status == "done" and job_video == video_id and isinstance(snap.result, comments_mod.AssistResult):
            found = snap.result
            total = found.input_tokens + found.output_tokens
            cost = agent.estimate_cost(found.model, found.input_tokens, found.cached_tokens, found.output_tokens)
            result = {
                "counts": [(comments_mod.CATEGORIES[key], found.counts.get(key, 0))
                           for key in comments_mod.CATEGORIES],
                "drafts": found.drafts, "notes": list(found.notes), "model": found.model,
                "tokens": f"{total:,}", "cost": format_cost(cost),
            }
        panel.update({"has_key": has_key, "has_comments": has_comments, "model": text_model_name(), "result": result})
        return panel

    def channel_fetch_panel(has_write: bool, videos: int) -> dict[str, Any]:
        snap = state.channel_comments_state()
        running = snap.running
        failed = snap.message if snap.status == "failed" else ""
        message = snap.message if running else ""
        result = None
        if snap.status == "done" and isinstance(snap.result, comments_mod.ChannelFetch):
            found = snap.result
            n = found.fetched
            v = found.videos
            result = {"text": f"Fetched {n} comment{'' if n == 1 else 's'} from {v} video{'' if v == 1 else 's'}.", "notes": list(found.notes)}
        return {"running": running, "message": message, "failed": failed, "has_write": has_write, "videos": videos, "units": comments_mod.channel_fetch_units(videos), "result": result}

    def channel_sort_panel(has_key: bool, videos: int) -> dict[str, Any]:
        snap = state.channel_sort_state()
        running = snap.running
        failed = snap.message if snap.status == "failed" else ""
        message = snap.message if running else ""
        result = None
        if snap.status == "done" and isinstance(snap.result, comments_mod.ChannelSort):
            found = snap.result
            total = found.input_tokens + found.output_tokens
            cost = agent.estimate_cost(found.model, found.input_tokens, found.cached_tokens, found.output_tokens)
            result = {
                "counts": [(comments_mod.CATEGORIES[key], found.counts.get(key, 0)) for key in comments_mod.CATEGORIES],
                "drafts": found.drafts, "notes": list(found.notes), "model": found.model,
                "tokens": f"{total:,}", "cost": format_cost(cost),
            }
        return {"running": running, "message": message, "failed": failed, "has_key": has_key, "videos": videos, "model": text_model_name(), "result": result}

    def video_not_found(request: Request) -> Response:
        return render(request, "error.html", {
            "layout": "shell.html", "active": "content",
            "heading": "Video not found",
            "text": "There is no video with this id in your pulled catalog. It may have been "
                    "removed from your channel, or the catalog is out of date.",
            "link_url": "/videos", "link_text": "Back to Content",
        }, status_code=404)

    # --- pages ------------------------------------------------------------------------------
    def week_card(stats: "channelstats_mod.ChannelStats | None", back: str, wid: str) -> dict[str, Any]:
        """What the shared "Channel this week" card needs besides the week itself."""
        snap = state.channel_stats_state()
        signin = state.searchgaps_signin_job.snapshot()
        return {
            "back": back, "wid": wid, "connected": auth.can_read_analytics(config),
            "stats_when": _local_time(stats.fetched_at) if stats else "",
            "stats_running": snap.running, "stats_message": snap.message,
            "stats_failed": snap.message if snap.status == "failed" else "",
            "signin_running": signin.running, "signin_message": signin.message,
            "poll": snap.running or signin.running,
        }

    def video_stats_card(video_id: str, duration: int | None) -> dict[str, Any]:
        """The video page's Analytics card: the sign-in, the refresh job (one video at a time), the saved numbers."""
        snap = state.video_stats_state()
        mine = state.video_stats_video == video_id
        saved = videostats_mod.load_video_stats(config.home, video_id)
        connected = auth.can_read_analytics(config)
        failed = snap.message if snap.status == "failed" and mine else ""
        return {
            "connected": connected,
            "running": snap.running and mine, "busy": snap.running and not mine, "message": snap.message,
            "failed": failed,
            "when": _local_time(saved.fetched_at) if saved else "",
            "view": build_video_analytics(saved, duration),
            "auto": connected and not snap.running and not failed and _stats_due(saved),
        }

    def _stats_due(saved: "videostats_mod.VideoStats | None") -> bool:
        """True when this video's numbers were never read or were read more than VIDEO_STATS_AUTO_HOURS ago."""
        if saved is None:
            return True
        try:
            read = datetime.fromisoformat(saved.fetched_at)
        except ValueError:
            return True
        if read.tzinfo is None:
            read = read.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - read > timedelta(hours=VIDEO_STATS_AUTO_HOURS)

    @app.get("/")
    def index(request: Request) -> Response:
        data = catalog_data()
        if data is None:
            return back_to_setup()
        with Store(config.catalog_db) as store:
            previous = store.previous_view_counts()
            to_answer = store.comments_to_answer(limit=500)
            drafts = {r.comment_id for r in store.list_open_comment_replies() if r.comment_id}
        entries = data["entries"]
        fix_next = dashboard_model(entries)["fix_next"]
        stats = channelstats_mod.load_stats(config.home)
        today = build_today(entries, previous, to_answer, drafts,
                            {e.video.video_id: e.video for e in entries}, stats,
                            fix_next, datetime.now(timezone.utc))
        card = week_card(stats, "/", "today-week")
        pull = state.pull_job.snapshot()
        return render(request, "today.html", {
            "active": "today", "today": today, "latest_pull": data["latest_pull"], "week": today.week,
            **card,
            "pull_running": pull.running, "pull_message": pull.message,
            "pull_failed": pull.message if pull.status == "failed" else "",
            "poll": card["poll"] or pull.running,
        })

    @app.get("/analytics")
    def analytics_page(request: Request) -> Response:
        stats = channelstats_mod.load_stats(config.home)
        an = build_analytics(stats)
        return render(request, "analytics.html", {
            "active": "analytics", "an": an, "week": an.week,
            **week_card(stats, "/analytics", "analytics-week"),
        })

    @app.get("/seo")
    def seo_overview(request: Request) -> Response:
        data = catalog_data()
        if data is None:
            return back_to_setup()
        model = dashboard_model(data["entries"])
        return render(request, "dashboard.html", {
            "active": "dashboard", "latest_pull": data["latest_pull"], "quota": data["quota"],
            "outlier_count": outlier_count(outlier_scores(data)),
            "outlier_high": outliers_mod.ratio_label(outliers_mod.OUTLIER_THRESHOLD),
            "sg": suggest_panel(), "poll": state.suggest_job.snapshot().running,
            **model,
        })

    @app.get("/videos")
    def videos_page(request: Request) -> Response:
        data = catalog_data()
        if data is None:
            return back_to_setup()
        scores = {s.video_id: s for s in outlier_scores(data)}
        listing = build_listing(data["entries"], dict(request.query_params), scores)
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
        pk = packaging_panel(video_id)
        with Store(config.catalog_db) as store:
            review = store.latest_packaging_review(video_id)
            made = store.list_thumbnail_variants(review.id) if review is not None else []
        text = ((row.title, row.description, row.tags) if row is not None
                else (video.title, video.description, tuple(video.tags)))
        score = next((s for s in outlier_scores(data) if s.video_id == video_id), None)
        ac = adcheck_panel(video_id, pk["has_tools"])  # one key check per page load
        ct = category_panel(video_id, pk["has_tools"])
        pt = pinned_panel(video_id, pk["has_tools"])
        with Store(config.catalog_db) as store:
            stored_shorts = store.latest_shorts_find(video_id)
            transcript = store.latest_transcript(video_id)
        sh = shorts_panel(video_id, tr["model"], tr["provider"], transcript is not None)
        with Store(config.catalog_db) as store:
            all_comments = store.list_comments(video_id)
            replies = store.list_comment_replies(video_id)
            reply_templates = store.list_reply_templates()
        cv = build_comments_view(video_id, all_comments, replies, reply_templates, dict(request.query_params))
        cm = comments_panel(video_id, state.write_credentials() is not None)
        am = assist_panel(video_id, pk["has_tools"], bool(all_comments))
        with Store(config.catalog_db) as store:
            stored_check = store.latest_ad_check(video_id)
            category_record = store.category_check(video_id)
            pinned_record = store.pinned_comment(video_id)
            current_hash = adcheck_mod.current_source_hash(store, video_id)
        vs = video_stats_card(video_id, parse_duration(video.duration))
        return render(request, "video.html", {
            "ac": ac, "av": build_adcheck_view(stored_check, current_hash, _local_time),
            "sp": sh, "sr": build_shorts_view(video_id, stored_shorts, transcript.id if transcript else 0,
                            _local_time),
            "cm": cm, "am": am, "cv": cv,
            "outlier_line": outlier_line(score),
            "facts": build_facts(video, entry.card),
            "vs": vs,
            "ct": ct, "cg": build_category_view(
                video_category_id(video), category_record,
                next((c for c in entry.card.checks if c.check_id == "category"), None), _local_time),
            "pt": pt, "pn": build_pinned_view(
                pinned_record, next((c for c in entry.card.checks if c.check_id == "pinned_comment"), None),
                _local_time, all_comments),
            "change": card,
            # the draft changes the title or description but keeps translations made from the old text
            "tr_draft_stale": next((i.message for i in (card.warnings if card else ())
                                    if i.code == "translations_out_of_date"), ""),
            "form": {"title": text[0], "description": text[1], "tags": ", ".join(text[2])},
            "tag_count": len(text[2]), "tag_chars": tags_char_count(text[2]), "tags_limit": TAGS_CHAR_LIMIT,
            # Enough rows for every tag (about 70 characters a row); CSS grows the box as you type.
            "tags_rows": max(2, tags_char_count(text[2]) // 70 + 1),
            "title_limit": TITLE_LIMIT, "description_limit": DESCRIPTION_BYTE_LIMIT,
            "active": "content", "entry": entry, "video": video, "card": entry.card,
            "groups": group_checks(entry.card),
            "big_thumb": thumb_url(video, ("high", "medium", "default")),
            "views": format_count(video.view_count), "likes": format_count(video.like_count),
            "comments": format_count(video.comment_count),
            "published": format_date(video.published_at),
            "sg": panel, "latest_run": latest,
            "tr": tr, "tv": build_translate_view(video, row, records, stale_youtube_keys(video, records)), "ts": ts, "tt": tt,
            "sv": sv, "sb": sb,
            "pk": pk, "pv": build_packaging_view(
                video_id, text[0], thumb_url(video, ("high", "medium", "default")), review, made,
                _local_time),
            "poll": (panel["running"] or panel["busy"] or tr["running"] or tr["busy"]
                     or ts["transcribe"]["running"] or ts["transcribe"]["busy"]
                     or ts["chapters"]["running"] or ts["chapters"]["busy"]
                     or sb["running"] or sb["busy"]
                     or pk["check"]["running"] or pk["check"]["busy"]
                     or pk["variants"]["running"] or pk["variants"]["busy"]
                     or ac["running"] or ac["busy"]
                     or sh["running"] or sh["busy"]
                     or cm["running"] or cm["busy"] or am["running"] or am["busy"] or vs["running"] or ct["running"] or pt["running"]),
        })

    def usable_variant(video_id: str, raw: str) -> ThumbnailVariant | None:
        """The stored variant with this id if it belongs to this video and has a picture, else None.
        The id is the only thing taken from the request."""
        if not raw.isascii() or not raw.isdigit() or len(raw) > MAX_ID_DIGITS:
            return None
        with Store(config.catalog_db) as store:
            variant = store.get_thumbnail_variant(int(raw))
        if variant is None or variant.video_id != video_id or variant.status == "failed":
            return None
        return variant

    @app.get("/videos/{video_id}/packaging")
    def packaging_preview(request: Request, video_id: str) -> Response:
        """The video among the channel's own top videos in a mock feed. No model is called."""
        data = catalog_data()
        if data is None:
            return back_to_setup()
        entry: Entry | None = next((e for e in data["entries"] if e.video.video_id == video_id),
                                   None)
        if entry is None:
            return video_not_found(request)
        video = entry.video
        chosen_raw = request.query_params.get("variant")
        chosen = usable_variant(video_id, chosen_raw) if chosen_raw is not None else None
        with Store(config.catalog_db) as store:
            row = store.get_draft(video_id)
            review = store.latest_packaging_review(video_id)
            variants = store.list_thumbnail_variants(review.id) if review is not None else []
        base = f"/videos/{quote(video_id)}/packaging"
        shown = [v for v in variants if v.status != "failed"]
        if chosen is not None and all(v.id != chosen.id for v in shown):
            shown.append(chosen)
        choices = [PreviewChoice("Current thumbnail", base, chosen is None)]
        choices += [PreviewChoice(f"Variant {n}", f"{base}?variant={v.id}",
                                  chosen is not None and v.id == chosen.id)
                    for n, v in enumerate(shown, 1)]
        image = (f"{base}/variants/{chosen.id}.jpg" if chosen is not None
                 else thumb_url(video, ("maxres", "medium", "default")))
        mine = preview_card(video, mine=True, title=row.title if row is not None else video.title,
                            thumb=image)
        feed = build_preview(mine, top_other_videos(video_id, [e.video for e in data["entries"]]))
        return render(request, "packaging_preview.html", {
            "active": "content", "video": video, "feed": feed, "choices": choices,
            "chosen": chosen, "bad_variant": chosen_raw is not None and chosen is None,
            "draft_title": row is not None,
        })

    @app.get("/videos/{video_id}/packaging/variants/{variant_id}.jpg")
    def packaging_variant_image(request: Request, video_id: str, variant_id: str) -> Response:
        """A variant's picture, by its stored id. The file name comes from the database."""
        missing = Response("There is no such picture.", status_code=404, media_type="text/plain")
        variant = usable_variant(video_id, variant_id)
        if variant is None:
            return missing
        try:
            path = packaging_mod.stored_image_path(state.images_dir, video_id, variant.image_file)
        except packaging_mod.PackagingError:
            return missing
        if not path.is_file():
            return missing
        headers = {}
        if request.query_params.get("download") == "1":
            headers["Content-Disposition"] = (
                f'attachment; filename="{video_id}-variant-{variant.id}.jpg"')
        return FileResponse(path, media_type="image/jpeg", headers=headers)

    def thumb_page(request: Request, *, video: Any, page: str, heading: str, new_image: str,
                   new_alt: str, back_url: str, back_text: str) -> Response:
        """The confirmation page for one thumbnail swap, with the job's state for this page."""
        with Store(config.catalog_db) as store:
            preview = QuotaGuard(store, config.daily_quota_budget).preview(thumbswap_mod.SWAP_PLAN)
        plan = {"lines": plan_lines(preview.lines), "total": preview.total, "used": preview.used,
                "remaining": preview.remaining, "fits": preview.fits,
                "budget": config.daily_quota_budget}
        snap, started = state.thumb_state()
        mine = started == page
        running = snap.running and mine
        busy = snap.running and not mine
        done = snap.status == "done" and mine
        failed = snap.message if snap.status == "failed" and mine else ""
        result = snap.result if done else None
        message = snap.message
        poll = snap.running
        if snap.status == "failed" and mine:
            state.dismiss_thumb_result()
        return render(request, "thumb_swap.html", {
            "active": "content", "video": video, "heading": heading,
            "current_image": thumb_url(video, ("maxres", "high", "medium", "default")),
            "new_image": new_image, "new_alt": new_alt, "action": page,
            "back_url": back_url, "back_text": back_text, "units": thumbswap_mod.swap_units(),
            "plan": plan, "has_write": state.write_credentials() is not None,
            "running": running, "busy": busy, "done": done, "failed": failed, "result": result,
            "message": message, "poll": poll,
        })

    @app.get("/videos/{video_id}/packaging/variants/{variant_id}/replace")
    def packaging_replace_page(request: Request, video_id: str, variant_id: str) -> Response:
        data = catalog_data()
        if data is None:
            return back_to_setup()
        entry = next((e for e in data["entries"] if e.video.video_id == video_id), None)
        if entry is None:
            return video_not_found(request)
        video = entry.video
        variant = usable_variant(video_id, variant_id)
        if variant is None:
            flash(request, "error", "There is no such variant idea.")
            return see_other(f"/videos/{quote(video_id)}#thumbnail")
        page = f"/videos/{quote(video_id)}/packaging/variants/{variant.id}/replace"
        return thumb_page(
            request, video=video, page=page, heading="Replace the current thumbnail",
            new_image=f"/videos/{quote(video_id)}/packaging/variants/{variant.id}.jpg",
            new_alt="Variant idea: " + variant.instruction,
            back_url=f"/videos/{quote(video_id)}#thumbnail", back_text="Back to the video")

    @app.post("/videos/{video_id}/packaging/variants/{variant_id}/replace")
    async def packaging_replace_do(request: Request, video_id: str, variant_id: str) -> Response:
        form = await read_form(request)
        variant = usable_variant(video_id, variant_id)
        if variant is None:
            flash(request, "error", "There is no such variant idea.")
            return see_other(f"/videos/{quote(video_id)}#thumbnail")
        page = f"/videos/{quote(video_id)}/packaging/variants/{variant.id}/replace"
        back = see_other(page)
        if form.get("confirm") != "yes":
            flash(request, "error",
                  "Tick the box to confirm that you want to change the thumbnail on YouTube. "
                  "Nothing was sent to YouTube.")
            return back
        creds = await run_in_threadpool(state.write_credentials)
        if creds is None:
            flash(request, "error",
                  "Yezatas does not have permission to edit your videos yet. Allow it on the Push "
                  "page first. Nothing was sent to YouTube.")
            return back
        try:
            data_bytes = packaging_mod.stored_image_path(
                state.images_dir, video_id, variant.image_file).read_bytes()
        except (packaging_mod.PackagingError, OSError):
            flash(request, "error", "The variant's picture is gone. Make the variants again.")
            return back
        ok = state.start_thumb_swap(creds, video_id, data_bytes, kind="replace",
                                    new_file=variant.image_file, variant_id=variant.id, page=page)
        if not ok:
            flash(request, "error", "A thumbnail change is already running.")
        return back

    @app.get("/history/thumbnails/{swap_id}/backup.jpg")
    def history_thumb_backup(request: Request, swap_id: int) -> Response:
        missing = Response("There is no such picture.", status_code=404, media_type="text/plain")
        swaps = thumbswap_mod.load_swaps(state.images_dir)
        swap = next((s for s in swaps if s.swap_id == swap_id), None)
        if swap is None:
            return missing
        try:
            path = packaging_mod.stored_image_path(state.images_dir, swap.video_id, swap.backup_file)
        except packaging_mod.PackagingError:
            return missing
        if not path.is_file():
            return missing
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/history/thumbnails/{swap_id}/restore")
    def history_thumb_restore_page(request: Request, swap_id: int) -> Response:
        swaps = thumbswap_mod.load_swaps(state.images_dir)
        swap = next((s for s in swaps if s.swap_id == swap_id), None)
        if swap is None:
            flash(request, "error", "There is no such thumbnail change.")
            return see_other("/history#thumbnails")
        data = catalog_data()
        if data is None:
            return back_to_setup()
        entry = next((e for e in data["entries"] if e.video.video_id == swap.video_id), None)
        if entry is None:
            return video_not_found(request)
        video = entry.video
        page = f"/history/thumbnails/{swap.swap_id}/restore"
        return thumb_page(
            request, video=video, page=page, heading="Put the old thumbnail back",
            new_image=f"/history/thumbnails/{swap.swap_id}/backup.jpg",
            new_alt=f"The thumbnail saved before change #{swap.swap_id}",
            back_url="/history#thumbnails", back_text="Back to History")

    @app.post("/history/thumbnails/{swap_id}/restore")
    async def history_thumb_restore_do(request: Request, swap_id: int) -> Response:
        form = await read_form(request)
        page = f"/history/thumbnails/{swap_id}/restore"
        back = see_other(page)
        if form.get("confirm") != "yes":
            flash(request, "error",
                  "Tick the box to confirm that you want to change the thumbnail on YouTube. "
                  "Nothing was sent to YouTube.")
            return back
        swaps = thumbswap_mod.load_swaps(state.images_dir)
        swap = next((s for s in swaps if s.swap_id == swap_id), None)
        if swap is None:
            flash(request, "error", "There is no such thumbnail change.")
            return see_other("/history#thumbnails")
        creds = await run_in_threadpool(state.write_credentials)
        if creds is None:
            flash(request, "error",
                  "Yezatas does not have permission to edit your videos yet. Allow it on the Push "
                  "page first. Nothing was sent to YouTube.")
            return back
        try:
            data_bytes = thumbswap_mod.backup_image(state.images_dir, swap)
        except thumbswap_mod.ThumbSwapError as exc:
            flash(request, "error", str(exc))
            return back
        ok = state.start_thumb_swap(creds, swap.video_id, data_bytes, kind="restore",
                                    new_file=swap.backup_file, variant_id=None, page=page)
        if not ok:
            flash(request, "error", "A thumbnail change is already running.")
        return back

    @app.post("/thumbnails/done")
    async def dismiss_thumb(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_thumb_result()
        back = str(form.get("back") or "")
        if back.startswith("/") and not back.startswith(("//", "/\\")):
            return see_other(back)
        return see_other("/history#thumbnails")

    @app.get("/outliers")
    def outliers_page(request: Request) -> Response:
        """The creator's own videos against their channel's norm for their age. No model is called."""
        with Store(config.catalog_db) as store:
            pull = store.latest_pull()
            videos = store.list_videos() if pull is not None else []
            report = store.latest_outlier_report() if pull is not None else None
        if pull is None:
            return back_to_setup()
        as_of = pull_moment(pull)
        view = build_outlier_view(videos, outliers_mod.score_videos(videos, as_of), report,
                                  int(pull["id"]), _local_time)
        snap = state.outliers_state()
        return render(request, "outliers.html", {
            "active": "outliers", "ov": view, "as_of": _local_time(as_of.isoformat()),
            "has_key": has_packaging_key(), "model": text_model_name(),
            "running": snap.running, "message": snap.message,
            "failed": snap.message if snap.status == "failed" else "",
            "poll": snap.running,
        })

    @app.get("/strategy")
    def strategy_page(request: Request) -> Response:
        """What works, what is missing and what to make next. Showing it calls no model."""
        with Store(config.catalog_db) as store:
            pull = store.latest_pull()
            if pull is None:
                return back_to_setup()
            videos = store.list_videos()
            as_of = pull_moment(pull)
            saved = strategy_mod.latest_report(config.home)
            view = build_strategy_view(saved, videos, outliers_mod.score_videos(videos, as_of),
                                       as_of, _local_time)
            estimate = ""
            context_error = ""
            if videos:
                try:
                    context = strategy_mod.build_context(store, as_of)
                except strategy_mod.StrategyError as exc:
                    context_error = str(exc)
                else:
                    input_tokens = int(len(context.text) / 4)
                    estimate = format_cost(agent.estimate_cost(
                        strategy_mod.STRATEGY_MODEL, input_tokens, 0, 4000))
            else:
                context_error = "There is no pulled catalog yet. Pull it in Configuration first."
        snap = state.strategy_state()
        return render(request, "strategy.html", {
            "active": "strategy", "view": view, "has_key": has_packaging_key(),
            "model": (config.verifier_model if primaryai.is_local(config)
                      else strategy_mod.STRATEGY_MODEL), "estimate": estimate,
            "estimate_cost": estimate, "context_error": context_error,
            "running": snap.running, "message": snap.message,
            "failed": snap.message if snap.status == "failed" else "",
            "poll": snap.running,
        })

    # --- search gaps: real search terms missing from titles (C5) ---
    @app.get("/search-gaps")
    def searchgaps_page(request: Request) -> Response:
        terms = searchgaps_mod.load_terms(config.home)
        with Store(config.catalog_db) as store:
            if store.latest_pull() is None:
                videos = []
            else:
                videos = store.list_videos("video_id")
        gaps = searchgaps_mod.find_gaps(terms, videos) if terms else []
        signin_snap, read_snap, rank_snap = state.searchgaps_state()
        return render(request, "searchgaps.html", {
            "active": "search",
            "connected": auth.can_read_analytics(config),
            "signed_in": config.token_path.is_file(),
            "rows": build_gap_rows(gaps),
            "gap_count": len(gaps),
            "terms_when": _local_time(terms.fetched_at) if terms else "",
            "terms_range": f"{terms.start_date} to {terms.end_date}" if terms else "",
            "calls": terms.calls if terms else 0,
            "ranking": build_ranking_view(searchgaps_mod.latest_ranking(config.home), _local_time),
            "has_key": has_packaging_key(),
            "model": text_model_name(),
            "days": searchgaps_mod.ANALYTICS_DAYS,
            "max_videos": searchgaps_mod.MAX_VIDEOS,
            "max_sent": searchgaps_mod.MAX_SENT,
            "max_picks": searchgaps_mod.MAX_PICKS,
            "estimate": format_cost(agent.estimate_cost(config.model, 2500, 0, 1200)),
            "signin_running": signin_snap.running,
            "signin_message": signin_snap.message,
            "signin_failed": signin_snap.message if signin_snap.status == "failed" else "",
            "read_running": read_snap.running,
            "read_message": read_snap.message,
            "read_failed": read_snap.message if read_snap.status == "failed" else "",
            "rank_running": rank_snap.running,
            "rank_message": rank_snap.message,
            "rank_failed": rank_snap.message if rank_snap.status == "failed" else "",
            "poll": signin_snap.running or read_snap.running or rank_snap.running,
        })

    @app.post("/search-gaps/connect")
    async def connect_searchgaps(request: Request) -> Response:
        await read_form(request)
        client, _ = read_saved_client(config)
        if client is None:
            flash(request, "error", "Add the OAuth client file first, in Configuration (step 5).")
        elif not state.start_analytics_sign_in():
            flash(request, "error", "A sign-in is already waiting for you in the browser.")
        return see_other("/search-gaps")

    @app.post("/search-gaps/connect/cancel")
    async def cancel_connect_searchgaps(request: Request) -> Response:
        await read_form(request)
        state.searchgaps_signin_job.cancel("Sign-in cancelled.")
        return see_other("/search-gaps")

    @app.post("/search-gaps/read")
    async def read_searchgaps(request: Request) -> Response:
        await read_form(request)
        creds = await run_in_threadpool(state.analytics_credentials)
        if creds is None:
            flash(request, "error", "Reading search terms needs the search-terms sign-in. "
                        "Click Connect search terms first.")
        elif not state.start_search_terms(creds):
            flash(request, "error", "The search terms are already being read. Wait for it to finish.")
        return see_other("/search-gaps")

    @app.post("/search-gaps/rank")
    async def rank_searchgaps(request: Request) -> Response:
        await read_form(request)
        has_key = await run_in_threadpool(has_packaging_key)
        if not has_key:
            flash(request, "error", NO_PACKAGING_KEY, "ai")
            return see_other("/search-gaps")
        terms = await run_in_threadpool(searchgaps_mod.load_terms, config.home)
        if terms is None:
            flash(request, "error", "Read the search terms first.")
            return see_other("/search-gaps")
        if not state.start_gap_rank():
            flash(request, "error", "The gaps are already being ranked. Wait for it to finish.")
        return see_other("/search-gaps")

    @app.post("/search-gaps/done")
    async def dismiss_searchgaps(request: Request) -> Response:
        await read_form(request)
        state.dismiss_searchgaps_results()
        return see_other("/search-gaps")

    def analytics_back(form: Any) -> str:
        back = str(form.get("back") or "")
        return back if back in ("/", "/analytics") else "/"

    @app.post("/analytics/connect")
    async def connect_analytics(request: Request) -> Response:
        form = await read_form(request)
        client, _ = read_saved_client(config)
        if client is None:
            flash(request, "error", "Add the OAuth client file first, in Configuration (step 5).")
        elif not state.start_analytics_sign_in():
            flash(request, "error", "A sign-in is already waiting for you in the browser.")
        return see_other(analytics_back(form))

    @app.post("/analytics/refresh")
    async def refresh_analytics(request: Request) -> Response:
        form = await read_form(request)
        creds = await run_in_threadpool(state.analytics_credentials)
        if creds is None:
            flash(request, "error", "Reading the channel's numbers needs the YouTube Analytics sign-in. "
                        "Click Connect Analytics first.")
        elif not state.start_channel_stats(creds):
            flash(request, "error", "The channel's numbers are already being read. Wait for it to finish.")
        return see_other(analytics_back(form))

    @app.post("/analytics/done")
    async def dismiss_analytics(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_channel_stats()
        return see_other(analytics_back(form))

    @app.get("/competitors")
    def competitors_page(request: Request) -> Response:
        cards = build_competitor_cards(competitors_mod.load_channels(config.home), _local_time)
        insight = build_insight_view(competitors_mod.latest_insight(config.home), _local_time)
        fetch_snap, explain_snap = state.competitors_state()
        return render(request, "competitors.html", {
            "active": "competitors",
            "cards": cards,
            "insight": insight,
            "signed_in": config.token_path.is_file(),
            "has_key": has_packaging_key(),
            "model": text_model_name(),
            "max_channels": competitors_mod.MAX_CHANNELS,
            "max_videos": competitors_mod.MAX_VIDEOS,
            "units": competitors_mod.estimate_units(),
            "remaining": remaining_quota(),
            "estimate": format_cost(agent.estimate_cost(config.model, 3000, 0, 1500)),
            "full": len(cards) >= competitors_mod.MAX_CHANNELS,
            "fetch_running": fetch_snap.running,
            "fetch_message": fetch_snap.message,
            "fetch_failed": fetch_snap.message if fetch_snap.status == "failed" else "",
            "explain_running": explain_snap.running,
            "explain_message": explain_snap.message,
            "explain_failed": explain_snap.message if explain_snap.status == "failed" else "",
            "poll": fetch_snap.running or explain_snap.running,
        })

    @app.get("/copilot")
    def copilot_page(request: Request) -> Response:
        """The chat about the channel. Showing it calls no model."""
        with Store(config.catalog_db) as store:
            pull = store.latest_pull()
            if pull is None:
                return back_to_setup()
            chat_id = state.copilot_chat(store)
            turns = store.list_copilot_turns(chat_id) if chat_id is not None else []
            proposals = store.list_copilot_proposals([t.id for t in turns])
            videos = {v.video_id: v for v in store.list_videos()}
            drafts = frozenset(row.video_id for row in store.list_drafts())
        snap = state.copilot_state()
        failed = snap.message if snap.status == "failed" else ""
        if snap.status == "failed":
            state.dismiss_copilot_result()  # shown once
        return render(request, "copilot.html", {
            "active": "copilot",
            "turns": build_copilot_turns(turns, proposals, videos, drafts, snap.running,
                                         _local_time),
            "as_of": _local_time(pull_moment(pull).isoformat()), "model": text_model_name(),
            "has_key": has_packaging_key(), "starters": STARTER_QUESTIONS,
            "limit": copilot_mod.MAX_QUESTION_CHARS, "running": snap.running,
            "message": snap.message, "failed": failed, "poll": snap.running,
        })

    @app.get("/links")
    def links_page(request: Request) -> Response:
        """Every link in the descriptions, ranked by reach. Showing it visits nothing."""
        with Store(config.catalog_db) as store:
            pull = store.latest_pull()
            videos = store.list_videos() if pull is not None else []
            checks = store.latest_link_checks() if pull is not None else {}
        if pull is None:
            return back_to_setup()
        view = build_links_view(links_mod.inventory(videos, checks), videos,
                                dict(request.query_params), _local_time)
        snap = state.links_state()
        summary = ""
        failed = snap.message if snap.status == "failed" else ""
        if snap.status == "done" and isinstance(snap.result, links_mod.LinkCheckSummary):
            summary = links_check_line(snap.result)
        if snap.status in ("done", "failed"):
            state.dismiss_links_result()  # shown once
        return render(request, "links.html", {
            "active": "links", "lv": view, "running": snap.running, "message": snap.message,
            "check_summary": summary, "failed": failed, "poll": snap.running,
            "max_links": links_mod.MAX_LINKS,
        })

    @app.get("/setup")
    def old_setup(request: Request) -> Response:
        """The Configuration page's old address: bookmarks and older links still land there."""
        query = request.url.query
        return RedirectResponse("/configuration" + (f"?{query}" if query else ""), status_code=307)

    @app.get("/configuration")
    def setup(request: Request) -> Response:
        data = page_data()
        data["flash"] = state.pop_flash(request.state.session_id)
        _, verifier = tools_or_none()
        data.update({
            "translator_key": baseten_key_source(),
            "translator_used": translate.translator_key_source(config),
            "translator_env": translate.TRANSLATOR_KEY_ENV,
            "translator_model": config.translator_model,
            "modal_key": translate.modal_key_source(config),
            "modal_env": translate.MODAL_KEY_ENV,
            "modal_url": translate.modal_url(config), "modal_model": config.modal_model,
            "modal_url_env": "MODAL_ENDPOINT_URL",
            "primary_ai": primaryai.load_primary_ai(config),
            "primary_ai_source": primaryai.primary_ai_source(config),
            "primary_ai_env": primaryai.PRIMARY_AI_ENV,
            "meta_key_set": bool(agent.load_api_key(config)),
            "asr_price": f"${transcribe_mod.ASR_PRICE_PER_HOUR:.2f}",
            "panns": soundtags_installed(),
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
            "packaging": _job_json(state.packaging_job.snapshot()),
            "outliers": _job_json(state.outliers_job.snapshot()),
            "strategy": _job_json(state.strategy_job.snapshot()),
            "competitors": _job_json(state.competitors_job.snapshot()),
            "competitors_explain": _job_json(state.competitors_explain_job.snapshot()),
            "searchgaps_signin": _job_json(state.searchgaps_signin_job.snapshot()),
            "searchgaps": _job_json(state.searchgaps_job.snapshot()),
            "searchgaps_rank": _job_json(state.searchgaps_rank_job.snapshot()),
            "channel_stats": _job_json(state.channel_stats_job.snapshot()),
            "video_stats": _job_json(state.video_stats_job.snapshot()),
            "copilot": _job_json(state.copilot_job.snapshot()),
            "bulk_agent": _job_json(state.bulk_agent_job.snapshot()),
            "links": _job_json(state.links_job.snapshot()),
            "adcheck": _job_json(state.adcheck_job.snapshot()),
            "category": _job_json(state.category_job.snapshot()),
            "pinned": _job_json(state.pinned_job.snapshot()),
            "shorts": _job_json(state.shorts_job.snapshot()),
            "comments": _job_json(state.comments_job.snapshot()),
            "assist": _job_json(state.assist_job.snapshot()),
            "channel_comments": _job_json(state.channel_comments_job.snapshot()),
            "channel_sort": _job_json(state.channel_sort_job.snapshot()),
            "variants": _job_json(state.variants_job.snapshot()),
            "thumb": _job_json(state.thumb_job.snapshot()),
            "check": None if check is None else {"ok": check.ok, "kind": check.kind},
        })

    @app.get("/api/activity")
    def api_activity() -> JSONResponse:
        """What the top bar shows on every page: the running and finished jobs, with the video's title."""
        items = state.activity()
        ids = {i["video_id"] for i in items if i["video_id"]}
        titles: dict[str, str] = {}
        if ids and config.catalog_db.exists():
            with Store(config.catalog_db) as store:
                for vid in ids:
                    v = store.get_video(vid)
                    if v is not None:
                        titles[vid] = v.title
        for i in items:
            i["title"] = titles.get(i["video_id"], "")
        return JSONResponse({"jobs": items})

    # --- actions (POST only, CSRF checked) ----------------------------------------------------
    @app.post("/configuration/client")
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

    @app.post("/configuration/signin")
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

    @app.post("/configuration/signin/cancel")
    async def cancel_sign_in(request: Request) -> Response:
        await read_form(request)
        state.signin_job.cancel("Sign-in cancelled.")
        return back_to_setup("signin")

    @app.post("/configuration/signout")
    async def sign_out(request: Request) -> Response:
        await read_form(request)
        removed = await run_in_threadpool(state.sign_out)
        state.add_flash(request.state.session_id, "ok",
                        "Signed out. The saved token was deleted." if removed
                        else "There was no saved sign-in.", "signin")
        return back_to_setup("signin")

    @app.post("/configuration/check")
    async def check_connection(request: Request) -> Response:
        await read_form(request)
        sid = request.state.session_id
        if not config.token_path.is_file():
            state.add_flash(sid, "error", "Sign in first (step 6).", "signin")
            return back_to_setup("signin")
        await run_in_threadpool(state.run_check)
        return back_to_setup("check")

    @app.post("/configuration/pull")
    async def start_pull(request: Request) -> Response:
        form = await read_form(request)
        sid = request.state.session_id
        from_today = str(form.get("back") or "") == "/"  # Today's "Pull again"

        def go_back(step: str) -> RedirectResponse:
            return see_other("/") if from_today else back_to_setup(step)

        check = state.check
        if check is None and config.token_path.is_file():
            # After a restart: check again on its own (1 quota unit), then pull.
            check = await run_in_threadpool(state.run_check)
        if check is None or not check.ok or not check.has_uploads:
            state.add_flash(sid, "error", "Check the connection first (step 7).", "check")
            return go_back("check")
        remaining = await run_in_threadpool(remaining_quota)
        needed = estimate_pull_units(check.video_count)
        if needed > remaining:
            state.add_flash(sid, "error",
                            f"The pull needs about {needed} quota units but only {remaining} are "
                            "left today. Quota resets at midnight Pacific time.", "pull")
            return go_back("pull")
        if not state.start_pull():
            state.add_flash(sid, "error", "A pull is already running.", "pull")
        return go_back("pull")

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
            return change_cards(edits.pending(store), lambda vid: store.latest_translations(vid).values())

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
            flash(request, "error", "Add the OAuth client file first, in Configuration (step 5).")
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
        swaps = thumbswap_mod.load_swaps(state.images_dir)
        with Store(config.catalog_db) as store:
            records = store.list_pushes(limit=HISTORY_LIMIT)
            run_ids = [r.run_id for r in records if r.run_id is not None]
            runs = store.get_push_runs(run_ids)
            run_records = {run_id: store.list_pushes_of_run(run_id) for run_id in runs}
            drafts = {row.video_id for row in store.list_drafts()}
            titles = {}
            for record in [*records, *(m for ms in run_records.values() for m in ms)]:
                video = store.get_video(record.video_id)
                if video is not None:
                    titles[record.video_id] = video.title
            swap_titles: dict[str, str] = {}
            for s in swaps:
                v = store.get_video(s.video_id)
                if v is not None:
                    swap_titles[s.video_id] = v.title
            posted = store.list_posted_comment_replies(limit=HISTORY_LIMIT)
            posted_titles: dict[str, str] = {}
            posted_authors: dict[str, str] = {}
            for r in posted:
                v = store.get_video(r.video_id)
                if v is not None:
                    posted_titles[r.video_id] = v.title
                answered = store.get_comment(r.comment_id) if r.comment_id else None
                if answered is not None:
                    posted_authors[r.comment_id] = answered.author
        return render(request, "history.html", {
            "active": "history",
            "groups": history_groups(records, runs, run_records, titles, _local_time, drafts),
            "limit": HISTORY_LIMIT, "shown": len(records),
            "swaps": build_swap_rows(swaps, swap_titles, _local_time),
            "posted": build_posted_rows(posted, posted_titles, posted_authors, _local_time),
        })

    @app.post("/history/runs/{run_id}/undo")
    async def undo_run(request: Request, run_id: int) -> Response:
        await read_form(request)

        def work() -> tuple[str, str, str]:
            with Store(config.catalog_db) as store:
                try:
                    result = edits.restore_run(store, run_id)
                except edits.DraftError as exc:
                    return "error", str(exc), "/history"
                titles = {}
                for item in result.items:
                    video = store.get_video(item.video_id)
                    if video is not None:
                        titles[item.video_id] = video.title
            made = result.count("drafted")
            message = run_restore_summary(result, titles)
            if made:
                return "ok", message + " Nothing changes on YouTube until you approve and push them.", "/changes"
            return "info", message, "/history"

        kind, message, target = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other(target)

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

    # --- bulk edit: the plain-words agent, and two forms that use no model -------------------------
    def agent_data() -> dict[str, Any]:
        """What the agent part of the Bulk edit page shows (worker thread: it opens its own Store)."""
        snap = state.bulk_agent_state()
        failed = snap.message if snap.status == "failed" else ""
        if snap.status == "failed":
            state.dismiss_bulk_agent_result()  # shown once
        session = state.bulk_session()
        data: dict[str, Any] = {
            "has_key": has_packaging_key(), "model": text_model_name(), "failed": failed,
            "request_max": bulkagent_mod.REQUEST_MAX, "examples": BULK_EXAMPLES, "session": None,
            "running": snap.running, "message": snap.message, "poll": snap.running,
            "review": None, "max_bulk": bulk.MAX_BULK}
        if session is None:
            return data
        with Store(config.catalog_db) as store:
            titles = {v.video_id: v.title for v in store.list_videos()}
            review = (None if snap.running
                      else build_agent_review(bulkagent_mod.check_staged(store, session)))
        data["session"] = build_agent_session(session, titles, snap.running)
        data["review"] = review
        return data

    def bulk_data(form: Any, error: str, preview: Any) -> dict[str, Any]:
        return {"active": "bulk", "form": form, "error": error, "preview": preview,
                "hidden": form.pairs(), "max_bulk": bulk.MAX_BULK, "agent": agent_data(),
                "poll": state.bulk_agent_job.snapshot().running}

    @app.get("/bulk")
    def bulk_page(request: Request) -> Response:
        params = request.query_params
        form = parse_bulk_form(params, submitted="preview" in params)
        if not form.submitted and "find" in params:  # a link from the Links page: prefill only
            form = replace(parse_bulk_form(params, submitted=True), submitted=False)
        error, preview = "", None
        if form.submitted:
            with Store(config.catalog_db) as store:
                try:
                    plan = bulk.plan_bulk(store, form.spec())
                except ValueError as exc:
                    error, plan = str(exc), []
                videos = {c.video_id: store.get_video(c.video_id) for c in plan}
            if not error:
                preview = build_bulk_preview(plan, {k: v for k, v in videos.items() if v})
        return render(request, "bulk.html", bulk_data(form, error, preview))



    @app.post("/bulk/apply")
    async def bulk_apply(request: Request) -> Response:
        posted = await read_form(request)
        form = parse_bulk_form(posted, submitted=True)
        ids = [str(i) for i in posted.getlist("video_id") if str(i)]

        def work() -> tuple[str, str, str]:
            if not ids:
                return "error", "Tick at least one video to make drafts.", form.preview_url()
            with Store(config.catalog_db) as store:
                try:
                    result = bulk.apply_bulk(store, form.spec(), ids)
                except ValueError as exc:
                    return "error", str(exc), form.preview_url()
            message = bulk_apply_message(result)
            if result.drafted:
                return "ok", message, "/changes"
            return "info", message, form.preview_url()

        kind, message, target = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other(target)

    # --- the bulk edit agent: every route is a POST with the csrf check; drafts only ---------------
    BULK_BUSY = "The bulk edit agent is still working. Wait for it to finish, or press Stop."

    def check_words(request: Request, text: str) -> bool:
        """Flash why the words cannot be sent; True when they are fine."""
        limit = bulkagent_mod.REQUEST_MAX
        if not text:
            flash(request, "error", "Describe the change in a few words first. Nothing was sent.")
        elif len(text) > limit:
            flash(request, "error", f"Your words have {len(text):,} characters; at most {limit:,} are "
                  "allowed. Nothing was sent.")
        elif not has_packaging_key():
            flash(request, "error", NO_PACKAGING_KEY, "ai")
        else:
            return True
        return False

    def has_pull() -> bool:
        with Store(config.catalog_db) as store:
            return store.latest_pull() is not None

    @app.post("/bulk/agent")
    async def bulk_agent_start(request: Request) -> Response:
        """Start a new session from the creator's words."""
        posted = await read_form(request)
        text = str(posted.get("request") or "").strip()
        if not await run_in_threadpool(has_pull):
            return back_to_setup()
        if state.bulk_agent_job.snapshot().running:
            flash(request, "error", BULK_BUSY)
        elif state.bulk_session() is not None:
            flash(request, "error", "A session is already open. Reply to it, or discard it first.")
        elif await run_in_threadpool(check_words, request, text):
            problem = state.start_bulk_agent(text, new=True)
            if problem:
                flash(request, "error", problem)
        return see_other("/bulk")

    @app.post("/bulk/agent/continue")
    async def bulk_agent_continue(request: Request) -> Response:
        """A reply in the same session: the agent goes on from what is staged."""
        posted = await read_form(request)
        text = str(posted.get("reply") or "").strip()
        if not await run_in_threadpool(has_pull):
            return back_to_setup()
        if state.bulk_agent_job.snapshot().running:
            flash(request, "error", BULK_BUSY)
        elif state.bulk_session() is None:
            flash(request, "error", "There is no session to continue. Start with your words.")
        elif await run_in_threadpool(check_words, request, text):
            problem = state.start_bulk_agent(text, new=False)
            if problem:
                flash(request, "error", problem)
        return see_other("/bulk")

    @app.post("/bulk/agent/stop")
    async def bulk_agent_stop(request: Request) -> Response:
        """Stop after the current step. Everything staged so far is kept."""
        await read_form(request)
        if state.stop_bulk_agent():
            flash(request, "info", "Stopping after the current step. What is staged is kept.")
        return see_other("/bulk")

    @app.post("/bulk/agent/apply")
    async def bulk_agent_apply(request: Request) -> Response:
        """Make drafts for the ticked videos. Drafts only: they still need approval on Changes."""
        posted = await read_form(request)
        ids = [str(i) for i in posted.getlist("video_id") if str(i)]
        session = state.bulk_session()

        def work() -> tuple[str, str, str]:
            if session is None:
                return "error", "There is nothing to make drafts from. Start with your words.", "/bulk"
            if state.bulk_agent_job.snapshot().running:
                return "error", BULK_BUSY, "/bulk"
            if not ids:
                return "error", "Tick at least one video to make drafts.", "/bulk"
            with Store(config.catalog_db) as store:
                try:
                    result = bulkagent_mod.apply(store, session, ids)
                except bulkagent_mod.BulkAgentError as exc:
                    return "error", str(exc), "/bulk"
            message = bulk_apply_message(result)
            if result.drafted:
                return "ok", message, "/changes"
            return "info", message, "/bulk"

        kind, message, target = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other(target)

    @app.post("/bulk/agent/discard")
    async def bulk_agent_discard(request: Request) -> Response:
        """Forget the session and what is staged in it. Nothing was written, so nothing is undone."""
        await read_form(request)
        if not state.discard_bulk_agent():
            flash(request, "error", "The agent is still working. Press Stop first.")
        return see_other("/bulk")

    @app.get("/api/bulk-agent")
    def api_bulk_agent(after: int = 0) -> JSONResponse:
        """The log lines written since line `after`, for the page that follows a running turn."""
        snap = state.bulk_agent_job.snapshot()
        session = state.bulk_session()
        lines, total, staged = ([], 0, 0) if session is None else session.log_page(max(after, 0))
        return JSONResponse({"status": snap.status, "message": snap.message, "log": lines,
                             "total": total, "staged": staged,
                             "stopping": bool(session is not None and session.stop.is_set())})

    # --- comments of the whole channel (/comments) ------------------------------------------------
    @app.get("/comments")
    def comments_page(request: Request) -> Response:
        with Store(config.catalog_db) as store:
            stored = store.list_recent_comments()
            replies = [*store.list_open_comment_replies(), *store.list_posted_comment_replies()]
            videos = {v.video_id: v for v in store.list_videos()}
            templates = store.list_reply_templates()
            to_fetch = comments_mod.newest_video_ids(store)
            to_sort = comments_mod.unsorted_video_ids(store.list_recent_comments(limit=5000))
        cv = build_channel_comments_view(stored, replies, videos, templates, dict(request.query_params))
        fetch = channel_fetch_panel(state.write_credentials() is not None, len(to_fetch))
        sort = channel_sort_panel(has_packaging_key(), len(to_sort))
        return render(request, "comments.html", {
            "active": "comments", "cv": cv, "fetch": fetch, "sort": sort,
            "poll": fetch["running"] or sort["running"]})

    # --- reply templates (G5) -------------------------------------------------------------------
    @app.get("/replies")
    def replies_page(request: Request) -> Response:
        with Store(config.catalog_db) as store:
            templates = store.list_reply_templates()
        return render(request, "replies.html", {"active": "replies", "templates": templates})

    def _template_id(raw: str) -> int | None:
        if not raw.isascii() or not raw.isdigit() or len(raw) > MAX_ID_DIGITS:
            return None
        return int(raw)

    @app.post("/replies/add")
    async def add_reply_template(request: Request) -> Response:
        form = await read_form(request)
        name, body = str(form.get("name") or ""), str(form.get("body") or "")

        def work() -> tuple[str, str]:
            with Store(config.catalog_db) as store:
                try:
                    replytemplates_mod.save_template(store, name, body)
                except replytemplates_mod.TemplateError as exc:
                    return "error", str(exc)
            return "ok", "Template saved."

        kind, message = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other("/replies")

    @app.post("/replies/{template_id}/edit")
    async def edit_reply_template(request: Request, template_id: str) -> Response:
        form = await read_form(request)
        name, body = str(form.get("name") or ""), str(form.get("body") or "")
        wanted = _template_id(template_id)

        def work() -> tuple[str, str]:
            if wanted is None:
                return "error", "This template does not exist."
            with Store(config.catalog_db) as store:
                if all(t.id != wanted for t in store.list_reply_templates()):
                    return "error", "This template does not exist."
                try:
                    replytemplates_mod.save_template(store, name, body, template_id=wanted)
                except replytemplates_mod.TemplateError as exc:
                    return "error", str(exc)
            return "ok", "Template saved."

        kind, message = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other("/replies")

    @app.post("/replies/{template_id}/delete")
    async def delete_reply_template(request: Request, template_id: str) -> Response:
        await read_form(request)
        wanted = _template_id(template_id)

        def work() -> tuple[str, str]:
            if wanted is None:
                return "error", "This template does not exist."
            with Store(config.catalog_db) as store:
                if not store.delete_reply_template(wanted):
                    return "error", "This template does not exist."
            return "ok", "Template deleted."

        kind, message = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other("/replies")

    # --- affiliate links (G2) ---------------------------------------------------------------------
    def affiliate_path() -> Path:
        return config.home / "affiliate.json"

    def load_affiliate_rules() -> "tuple[list[affiliate_mod.AffiliateRule], str]":
        try:
            return affiliate_mod.load_rules(affiliate_path()), ""
        except affiliate_mod.AffiliateError as exc:
            return [], str(exc)

    def affiliate_preview_url(visibility: str) -> str:
        return "/affiliate?" + urlencode([("visibility", visibility), ("preview", "1")])

    @app.get("/affiliate")
    def affiliate_page(request: Request) -> Response:
        rules, rules_error = load_affiliate_rules()
        params = request.query_params
        visibility = params.get("visibility") if params.get("visibility") in ("public", "all") else "public"
        preview = None
        if "preview" in params and not rules_error:
            with Store(config.catalog_db) as store:
                plan = affiliate_mod.plan_affiliate(store, rules, visibility)
                videos = {v.video_id: v for v in store.list_videos()}
            preview = build_affiliate_preview(plan, videos)
        return render(request, "affiliate.html", {
            "active": "affiliate", "rules": rules, "rules_error": rules_error,
            "preview": preview, "visibility": visibility})

    @app.post("/affiliate/rules/add")
    async def add_affiliate_rule(request: Request) -> Response:
        form = await read_form(request)

        def work() -> tuple[str, str]:
            rules, error = load_affiliate_rules()
            if error:
                return "error", error
            try:
                rule = parse_affiliate_rule_form(form)
            except affiliate_mod.AffiliateError as exc:
                return "error", str(exc)
            affiliate_mod.save_rules(affiliate_path(), [*rules, rule])
            return "ok", "Rule saved."

        kind, message = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other("/affiliate")

    @app.post("/affiliate/rules/delete")
    async def delete_affiliate_rule(request: Request) -> Response:
        form = await read_form(request)
        raw = str(form.get("index") or "")

        def work() -> tuple[str, str]:
            rules, error = load_affiliate_rules()
            if error:
                return "error", error
            if not raw.isascii() or not raw.isdigit() or int(raw) >= len(rules):
                return "error", "This rule does not exist."
            del rules[int(raw)]
            affiliate_mod.save_rules(affiliate_path(), rules)
            return "ok", "Rule deleted."

        kind, message = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other("/affiliate")

    @app.post("/affiliate/apply")
    async def affiliate_apply(request: Request) -> Response:
        posted = await read_form(request)
        visibility = str(posted.get("visibility") or "public")
        ids = [str(i) for i in posted.getlist("video_id") if str(i)]
        back = affiliate_preview_url(visibility if visibility in ("public", "all") else "public")

        def work() -> tuple[str, str, str]:
            if visibility not in ("public", "all"):
                return "error", "Choose which videos to change: public only, or all.", "/affiliate"
            if not ids:
                return "error", "Tick at least one video to make drafts.", back
            rules, error = load_affiliate_rules()
            if error:
                return "error", error, "/affiliate"
            with Store(config.catalog_db) as store:
                try:
                    result = affiliate_mod.apply_affiliate(store, rules, ids, visibility=visibility)
                except ValueError as exc:
                    return "error", str(exc), back
            message = affiliate_apply_message(result)
            if result.drafted:
                return "ok", message, "/changes"
            return "info", message, back

        kind, message, target = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other(target)

    # --- translations: the glossary (E7) ---
    def translations_glossary_path() -> Path:
        return config.home / glossary_mod.GLOSSARY_FILE

    def load_glossary_terms() -> "tuple[list[glossary_mod.GlossaryTerm], str]":
        try:
            return glossary_mod.load_glossary(translations_glossary_path()), ""
        except glossary_mod.GlossaryError as exc:
            return [], str(exc)

    @app.get("/translations")
    def translations_page(request: Request) -> Response:
        terms, glossary_error = load_glossary_terms()
        rows = build_glossary_rows(terms)
        outdated = None
        with Store(config.catalog_db) as store:
            if store.latest_pull() is not None:
                outdated = build_outdated_view(store)
        return render(request, "translations.html", {
            "active": "translations", "terms": rows, "glossary_error": glossary_error, "outdated": outdated})

    @app.post("/translations/glossary/add")
    async def add_glossary_term(request: Request) -> Response:
        form = await read_form(request)

        def work() -> tuple[str, str]:
            terms, error = load_glossary_terms()
            if error:
                return "error", error
            try:
                term = parse_glossary_form(form)
            except glossary_mod.GlossaryError as exc:
                return "error", str(exc)
            lowered = term.term.lower()
            for existing in terms:
                if existing.term.lower() == lowered:
                    return "error", "That term is already in the glossary."
            glossary_mod.save_glossary(translations_glossary_path(), [*terms, term])
            return "ok", f'Added the term "{term.term}" to the glossary.'

        kind, message = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other("/translations#glossary")

    @app.post("/translations/glossary/delete")
    async def delete_glossary_term(request: Request) -> Response:
        form = await read_form(request)
        raw = str(form.get("index") or "")

        def work() -> tuple[str, str]:
            terms, error = load_glossary_terms()
            if error:
                return "error", error
            if not raw.isascii() or not raw.isdigit():
                return "error", "That term is no longer in the glossary."
            idx = int(raw)
            if idx < 0 or idx >= len(terms):
                return "error", "That term is no longer in the glossary."
            removed = terms[idx]
            del terms[idx]
            glossary_mod.save_glossary(translations_glossary_path(), terms)
            return "ok", f'Removed the term "{removed.term}".'

        kind, message = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other("/translations#glossary")

    # --- custom checks (G7) -----------------------------------------------------------------------
    def checks_data(proposal: dict[str, Any] | None = None) -> dict[str, Any]:
        with Store(config.catalog_db) as store:
            rules = store.list_custom_rules()
            titles = {v.video_id: v.title for v in store.list_videos()}
            results: dict[int, Any] = {}
            results_error = ""
            enabled = [rule for rule in rules if rule.enabled]
            if enabled:
                try:
                    results = {found.rule.id: found
                               for found in customchecks_mod.run_rules(store, enabled)}
                except customchecks_mod.RuleError as exc:
                    results_error = str(exc)
        return {"active": "checks", "has_key": has_packaging_key(), "model": text_model_name(),
                "request_max": customchecks_mod.REQUEST_MAX, "proposal": proposal,
                "checks": build_checks_view(rules, results, titles),
                "results_error": results_error}

    @app.get("/checks")
    def checks_page(request: Request) -> Response:
        return render(request, "checks.html", checks_data())

    @app.post("/checks/ask")
    async def ask_custom_check(request: Request) -> Response:
        form = await read_form(request)
        text = str(form.get("request") or "")
        if not await run_in_threadpool(has_packaging_key):
            flash(request, "error", NO_PACKAGING_KEY, "ai")
            return see_other("/checks")
        try:
            draft = await run_in_threadpool(state.propose_rule, text)
        except (customchecks_mod.RuleError, agent.ModelError) as exc:
            flash(request, "error", str(exc))
            return see_other("/checks")
        total = draft.input_tokens + draft.output_tokens
        cost = agent.estimate_cost(draft.model, draft.input_tokens, draft.cached_tokens,
                                   draft.output_tokens)
        proposal = {"explain": draft.explain, "model": draft.model,
                    "tokens": f"{total:,}", "cost": format_cost(cost),
                    "name": draft.explain, "request": text.strip(),
                    "rule_json": json.dumps(draft.rule, ensure_ascii=False, sort_keys=True)}
        return render(request, "checks.html",
                      await run_in_threadpool(checks_data, proposal))

    @app.post("/checks/save")
    async def save_custom_check(request: Request) -> Response:
        form = await read_form(request)
        name_raw = str(form.get("name") or "").strip()
        request_text = str(form.get("request") or "").strip()
        rule_raw = str(form.get("rule") or "")

        def work() -> tuple[str, str]:
            try:
                rule, explain = parse_rule_proposal(rule_raw, request_text)
            except customchecks_mod.RuleError as exc:
                return "error", str(exc)
            with Store(config.catalog_db) as store:
                store.add_custom_rule(name_raw or explain, request_text, rule)
            return "ok", "Check saved."

        kind, message = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other("/checks")

    @app.post("/checks/discard")
    async def discard_custom_check(request: Request) -> Response:
        await read_form(request)
        return see_other("/checks")

    def _custom_rule_id(raw: str) -> int | None:
        if not raw.isascii() or not raw.isdigit() or len(raw) > MAX_ID_DIGITS:
            return None
        return int(raw)

    @app.post("/checks/{rule_id}/toggle")
    async def toggle_custom_check(request: Request, rule_id: str) -> Response:
        form = await read_form(request)
        wanted = _custom_rule_id(rule_id)
        enabled = str(form.get("enabled") or "") in ("1", "true")

        def work() -> tuple[str, str]:
            if wanted is None:
                return "error", "This check does not exist."
            with Store(config.catalog_db) as store:
                if all(rule.id != wanted for rule in store.list_custom_rules()):
                    return "error", "This check does not exist."
                store.set_custom_rule_enabled(wanted, enabled)
            return "ok", "Check turned on." if enabled else "Check turned off."

        kind, message = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other("/checks")

    @app.post("/checks/{rule_id}/delete")
    async def delete_custom_check(request: Request, rule_id: str) -> Response:
        await read_form(request)
        wanted = _custom_rule_id(rule_id)

        def work() -> tuple[str, str]:
            if wanted is None:
                return "error", "This check does not exist."
            with Store(config.catalog_db) as store:
                if not store.delete_custom_rule(wanted):
                    return "error", "This check does not exist."
            return "ok", "Check deleted."

        kind, message = await run_in_threadpool(work)
        flash(request, kind, message)
        return see_other("/checks")

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
        if not primaryai.text_ai_ready(config):
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
        back = see_other("/seo")
        raw = str(form.get("count") or "")
        count = int(raw) if raw.isdigit() and int(raw) in FIX_NEXT_COUNTS else 0
        if not count:
            flash(request, "error", "Choose 1, 3, 5 or 10 videos.")
            return back
        if not primaryai.text_ai_ready(config):
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
        return see_other(f"/videos/{quote(video_id)}" if video_id else "/seo")

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
            "active": "agent", "limit": AGENT_LOG_LIMIT, "model": text_model_name(),
            "rows": [run_view(r, titles, check_titles, _local_time, interrupted=not snap.running)
                     for r in runs],
        })

    # --- the Meta API key (optional, for the suggestions) --------------------------------------
    @app.post("/configuration/model-key")
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

    @app.post("/configuration/model-key/delete")
    async def delete_model_key(request: Request) -> Response:
        await read_form(request)
        removed = await run_in_threadpool(agent.delete_api_key, config)
        flash(request, "ok", "The saved key was deleted." if removed
              else "There was no saved key.", "ai")
        return back_to_setup("ai")

    # --- the Primary AI: Meta's Model API or Muse Glimmer on this computer ----------------------
    @app.post("/configuration/primary-ai")
    async def save_primary_ai(request: Request) -> Response:
        form = await read_form(request)
        choice = str(form.get("primary_ai") or "")
        if primaryai.primary_ai_source(config) == "environment":
            flash(request, "error", f"The choice comes from the {primaryai.PRIMARY_AI_ENV} "
                  "environment variable, so nothing was saved.", "ai")
        elif choice not in PRIMARY_AI_CHOICES:
            flash(request, "error", "Choose Meta's Model API or Muse Glimmer on this computer. "
                  "Nothing was saved.", "ai")
        else:
            try:
                await run_in_threadpool(primaryai.save_primary_ai, config, choice)
            except (ValueError, OSError):
                flash(request, "error", "The choice could not be saved. Check that "
                      f"{config.credentials_dir} is writable.", "ai")
            else:
                flash(request, "ok", "Primary AI is now "
                      + ("Muse Glimmer on this computer." if choice == "local"
                         else "Meta's Model API."), "ai")
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
    def video_language_of(video_id: str) -> tuple[bool, str]:
        """(the video exists, its audio or default language code or "")."""
        with Store(config.catalog_db) as store:
            video = store.get_video(video_id)
        if video is None:
            return False, ""
        return True, video.default_audio_language or video.default_language or ""

    @app.post("/videos/{video_id}/transcribe/captions")
    async def read_video_captions(request: Request, video_id: str) -> Response:
        """Use the captions already on YouTube: one background job, 250 quota units."""
        await read_form(request)
        back = see_other(f"/videos/{quote(video_id)}#chapters")
        found, language = await run_in_threadpool(video_language_of, video_id)
        if not found:
            return video_not_found(request)
        creds = await run_in_threadpool(state.write_credentials)
        if creds is None:
            flash(request, "error", "Reading captions needs the YouTube permission that pushing "
                  "uses. Nothing is changed on YouTube.")
        elif not state.start_captions(video_id, language, creds):
            flash(request, "error", "A transcription is already running. Wait for it to finish.")
        return back

    @app.post("/videos/{video_id}/transcribe")
    async def transcribe_video_file(request: Request, video_id: str) -> Response:
        """A video or audio file the browser uploads: speech goes to Muse Voice Transcribe, music
        and sounds are tagged on this computer. The file is a temporary copy, deleted when the
        job ends."""
        back = see_other(f"/videos/{quote(video_id)}#chapters")
        found, _ = await run_in_threadpool(video_language_of, video_id)
        if not found:
            return video_not_found(request)
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
            await read_form(request)  # the CSRF check; there is no file to take
            flash(request, "error", "Choose a video or audio file on this computer. "
                  "Nothing was sent.")
            return back

        def accept(fields: Any) -> None:
            """Runs when the file starts: nothing is written until the post is allowed."""
            verify_csrf(request, str(fields.get(CSRF_FIELD) or ""))
            kind = str(fields.get("kind") or "speech")
            if kind not in ("speech", "sounds"):
                raise upload_mod.UploadError("Choose whether the sound is speech or music and "
                                             "sounds. Nothing was sent.")
            if kind == "sounds" and not soundtags_mod.available():
                raise upload_mod.UploadError(soundtags_mod.MISSING_MESSAGE)
            if kind == "speech" and primaryai.is_local(config) and not agent.key_source(config):
                raise upload_mod.UploadError(LOCAL_NO_KEY_MESSAGE)
            if transcribe_mod.find_ffmpeg() is None:
                raise upload_mod.UploadError(transcribe_mod.NO_FFMPEG_MESSAGE)
            if state.transcribe_job.snapshot().running:
                raise upload_mod.UploadError("A transcription is already running. Wait for it "
                                             "to finish.")

        try:
            received = await upload_mod.receive_media(
                request, config.home / UPLOADS_DIR, suffixes=transcribe_mod.VIDEO_SUFFIXES,
                accept=accept)
        except upload_mod.UploadError as exc:
            flash(request, "error", str(exc))
            return back
        started = False
        try:
            fields = received.fields
            sounds = str(fields.get("kind") or "speech") == "sounds"
            words, problem = ([], "") if sounds else parse_keywords(str(fields.get("keywords") or ""))
            language = str(fields.get("language") or "")
            if problem:
                flash(request, "error", problem)
            elif not sounds and language not in transcribe_mod.ASR_LANGUAGES:
                flash(request, "error", "Choose the language that is spoken in the video. "
                      "Lithuanian and some other languages are not supported. Nothing was sent.")
            else:
                media = await run_in_threadpool(transcribe_mod.check_media_path, str(received.path))
                if sounds:
                    started = state.start_sound_tags(video_id, media, received.folder)
                else:
                    started = state.start_transcribe(video_id, media, language, words,
                                                     received.folder)
                if not started:
                    flash(request, "error", "A transcription is already running. Wait for it "
                          "to finish.")
        except transcribe_mod.TranscribeError as exc:
            flash(request, "error", f"{exc} Nothing was sent.")
        finally:
            if not started:
                received.discard()
        return back

    @app.post("/transcribe/done")
    async def dismiss_transcribe(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_transcribe_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}#chapters" if video_id else "/")

    @app.post("/videos/{video_id}/chapters")
    async def make_video_chapters(request: Request, video_id: str) -> Response:
        await read_form(request)
        back = see_other(f"/videos/{quote(video_id)}#chapters")

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
        return see_other(f"/videos/{quote(video_id)}#chapters" if video_id else "/")

    @app.post("/videos/{video_id}/analytics/refresh")
    async def refresh_video_analytics(request: Request, video_id: str) -> Response:
        form = await read_form(request)
        auto = str(form.get("auto") or "") == "1"  # sent by the page on opening: no jump to the card

        def exists() -> bool:
            with Store(config.catalog_db) as store:
                return store.get_video(video_id) is not None

        if not await run_in_threadpool(exists):
            flash(request, "error", "This video is not in your catalog. Pull your videos first.")
            return see_other("/videos")
        creds = await run_in_threadpool(state.analytics_credentials)
        if creds is None:
            flash(request, "error", "Reading this video's numbers needs the YouTube Analytics sign-in. "
                        "Connect Analytics on the Analytics page first.")
        elif not state.start_video_stats(creds, video_id):
            flash(request, "error", "Another video's numbers are being read. Wait for it to finish.")
        return see_other(f"/videos/{quote(video_id)}" + ("" if auto else "#video-analytics"))

    @app.post("/videos/{video_id}/analytics/done")
    async def dismiss_video_analytics(request: Request, video_id: str) -> Response:
        await read_form(request)
        state.dismiss_video_stats()
        return see_other(f"/videos/{quote(video_id)}#video-analytics")

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

    # --- packaging check and variant ideas ----------------------------------------------------
    @app.post("/videos/{video_id}/packaging")
    async def check_video_packaging(request: Request, video_id: str) -> Response:
        await read_form(request)
        back = see_other(f"/videos/{quote(video_id)}#thumbnail")

        def exists() -> bool:
            with Store(config.catalog_db) as store:
                return store.get_video(video_id) is not None

        if not await run_in_threadpool(exists):
            return video_not_found(request)
        if await run_in_threadpool(packaging_tools_or_none) is None:
            flash(request, "error", NO_PACKAGING_KEY, "ai")
        elif primaryai.meta_only_blocked(config):
            flash(request, "error", primaryai.meta_only_message("Subject and face sizes"), "ai")
        elif not state.start_packaging(video_id):
            flash(request, "error", "A packaging check is already running. Wait for it to finish.")
        return back

    @app.post("/packaging/done")
    async def dismiss_packaging(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_packaging_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}#thumbnail" if video_id else "/")

    # --- advertiser-friendly check (advice only) ---------------------------------------------
    @app.post("/videos/{video_id}/adcheck")
    async def check_video_ads(request: Request, video_id: str) -> Response:
        await read_form(request)

        def exists() -> bool:
            with Store(config.catalog_db) as store:
                return store.get_video(video_id) is not None

        if not await run_in_threadpool(exists):
            return video_not_found(request)
        if not await run_in_threadpool(has_packaging_key):
            flash(request, "error", NO_PACKAGING_KEY, "ai")
        elif not state.start_adcheck(video_id):
            flash(request, "error", "An advertiser-friendly check is already running. Wait for it "
                                    "to finish.")
        return see_other(f"/videos/{quote(video_id)}#ad-check")

    @app.post("/videos/{video_id}/shorts")
    async def start_shorts(request: Request, video_id: str) -> Response:
        await read_form(request)

        def exists() -> bool:
            with Store(config.catalog_db) as store:
                return store.get_video(video_id) is not None

        if not await run_in_threadpool(exists):
            return video_not_found(request)
        if not await run_in_threadpool(has_shorts_key):
            flash(request, "error", NO_SHORTS_KEY, "ai")
        elif not state.start_shorts(video_id):
            flash(request, "error", "A Shorts search is already running. Wait for it to finish.")
        return see_other(f"/videos/{quote(video_id)}#shorts")

    @app.post("/shorts/done")
    async def dismiss_shorts(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_shorts_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}#shorts" if video_id else "/")

    # --- the category check (V7) -------------------------------------------------------------------
    @app.post("/videos/{video_id}/category/check")
    async def check_video_category(request: Request, video_id: str) -> Response:
        await read_form(request)

        def exists() -> bool:
            with Store(config.catalog_db) as store:
                return store.get_video(video_id) is not None

        if not await run_in_threadpool(exists):
            return video_not_found(request)
        if not await run_in_threadpool(has_packaging_key):
            flash(request, "error", NO_PACKAGING_KEY, "ai")
        elif not state.start_category(video_id):
            flash(request, "error", "A category check is already running. Wait for it to finish.")
        return see_other(f"/videos/{quote(video_id)}#category")

    @app.post("/videos/{video_id}/category/correct")
    async def confirm_video_category(request: Request, video_id: str) -> Response:
        await read_form(request)

        def confirm() -> bool:
            with Store(config.catalog_db) as store:
                return store.confirm_category(video_id)

        if await run_in_threadpool(confirm):
            flash(request, "ok", "Marked the category as correct. The check passes now.")
        else:
            flash(request, "error", "Check the category first: there is no finished check to mark as correct.")
        return see_other(f"/videos/{quote(video_id)}#category")

    # --- the pinned comment (V8) -------------------------------------------------------------------
    @app.post("/videos/{video_id}/pinned/suggest")
    async def suggest_pinned_comment(request: Request, video_id: str) -> Response:
        await read_form(request)

        def exists() -> bool:
            with Store(config.catalog_db) as store:
                return store.get_video(video_id) is not None

        if not await run_in_threadpool(exists):
            return video_not_found(request)
        if not await run_in_threadpool(has_packaging_key):
            flash(request, "error", NO_PACKAGING_KEY, "ai")
        elif not state.start_pinned(video_id):
            flash(request, "error", "A pinned comment is already being written. Wait for it to finish.")
        return see_other(f"/videos/{quote(video_id)}#pinned-comment")

    @app.post("/videos/{video_id}/pinned/mark")
    async def mark_pinned_comment(request: Request, video_id: str) -> Response:
        await read_form(request)

        def mark() -> bool:
            with Store(config.catalog_db) as store:
                if store.get_video(video_id) is None:
                    return False
                store.mark_pinned(video_id)
                return True

        if not await run_in_threadpool(mark):
            return video_not_found(request)
        flash(request, "ok", "Marked: a pinned comment is up. The check passes now.")
        return see_other(f"/videos/{quote(video_id)}#pinned-comment")

    @app.post("/pinned/done")
    async def dismiss_pinned(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_pinned_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}#pinned-comment" if video_id else "/")

    @app.post("/category/done")
    async def dismiss_category(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_category_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}#category" if video_id else "/")

    @app.post("/adcheck/done")
    async def dismiss_adcheck(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_adcheck_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}#ad-check" if video_id else "/")

    # --- comments on one video (G4, G5) ---------------------------------------------------------
    @app.post("/videos/{video_id}/comments/fetch")
    async def fetch_video_comments(request: Request, video_id: str) -> Response:
        await read_form(request)
        back = see_other(f"/videos/{quote(video_id)}#comments")

        def exists() -> bool:
            with Store(config.catalog_db) as store:
                return store.get_video(video_id) is not None

        if not await run_in_threadpool(exists):
            return video_not_found(request)
        creds = await run_in_threadpool(state.write_credentials)
        if creds is None:
            flash(request, "error", "Reading comments needs the YouTube permission that pushing uses. Nothing is posted.")
        elif not state.start_comments(video_id, creds):
            flash(request, "error", "The comments are already being read. Wait for it to finish.")
        return back

    @app.post("/comments/fetch/done")
    async def dismiss_comments_fetch(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_comments_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}#comments" if video_id else "/")

    @app.post("/videos/{video_id}/comments/assist")
    async def assist_video_comments(request: Request, video_id: str) -> Response:
        await read_form(request)
        back = see_other(f"/videos/{quote(video_id)}#comments")

        def lookup() -> tuple[bool, bool]:
            with Store(config.catalog_db) as store:
                if store.get_video(video_id) is None:
                    return False, False
                return True, bool(store.list_comments(video_id))

        found, has_stored = await run_in_threadpool(lookup)
        if not found:
            return video_not_found(request)
        if not await run_in_threadpool(has_packaging_key):
            flash(request, "error", NO_PACKAGING_KEY, "ai")
        elif not has_stored:
            flash(request, "error", "There are no comments to sort yet. Fetch the comments first.")
        elif not state.start_assist(video_id):
            flash(request, "error", "The comments are already being sorted. Wait for it to finish.")
        return back

    @app.post("/comments/assist/done")
    async def dismiss_comments_assist(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_assist_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}#comments" if video_id else "/")

    @app.post("/comments/fetch")
    async def fetch_channel_comments_route(request: Request) -> Response:
        await read_form(request)
        creds = await run_in_threadpool(state.write_credentials)

        def newest() -> list[str]:
            with Store(config.catalog_db) as store:
                return comments_mod.newest_video_ids(store)

        ids = await run_in_threadpool(newest)
        if creds is None:
            flash(request, "error", "Reading comments needs the YouTube permission that pushing uses. Nothing is posted.")
        elif not ids:
            flash(request, "error", "There are no videos yet. Pull your videos first.")
        elif state.comments_job.snapshot().running or not state.start_channel_comments(ids, creds):
            flash(request, "error", "The comments are already being read. Wait for it to finish.")
        return see_other(CHANNEL_COMMENTS_PATH)

    @app.post("/comments/sort")
    async def sort_channel_comments_route(request: Request) -> Response:
        await read_form(request)
        if not await run_in_threadpool(has_packaging_key):
            flash(request, "error", NO_PACKAGING_KEY, "ai")
            return see_other(CHANNEL_COMMENTS_PATH)

        def need_sort() -> list[str]:
            with Store(config.catalog_db) as store:
                return comments_mod.unsorted_video_ids(store.list_recent_comments(limit=5000))

        ids = await run_in_threadpool(need_sort)
        if not ids:
            flash(request, "error", "Every stored comment is sorted already.")
        elif state.assist_job.snapshot().running or not state.start_channel_sort(ids):
            flash(request, "error", "The comments are already being sorted. Wait for it to finish.")
        return see_other(CHANNEL_COMMENTS_PATH)

    @app.post("/comments/channel/fetch/done")
    async def dismiss_channel_fetch(request: Request) -> Response:
        await read_form(request)
        state.dismiss_channel_comments_result()
        return see_other(CHANNEL_COMMENTS_PATH)

    @app.post("/comments/channel/sort/done")
    async def dismiss_channel_sort(request: Request) -> Response:
        await read_form(request)
        state.dismiss_channel_sort_result()
        return see_other(CHANNEL_COMMENTS_PATH)

    async def set_reply_status(request: Request, video_id: str, reply_id: str, status: str) -> Response:
        form = await read_form(request)
        back = see_other(comments_back(str(form.get("next") or ""))
                         or f"/videos/{quote(video_id)}#comments")

        def work() -> str:
            with Store(config.catalog_db) as store:
                if store.get_video(video_id) is None:
                    return "missing_video"
                if not reply_id.isascii() or not reply_id.isdigit() or len(reply_id) > MAX_ID_DIGITS:
                    return "missing_reply"
                wanted = int(reply_id)
                found = next((r for r in store.list_comment_replies(video_id) if r.id == wanted), None)
                if found is None:
                    return "missing_reply"
                if found.status == "posted":
                    return "posted"
                store.set_comment_reply_status(found.id, status)
                return "ok"

        outcome = await run_in_threadpool(work)
        if outcome == "missing_video":
            return video_not_found(request)
        if outcome == "missing_reply":
            flash(request, "error", "This draft does not exist.")
        elif outcome == "posted":
            flash(request, "error", "Yezatas already posted this reply, so it stays as it is.")
        return back

    # --- posting a reply draft: a confirmation page first, the post only from its button ----------
    def post_context(video_id: str, reply_id: str) -> tuple[str, Any, Any, Any, Any]:
        """("ok", video, draft, comment, quota preview), or (problem code, None, None, None, None)."""
        nothing: tuple[None, None, None, None] = (None, None, None, None)
        with Store(config.catalog_db) as store:
            video = store.get_video(video_id)
            if video is None:
                return ("missing_video", *nothing)
            if not reply_id.isascii() or not reply_id.isdigit() or len(reply_id) > MAX_ID_DIGITS:
                return ("missing_reply", *nothing)
            draft = store.get_comment_reply(int(reply_id))
            if draft is None or draft.video_id != video_id:
                return ("missing_reply", *nothing)
            if draft.status != "open":
                return (draft.status, *nothing)
            comment = store.get_comment(draft.comment_id) if draft.comment_id else None
            preview = QuotaGuard(store, config.daily_quota_budget).preview(
                [(postreply_mod.operation_for(not draft.comment_id), 1)])
        return "ok", video, draft, comment, preview

    def post_problem(request: Request, code: str, back: str) -> Response:
        if code == "missing_video":
            return video_not_found(request)
        if code == "missing_reply":
            flash(request, "error", "This draft does not exist.")
        elif code == "posted":
            flash(request, "info", "This reply was already posted. Nothing was sent.")
        else:
            flash(request, "error", "This draft was put away, so it cannot be posted.")
        return see_other(back)

    def post_page(request: Request, video: Any, draft: Any, comment: Any, preview: Any, text: str,
                  next_path: str, *, reconnect: bool = False) -> Response:
        has_write = not reconnect and state.write_credentials() is not None
        return render(request, "post_reply.html", {
            "active": "comments", "video": video, "draft": draft, "comment": comment, "text": text,
            "pinned_draft": not draft.comment_id, "next_path": next_path,
            "back_url": comments_back(next_path) or f"/videos/{quote(video.video_id)}#comments",
            "action": f"/videos/{quote(video.video_id)}/comments/replies/{draft.id}/post",
            "units": preview.total, "remaining": preview.remaining, "fits": preview.fits,
            "budget": config.daily_quota_budget, "has_write": has_write,
            "limit": postreply_mod.MAX_CHARS})

    @app.get("/videos/{video_id}/comments/replies/{reply_id}/post")
    def post_reply_page(request: Request, video_id: str, reply_id: str) -> Response:
        next_path = str(request.query_params.get("next") or "")
        back = comments_back(next_path) or f"/videos/{quote(video_id)}#comments"
        code, video, draft, comment, preview = post_context(video_id, reply_id)
        if code != "ok":
            return post_problem(request, code, back)
        return post_page(request, video, draft, comment, preview, draft.text, next_path)

    @app.post("/videos/{video_id}/comments/replies/{reply_id}/post")
    async def post_reply_route(request: Request, video_id: str, reply_id: str) -> Response:
        form = await read_form(request)
        next_path = str(form.get("next") or "")
        back = comments_back(next_path) or f"/videos/{quote(video_id)}#comments"
        text = str(form.get("text") or "")
        code, video, draft, comment, preview = await run_in_threadpool(post_context, video_id, reply_id)
        if code != "ok":
            return post_problem(request, code, back)

        def again(message: str, *, reconnect: bool = False) -> Response:
            flash(request, "error", message)
            return post_page(request, video, draft, comment, preview, text, next_path,
                             reconnect=reconnect)

        if form.get("confirm") != "yes":
            return again("Check the text and press Post to YouTube. Nothing was posted.")
        creds = await run_in_threadpool(state.write_credentials)
        if creds is None:
            return again(postreply_mod.SCOPE_MESSAGE, reconnect=True)

        def work() -> "postreply_mod.PostedReply":
            try:
                writer = state.make_comment_writer(creds)
            except Exception as exc:  # noqa: BLE001
                raise postreply_mod.PostReplyError(
                    "api", f"Could not start the connection to YouTube ({exc.__class__.__name__}). "
                           "Nothing was posted.") from exc
            with Store(config.catalog_db) as store:
                guard = QuotaGuard(store, config.daily_quota_budget)
                return postreply_mod.post_reply(store, writer, guard, draft.id, text)

        try:
            posted = await run_in_threadpool(work)
        except postreply_mod.PostReplyError as exc:
            if exc.kind in ("missing", "already", "closed"):
                flash(request, "error", str(exc))
                return see_other(back)
            return again(str(exc), reconnect=exc.kind == "scope")
        if posted.pinned_draft:
            flash(request, "ok", f"Posted the comment on YouTube ({posted.units} quota units). "
                                 "YouTube does not let apps pin comments: open YouTube Studio and "
                                 "pin it there.")
        else:
            who = f" to {comment.author}" if comment is not None and comment.author else ""
            flash(request, "ok", f"Posted your reply{who} on YouTube ({posted.units} quota units).")
        return see_other(back)

    @app.post("/videos/{video_id}/comments/replies/{reply_id}/used")
    async def mark_reply_used(request: Request, video_id: str, reply_id: str) -> Response:
        return await set_reply_status(request, video_id, reply_id, "used")

    @app.post("/videos/{video_id}/comments/replies/{reply_id}/dismiss")
    async def dismiss_reply(request: Request, video_id: str, reply_id: str) -> Response:
        return await set_reply_status(request, video_id, reply_id, "dismissed")

    # --- outliers: what the best videos share ------------------------------------------------------
    @app.post("/outliers/explain")
    async def explain_video_outliers(request: Request) -> Response:
        await read_form(request)

        def has_pull() -> bool:
            with Store(config.catalog_db) as store:
                return store.latest_pull() is not None

        if not await run_in_threadpool(has_pull):
            return back_to_setup()
        if await run_in_threadpool(packaging_tools_or_none) is None:
            flash(request, "error", NO_PACKAGING_KEY, "ai")
        elif not state.start_outliers():
            flash(request, "error", "An explanation is already running. Wait for it to finish.")
        return see_other("/outliers#why")

    @app.post("/outliers/done")
    async def dismiss_outliers(request: Request) -> Response:
        await read_form(request)
        state.dismiss_outliers_result()
        return see_other("/outliers#why")

    # --- strategy: what works, what is missing and what to make next ----------------------------
    @app.post("/strategy/run")
    async def run_strategy(request: Request) -> Response:
        await read_form(request)

        def has_pull() -> bool:
            with Store(config.catalog_db) as store:
                return store.latest_pull() is not None

        if not await run_in_threadpool(has_pull):
            return back_to_setup()
        if not await run_in_threadpool(has_packaging_key):
            flash(request, "error", NO_PACKAGING_KEY, "ai")
        elif not state.start_strategy():
            flash(request, "error", "A strategy report is already being made. Wait for it to finish.")
        return see_other("/strategy")

    @app.post("/strategy/done")
    async def dismiss_strategy(request: Request) -> Response:
        await read_form(request)
        state.dismiss_strategy_result()
        return see_other("/strategy")

    # --- competitors: outliers on other public channels (C8) -----------------------------------------
    @app.post("/competitors/add")
    async def add_competitor(request: Request) -> Response:
        form = await read_form(request)
        text = str(form.get("channel") or "")
        try:
            ref = competitors_mod.parse_channel(text)
        except competitors_mod.CompetitorError as exc:
            flash(request, "error", str(exc))
            return see_other("/competitors")
        count = await run_in_threadpool(competitors_mod.channel_count, config.home)
        if count >= competitors_mod.MAX_CHANNELS:
            flash(request, "error", "Yezatas follows at most 5 channels. Remove one first.")
            return see_other("/competitors")
        creds = await run_in_threadpool(state.read_credentials)
        if creds is None:
            flash(request, "error", "Reading other channels needs the YouTube sign-in. Sign in in Configuration first.")
            return see_other("/competitors")
        ok = state.start_competitor_fetch(ref, creds)
        if not ok:
            flash(request, "error", "A channel is already being read. Wait for it to finish.")
        return see_other("/competitors")

    @app.post("/competitors/refresh")
    async def refresh_competitor(request: Request) -> Response:
        form = await read_form(request)
        channel_id = str(form.get("channel_id") or "")
        loaded = await run_in_threadpool(competitors_mod.load_channels, config.home)
        if not any(c.channel_id == channel_id for c in loaded):
            flash(request, "error", "That channel is not on your list.")
            return see_other("/competitors")
        creds = await run_in_threadpool(state.read_credentials)
        if creds is None:
            flash(request, "error", "Reading other channels needs the YouTube sign-in. Sign in in Configuration first.")
            return see_other("/competitors")
        ok = state.start_competitor_fetch(competitors_mod.ChannelRef("id", channel_id), creds)
        if not ok:
            flash(request, "error", "A channel is already being read. Wait for it to finish.")
        return see_other("/competitors")

    @app.post("/competitors/remove")
    async def remove_competitor(request: Request) -> Response:
        form = await read_form(request)
        channel_id = str(form.get("channel_id") or "")
        ok = await run_in_threadpool(competitors_mod.remove_channel, config.home, channel_id)
        if ok:
            flash(request, "ok", "The channel was removed from your list.")
        else:
            flash(request, "error", "That channel is not on your list.")
        return see_other("/competitors")

    @app.post("/competitors/explain")
    async def explain_competitors(request: Request) -> Response:
        form = await read_form(request)
        has_key = await run_in_threadpool(has_packaging_key)
        if not has_key:
            flash(request, "error", NO_PACKAGING_KEY, "ai")
            return see_other("/competitors")
        ok = state.start_competitor_explain()
        if not ok:
            flash(request, "error", "The outliers are already being explained. Wait for it to finish.")
        return see_other("/competitors")

    @app.post("/competitors/done")
    async def dismiss_competitors(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_competitors_results()
        return see_other("/competitors")

    # --- links: visit each external link once -----------------------------------------------------
    @app.post("/links/check")
    async def check_links(request: Request) -> Response:
        await read_form(request)

        def has_pull() -> bool:
            with Store(config.catalog_db) as store:
                return store.latest_pull() is not None

        if not await run_in_threadpool(has_pull):
            return back_to_setup()
        if not state.start_links():
            flash(request, "error", "A link check is already running. Wait for it to finish.")
        return see_other("/links")

    # --- copilot: questions about the channel -----------------------------------------------------
    @app.post("/copilot/ask")
    async def ask_copilot(request: Request) -> Response:
        form = await read_form(request)
        question = str(form.get("question") or "").strip()
        limit = copilot_mod.MAX_QUESTION_CHARS

        def has_pull() -> bool:
            with Store(config.catalog_db) as store:
                return store.latest_pull() is not None

        def chat() -> int:
            with Store(config.catalog_db) as store:
                found = state.copilot_chat(store)
                return found if found is not None else store.next_copilot_chat_id()

        if not await run_in_threadpool(has_pull):
            return back_to_setup()
        if not question:
            flash(request, "error", "Type a question first. Nothing was sent.")
        elif len(question) > limit:
            flash(request, "error", f"The question has {len(question):,} characters; at most "
                  f"{limit:,} are allowed. Nothing was sent.")
        elif not await run_in_threadpool(has_packaging_key):
            flash(request, "error", NO_PACKAGING_KEY, "ai")
        elif not state.start_copilot(question, await run_in_threadpool(chat)):
            flash(request, "error", "An answer is already being written. Wait for it to finish.")
        return see_other("/copilot")

    @app.post("/copilot/new")
    async def new_copilot_chat(request: Request) -> Response:
        await read_form(request)

        def start() -> None:
            with Store(config.catalog_db) as store:
                state.new_copilot_chat(store)

        if state.copilot_job.snapshot().running:
            flash(request, "error", "An answer is being written. Wait for it to finish.")
        else:
            await run_in_threadpool(start)
        return see_other("/copilot")

    def proposal_turn(raw: str) -> tuple[int, int] | None:
        """(proposal id, id of its turn) for a posted id that names a stored proposal, else None."""
        if not raw.isascii() or not raw.isdigit() or len(raw) > MAX_ID_DIGITS:
            return None
        with Store(config.catalog_db) as store:
            found = store.get_copilot_proposal(int(raw))
        return (found.id, found.turn_id) if found is not None else None

    @app.post("/copilot/proposals/{proposal_id}/draft")
    async def draft_copilot_proposal(request: Request, proposal_id: str) -> Response:
        await read_form(request)
        found = await run_in_threadpool(proposal_turn, proposal_id)
        if found is None:
            flash(request, "error", "This proposal does not exist.")
            return see_other("/copilot")

        def make() -> tuple[str, str]:
            with Store(config.catalog_db) as store:
                try:
                    video_id = copilot_mod.make_draft(store, found[0])
                except copilot_mod.CopilotError as exc:
                    return "error", str(exc)
                video = store.get_video(video_id)
            name = video.title if video else video_id
            return "ok", (f"Draft saved for \"{name}\". Review it on the Changes page; nothing "
                          "changes on YouTube until you approve and push.")

        kind, text = await run_in_threadpool(make)
        flash(request, kind, text)
        return see_other(f"/copilot#turn-{found[1]}")

    @app.post("/copilot/proposals/{proposal_id}/dismiss")
    async def dismiss_copilot_proposal(request: Request, proposal_id: str) -> Response:
        await read_form(request)
        found = await run_in_threadpool(proposal_turn, proposal_id)
        if found is None:
            flash(request, "error", "This proposal does not exist.")
            return see_other("/copilot")

        def dismiss() -> None:
            with Store(config.catalog_db) as store:
                copilot_mod.dismiss_proposal(store, found[0])

        await run_in_threadpool(dismiss)
        return see_other(f"/copilot#turn-{found[1]}")

    @app.post("/videos/{video_id}/packaging/variants")
    async def make_video_variants(request: Request, video_id: str) -> Response:
        await read_form(request)
        back = see_other(f"/videos/{quote(video_id)}#thumbnail")

        def lookup() -> tuple[bool, int]:
            """(the video exists, the id of its newest check if it has ideas to work from, else 0)."""
            with Store(config.catalog_db) as store:
                if store.get_video(video_id) is None:
                    return False, 0
                review = store.latest_packaging_review(video_id)
                if review is None or review.status != "done" or not review.image_file:
                    return True, 0
                ideas = review.critique.get("ideas") if isinstance(review.critique, dict) else None
                return True, review.id if isinstance(ideas, list) and ideas else 0

        found, review_id = await run_in_threadpool(lookup)
        if not found:
            return video_not_found(request)
        if not review_id:
            flash(request, "error", "There is no packaging check with ideas for this video yet. "
                  "Run the check first. Nothing was sent.")
        elif await run_in_threadpool(packaging_tools_or_none) is None:
            flash(request, "error", NO_PACKAGING_KEY, "ai")
        elif primaryai.meta_only_blocked(config):
            flash(request, "error", primaryai.meta_only_message("Thumbnail variants"), "ai")
        elif not state.start_variants(video_id, review_id):
            flash(request, "error", "Variants are already being made. Wait for them to finish.")
        return back

    @app.post("/packaging/variants/done")
    async def dismiss_variants(request: Request) -> Response:
        form = await read_form(request)
        state.dismiss_variants_result()
        video_id = str(form.get("video_id") or "")
        return see_other(f"/videos/{quote(video_id)}#thumbnail" if video_id else "/")

    # --- the translation key (optional, Baseten; the Meta key is the fallback) -------------------
    @app.post("/configuration/translator-key")
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

    @app.post("/configuration/translator-key/delete")
    async def delete_translator_key(request: Request) -> Response:
        await read_form(request)
        removed = await run_in_threadpool(translate.delete_translator_key, config)
        flash(request, "ok", "The saved key was deleted." if removed
              else "There was no saved key.", "translate")
        return back_to_setup("translate")

    # --- the Modal key (optional): Spark Open on another host ------------------------------------
    @app.post("/configuration/modal-key")
    async def save_modal_key(request: Request) -> Response:
        form = await read_form(request)
        key = str(form.get("api_key") or "").strip()
        if translate.modal_key_source(config) == "environment":
            flash(request, "error", f"The key comes from the {translate.MODAL_KEY_ENV} "
                  "environment variable, so nothing was saved.", "translate")
        elif not key:
            flash(request, "error", "Paste your key first.", "translate")
        elif len(key) > MAX_KEY_CHARS or any(c.isspace() for c in key):
            flash(request, "error", "That does not look like a key: it must be one line with no "
                  "spaces. Nothing was saved.", "translate")
        else:
            try:
                await run_in_threadpool(translate.save_modal_key, config, key)
            except (ValueError, OSError):
                flash(request, "error", "The key could not be saved. Check that "
                      f"{config.credentials_dir} is writable.", "translate")
            else:
                flash(request, "ok", "Saved the Modal key on this computer. It is not shown again.",
                      "translate")
        return back_to_setup("translate")

    @app.post("/configuration/modal-key/delete")
    async def delete_modal_key(request: Request) -> Response:
        await read_form(request)
        removed = await run_in_threadpool(translate.delete_modal_key, config)
        flash(request, "ok", "The saved Modal key was deleted." if removed
              else "There was no saved Modal key.", "translate")
        return back_to_setup("translate")

    return app
