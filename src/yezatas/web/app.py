"""The Yezatas web app. Today: the setup screen that connects a YouTube channel.

Everything here is read-only towards YouTube. Background jobs open their own Store, because a
SQLite connection must not cross threads.
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

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from .. import auth
from ..catalog import NoChannelError, PullResult, pull_catalog
from ..config import Config, ensure_dirs
from ..quota import QuotaGuard
from ..store import Store
from ..youtube import GoogleYouTubeReader, YouTubeReader
from .clientfile import (MAX_CLIENT_FILE_BYTES, ClientFileError, parse_client_file,
                         read_saved_client, save_client_file)
from .guidance import NeedsSignIn, map_api_error, map_sign_in_error
from .jobs import Job, JobSnapshot, Report
from .security import CSRF_FIELD, Sessions, install_security, load_session_key, verify_csrf
from .steps import CheckState, compute_steps, estimate_pull_units

DEFAULT_PORT = 8765
SIGN_IN_TIMEOUT_SECONDS = 300.0
SIGN_IN_TIMEOUT_MESSAGE = (
    "The sign-in did not finish within 5 minutes. If Google showed \"Access blocked\", your "
    "account is probably not on the Test users list: add it in step 3. Otherwise just start "
    "the sign-in again."
)
UPLOAD_BODY_LIMIT = MAX_CLIENT_FILE_BYTES + 16 * 1024  # file plus multipart overhead

HERE = Path(__file__).parent

ReaderFactory = Callable[[Any], YouTubeReader]
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
    ) -> None:
        self.config = config
        self._sign_in = sign_in
        self._reader_factory = reader_factory
        self._credentials_loader = credentials_loader
        self.signin_job = Job(
            map_sign_in_error, sign_in_timeout, SIGN_IN_TIMEOUT_MESSAGE, clock
        )
        self.pull_job = Job(map_api_error)
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

    def sign_out(self) -> bool:
        removed = auth.sign_out(self.config)
        self.reset_check()
        return removed


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
) -> FastAPI:
    """Build the app. `app.state.start_url` is the one-time link that signs the browser in."""
    ensure_dirs(config)
    token = token or secrets.token_urlsafe(32)
    start_url = f"http://127.0.0.1:{port}/?t={token}"
    state = WebState(config, sign_in, reader_factory, credentials_loader, clock, sign_in_timeout)
    sessions = Sessions(token, port, load_session_key(config.credentials_dir / "session.key"))
    templates = Jinja2Templates(directory=str(HERE / "templates"))

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
        # Steps 1 to 3 cannot be detected, so once the client file is valid the next action is
        # further down the list.
        candidates = steps[4:] if client else steps
        current = next((s.key for s in candidates if s.status != "done" and s.enabled), "")
        estimate = None
        if check is not None and check.ok:
            units = estimate_pull_units(check.video_count)
            estimate = {"units": units, "fits": units <= remaining}
        return {
            "steps": {s.key: s for s in steps}, "current": current,
            "client": client, "client_exists": config.client_secrets_path.exists(),
            "client_path": str(config.client_secrets_path),
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

    def render(request: Request, name: str, extra: dict[str, Any]) -> Response:
        context = {"csrf": request.state.csrf, **extra}
        return templates.TemplateResponse(request, name, context)

    # --- pages ------------------------------------------------------------------------------
    @app.get("/")
    def index(request: Request) -> Response:
        data = page_data()
        if data["latest_pull"] is None:
            return RedirectResponse("/setup", status_code=303)
        return render(request, "home.html", data)

    @app.get("/setup")
    def setup(request: Request) -> Response:
        data = page_data()
        data["flash"] = state.pop_flash(request.state.session_id)
        return render(request, "setup.html", data)

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        check = state.check
        return JSONResponse({
            "signin": _job_json(state.signin_job.snapshot()),
            "pull": _job_json(state.pull_job.snapshot()),
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
            text += " Because the client changed, you were signed out: sign in again in step 5."
        state.add_flash(sid, "ok", text, "client")
        return back_to_setup("signin" if existing is not None else "client")

    @app.post("/setup/signin")
    async def start_sign_in(request: Request) -> Response:
        await read_form(request)
        sid = request.state.session_id
        client, _ = read_saved_client(config)
        if client is None:
            state.add_flash(sid, "error", "Add the OAuth client file first (step 4).", "client")
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
            state.add_flash(sid, "error", "Sign in first (step 5).", "signin")
            return back_to_setup("signin")
        await run_in_threadpool(state.run_check)
        return back_to_setup("check")

    @app.post("/setup/pull")
    async def start_pull(request: Request) -> Response:
        await read_form(request)
        sid = request.state.session_id
        check = state.check
        if check is None or not check.ok or not check.has_uploads:
            state.add_flash(sid, "error", "Check the connection first (step 6).", "check")
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

    return app
