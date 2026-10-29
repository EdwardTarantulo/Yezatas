"""Local-only protection for the web UI: Host check, per-run token, session cookie, CSRF.

The server binds 127.0.0.1 only. On top of that:
- the Host header must be 127.0.0.1:PORT or localhost:PORT (blocks DNS rebinding);
- every page needs a session cookie, which is only issued in exchange for the per-run token that
  the CLI prints and puts in the URL it opens. The cookie is signed with a key kept on disk, so
  restarting Yezatas does not sign out tabs that are already open;
- state-changing requests are POST only and carry a per-session CSRF token.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sys
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import parse_qsl, urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, RedirectResponse, Response
from starlette.templating import Jinja2Templates

COOKIE_PREFIX = "yezatas_session_"
TOKEN_PARAM = "t"
CSRF_FIELD = "csrf"
CSRF_HEADER = "x-csrf-token"
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    # Not "no-referrer": with it, Chromium browsers send `Origin: null` on form posts, which the
    # Origin check below rightly refuses. "same-origin" still keeps the referrer off other sites.
    "Referrer-Policy": "same-origin",
}
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
KEY_BYTES = 32


def load_session_key(path: Path) -> bytes:
    """The cookie-signing key, created on first use. Unreadable or damaged: a key for this run only."""
    try:
        key = path.read_bytes()
        if len(key) == KEY_BYTES:
            return key
    except OSError:
        pass
    key = secrets.token_bytes(KEY_BYTES)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(key)
        if sys.platform != "win32":
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except OSError:
        pass
    return key


class Sessions:
    """Signed session cookies. Stateless: the CSRF token is derived from the session id and key."""

    def __init__(self, token: str, port: int, key: bytes) -> None:
        self.token = token
        self.key = key
        # Browsers share cookies between ports, so two instances must not overwrite each other's.
        self.cookie_name = f"{COOKIE_PREFIX}{port}"
        self.allowed_hosts = frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})
        self.allowed_origins = frozenset(f"http://{h}" for h in self.allowed_hosts)

    def token_ok(self, candidate: str) -> bool:
        return hmac.compare_digest(candidate.encode(), self.token.encode())

    def _sign(self, label: str, sid: str) -> str:
        return hmac.new(self.key, f"{label}:{sid}".encode(), hashlib.sha256).hexdigest()

    def new_session(self) -> str:
        """A new cookie value: `sid.signature`."""
        sid = secrets.token_urlsafe(32)
        return f"{sid}.{self._sign('session', sid)}"

    def session_id(self, cookie: str | None) -> str | None:
        """The session id if the cookie was signed with our key, else None."""
        if not cookie or "." not in cookie:
            return None
        sid, sig = cookie.rsplit(".", 1)
        if not sid or not hmac.compare_digest(sig.encode(), self._sign("session", sid).encode()):
            return None
        return sid

    def csrf_for(self, sid: str) -> str:
        return self._sign("csrf", sid)


def verify_csrf(request: Request, submitted: str | None) -> None:
    """Raise 403 unless `submitted` (form field) or the X-CSRF-Token header matches the session."""
    expected: str | None = getattr(request.state, "csrf", None)
    candidate = submitted or request.headers.get(CSRF_HEADER) or ""
    if not expected or not hmac.compare_digest(candidate.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="Missing or wrong CSRF token.")


def install_security(app: FastAPI, sessions: Sessions, templates: Jinja2Templates) -> None:
    def forbidden(request: Request) -> Response:
        return templates.TemplateResponse(request, "forbidden.html", {}, status_code=403)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> Response:
        if exc.status_code == 403:
            return forbidden(request)
        return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await _guard(request, call_next)
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        return response

    async def _guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.headers.get("host", "").lower() not in sessions.allowed_hosts:
            return PlainTextResponse("Bad Host header.", status_code=400)
        if request.method in UNSAFE_METHODS:
            origin = request.headers.get("origin")
            if origin is not None and origin.lower() not in sessions.allowed_origins:
                return forbidden(request)
        if request.url.path.startswith("/static/"):
            return await call_next(request)

        supplied = request.query_params.get(TOKEN_PARAM)
        if supplied is not None and request.method == "GET":
            if not sessions.token_ok(supplied):
                return forbidden(request)
            rest = [(k, v) for k, v in parse_qsl(request.url.query) if k != TOKEN_PARAM]
            target = request.url.path + (f"?{urlencode(rest)}" if rest else "")
            redirect = RedirectResponse(target, status_code=303)
            redirect.set_cookie(
                sessions.cookie_name, sessions.new_session(), httponly=True, samesite="strict",
                path="/",
            )
            return redirect

        sid = sessions.session_id(request.cookies.get(sessions.cookie_name))
        if sid is None:
            return forbidden(request)
        request.state.session_id = sid
        request.state.csrf = sessions.csrf_for(sid)
        return await call_next(request)
