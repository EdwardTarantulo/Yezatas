"""Shared helpers for the web UI tests: a signed-in TestClient with fakes only."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import httplib2
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError

from fakes import FakeYouTubeReader, ScriptedVerifier
from yezatas import auth
from yezatas.config import Config, load_config
from yezatas.store import Store, Video
from yezatas.web.app import create_app

PORT = 8765
BASE_URL = f"http://127.0.0.1:{PORT}"
TOKEN = "test-run-token"
FAKE_SECRET = "FAKE-SECRET-VALUE-9f3a"
FAKE_CLIENT_ID = "123456789012-abcdefgh.apps.example.invalid"


def client_json(project_id: str = "demo-project", **overrides: Any) -> bytes:
    installed = {
        "client_id": FAKE_CLIENT_ID,
        "project_id": project_id,
        "auth_uri": "https://example.invalid/auth",
        "token_uri": "https://example.invalid/token",
        "client_secret": FAKE_SECRET,
        "redirect_uris": ["http://localhost"],
    }
    installed.update(overrides)
    return json.dumps({"installed": installed}).encode()


def fake_creds(scopes: list[str] | None = None) -> Credentials:
    return Credentials(
        token="fake-access-token",
        refresh_token="fake-refresh-token",
        token_uri="https://example.invalid/token",
        client_id="fake-client-id",
        client_secret="fake-client-secret",
        scopes=scopes or [auth.READONLY],
        expiry=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1),
    )


def http_error(status: int, reason: str, message: str = "Something went wrong.",
               detail_reason: str | None = None) -> HttpError:
    body: dict[str, Any] = {"error": {"code": status, "message": message,
                                      "errors": [{"reason": reason, "message": message}]}}
    if detail_reason:
        body["error"]["details"] = [{"@type": "type.example/ErrorInfo", "reason": detail_reason}]
    return HttpError(httplib2.Response({"status": str(status)}), json.dumps(body).encode())


def make_config(tmp_path: Path, **changes: Any) -> Config:
    config = load_config({"YEZATAS_HOME": str(tmp_path / "home")})
    return replace(config, **changes) if changes else config


def write_client_file(config: Config, data: bytes | None = None) -> None:
    config.credentials_dir.mkdir(parents=True, exist_ok=True)
    config.client_secrets_path.write_bytes(data if data is not None else client_json())


def write_token(config: Config, scopes: list[str] | None = None) -> None:
    auth.save_credentials(config, fake_creds(scopes))


def fake_sign_in_ok(config: Config) -> Any:
    creds = fake_creds()
    auth.save_credentials(config, creds)
    return creds


@dataclass
class Ui:
    app: FastAPI
    client: TestClient
    config: Config
    reader: FakeYouTubeReader

    @property
    def web(self) -> Any:
        return self.app.state.web

    def csrf(self) -> str:
        html = self.client.get("/configuration").text
        match = re.search(r'name="csrf" value="([^"]+)"', html)
        assert match, "no csrf field on /setup"
        return match.group(1)

    def post(self, path: str, **kwargs: Any) -> Any:
        data = dict(kwargs.pop("data", {}) or {})
        data.setdefault("csrf", self.csrf())
        return self.client.post(path, data=data, **kwargs)

    def upload(self, content: bytes, replace_confirmed: bool = False) -> Any:
        data = {"csrf": self.csrf()}
        if replace_confirmed:
            data["replace"] = "yes"
        return self.client.post("/configuration/client", data=data,
                                files={"client_file": ("client.json", content, "application/json")})

    def setup_html(self) -> str:
        return self.client.get("/configuration").text


def make_ui(tmp_path: Path, *, signed_in_browser: bool = True,
            reader: FakeYouTubeReader | None = None,
            sign_in: Callable[[Config], Any] = fake_sign_in_ok,
            config: Config | None = None, **kwargs: Any) -> Ui:
    config = config or make_config(tmp_path)
    reader = reader or FakeYouTubeReader(120)
    # never reach for the network or a local Ollama: no translator, a checker that is down
    kwargs.setdefault("translate_tools", lambda cfg: (None, ScriptedVerifier(up=False)))
    app = create_app(config, port=PORT, token=TOKEN, sign_in=sign_in,
                     reader_factory=lambda creds: reader, open_browser=False, **kwargs)
    client = TestClient(app, base_url=BASE_URL)
    if signed_in_browser:
        assert client.get(f"/?t={TOKEN}").status_code == 200
    return Ui(app, client, config, reader)


def step_class(html: str, key: str) -> str:
    """The CSS classes of a step card, for example 'card done current'."""
    match = re.search(rf'<section class="([^"]*)" id="step-{key}"', html)
    assert match, f"step {key} not found"
    return match.group(1)


# --- a seeded catalog (made-up videos) for the dashboard, content and video pages -------------
PRIVACY_CYCLE = ("public", "public", "unlisted", "private")


def make_video(n: int, **overrides: Any) -> Video:
    """A made-up video whose title/description/tags vary with `n`, so scores differ."""
    fields: dict[str, Any] = {
        "video_id": f"vid{n:06d}",
        "channel_id": "UC_fake_channel",
        "title": f"Repotting cactus {'tips ' * (n % 9)}#{n}",
        "description": ("How to repot a cactus. " * (n % 13)).strip(),
        "tags": ["cactus", "garden", f"topic{n}"][: (n % 3) + 1],
        "privacy_status": PRIVACY_CYCLE[n % len(PRIVACY_CYCLE)],
        "published_at": f"2026-{(n % 12) + 1:02d}-{(n % 27) + 1:02d}T10:00:00Z",
        "duration": "PT5M30S",
        "view_count": (n * 37) % 1000 * 25,
        "like_count": n,
        "comment_count": n % 5,
        "thumbnails": {"medium": {"url": f"https://i.ytimg.com/vi/vid{n:06d}/mqdefault.jpg"}},
    }
    fields.update(overrides)
    return Video(**fields)


def seed_catalog(config: Config, videos: list[Video]) -> None:
    """Store `videos` through one finished pull, the way a real pull would."""
    with Store(config.catalog_db) as store:
        pull_id = store.begin_pull("UC_fake_channel")
        for video in videos:
            store.upsert_video(video, pull_id)
        store.finish_pull(pull_id, len(videos))


# --- a seeded catalog with a fake channel that also accepts updates (edit / changes / push) ----
def make_push_ui(tmp_path: Path, count: int = 3, write_access: bool = True,
                 **kwargs: Any) -> "tuple[Ui, Any]":
    """A Ui whose catalog was pulled from a FakeYouTubeWriter; returns (ui, writer).

    With `write_access` the write credentials loader returns fake credentials, otherwise None.
    """
    from fakes import FakeYouTubeWriter
    from yezatas.catalog import pull_catalog
    from yezatas.quota import QuotaGuard

    writer = FakeYouTubeWriter(count)
    loader = (lambda config: fake_creds()) if write_access else (lambda config: None)
    kwargs.setdefault("write_credentials_loader", loader)
    kwargs.setdefault("writer_factory", lambda creds: writer)
    ui = make_ui(tmp_path, reader=writer, **kwargs)
    with Store(ui.config.catalog_db) as store:
        pull_catalog(writer, store, QuotaGuard(store, ui.config.daily_quota_budget))
    writer.calls.clear()
    return ui, writer


# --- a seeded transcript (made-up words) for the transcript card ---------------------------------
def seed_transcript(config: Config, video_id: str, *, segments: int = 6, language: str = "en",
                    problems: list[str] | None = None, duration_ms: int | None = None,
                    source_name: str = "clip.mp4", texts: list[str] | None = None) -> int:
    """Store a transcript of `segments` pieces, each 50 seconds long, and return its id."""
    from yezatas.store import TranscriptRecord

    parts = [{"start_ms": i * 50_000, "end_ms": (i + 1) * 50_000,
              "text": texts[i] if texts is not None else f"Piece {i} talks about topic {i}."}
             for i in range(segments)]
    record = TranscriptRecord(0, video_id, language, "voice-test", source_name,
                              duration_ms if duration_ms is not None else segments * 50_000,
                              parts, problems or [], "")
    with Store(config.catalog_db) as store:
        return store.add_transcript(record)
