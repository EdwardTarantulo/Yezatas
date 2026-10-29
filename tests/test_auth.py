import json
import os
import stat
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials

from yezatas.auth import (READONLY, WRITE, MissingClientSecrets, load_credentials,
                          save_credentials, sign_in, sign_out)
from yezatas.config import Config, load_config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return load_config({"YEZATAS_HOME": str(tmp_path / "home")})


def fake_creds(scopes: list[str] | None = None, expired: bool = False,
               refresh_token: str | None = "fake-refresh-token") -> Credentials:
    expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=-1 if expired else 1)
    return Credentials(
        token="fake-access-token",
        refresh_token=refresh_token,
        token_uri="https://example.invalid/token",
        client_id="fake-client-id",
        client_secret="fake-client-secret",
        scopes=scopes if scopes is not None else [READONLY],
        expiry=expiry,
    )


class FakeFlow:
    def __init__(self, creds: Credentials) -> None:
        self.creds = creds
        self.kwargs: dict[str, Any] = {}

    def run_local_server(self, **kwargs: Any) -> Credentials:
        self.kwargs = kwargs
        return self.creds


def test_sign_in_without_client_secrets(config: Config) -> None:
    with pytest.raises(MissingClientSecrets) as exc:
        sign_in(config, flow_factory=lambda *_: pytest.fail("flow must not start"))
    assert "SETUP_GOOGLE.md" in str(exc.value)
    assert str(config.client_secrets_path) in str(exc.value)
    assert not config.token_path.exists()


def test_sign_in_saves_token(config: Config) -> None:
    config.credentials_dir.mkdir(parents=True)
    config.client_secrets_path.write_text("{}", encoding="utf-8")
    seen: dict[str, Any] = {}
    flow = FakeFlow(fake_creds())

    def factory(path: str, scopes: list[str]) -> FakeFlow:
        seen["path"], seen["scopes"] = path, scopes
        return flow

    creds = sign_in(config, flow_factory=factory)
    assert creds is flow.creds
    assert seen == {"path": str(config.client_secrets_path), "scopes": [READONLY]}
    assert flow.kwargs == {"host": "localhost", "port": 0, "open_browser": True}
    saved = json.loads(config.token_path.read_text(encoding="utf-8"))
    assert saved["refresh_token"] == "fake-refresh-token"
    assert saved["scopes"] == [READONLY]


def test_load_returns_none_when_missing_or_unreadable(config: Config) -> None:
    assert load_credentials(config) is None
    config.credentials_dir.mkdir(parents=True)
    config.token_path.write_text("not json", encoding="utf-8")
    assert load_credentials(config) is None
    config.token_path.write_text("{}", encoding="utf-8")
    assert load_credentials(config) is None


def test_load_valid_token(config: Config) -> None:
    save_credentials(config, fake_creds())
    creds = load_credentials(config, refresh=lambda c: pytest.fail("no refresh needed"))
    assert creds is not None and creds.token == "fake-access-token"


def test_load_none_when_scopes_insufficient(config: Config) -> None:
    save_credentials(config, fake_creds([READONLY]))
    assert load_credentials(config, scopes=(WRITE,)) is None
    assert load_credentials(config, scopes=(READONLY,)) is not None
    save_credentials(config, fake_creds([WRITE]))
    assert load_credentials(config, scopes=(READONLY,)) is not None  # force-ssl covers read
    assert load_credentials(config, scopes=(WRITE,)) is not None


def test_load_refreshes_expired_and_saves(config: Config) -> None:
    save_credentials(config, fake_creds(expired=True))

    def refresh(c: Credentials) -> None:
        c.token = "fake-new-token"
        c.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)

    creds = load_credentials(config, refresh=refresh)
    assert creds is not None and creds.token == "fake-new-token"
    assert json.loads(config.token_path.read_text(encoding="utf-8"))["token"] == "fake-new-token"


def test_load_refresh_failure_returns_none(config: Config) -> None:
    save_credentials(config, fake_creds(expired=True))

    def refresh(c: Credentials) -> None:
        raise RefreshError("invalid_grant")

    assert load_credentials(config, refresh=refresh) is None


def test_load_expired_without_refresh_token(config: Config) -> None:
    save_credentials(config, fake_creds(expired=True, refresh_token=None))
    # from_authorized_user_info needs a refresh token, so this is simply "not signed in"
    assert load_credentials(config, refresh=lambda c: pytest.fail("cannot refresh")) is None


def test_save_is_atomic_and_leaves_no_temp_files(config: Config) -> None:
    creds = fake_creds()
    save_credentials(config, creds)
    first = config.token_path.read_text(encoding="utf-8")
    save_credentials(config, creds)
    assert config.token_path.read_text(encoding="utf-8") == first
    assert [p.name for p in config.credentials_dir.iterdir()] == ["token.json"]


def test_failed_save_keeps_old_token(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    save_credentials(config, fake_creds())
    before = config.token_path.read_text(encoding="utf-8")

    def boom(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        save_credentials(config, fake_creds(["https://example.invalid/other"]))
    assert config.token_path.read_text(encoding="utf-8") == before
    assert [p.name for p in config.credentials_dir.iterdir()] == ["token.json"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")
def test_token_file_mode_0600(config: Config) -> None:
    save_credentials(config, fake_creds())
    assert stat.S_IMODE(config.token_path.stat().st_mode) == 0o600


def test_sign_out(config: Config) -> None:
    assert sign_out(config) is False
    save_credentials(config, fake_creds())
    assert sign_out(config) is True
    assert not config.token_path.exists()
    assert load_credentials(config) is None
