"""YouTube sign-in: OAuth through the user's own Google Cloud project, token kept locally.

Nothing here logs or prints token contents.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Callable, Iterable

from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from .config import Config, ensure_dirs

READONLY = "https://www.googleapis.com/auth/youtube.readonly"
WRITE = "https://www.googleapis.com/auth/youtube.force-ssl"


class MissingClientSecrets(FileNotFoundError):
    """client_secrets.json is not where Yezatas expects it."""


def _default_flow_factory(secrets_path: str, scopes: list[str]) -> Any:
    return InstalledAppFlow.from_client_secrets_file(secrets_path, scopes=scopes)


def _default_refresh(creds: Credentials) -> None:
    creds.refresh(Request())


def _covers(granted: Iterable[str] | None, wanted: Iterable[str]) -> bool:
    """True if the granted scopes include every wanted one (force-ssl also grants read access)."""
    have = set(granted or ())
    return all(s in have or (s == READONLY and WRITE in have) for s in wanted)


def save_credentials(config: Config, creds: Credentials) -> None:
    """Write the token JSON atomically (temp file in the same folder, then os.replace).

    On POSIX the file is chmod 0o600. On Windows chmod does not restrict access, so we rely on
    the default ACL of the per-user LOCALAPPDATA folder that holds the credentials directory.
    """
    ensure_dirs(config)
    fd, tmp_name = tempfile.mkstemp(dir=config.credentials_dir, prefix="token.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(creds.to_json())
        if sys.platform != "win32":
            os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, config.token_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_credentials(
    config: Config,
    scopes: tuple[str, ...] = (READONLY,),
    refresh: Callable[[Credentials], None] = _default_refresh,
) -> Credentials | None:
    """Saved credentials covering `scopes`, or None when the user needs to sign in (again)."""
    try:
        info = json.loads(config.token_path.read_text(encoding="utf-8"))
        creds = Credentials.from_authorized_user_info(info)
    except (OSError, ValueError, KeyError):
        return None
    if not _covers(creds.scopes, scopes):
        return None
    if creds.expired or not creds.token:
        if not creds.refresh_token:
            return None
        try:
            refresh(creds)
        except GoogleAuthError:
            return None
        save_credentials(config, creds)
    return creds


def sign_in(
    config: Config,
    scopes: tuple[str, ...] = (READONLY,),
    flow_factory: Callable[[str, list[str]], Any] = _default_flow_factory,
) -> Credentials:
    """Run the browser sign-in on localhost and save the token."""
    if not config.client_secrets_path.is_file():
        raise MissingClientSecrets(
            f"client_secrets.json not found at {config.client_secrets_path}. "
            "Create an OAuth client and put the file there: see docs/SETUP_GOOGLE.md."
        )
    flow = flow_factory(str(config.client_secrets_path), list(scopes))
    creds = flow.run_local_server(host="localhost", port=0, open_browser=True)
    save_credentials(config, creds)
    return creds


def sign_out(config: Config) -> bool:
    """Delete the saved token. Returns True if there was one."""
    try:
        config.token_path.unlink()
    except FileNotFoundError:
        return False
    return True
