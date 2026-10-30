"""Validate and save the Google OAuth client file (client_secrets.json) the user uploads.

Only the project id and a shortened client id are ever shown; the secret is never echoed.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

from ..config import Config, ensure_dirs

MAX_CLIENT_FILE_BYTES = 64 * 1024
REQUIRED_KEYS = ("client_id", "client_secret", "auth_uri", "token_uri")


class ClientFileError(ValueError):
    """The file is not a usable Desktop-app OAuth client. The message is safe to show."""


@dataclass(frozen=True)
class ClientInfo:
    project_id: str
    client_id_short: str


def shorten_client_id(client_id: str) -> str:
    return client_id[:12] + "…"


def parse_client_file(data: bytes) -> ClientInfo:
    """Check the bytes are a Desktop-app client file; return display-safe info or raise."""
    if len(data) > MAX_CLIENT_FILE_BYTES:
        raise ClientFileError(
            f"This file is larger than {MAX_CLIENT_FILE_BYTES // 1024} KB, so it is not an "
            "OAuth client file. Upload the JSON you downloaded for your Desktop app client."
        )
    try:
        parsed: Any = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError):
        raise ClientFileError(
            "This is not a JSON file. Upload the JSON you downloaded when you created the "
            "OAuth client."
        ) from None
    not_client = ClientFileError(
        "This is not an OAuth client file. Upload the JSON you downloaded for your "
        "Desktop app client."
    )
    if not isinstance(parsed, dict):
        raise not_client
    installed = parsed.get("installed")
    if installed is None:
        if isinstance(parsed.get("web"), dict):
            raise ClientFileError(
                "This is a Web application client. Create a Desktop app client instead "
                "(step 5, application type: Desktop app) and upload that file."
            )
        if parsed.get("type") == "service_account":
            raise ClientFileError(
                "This is a service account key, which cannot sign in to a YouTube channel. "
                "Create an OAuth client of type Desktop app instead."
            )
        raise not_client
    if not isinstance(installed, dict):
        raise not_client
    missing = [k for k in REQUIRED_KEYS if not isinstance(installed.get(k), str) or not installed[k]]
    if missing:
        raise ClientFileError(
            "This does not look like an OAuth client file: the \"installed\" section is "
            f"missing {', '.join(missing)}. Download the JSON again from the Clients page."
        )
    project_id = installed.get("project_id")
    return ClientInfo(
        project_id=project_id if isinstance(project_id, str) and project_id else "(not given)",
        client_id_short=shorten_client_id(installed["client_id"]),
    )


def read_saved_client(config: Config) -> tuple[ClientInfo | None, str]:
    """The saved client file's info, or (None, problem). (None, "") means there is no file."""
    try:
        data = config.client_secrets_path.read_bytes()
    except FileNotFoundError:
        return None, ""
    except OSError as exc:
        return None, f"The saved client file cannot be read ({exc.__class__.__name__})."
    try:
        return parse_client_file(data), ""
    except ClientFileError as exc:
        return None, str(exc)


def save_client_file(config: Config, data: bytes) -> None:
    """Write the (already validated) bytes atomically; 0o600 on POSIX, like the token file."""
    ensure_dirs(config)
    fd, tmp_name = tempfile.mkstemp(dir=config.credentials_dir, prefix="client.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        if sys.platform != "win32":
            os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, config.client_secrets_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
