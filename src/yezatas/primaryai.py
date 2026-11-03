"""Primary AI: which model answers the text features.

"meta" (the default) is Muse Spark through Meta's Model API with the creator's Meta key. "local" is
Muse Glimmer through Ollama on this computer: no key, nothing leaves the machine. This is the ONE
place that decides the text model's address, model and key; every text feature asks `text_model`
or `text_client`. Three features always use Meta's API and key, in both modes: transcripts from a
video file (Muse Voice Transcribe), subject and face sizes (SAM 3.1) and thumbnail variants
(Muse Image). The key is only handed to the client; it is never logged or put in a message.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from . import agent
from .config import DEFAULT_PRIMARY_AI, PRIMARY_AI_CHOICES, Config, ensure_dirs

if TYPE_CHECKING:
    from .modelclient import MetaModelClient

PRIMARY_AI_ENV = "YEZATAS_PRIMARY_AI"
LOCAL_KEY = "ollama"  # Ollama ignores the key, but the OpenAI client wants a non-empty one
LOCAL_TIMEOUT = 600  # a 30B model on a home computer is slow

NO_KEY_MESSAGE = "There is no Meta API key. Add one in Configuration. Nothing was sent."


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def primary_ai_source(config: Config, env: Mapping[str, str] | None = None) -> str:
    """Where the choice comes from: "environment", "saved" or "" (the default)."""
    if _env(env).get(PRIMARY_AI_ENV, "").strip().lower() in PRIMARY_AI_CHOICES:
        return "environment"
    return "saved" if _saved(config) else ""


def _saved(config: Config) -> str:
    if config.primary_ai_path is None:
        return ""
    try:
        choice = config.primary_ai_path.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return ""
    return choice if choice in PRIMARY_AI_CHOICES else ""


def load_primary_ai(config: Config, env: Mapping[str, str] | None = None) -> str:
    """"meta" or "local": YEZATAS_PRIMARY_AI wins, then the saved choice, then "meta"."""
    choice = _env(env).get(PRIMARY_AI_ENV, "").strip().lower()
    if choice in PRIMARY_AI_CHOICES:
        return choice
    return _saved(config) or DEFAULT_PRIMARY_AI


def is_local(config: Config, env: Mapping[str, str] | None = None) -> bool:
    return load_primary_ai(config, env) == "local"


def save_primary_ai(config: Config, choice: str) -> None:
    """Save the choice atomically next to the keys (same folder, same protection)."""
    if choice not in PRIMARY_AI_CHOICES:
        raise ValueError("The Primary AI must be Meta's Model API or Muse Glimmer on this computer.")
    if config.primary_ai_path is None:
        raise ValueError("This configuration has no place for the choice.")
    ensure_dirs(config)
    fd, tmp_name = tempfile.mkstemp(dir=config.credentials_dir, prefix="primary_ai.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(choice)
        if sys.platform != "win32":
            os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, config.primary_ai_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class TextModel:
    """Where the text model lives. `api_key` is never shown or logged (hidden from the repr)."""

    base_url: str
    model: str
    api_key: str
    local: bool

    def __repr__(self) -> str:
        return f"TextModel(base_url={self.base_url!r}, model={self.model!r}, local={self.local})"


def local_text_model(config: Config) -> TextModel:
    return TextModel(f"{config.verifier_url}/v1", config.verifier_model, LOCAL_KEY, True)


def text_model(config: Config, env: Mapping[str, str] | None = None,
               model: str = "") -> TextModel | None:
    """The base URL, model and key of the text model for the current mode, or None when the mode is
    Meta and no Meta key is set. `model` picks another Meta model (Local mode ignores it)."""
    if is_local(config, env):
        return local_text_model(config)
    key = agent.load_api_key(config, env)
    if not key:
        return None
    return TextModel(agent.MODEL_BASE_URL, model or config.model, key, False)


def text_ai_ready(config: Config, env: Mapping[str, str] | None = None) -> bool:
    """Whether the text features can start: Local needs nothing saved; Meta needs a key."""
    return is_local(config, env) or bool(agent.load_api_key(config, env))


def text_client(config: Config, env: Mapping[str, str] | None = None,
                model: str = "") -> "MetaModelClient":
    """A chat client for the text model. Raises `agent.ModelError("auth", ...)` without a key."""
    found = text_model(config, env, model)
    if found is None:
        raise agent.ModelError("auth", NO_KEY_MESSAGE)
    from .modelclient import MetaModelClient

    if not found.local:
        return MetaModelClient(found.api_key, found.model)
    return MetaModelClient(found.api_key, found.model, found.base_url, timeout=LOCAL_TIMEOUT,
                           local=True)


def meta_only_blocked(config: Config, env: Mapping[str, str] | None = None) -> bool:
    """True when a Meta-only feature (SAM sizes, thumbnail variants, transcripts from a file)
    cannot run: Primary AI is Local and there is no Meta key."""
    return is_local(config, env) and not agent.load_api_key(config, env)


def meta_only_message(feature: str) -> str:
    """The sentence shown instead of a button, e.g. "Thumbnail variants need Meta's API. ..."."""
    return (f"{feature} need Meta's API. Primary AI is Local and no Meta key is saved. "
            "Add a key in Configuration.")
