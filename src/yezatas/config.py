"""Configuration: where Yezatas keeps its data. Loading never touches the disk."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT_DAILY_QUOTA = 10000
# Standard mode: Meta does not train on the prompts. Contributor models (`*-contributor`) may, so
# Yezatas never sends channel data to them.
DEFAULT_MODEL = "muse-spark-1.3"


class ConfigError(ValueError):
    """Raised when an environment setting is invalid."""


@dataclass(frozen=True)
class Config:
    home: Path
    catalog_db: Path
    credentials_dir: Path
    client_secrets_path: Path
    token_path: Path
    daily_quota_budget: int = DEFAULT_DAILY_QUOTA
    model: str = DEFAULT_MODEL
    model_key_path: Path | None = None  # the saved Meta API key (MODEL_API_KEY in the environment wins)


def _default_home(env: Mapping[str, str]) -> Path:
    if sys.platform == "win32":
        local = env.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / "Yezatas"
    xdg = env.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "yezatas"


def _parse_quota(raw: str) -> int:
    try:
        value = int(raw.strip())
    except ValueError:
        raise ConfigError(f"YEZATAS_DAILY_QUOTA must be a positive integer, got {raw!r}") from None
    if value <= 0:
        raise ConfigError(f"YEZATAS_DAILY_QUOTA must be a positive integer, got {raw!r}")
    return value


def _parse_model(raw: str) -> str:
    model = raw.strip()
    if not model:
        raise ConfigError("YEZATAS_MODEL must not be empty")
    if model.endswith("-contributor"):
        raise ConfigError(
            f"YEZATAS_MODEL {model!r} is a Contributor model, which may train on your data. "
            "Use a Standard model such as muse-spark-1.3."
        )
    return model


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Build a Config from the environment (or the mapping given). Creates nothing."""
    if env is None:
        env = os.environ
    explicit = env.get("YEZATAS_HOME")
    home = Path(explicit) if explicit else _default_home(env)
    credentials_dir = home / "credentials"
    raw_quota = env.get("YEZATAS_DAILY_QUOTA")
    budget = _parse_quota(raw_quota) if raw_quota is not None else DEFAULT_DAILY_QUOTA
    raw_model = env.get("YEZATAS_MODEL")
    model = _parse_model(raw_model) if raw_model is not None else DEFAULT_MODEL
    return Config(
        home=home,
        catalog_db=home / "catalog" / "catalog.sqlite",
        credentials_dir=credentials_dir,
        client_secrets_path=credentials_dir / "client_secrets.json",
        token_path=credentials_dir / "token.json",
        daily_quota_budget=budget,
        model=model,
        model_key_path=credentials_dir / "model_api_key",
    )


def ensure_dirs(config: Config) -> None:
    """Create the data directories; restrict the credentials dir on POSIX."""
    config.home.mkdir(parents=True, exist_ok=True)
    config.catalog_db.parent.mkdir(parents=True, exist_ok=True)
    config.credentials_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        os.chmod(config.credentials_dir, 0o700)
