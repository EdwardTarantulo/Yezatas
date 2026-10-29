from pathlib import Path

import pytest

from yezatas.config import ConfigError, ensure_dirs, load_config


def test_yezatas_home_override(tmp_path: Path) -> None:
    cfg = load_config({"YEZATAS_HOME": str(tmp_path / "h")})
    home = tmp_path / "h"
    assert cfg.home == home
    assert cfg.catalog_db == home / "catalog" / "catalog.sqlite"
    assert cfg.credentials_dir == home / "credentials"
    assert cfg.client_secrets_path == home / "credentials" / "client_secrets.json"
    assert cfg.token_path == home / "credentials" / "token.json"
    assert cfg.daily_quota_budget == 10000


def test_quota_override() -> None:
    cfg = load_config({"YEZATAS_HOME": "x", "YEZATAS_DAILY_QUOTA": "2500"})
    assert cfg.daily_quota_budget == 2500


@pytest.mark.parametrize("bad", ["0", "-5", "abc", "", "1.5"])
def test_invalid_quota_rejected(bad: str) -> None:
    with pytest.raises(ConfigError):
        load_config({"YEZATAS_HOME": "x", "YEZATAS_DAILY_QUOTA": bad})


def test_default_home_is_outside_repo_and_named(tmp_path: Path) -> None:
    cfg = load_config({"LOCALAPPDATA": str(tmp_path), "XDG_DATA_HOME": str(tmp_path)})
    assert cfg.home.parent == tmp_path
    assert cfg.home.name.lower() == "yezatas"


def test_load_does_not_create_anything(tmp_path: Path) -> None:
    home = tmp_path / "nope"
    load_config({"YEZATAS_HOME": str(home)})
    assert not home.exists()


def test_ensure_dirs_creates(tmp_path: Path) -> None:
    cfg = load_config({"YEZATAS_HOME": str(tmp_path / "h")})
    ensure_dirs(cfg)
    assert cfg.catalog_db.parent.is_dir()
    assert cfg.credentials_dir.is_dir()
    ensure_dirs(cfg)  # idempotent


def test_config_is_frozen(tmp_path: Path) -> None:
    cfg = load_config({"YEZATAS_HOME": str(tmp_path)})
    with pytest.raises(Exception):
        cfg.daily_quota_budget = 1  # type: ignore[misc]
