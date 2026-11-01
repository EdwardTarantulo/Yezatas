import json

from webhelpers import make_config

from yezatas.auth import (
    ANALYTICS,
    READONLY,
    WRITE,
    analytics_scopes,
    can_read_analytics,
    granted_scopes,
)


def _write_token(config, data: str | dict):
    config.token_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        config.token_path.write_text(data, encoding="utf-8")
    else:
        config.token_path.write_text(
            json.dumps(data), encoding="utf-8"
        )


def test_granted_scopes_missing(tmp_path):
    cfg = make_config(tmp_path)
    assert granted_scopes(cfg) == ()


def test_granted_scopes_not_json(tmp_path):
    cfg = make_config(tmp_path)
    _write_token(cfg, "not json")
    assert granted_scopes(cfg) == ()


def test_granted_scopes_not_object(tmp_path):
    cfg = make_config(tmp_path)
    _write_token(cfg, [])
    assert granted_scopes(cfg) == ()


def test_granted_scopes_no_list(tmp_path):
    cfg = make_config(tmp_path)
    _write_token(cfg, {})
    assert granted_scopes(cfg) == ()


def test_granted_scopes_filters_non_strings(tmp_path):
    cfg = make_config(tmp_path)
    _write_token(cfg, {"scopes": [READONLY, 123, WRITE]})
    assert granted_scopes(cfg) == (READONLY, WRITE)


def test_can_read_analytics(tmp_path):
    cfg = make_config(tmp_path)
    _write_token(cfg, {"scopes": [READONLY]})
    assert can_read_analytics(cfg) is False
    _write_token(cfg, {"scopes": [ANALYTICS]})
    assert can_read_analytics(cfg) is True


def test_analytics_scopes_without_write(tmp_path):
    cfg = make_config(tmp_path)
    _write_token(cfg, {"scopes": [READONLY]})
    assert analytics_scopes(cfg) == (READONLY, ANALYTICS)


def test_analytics_scopes_with_write(tmp_path):
    cfg = make_config(tmp_path)
    _write_token(cfg, {"scopes": [WRITE]})
    assert analytics_scopes(cfg) == (WRITE, ANALYTICS)
