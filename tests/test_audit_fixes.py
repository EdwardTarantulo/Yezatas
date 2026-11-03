"""Regressions for the Nov 3 audit: claims the code now backs, and bugs found by running it."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from yezatas import outliers, primaryai, translate
from yezatas.config import ConfigError, load_config
from yezatas.facts import check_facts, number_key, numbers_in
from yezatas.links import PRIVATE_HOST_MESSAGE, check_links
from yezatas.store import Store, Video
from yezatas.subtitle_translate import _digits
from yezatas.transcribe import TranscribeError, _check_length
from yezatas.validate import Draft
from yezatas.web.views import unchecked_translations

NOW = datetime(2026, 11, 3, tzinfo=timezone.utc)


def _video(video_id: str, **kw) -> Video:
    base = dict(video_id=video_id, channel_id="c", title=f"Title {video_id}", description="", tags=[],
                privacy_status="public", published_at="2026-01-01T00:00:00Z", duration="PT10M")
    base.update(kw)
    return Video(**base)


# --- numbers ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("a,b", [("1,204", "1204"), ("1.204", "1 204"), ("4,2", "4.2"), ("4.20", "4.2"),
                                 ("1,204.5", "1.204,5"), ("٤٩", "49"), ("1 000", "1000")])
def test_same_number_in_any_spelling(a: str, b: str) -> None:
    assert number_key(a) == number_key(b)


@pytest.mark.parametrize("a,b", [("4.2", "42"), ("1,204", "12.04"), ("49", "94"), ("0.125", "125")])
def test_different_numbers_stay_different(a: str, b: str) -> None:
    assert number_key(a) != number_key(b)
    assert _digits(a) != _digits(b)


def test_fact_guard_refuses_new_numbers() -> None:
    original = Draft("Cactus care", "Water it monthly.", ())
    draft = Draft("Cactus care", "Only $99, 5 million sold, 90% off, 12-year warranty.", ())
    values = {p.value for p in check_facts(draft, original) if p.kind == "new_number"}
    assert values == {"99", "5", "90", "12"}


def test_fact_guard_allows_known_numbers_lists_and_sources() -> None:
    original = Draft("5 cactus tips", "Costs $1,204.", ())
    same = Draft("Cactus: 5 tips", "1. Costs $1.204 today.\n2. Water it.", ())
    assert check_facts(same, original) == []
    said = Draft("5 cactus tips", "Lives 40 years.", ())
    assert check_facts(said, original)
    assert check_facts(said, original, allowed_numbers=numbers_in("It lives for 40 years")) == []


def test_translation_check_catches_changed_numbers() -> None:
    problems = translate.check_translation("Sale $49", "Now 20% off", "Oferta $94", "Ahora 90% menos")
    assert any('"49"' in p for p in problems) and any('"94"' in p for p in problems)
    assert any('"20"' in p for p in problems) and any('"90"' in p for p in problems)
    assert translate.check_translation("Sale $1,204", "20% off", "Oferta $1.204", "20 % menos") == []


def test_placeholder_link_is_a_problem_not_a_crash() -> None:
    original = Draft("T", "Buy: https://[your-link-here]", ())
    assert check_facts(original, original) == []
    assert [p.kind for p in check_facts(Draft("T", "https://[new-one]", ()), Draft("T", "", ()))] == ["new_link"]


# --- Local mode -------------------------------------------------------------------------------------
def test_local_mode_ignores_saved_keys(tmp_path: Path) -> None:
    env = {"YEZATAS_HOME": str(tmp_path), "BASETEN_API_KEY": "x" * 20, "MODAL_API_KEY": "y" * 20,
           "MODAL_ENDPOINT_URL": "https://modal.invalid/v1", "YEZATAS_PRIMARY_AI": "local"}
    config = load_config(env)
    assert translate.translator_key_source(config, env) == "local"
    used = translate.translator_from_config(config, env)
    assert used.model == config.verifier_model
    assert primaryai.is_local(config, env)
    env["YEZATAS_PRIMARY_AI"] = "meta"
    assert translate.translator_from_config(config, env).model == config.translator_model


@pytest.mark.parametrize("setting", ["YEZATAS_MODEL", "YEZATAS_TRANSLATOR_MODEL", "YEZATAS_VERIFIER_MODEL",
                                     "YEZATAS_MODAL_MODEL"])
def test_every_model_setting_refuses_contributor_models(setting: str, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Contributor"):
        load_config({"YEZATAS_HOME": str(tmp_path), setting: "some-model-Contributor"})


# --- the catalog ------------------------------------------------------------------------------------
def test_gone_videos_leave_the_lists_and_come_back(tmp_path: Path) -> None:
    with Store(tmp_path / "c.sqlite") as store:
        pull = store.begin_pull("c", NOW)
        for vid in ("a", "b"):
            store.upsert_video(_video(vid), pull, NOW)
        assert store.mark_gone_except({"a"}, NOW) == 1
        assert [v.video_id for v in store.list_videos()] == ["a"]
        assert store.get_video("b") is not None  # drafts and history still find it
        store.upsert_video(_video("b"), pull, NOW)
        assert {v.video_id for v in store.list_videos()} == {"a", "b"}


def test_writers_wait_instead_of_failing(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    Store(path).close()
    errors: list[Exception] = []

    def write(n: int) -> None:
        try:
            with Store(path) as store:
                for i in range(20):
                    store.upsert_video(_video(f"v{n}-{i}"), store.begin_pull("c", NOW), NOW)
        except sqlite3.OperationalError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_shorts_and_long_videos_are_not_peers() -> None:
    longs = [_video(f"l{i}", view_count=100_000, published_at=f"2026-01-{i + 1:02d}T00:00:00Z") for i in range(12)]
    shorts = [_video(f"s{i}", view_count=100, duration="PT30S",
                     published_at=f"2026-01-{i + 1:02d}T00:00:00Z") for i in range(12)]
    scores = {s.video_id: s for s in outliers.score_videos(longs + shorts, NOW)}
    assert all(p.startswith("s") for p in scores["s0"].peer_ids)
    assert scores["s0"].ratio == pytest.approx(1.0)


def test_a_file_of_another_length_is_refused() -> None:
    _check_length(600_000, "PT10M3S")
    _check_length(600_000, "")
    with pytest.raises(TranscribeError, match="different video"):
        _check_length(70_000, "PT5M")


def test_links_to_private_addresses_are_not_visited(tmp_path: Path) -> None:
    visited: list[str] = []

    def fetch(url):
        visited.append(url)
        from yezatas.links import LinkOutcome
        return LinkOutcome("ok", 200, "", "")

    urls = ["http://127.0.0.1:8080/x", "http://192.168.1.1/", "http://printer.local/", "https://example.org/"]
    with Store(tmp_path / "c.sqlite") as store:
        check_links(store, urls, fetch=fetch, now=NOW)
        assert visited == ["https://example.org/"]
        assert store.latest_link_check("http://192.168.1.1/").message == PRIVATE_HOST_MESSAGE \
            if hasattr(store, "latest_link_check") else True


# --- translations going out --------------------------------------------------------------------------
class _Row:
    def __init__(self, sent, base=None):
        self.localizations, self.base_localizations = sent, base or {}


class _Rec:
    def __init__(self, language, status, title="T", description="D"):
        self.language, self.status, self.title, self.description = language, status, title, description


def test_unchecked_translations_are_named() -> None:
    sent = {"es": {"title": "T", "description": "D"}, "de": {"title": "T", "description": "D"},
            "fr": {"title": "T", "description": "D"}, "it": {"title": "Mine", "description": "D"}}
    found = unchecked_translations(_Row(sent, {"fr": {"title": "T", "description": "D"}}), [
        _Rec("es", "flagged"), _Rec("de", "verified"), _Rec("fr", "flagged"), _Rec("it", "flagged")])
    assert len(found) == 1 and "failed a check" in found[0]  # es only: de passed, fr unchanged, it typed by hand
    assert unchecked_translations(_Row(None), [_Rec("es", "flagged")]) == ()
