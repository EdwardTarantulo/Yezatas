"""The Category card on the video page: its states, the check job with a scripted model, "It's correct",
the fix rail and the quick-info tile, a failed check and the no-key path. No model or network call."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient
from webhelpers import make_ui, make_video, seed_catalog
from yezatas import agent, categorycheck
from yezatas.agent import ModelReply
from yezatas.store import CategoryCheck, Store

V1 = "vid00000001"


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


class FakeCategory:
    """The injected check: the real judge with a scripted model."""

    def __init__(self, suggested: str = "26") -> None:
        answer = json.dumps({"category_id": suggested, "reason": "A gardening how-to."})
        self.client = FakeModelClient([ModelReply(answer, (), {}, 900, 0, 40)])
        self.calls: list[str] = []

    def __call__(self, store: Store, video_id: str, now: datetime, report: Any) -> CategoryCheck:
        self.calls.append(video_id)
        return categorycheck.judge_category(store, self.client, video_id, now=now, report=report)


def make_cat_ui(tmp_path: Path, fn: Any = None, key: bool = True) -> Any:
    fn = fn or FakeCategory()
    ui = make_ui(tmp_path, category=fn, packaging_key=lambda cfg: key)
    seed_catalog(ui.config, [make_video(1, video_id=V1, raw={"snippet": {"categoryId": "20"}})])
    ui.fn = fn
    return ui


def page(ui: Any) -> str:
    resp = ui.client.get(f"/videos/{V1}")
    assert resp.status_code == 200
    return resp.text


def card(ui: Any) -> str:
    return page(ui).split('id="category"')[1].split("</section>")[0]


def run_check(ui: Any) -> Any:
    resp = ui.post(f"/videos/{V1}/category/check", follow_redirects=False)
    ui.web.category_job.wait()
    return resp


def test_never_checked_shows_the_button_and_loses_no_points(tmp_path: Path) -> None:
    ui = make_cat_ui(tmp_path)
    part = card(ui)
    assert "Set as <strong>Gaming</strong>" in part and ">Check category</button>" in part
    assert ">Not checked</span>" in part and "It's correct" not in part
    html = page(ui)
    assert 'href="#check-category"' not in html
    assert not re.search(r'id="fact-category" data-tone="warn"', html)
    assert " style=" not in part


def test_a_different_category_fails_with_both_actions(tmp_path: Path) -> None:
    ui = make_cat_ui(tmp_path)
    resp = run_check(ui)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#category"
    assert ui.fn.calls == [V1]
    part = card(ui)
    assert "Muse Spark suggests Howto &amp; Style" in part and "A gardening how-to." in part
    assert f'href="https://studio.youtube.com/video/{V1}/edit"' in part
    assert "It's correct</button>" in part and ">Check again</button>" in part and "Checked " in part
    html = page(ui)
    assert 'href="#check-category"' in html and 'href="#category">Category</a>' in html
    assert re.search(r'id="fact-category" data-tone="warn"', html)


def test_its_correct_makes_the_check_pass(tmp_path: Path) -> None:
    ui = make_cat_ui(tmp_path)
    run_check(ui)
    resp = ui.post(f"/videos/{V1}/category/correct", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#category"
    part = card(ui)
    assert ">Marked correct</span>" in part and "It's correct</button>" not in part
    assert "you marked Gaming as correct" in part
    assert 'href="#check-category"' not in page(ui)


def test_agreeing_shows_fits(tmp_path: Path) -> None:
    ui = make_cat_ui(tmp_path, FakeCategory("20"))
    run_check(ui)
    part = card(ui)
    assert ">Fits</span>" in part and "Muse Spark agrees." in part and "It's correct" not in part


def test_its_correct_before_any_check_is_refused(tmp_path: Path) -> None:
    ui = make_cat_ui(tmp_path)
    html = ui.post(f"/videos/{V1}/category/correct").text
    assert "no finished check" in html
    with Store(ui.config.catalog_db) as store:
        assert store.category_check(V1) is None


def test_a_failed_check_shows_its_error(tmp_path: Path) -> None:
    ui = make_cat_ui(tmp_path, FakeCategory("99"))
    run_check(ui)
    part = card(ui)
    assert 'id="category-error"' in part and "category YouTube does not have" in part
    assert ">Not checked</span>" in part


def test_without_a_key_there_is_no_button_and_the_post_refuses(tmp_path: Path) -> None:
    ui = make_cat_ui(tmp_path, key=False)
    part = card(ui)
    assert 'id="category-no-key"' in part and "/category/check" not in part
    ui.post(f"/videos/{V1}/category/check")
    assert ui.fn.calls == []


def test_an_unknown_video_is_not_checked(tmp_path: Path) -> None:
    ui = make_cat_ui(tmp_path)
    ui.post("/videos/nope/category/check")
    assert ui.fn.calls == []
