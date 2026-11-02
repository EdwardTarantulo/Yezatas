"""Category check: the stored verdict, the seo check built on it, and the Muse Spark judge (a scripted
model). No test touches the network."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fakes import FakeModelClient
from yezatas.agent import ModelError, ModelReply
from yezatas.categorycheck import CATEGORY_FORMAT, judge_category
from yezatas.seo import CategoryJudgment, ChannelContext, check_category, score_catalog, video_category_id
from yezatas.store import Store, Video

NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)
VID = "vid000001"


def video(category: str | None = "20") -> Video:
    return Video(video_id=VID, title="Repotting a cactus in five steps",
                 description="How to repot a cactus without hurting the roots.", tags=["cactus", "repotting"],
                 privacy_status="public", duration="PT6M",
                 raw={"snippet": {"categoryId": category}} if category else {})


def make_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "c.sqlite")
    pull = store.begin_pull("chan0000001", now=NOW)
    store.upsert_video(video(), pull, now=NOW)
    store.finish_pull(pull, 1, now=NOW)
    return store


def reply(data: dict) -> ModelReply:
    return ModelReply(json.dumps(data), (), {}, 900, 0, 40)


def judged(v: Video, judgment: CategoryJudgment | None):
    return check_category(v, ChannelContext(category_checks={VID: judgment} if judgment else {}))


def test_the_category_id_comes_from_the_stored_snippet() -> None:
    assert video_category_id(video("26")) == "26"
    assert video_category_id(video(None)) == ""


def test_never_checked_costs_nothing() -> None:
    result = judged(video(), None)
    assert not result.applicable and result.max_points == 6 and "Check category" in result.detail
    assert not check_category(video(), None).applicable


def test_a_different_category_fails_until_marked_correct() -> None:
    result = judged(video("20"), CategoryJudgment("20", "26", False, "A gardening how-to."))
    assert result.applicable and result.points == 0
    assert "Set as Gaming" in result.detail and "Howto & Style" in result.detail
    assert "YouTube Studio" in result.fix
    confirmed = judged(video("20"), CategoryJudgment("20", "26", True))
    assert confirmed.passed and "marked it as correct" in confirmed.detail


def test_agreeing_or_changed_to_the_suggestion_passes() -> None:
    assert judged(video("26"), CategoryJudgment("26", "26", False)).passed
    assert judged(video("26"), CategoryJudgment("20", "26", False)).passed


def test_changed_to_something_else_since_the_check_asks_again() -> None:
    result = judged(video("24"), CategoryJudgment("20", "26", True))
    assert not result.applicable and "Check again" in result.detail


def test_no_category_in_the_data_is_not_run() -> None:
    assert not judged(video(None), CategoryJudgment("", "26", False)).applicable


def test_score_catalog_passes_the_verdicts_on() -> None:
    v = video("20")
    cards = score_catalog([v], category_checks={VID: CategoryJudgment("20", "26", False)})
    found = next(c for c in cards[VID].checks if c.check_id == "category")
    assert found.applicable and not found.passed
    plain = next(c for c in score_catalog([v])[VID].checks if c.check_id == "category")
    assert not plain.applicable


def test_the_judge_stores_the_verdict(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    client = FakeModelClient([reply({"category_id": "26", "reason": "A step-by-step <b>gardening</b> how-to."})])
    record = judge_category(store, client, VID, now=NOW)
    assert (record.status, record.current_id, record.suggested_id) == ("done", "20", "26")
    assert record.reason == "A step-by-step b gardening /b how-to."
    assert store.category_check(VID) == record and record.input_tokens == 900
    call = client.calls[0]
    assert call["response_format"] == CATEGORY_FORMAT
    asked = call["messages"][1]["content"]
    assert "Repotting a cactus" in asked and "Gaming" in asked
    store.close()


def test_an_unknown_category_or_a_model_error_is_stored_as_failed(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    bad = judge_category(store, FakeModelClient([reply({"category_id": "99", "reason": "x"})]), VID, now=NOW)
    assert bad.status == "failed" and bad.suggested_id == "" and "category" in bad.error
    err = judge_category(store, FakeModelClient([ModelError("network", "No connection.")]), VID, now=NOW)
    assert err.status == "failed" and err.error == "No connection."
    store.close()


def test_an_unknown_video_raises_before_any_call(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    client = FakeModelClient([])
    with pytest.raises(ValueError):
        judge_category(store, client, "nope", now=NOW)
    assert client.calls == []
    store.close()


def test_its_correct_and_a_new_check_clears_it(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.confirm_category(VID, now=NOW) is False  # nothing checked yet
    judge_category(store, FakeModelClient([reply({"category_id": "26", "reason": "How-to."})]), VID, now=NOW)
    assert store.confirm_category(VID, now=NOW) is True
    assert store.category_check(VID).confirmed_at != ""
    assert set(store.category_checks()) == {VID}
    judge_category(store, FakeModelClient([reply({"category_id": "26", "reason": "How-to."})]), VID, now=NOW)
    assert store.category_check(VID).confirmed_at == ""
    store.close()
