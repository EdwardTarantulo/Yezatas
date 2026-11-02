"""Pinned comment: the channel's own comments, the stored mark and suggestion, the seo check, and the
Muse Spark suggestion with its guards (a scripted model). No test touches the network."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fakes import FakeModelClient
from yezatas import comments
from yezatas.agent import ModelError, ModelReply
from yezatas.pinnedcomment import PINNED_FORMAT, suggest_pinned
from yezatas.seo import ChannelContext, check_pinned_comment, score_catalog
from yezatas.store import Comment, Store, Video

NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)
VID, OWNER = "vid000001", "UC_owner_000"
DESCRIPTION = "How to repot a cactus in 5 steps.\nSoil mix: https://example.com/soil"


def video() -> Video:
    return Video(video_id=VID, channel_id=OWNER, title="Repotting a cactus", description=DESCRIPTION,
                 privacy_status="public", duration="PT6M")


def make_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "c.sqlite")
    pull = store.begin_pull(OWNER, now=NOW)
    store.upsert_video(video(), pull, now=NOW)
    store.finish_pull(pull, 1, now=NOW)
    return store


def reply(text: str) -> ModelReply:
    return ModelReply(json.dumps({"text": text}), (), {}, 800, 0, 60)


def item(comment_id: str, author_channel: str) -> dict:
    return {"snippet": {"topLevelComment": {"id": comment_id, "snippet": {
        "textOriginal": "Nice video", "authorDisplayName": "Someone",
        "authorChannelId": {"value": author_channel}}}}}


def judged(state: str | None):
    return check_pinned_comment(video(), ChannelContext(pinned={VID: state} if state else {}))


def test_a_comment_by_the_channel_itself_is_marked() -> None:
    own = comments._comment_from_item(item("c1", OWNER), VID, OWNER)
    other = comments._comment_from_item(item("c2", "UC_viewer"), VID, OWNER)
    assert own is not None and own.by_owner
    assert other is not None and not other.by_owner
    assert not comments._comment_from_item(item("c3", OWNER), VID, "").by_owner


def test_pinned_states_come_from_the_comments_and_the_mark(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.pinned_states() == {}
    store.upsert_comments([Comment("c1", VID, author="viewer", text="Hi")], now=NOW)
    assert store.pinned_states() == {VID: "none"}
    store.upsert_comments([Comment("c2", VID, author="me", text="Pinned", by_owner=True)], now=NOW)
    assert store.pinned_states() == {VID: "own"}
    assert {c.comment_id for c in store.list_comments(VID) if c.by_owner} == {"c2"}
    store.mark_pinned(VID, now=NOW)
    assert store.pinned_states() == {VID: "marked"}
    assert store.pinned_comment(VID).marked_at != ""
    store.close()


def test_the_check_by_state() -> None:
    assert not judged(None).applicable and "Fetch" in judged(None).detail
    assert not check_pinned_comment(video(), None).applicable
    none = judged("none")
    assert none.applicable and none.points == 0 and none.max_points == 5 and "question" in none.fix
    assert judged("own").passed and "make sure yours is" in judged("own").detail
    assert judged("marked").passed


def test_score_catalog_passes_the_states_on() -> None:
    found = next(c for c in score_catalog([video()], pinned={VID: "none"})[VID].checks
                 if c.check_id == "pinned_comment")
    assert found.applicable and not found.passed


def test_a_suggestion_is_stored_and_another_differs_from_it(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    client = FakeModelClient([reply("Which cactus should I repot next? Soil mix: https://example.com/soil"),
                              reply("Repotted in 5 steps. Which one is next?")])
    first = suggest_pinned(store, client, VID, now=NOW)
    assert first.suggestion.startswith("Which cactus") and first.error == "" and first.input_tokens == 800
    assert client.calls[0]["response_format"] == PINNED_FORMAT
    assert "earlier suggestion" not in client.calls[0]["messages"][1]["content"]
    second = suggest_pinned(store, client, VID, now=NOW)
    assert second.suggestion.startswith("Repotted in 5")
    assert "Which cactus should I repot next?" in client.calls[1]["messages"][1]["content"]
    store.mark_pinned(VID, now=NOW)
    assert store.pinned_comment(VID).suggestion == second.suggestion
    store.close()


def test_invented_links_or_numbers_are_refused_and_the_old_suggestion_stays(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    client = FakeModelClient([reply("Which cactus is next?"), reply("Get it at https://shop.invalid/x"),
                              reply("Over 900 people asked for this."), ModelError("network", "No connection.")])
    suggest_pinned(store, client, VID, now=NOW)
    link = suggest_pinned(store, client, VID, now=NOW)
    assert "https://shop.invalid/x" in link.error and link.suggestion == "Which cactus is next?"
    number = suggest_pinned(store, client, VID, now=NOW)
    assert '"900"' in number.error and number.suggestion == "Which cactus is next?"
    down = suggest_pinned(store, client, VID, now=NOW)
    assert down.error == "No connection." and down.suggestion == "Which cactus is next?"
    store.close()


def test_an_unknown_video_raises_before_any_call(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    client = FakeModelClient([])
    with pytest.raises(ValueError):
        suggest_pinned(store, client, "nope", now=NOW)
    assert client.calls == []
    store.close()
