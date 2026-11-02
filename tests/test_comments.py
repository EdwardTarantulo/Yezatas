"""Comments assistant: fetching, sorting and draft guards. No test touches the network or a real model."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient, make_http_error
from yezatas import comments as comments_mod
from yezatas.agent import ModelError, ModelReply
from yezatas.comments import ASSIST_FORMAT, CATEGORIES, CommentsError, assist, fetch_comments
from yezatas.quota import QuotaExceeded, QuotaGuard, quota_day
from yezatas.store import Comment, Store, Video

NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)
VID = "vid00000001"
TITLE = "How to repot a cactus in 5 steps"
DESCRIPTION = "How to repot a cactus.\nSee https://example.com/guide for the soil mix.\nUse a 10 cm pot."


def make_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "c.sqlite")
    pull = store.begin_pull("chan0000001", now=NOW)
    store.upsert_video(Video(video_id=VID, title=TITLE, description=DESCRIPTION,
                             tags=["cactus"], privacy_status="public", duration="PT6M"), pull, now=NOW)
    store.finish_pull(pull, 1, now=NOW)
    return store


@pytest.fixture
def store(tmp_path: Path):
    s = make_store(tmp_path)
    yield s
    s.close()


def comment_item(cid: str, text: str, published: str, *, author: str = "Viewer One",
                 likes: int = 3, replies: int = 2, use_display: bool = False) -> dict[str, Any]:
    snippet: dict[str, Any] = {"authorDisplayName": author, "publishedAt": published,
                               "likeCount": likes}
    if use_display:
        snippet["textDisplay"] = text
    else:
        snippet["textOriginal"] = text
    return {"snippet": {"totalReplyCount": replies,
                        "topLevelComment": {"id": cid, "snippet": snippet}}}


class FakeCommentReader:
    """Pages by page token (None is the first page); a value may be an exception to raise."""

    def __init__(self, pages: dict[Any, Any]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, Any]] = []

    def comment_threads(self, video_id: str, page_token: str | None) -> dict[str, Any]:
        self.calls.append((video_id, page_token))
        item = self.pages[page_token]
        if isinstance(item, Exception):
            raise item
        return item


def seed_comment(store: Store, cid: str, text: str, published: str) -> None:
    store.upsert_comments([Comment(comment_id=cid, video_id=VID, author="Viewer One", text=text,
                                   published_at=published, like_count=1, reply_count=0)], now=NOW)


def reply(data: Any, tokens: tuple[int, int, int] = (1000, 100, 200)) -> ModelReply:
    text = data if isinstance(data, str) else json.dumps(data)
    return ModelReply(text, (), {}, *tokens)


def good_answer(**override: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "categories": [{"ref": "C1", "category": "question"},
                       {"ref": "C2", "category": "feedback"},
                       {"ref": "C3", "category": "praise"}],
        "replies": [{"ref": "C1", "text": "Water when the soil is dry."},
                    {"ref": "C2", "text": "Try the mix at https://example.com/guide ."}],
        "pinned": "Thanks for watching. See https://example.com/guide .",
    }
    data.update(override)
    return data


def seed_three(store: Store) -> None:
    # newest first in the store: C1, C2, C3 in the request
    seed_comment(store, "cmt00000001", "How often should I water it?", "2026-06-03T10:00:00Z")
    seed_comment(store, "cmt00000002", "The soil mix did not work for me.", "2026-06-02T10:00:00Z")
    seed_comment(store, "cmt00000003", "Great video, thank you!", "2026-06-01T10:00:00Z")


# --- fetching --------------------------------------------------------------------------------------
def test_fetch_stores_two_pages_newest_first(store: Store) -> None:
    reader = FakeCommentReader({
        None: {"items": [comment_item("cmt00000002", "Second newest.", "2026-06-02T10:00:00Z"),
                         comment_item("cmt00000003", "Oldest.", "2026-06-01T10:00:00Z")],
               "nextPageToken": "page2"},
        "page2": {"items": [comment_item("cmt00000001", "Newest.", "2026-06-03T10:00:00Z",
                                         use_display=True)]},
    })
    stored = fetch_comments(store, reader, VID, guard=None, now=NOW)
    assert stored == 3
    assert reader.calls == [(VID, None), (VID, "page2")]
    got = store.list_comments(VID)
    assert [c.comment_id for c in got] == ["cmt00000001", "cmt00000002", "cmt00000003"]
    assert got[0].text == "Newest."  # textDisplay fallback
    assert got[1].author == "Viewer One" and got[1].like_count == 3 and got[1].reply_count == 2


def test_fetch_slices_the_last_page_at_max_comments(store: Store) -> None:
    reader = FakeCommentReader({
        None: {"items": [comment_item("cmt00000001", "One.", "2026-06-03T10:00:00Z"),
                         comment_item("cmt00000002", "Two.", "2026-06-02T10:00:00Z"),
                         comment_item("cmt00000003", "Three.", "2026-06-01T10:00:00Z")],
               "nextPageToken": "page2"},
        "page2": {"items": [comment_item("cmt00000004", "Four.", "2026-05-01T10:00:00Z")]},
    })
    assert fetch_comments(store, reader, VID, guard=None, max_comments=2, now=NOW) == 2
    assert [c.comment_id for c in store.list_comments(VID)] == ["cmt00000001", "cmt00000002"]
    assert reader.calls == [(VID, None)]  # the second page is never fetched


def test_fetch_skips_malformed_items(store: Store) -> None:
    reader = FakeCommentReader({None: {"items": [
        {"snippet": {"topLevelComment": {"snippet": {"textOriginal": "No id."}}}},
        {"snippet": {"topLevelComment": {"id": "cmt00000009", "snippet": {}}}},
        comment_item("cmt00000001", "Good one.", "2026-06-01T10:00:00Z"),
        "junk",
    ]}})
    assert fetch_comments(store, reader, VID, guard=None, now=NOW) == 1
    assert [c.comment_id for c in store.list_comments(VID)] == ["cmt00000001"]


def test_fetch_charges_quota_before_each_page(store: Store) -> None:
    reader = FakeCommentReader({
        None: {"items": [comment_item("cmt00000001", "One.", "2026-06-02T10:00:00Z")],
               "nextPageToken": "page2"},
        "page2": {"items": [comment_item("cmt00000002", "Two.", "2026-06-01T10:00:00Z")]},
    })
    guard = QuotaGuard(store, 10, clock=lambda: NOW)
    assert fetch_comments(store, reader, VID, guard=guard, now=NOW) == 2
    day = quota_day(NOW)
    assert store.quota_used(day) == 2
    entries = store.quota_entries(day)
    assert [e["operation"] for e in entries] == ["commentThreads.list", "commentThreads.list"]
    assert all(e["note"] == "fetch comments" for e in entries)


def test_fetch_disabled_comments_raises_a_plain_error(store: Store) -> None:
    reader = FakeCommentReader({None: make_http_error(403, "commentsDisabled")})
    with pytest.raises(CommentsError) as excinfo:
        fetch_comments(store, reader, VID, guard=None, now=NOW)
    assert str(excinfo.value) == "Comments are turned off on this video."
    assert store.list_comments(VID) == []


def test_fetch_unknown_video_raises_before_any_call(store: Store) -> None:
    reader = FakeCommentReader({None: {"items": []}})
    guard = QuotaGuard(store, 10, clock=lambda: NOW)
    with pytest.raises(ValueError) as excinfo:
        fetch_comments(store, reader, "missing0001", guard=guard, now=NOW)
    assert str(excinfo.value) == "This video is not in your catalog. Pull your videos first."
    assert reader.calls == []
    assert store.quota_used(quota_day(NOW)) == 0


def test_fetch_quota_stop_keeps_the_pages_already_read(store: Store) -> None:
    reader = FakeCommentReader({
        None: {"items": [comment_item("cmt00000001", "First page.", "2026-06-02T10:00:00Z")],
               "nextPageToken": "page2"},
        "page2": {"items": [comment_item("cmt00000002", "Second page.", "2026-06-01T10:00:00Z")]},
    })
    guard = QuotaGuard(store, 1, clock=lambda: NOW)
    with pytest.raises(QuotaExceeded):
        fetch_comments(store, reader, VID, guard=guard, now=NOW)
    assert [c.comment_id for c in store.list_comments(VID)] == ["cmt00000001"]


def test_fetch_api_failure_keeps_the_pages_already_read(store: Store) -> None:
    reader = FakeCommentReader({
        None: {"items": [comment_item("cmt00000001", "First page.", "2026-06-02T10:00:00Z")],
               "nextPageToken": "page2"},
        "page2": RuntimeError("boom"),
    })
    with pytest.raises(RuntimeError, match="boom"):
        fetch_comments(store, reader, VID, guard=None, now=NOW)
    assert [c.comment_id for c in store.list_comments(VID)] == ["cmt00000001"]


# --- assist: the good path -------------------------------------------------------------------------
def test_assist_good_path_saves_categories_and_drafts(store: Store) -> None:
    seed_three(store)
    client = FakeModelClient([reply(good_answer())])
    result = assist(store, client, VID, now=NOW)
    assert result.video_id == VID and result.model == "muse-spark-1.3"
    assert result.counts == {"question": 1, "feedback": 1, "praise": 1, "spam": 0, "other": 0}
    assert result.drafts == 3 and result.notes == ()
    assert (result.input_tokens, result.cached_tokens, result.output_tokens) == (1000, 100, 200)
    by_id = {c.comment_id: c.category for c in store.list_comments(VID)}
    assert by_id == {"cmt00000001": "question", "cmt00000002": "feedback", "cmt00000003": "praise"}
    drafts = store.list_comment_replies(VID)
    assert [(d.comment_id, d.text) for d in drafts] == [
        ("cmt00000001", "Water when the soil is dry."),
        ("cmt00000002", "Try the mix at https://example.com/guide ."),
        ("", "Thanks for watching. See https://example.com/guide ."),
    ]
    assert all(d.model == "muse-spark-1.3" and d.status == "open" for d in drafts)


def test_assist_request_uses_refs_not_real_ids_and_cuts_long_comments(store: Store) -> None:
    seed_comment(store, "cmt00000001", "a" * 700, "2026-06-02T10:00:00Z")
    seed_comment(store, "cmt00000002", "Short question?", "2026-06-01T10:00:00Z")
    client = FakeModelClient([reply({"categories": [], "replies": [], "pinned": ""})])
    result = assist(store, client, VID, now=NOW)
    assert result.counts == {key: 0 for key in CATEGORIES} and result.drafts == 0
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["response_format"] is ASSIST_FORMAT
    schema = ASSIST_FORMAT["json_schema"]["schema"]
    assert ASSIST_FORMAT["json_schema"]["strict"] is True
    assert schema["properties"]["categories"]["items"]["properties"]["category"]["enum"] == list(CATEGORIES)
    user = call["messages"][1]["content"]
    system = call["messages"][0]["content"]
    assert "at most 20" in system and "question or" in system and "Never invent" in system
    assert f"Title: {TITLE}" in user and "https://example.com/guide" in user
    assert "C1:" in user and "C2:" in user
    assert "cmt00000001" not in user and "cmt00000002" not in user
    assert "a" * 600 in user and "a" * 601 not in user


def test_assist_long_draft_is_cut_to_1000_characters(store: Store) -> None:
    seed_three(store)
    long_text = "Thank you for asking. " + "word " * 300
    client = FakeModelClient([reply(good_answer(
        replies=[{"ref": "C1", "text": long_text}], pinned=""))])
    result = assist(store, client, VID, now=NOW)
    assert result.drafts == 1
    drafts = store.list_comment_replies(VID)
    assert len(drafts) == 1 and len(drafts[0].text) == 1000
    assert drafts[0].text == long_text.strip()[:1000]


# --- assist: the guards ----------------------------------------------------------------------------
def test_assist_made_up_url_is_dropped_with_a_note(store: Store) -> None:
    seed_three(store)
    client = FakeModelClient([reply(good_answer(
        replies=[{"ref": "C1", "text": "See https://example.com/madeup for more."}], pinned=""))])
    result = assist(store, client, VID, now=NOW)
    assert result.drafts == 0 and store.list_comment_replies(VID) == []
    assert result.notes == (
        'A reply was dropped: it links to "https://example.com/madeup", which is not in the '
        "video's description.",)
    # the categories were still saved: only the draft failed the guard
    assert {c.comment_id: c.category for c in store.list_comments(VID)}["cmt00000001"] == "question"


def test_assist_made_up_number_is_dropped_and_a_number_from_the_comment_is_allowed(
        store: Store) -> None:
    seed_comment(store, "cmt00000001", "I tried 3 times and it fell over.",
                 "2026-06-02T10:00:00Z")
    seed_comment(store, "cmt00000002", "How do I fix it?", "2026-06-01T10:00:00Z")
    client = FakeModelClient([reply({
        "categories": [{"ref": "C1", "category": "feedback"}, {"ref": "C2", "category": "question"}],
        "replies": [{"ref": "C1", "text": "Trying 3 times is a good start."},
                    {"ref": "C2", "text": "Use 99 pots at once."}],
        "pinned": "",
    })])
    result = assist(store, client, VID, now=NOW)
    assert [d.comment_id for d in store.list_comment_replies(VID)] == ["cmt00000001"]
    assert result.drafts == 1
    assert result.notes == (
        'A reply was dropped: it uses the number "99", which is not in the comment or the '
        "video's text.",)


def test_assist_unknown_ref_and_unknown_category_are_dropped(store: Store) -> None:
    seed_three(store)
    client = FakeModelClient([reply({
        "categories": [{"ref": "C99", "category": "question"},
                       {"ref": "C1", "category": "angry"},
                       {"ref": "C2", "category": "feedback"}],
        "replies": [{"ref": "C99", "text": "Hello there."},
                    {"ref": "C2", "text": "Thanks for the feedback."}],
        "pinned": "",
    })])
    result = assist(store, client, VID, now=NOW)
    assert result.counts == {"question": 0, "feedback": 1, "praise": 0, "spam": 0, "other": 0}
    assert {c.comment_id: c.category for c in store.list_comments(VID)} == {
        "cmt00000001": "", "cmt00000002": "feedback", "cmt00000003": ""}
    assert [d.comment_id for d in store.list_comment_replies(VID)] == ["cmt00000002"]
    assert any("A category was dropped" in n for n in result.notes)
    assert any("A reply was dropped: it names a comment that was not listed." in n
               for n in result.notes)


def test_assist_a_reply_to_praise_is_dropped(store: Store) -> None:
    seed_three(store)
    client = FakeModelClient([reply(good_answer(
        replies=[{"ref": "C3", "text": "Thank you so much!"}], pinned=""))])
    result = assist(store, client, VID, now=NOW)
    assert result.drafts == 0 and store.list_comment_replies(VID) == []
    assert result.notes == (
        "A reply was dropped: it does not answer a question or feedback comment.",)


def test_assist_pinned_with_a_made_up_number_is_dropped(store: Store) -> None:
    seed_three(store)
    client = FakeModelClient([reply(good_answer(
        replies=[], pinned="Watch all 42 videos in the series."))])
    result = assist(store, client, VID, now=NOW)
    assert result.drafts == 0 and store.list_comment_replies(VID) == []
    assert result.notes == (
        'The pinned comment draft was dropped: it uses the number "42", which is not in the '
        "video's text.",)


def test_assist_replies_beyond_twenty_are_dropped(store: Store) -> None:
    for n in range(25):
        seed_comment(store, f"cmt{n:08d}", f"Question {n} about watering?",
                     f"2026-06-{25 - n:02d}T10:00:00Z")
    categories = [{"ref": f"C{n}", "category": "question"} for n in range(1, 26)]
    replies = [{"ref": f"C{n}", "text": "Water when dry."} for n in range(1, 26)]
    client = FakeModelClient([reply({"categories": categories, "replies": replies, "pinned": ""})])
    result = assist(store, client, VID, now=NOW)
    assert result.drafts == 20 and len(store.list_comment_replies(VID)) == 20
    assert sum("only the first 20" in n for n in result.notes) == 5


# --- assist: failures store nothing ------------------------------------------------------------------
def test_assist_model_error_stores_nothing(store: Store) -> None:
    seed_three(store)
    client = FakeModelClient([ModelError("rate_limit", "Slow down, then try again.")])
    with pytest.raises(CommentsError) as excinfo:
        assist(store, client, VID, now=NOW)
    assert str(excinfo.value) == "Slow down, then try again."
    assert all(c.category == "" for c in store.list_comments(VID))
    assert store.list_comment_replies(VID) == []


@pytest.mark.parametrize("content", [
    "not json at all",
    json.dumps({"categories": [], "replies": []}),
    json.dumps({"categories": "none", "replies": [], "pinned": ""}),
    json.dumps({"categories": [], "replies": [], "pinned": 3}),
])
def test_assist_bad_answer_stores_nothing(store: Store, content: str) -> None:
    seed_three(store)
    client = FakeModelClient([reply(content)])
    with pytest.raises(CommentsError):
        assist(store, client, VID, now=NOW)
    assert all(c.category == "" for c in store.list_comments(VID))
    assert store.list_comment_replies(VID) == []


def test_assist_unknown_video_raises_before_any_model_call(store: Store) -> None:
    client = FakeModelClient([])
    with pytest.raises(ValueError) as excinfo:
        assist(store, client, "missing0001", now=NOW)
    assert str(excinfo.value) == "This video is not in your catalog. Pull your videos first."
    assert client.calls == []


def test_public_constants() -> None:
    assert CATEGORIES == {"question": "Question", "feedback": "Feedback", "praise": "Praise",
                          "spam": "Spam", "other": "Other"}
    assert comments_mod.MAX_REPLIES == 20


def test_assist_second_run_dismisses_leftover_open_drafts(store: Store) -> None:
    seed_three(store)
    first = assist(store, FakeModelClient([reply(good_answer())]), VID, now=NOW)
    assert first.drafts == 3
    drafts = store.list_comment_replies(VID)
    store.set_comment_reply_status(drafts[0].id, "used")
    second_answer = good_answer(
        replies=[{"ref": "C1", "text": "Water only when the soil is dry."},
                 {"ref": "C2", "text": "The guide mix is worth another try."}],
        pinned="Welcome to the comments.")
    second = assist(store, FakeModelClient([reply(second_answer)]), VID, now=NOW)
    assert second.drafts == 3
    by_text = {d.text: d.status for d in store.list_comment_replies(VID)}
    assert by_text == {
        "Water when the soil is dry.": "used",
        "Try the mix at https://example.com/guide .": "dismissed",
        "Thanks for watching. See https://example.com/guide .": "dismissed",
        "Water only when the soil is dry.": "open",
        "The guide mix is worth another try.": "open",
        "Welcome to the comments.": "open",
    }


@pytest.mark.parametrize("bad", [
    ModelError("rate_limit", "Slow down, then try again."),
    reply("not json at all"),
])
def test_assist_failed_second_run_keeps_first_run_drafts_open(
        store: Store, bad: Any) -> None:
    seed_three(store)
    assist(store, FakeModelClient([reply(good_answer())]), VID, now=NOW)
    with pytest.raises(CommentsError):
        assist(store, FakeModelClient([bad]), VID, now=NOW)
    assert [d.status for d in store.list_comment_replies(VID)] == ["open"] * 3


def test_assist_with_no_comments_makes_no_call(store: Store) -> None:
    client = FakeModelClient([])
    with pytest.raises(CommentsError, match="no comments"):
        assist(store, client, VID)
    assert client.calls == []


def test_fetch_marks_comments_the_channel_replied_to(tmp_path: Path) -> None:
    store = Store(tmp_path / "o.sqlite")
    pull = store.begin_pull("chan0000001", now=NOW)
    store.upsert_video(Video(video_id=VID, channel_id="chan0000001", title=TITLE, description=DESCRIPTION,
                             privacy_status="public"), pull, now=NOW)
    store.finish_pull(pull, 1, now=NOW)
    mine = comment_item("cmt00000001", "Which pot size?", "2026-06-01T10:00:00Z")
    mine["replies"] = {"comments": [{"snippet": {"authorChannelId": {"value": "chan0000001"}}}]}
    other = comment_item("cmt00000002", "And the soil?", "2026-06-01T09:00:00Z")
    other["replies"] = {"comments": [{"snippet": {"authorChannelId": {"value": "chanSomeoneElse"}}}]}
    plain = comment_item("cmt00000003", "Nice video", "2026-06-01T08:00:00Z")
    reader = FakeCommentReader({None: {"items": [mine, other, plain]}})
    assert fetch_comments(store, reader, VID, guard=None, now=NOW) == 3
    replied = {c.comment_id: c.owner_replied for c in store.list_comments(VID)}
    assert replied == {"cmt00000001": True, "cmt00000002": False, "cmt00000003": False}
    store.close()
