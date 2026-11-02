"""Channel-wide comments: store queries and helpers. No test touches the network or a real model."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fakes import make_http_error
from yezatas.comments import (
    AssistResult,
    CATEGORIES,
    CommentsError,
    channel_fetch_units,
    fetch_channel_comments,
    newest_video_ids,
    sort_channel_comments,
    unsorted_video_ids,
)
from yezatas.quota import QuotaGuard, quota_day
from yezatas.store import Comment, Store, Video

NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)

VIDS = ["vidchan0001", "vidchan0002", "vidchan0003"]
PUBLISHED = ["2026-05-01T10:00:00Z", "2026-05-03T10:00:00Z", "2026-05-02T10:00:00Z"]
TITLES = ["Cactus one", "Cactus two", "Cactus three"]


def make_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "c.sqlite")
    pull = store.begin_pull("chan0000001", now=NOW)
    for vid, pub, title in zip(VIDS, PUBLISHED, TITLES):
        store.upsert_video(
            Video(
                video_id=vid,
                title=title,
                description="desc",
                tags=[],
                privacy_status="public",
                duration="PT1M",
                published_at=pub,
            ),
            pull,
            now=NOW,
        )
    store.finish_pull(pull, 3, now=NOW)
    return store


def comment_item(cid: str, text: str, published: str) -> dict[str, Any]:
    return {
        "snippet": {
            "totalReplyCount": 0,
            "topLevelComment": {
                "id": cid,
                "snippet": {
                    "authorDisplayName": "Viewer",
                    "publishedAt": published,
                    "likeCount": 1,
                    "textOriginal": text,
                },
            },
        }
    }


class ByVideoReader:
    def __init__(self, pages: dict[str, Any]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def comment_threads(self, video_id: str, page_token: str | None) -> dict[str, Any]:
        self.calls.append(video_id)
        item = self.pages[video_id]
        if isinstance(item, Exception):
            raise item
        return item


def test_list_recent_comments_newest_first_and_limit(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        store.upsert_comments(
            [
                Comment(comment_id="c1", video_id=VIDS[0], author="A", text="t1", published_at="2026-05-01T11:00:00Z"),
                Comment(comment_id="c2", video_id=VIDS[1], author="A", text="t2", published_at="2026-05-03T12:00:00Z"),
                Comment(comment_id="c3", video_id=VIDS[2], author="A", text="t3", published_at="2026-05-02T11:00:00Z"),
            ],
            now=NOW,
        )
        got = store.list_recent_comments()
        assert [c.comment_id for c in got] == ["c2", "c3", "c1"]
        got2 = store.list_recent_comments(limit=2)
        assert [c.comment_id for c in got2] == ["c2", "c3"]
    finally:
        store.close()


def test_list_open_comment_replies(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        a = store.add_comment_reply(VIDS[0], "c1", "draft a", "m", [], now=NOW)
        b = store.add_comment_reply(VIDS[1], "c2", "draft b", "m", [], now=NOW)
        c = store.add_comment_reply(VIDS[2], "c3", "draft c", "m", [], now=NOW)
        store.set_comment_reply_status(b, "dismissed")
        got = store.list_open_comment_replies()
        assert [r.id for r in got] == [a, c]
    finally:
        store.close()


def test_newest_video_ids(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        assert newest_video_ids(store) == ["vidchan0002", "vidchan0003", "vidchan0001"]
        assert newest_video_ids(store, count=2) == ["vidchan0002", "vidchan0003"]
    finally:
        store.close()


def test_channel_fetch_units() -> None:
    assert channel_fetch_units(10) == 10
    assert channel_fetch_units(10, per_video=250) == 30
    assert channel_fetch_units(0) == 0


def test_fetch_channel_comments_stores_and_charges(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        reader = ByVideoReader(
            {
                VIDS[0]: {"items": [comment_item("c1", "hello one", "2026-05-01T11:00:00Z")]},
                VIDS[1]: {"items": [comment_item("c2", "hello two", "2026-05-03T11:00:00Z")]},
                VIDS[2]: {"items": [comment_item("c3", "hello three", "2026-05-02T11:00:00Z")]},
            }
        )
        guard = QuotaGuard(store, 10, clock=lambda: NOW)
        reports: list[str] = []
        result = fetch_channel_comments(
            store, reader, VIDS, guard=guard, per_video=100, now=NOW, report=reports.append
        )
        assert result.fetched == 3
        assert result.videos == 3
        assert result.notes == ()
        assert store.quota_used(quota_day(NOW)) == 3
        assert reports == [
            "Fetching comments (1 of 3)",
            "Fetching comments (2 of 3)",
            "Fetching comments (3 of 3)",
        ]
        assert reader.calls == VIDS
        assert len(store.list_recent_comments()) == 3
    finally:
        store.close()


def test_fetch_channel_comments_skips_disabled(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        reader = ByVideoReader(
            {
                VIDS[0]: {"items": [comment_item("c1", "hello one", "2026-05-01T11:00:00Z")]},
                VIDS[1]: make_http_error(403, "commentsDisabled"),
                VIDS[2]: {"items": [comment_item("c3", "hello three", "2026-05-02T11:00:00Z")]},
            }
        )
        result = fetch_channel_comments(store, reader, VIDS, guard=None, now=NOW)
        assert result.fetched == 2
        assert result.notes == ('Skipped "Cactus two": Comments are turned off on this video.',)
        assert len(store.list_recent_comments()) == 2
    finally:
        store.close()


def test_fetch_channel_comments_quota_stops(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        reader = ByVideoReader(
            {
                VIDS[0]: {"items": [comment_item("c1", "hello one", "2026-05-01T11:00:00Z")]},
                VIDS[1]: {"items": [comment_item("c2", "hello two", "2026-05-03T11:00:00Z")]},
                VIDS[2]: {"items": [comment_item("c3", "hello three", "2026-05-02T11:00:00Z")]},
            }
        )
        guard = QuotaGuard(store, 1, clock=lambda: NOW)
        try:
            fetch_channel_comments(store, reader, VIDS, guard=guard, now=NOW)
            assert False, "should have raised"
        except Exception as exc:
            from yezatas.quota import QuotaExceeded

            assert isinstance(exc, QuotaExceeded)
        assert len(store.list_recent_comments()) == 1
        assert reader.calls == [VIDS[0]]
    finally:
        store.close()


def test_unsorted_video_ids(tmp_path: Path) -> None:
    comments = [
        Comment(comment_id="c2", video_id=VIDS[1], author="A", text="t", published_at="2026-05-03T12:00:00Z", category="question"),
        Comment(comment_id="c3", video_id=VIDS[2], author="A", text="t", published_at="2026-05-02T12:00:00Z", category=""),
        Comment(comment_id="c1", video_id=VIDS[0], author="A", text="t", published_at="2026-05-01T12:00:00Z", category=""),
        Comment(comment_id="c4", video_id=VIDS[2], author="A", text="t", published_at="2026-05-02T11:00:00Z", category=""),
    ]
    assert unsorted_video_ids(comments) == [VIDS[2], VIDS[0]]


def test_sort_channel_comments(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        def run(video_id: str) -> AssistResult:
            if video_id == VIDS[1]:
                raise CommentsError("There are no comments to sort yet. Fetch the comments first.")
            if video_id == VIDS[0]:
                return AssistResult(
                    video_id=video_id,
                    counts={k: (1 if k == "question" else 0) for k in CATEGORIES},
                    drafts=1,
                    notes=(),
                    model="m",
                    input_tokens=10,
                    cached_tokens=1,
                    output_tokens=2,
                )
            return AssistResult(
                video_id=video_id,
                counts={k: (1 if k == "praise" else 0) for k in CATEGORIES},
                drafts=2,
                notes=(),
                model="m",
                input_tokens=20,
                cached_tokens=2,
                output_tokens=3,
            )

        reports: list[str] = []
        result = sort_channel_comments(run, VIDS, report=reports.append)
        assert result.videos == 2
        assert result.drafts == 3
        assert result.counts["question"] == 1
        assert result.counts["praise"] == 1
        assert result.input_tokens == 30
        assert result.cached_tokens == 3
        assert result.output_tokens == 5
        assert "A video was skipped: There are no comments to sort yet. Fetch the comments first." in result.notes
        assert reports == [
            "Sorting comments (1 of 3)",
            "Sorting comments (2 of 3)",
            "Sorting comments (3 of 3)",
        ]
    finally:
        store.close()
