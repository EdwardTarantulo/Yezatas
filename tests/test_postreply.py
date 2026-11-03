"""postreply: the text check, the YouTube error messages, the quota charge and the single post.

Fakes only; nothing reaches YouTube.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from webhelpers import http_error
from yezatas.postreply import (MAX_CHARS, PostReplyError, api_failure, check_text, post_reply,
                               posted_url)
from yezatas.quota import QuotaGuard, quota_day
from yezatas.store import Store

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
VID = "vid00000001"


class Writer:
    def __init__(self, error: Exception | None = None, response: Any = None,
                 gate: threading.Event | None = None) -> None:
        self.error, self.response, self.gate = error, response, gate
        self.calls: list[tuple[Any, ...]] = []
        self.entered = threading.Event()

    def _answer(self, default: Any) -> Any:
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.error is not None:
            raise self.error
        return default if self.response is None else self.response

    def insert_reply(self, parent_id: str, text: str) -> dict[str, Any]:
        self.calls.append(("reply", parent_id, text))
        return self._answer({"id": "UgzR.1"})

    def insert_top_level(self, video_id: str, text: str) -> dict[str, Any]:
        self.calls.append(("thread", video_id, text))
        return self._answer({"id": "UgzT", "snippet": {"topLevelComment": {"id": "UgzTop"}}})


@pytest.fixture
def store(tmp_path: Path) -> Store:
    with Store(tmp_path / "c.sqlite") as s:
        yield s


def guard(store: Store, budget: int = 10_000) -> QuotaGuard:
    return QuotaGuard(store, budget, clock=lambda: NOW)


def draft(store: Store, comment_id: str = "cmt1", text: str = "Thanks.") -> int:
    return store.add_comment_reply(VID, comment_id, text, "muse-spark-1.3", [], now=NOW)


def test_check_text_trims_and_limits() -> None:
    assert check_text("  hi \n") == "hi"
    assert check_text("a" * MAX_CHARS) == "a" * MAX_CHARS
    for bad in ("", "  \n\t "):
        with pytest.raises(PostReplyError) as empty:
            check_text(bad)
        assert empty.value.kind == "invalid" and "empty" in str(empty.value)
    with pytest.raises(PostReplyError) as long:
        check_text("a" * (MAX_CHARS + 1))
    assert long.value.kind == "invalid" and "10,000" in str(long.value)


def test_posted_url() -> None:
    assert posted_url(VID, "UgzR.1") == f"https://www.youtube.com/watch?v={VID}&lc=UgzR.1"
    assert posted_url(VID, "") == ""


def test_reply_goes_through_comments_insert_and_is_charged_and_stored(store: Store) -> None:
    rid = draft(store)
    writer = Writer()
    done = post_reply(store, writer, guard(store), rid, " Thanks a lot. ", now=NOW)
    assert writer.calls == [("reply", "cmt1", "Thanks a lot.")]
    assert (done.youtube_id, done.pinned_draft, done.units, done.text) == ("UgzR.1", False, 50, "Thanks a lot.")
    assert done.url.endswith("lc=UgzR.1")
    row = store.get_comment_reply(rid)
    assert (row.status, row.text, row.posted_comment_id, row.posted_at) == (
        "posted", "Thanks a lot.", "UgzR.1", NOW.isoformat())
    entries = store.quota_entries(quota_day(NOW))
    assert [(e["operation"], e["units"]) for e in entries] == [("comments.insert", 50)]


def test_pinned_draft_goes_through_comment_threads_insert(store: Store) -> None:
    rid = draft(store, comment_id="", text="Welcome!")
    writer = Writer()
    done = post_reply(store, writer, guard(store), rid, "Welcome!", now=NOW)
    assert writer.calls == [("thread", VID, "Welcome!")]
    assert done.pinned_draft and done.youtube_id == "UgzTop"
    assert [e["operation"] for e in store.quota_entries(quota_day(NOW))] == ["commentThreads.insert"]


def test_a_response_without_an_id_still_counts_as_posted(store: Store) -> None:
    rid = draft(store)
    done = post_reply(store, Writer(response={"kind": "youtube#comment"}), guard(store), rid, "Hi", now=NOW)
    assert done.youtube_id == "" and done.url == ""
    assert store.get_comment_reply(rid).status == "posted"


def test_bad_text_never_reaches_the_api_or_the_ledger(store: Store) -> None:
    rid = draft(store)
    writer = Writer()
    for bad in ("", "x" * (MAX_CHARS + 1)):
        with pytest.raises(PostReplyError):
            post_reply(store, writer, guard(store), rid, bad, now=NOW)
    assert writer.calls == [] and store.quota_used(quota_day(NOW)) == 0
    assert store.get_comment_reply(rid).status == "open"


@pytest.mark.parametrize("status", ["posted", "used", "dismissed"])
def test_only_an_open_draft_is_posted(store: Store, status: str) -> None:
    rid = draft(store)
    store.set_comment_reply_status(rid, status)
    writer = Writer()
    with pytest.raises(PostReplyError) as err:
        post_reply(store, writer, guard(store), rid, "Hi", now=NOW)
    assert err.value.kind == ("already" if status == "posted" else "closed")
    assert writer.calls == [] and store.quota_used(quota_day(NOW)) == 0


def test_an_unknown_draft_is_refused(store: Store) -> None:
    with pytest.raises(PostReplyError) as err:
        post_reply(store, Writer(), guard(store), 99, "Hi", now=NOW)
    assert err.value.kind == "missing"


def test_the_budget_is_checked_before_the_call(store: Store) -> None:
    rid = draft(store)
    writer = Writer()
    with pytest.raises(PostReplyError) as err:
        post_reply(store, writer, guard(store, budget=40), rid, "Hi", now=NOW)
    assert err.value.kind == "quota" and writer.calls == []
    assert store.get_comment_reply(rid).status == "open"


@pytest.mark.parametrize("error, kind", [
    (http_error(403, "quotaExceeded"), "quota"),
    (http_error(403, "rateLimitExceeded"), "quota"),
    (http_error(403, "commentsDisabled"), "disabled"),
    (http_error(403, "insufficientPermissions"), "scope"),
    (http_error(401, "authError"), "scope"),
    (http_error(403, "forbidden", "Request had insufficient authentication scopes."), "scope"),
    (http_error(404, "commentNotFound"), "deleted"),
    (http_error(404, "videoNotFound"), "deleted"),
    (http_error(400, "processingFailure", "Bad"), "api"),
    (http_error(403, "forbidden", "Not allowed"), "api"),
    (ConnectionError("down"), "network"),
    (TimeoutError("slow"), "network"),
    (ValueError("bug"), "api"),
])
def test_failures_are_plain_sentences_and_keep_the_draft_open(store: Store, error: Exception, kind: str) -> None:
    rid = draft(store)
    with pytest.raises(PostReplyError) as err:
        post_reply(store, Writer(error=error), guard(store), rid, "Hi", now=NOW)
    assert err.value.kind == kind
    assert str(err.value).endswith((".", "again.")) or "It said" in str(err.value)
    assert "Traceback" not in str(err.value) and "HttpError" not in str(err.value)
    row = store.get_comment_reply(rid)
    assert (row.status, row.posted_comment_id) == ("open", "")


def test_a_pinned_draft_failure_talks_about_the_comment() -> None:
    assert "comment was not posted" in str(api_failure(http_error(403, "commentsDisabled"), True))
    assert "video" in str(api_failure(http_error(404, "videoNotFound"), True))


def test_two_posts_at_once_make_one_call(tmp_path: Path) -> None:
    path = tmp_path / "race.sqlite"
    with Store(path) as seed:
        rid = draft(seed)
    gate = threading.Event()
    writer = Writer(gate=gate)
    results: list[str] = []

    def run() -> None:
        with Store(path) as mine:  # each request opens its own store
            try:
                post_reply(mine, writer, guard(mine), rid, "Hi", now=NOW)
                results.append("posted")
            except PostReplyError as exc:
                results.append(exc.kind)

    first = threading.Thread(target=run)
    first.start()
    assert writer.entered.wait(10)
    second = threading.Thread(target=run)
    second.start()
    gate.set()
    first.join(10)
    second.join(10)
    assert sorted(results) == ["already", "posted"]
    assert len(writer.calls) == 1
    with Store(path) as check:
        assert check.quota_used(quota_day(NOW)) == 50
