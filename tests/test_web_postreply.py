"""Posting a reply draft to YouTube from the Comments page and the video page.

Fakes only: a fake comment writer records every call, nothing reaches YouTube. A post happens only
from the confirmation page's button, with the text shown there.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from test_web_comments import V1, V2, make_comments_ui, nf, page, seed_comment, seed_reply
from webhelpers import http_error
from yezatas import agent
from yezatas.quota import quota_day
from yezatas.store import Store
from datetime import datetime, timezone


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


class FakeCommentWriter:
    """Records every call; returns scripted ids or raises the scripted error."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str, str]] = []

    def insert_reply(self, parent_id: str, text: str) -> dict[str, Any]:
        self.calls.append(("reply", parent_id, text))
        if self.error is not None:
            raise self.error
        return {"id": "UgzNewReply.AAA", "snippet": {"parentId": parent_id, "textOriginal": text}}

    def insert_top_level(self, video_id: str, text: str) -> dict[str, Any]:
        self.calls.append(("thread", video_id, text))
        if self.error is not None:
            raise self.error
        return {"id": "UgzNewThread", "snippet": {"videoId": video_id, "topLevelComment": {
            "id": "UgzNewThread", "snippet": {"textOriginal": text}}}}


def make_post_ui(tmp_path: Path, error: Exception | None = None, write: bool = True) -> Any:
    writer = FakeCommentWriter(error)
    ui = make_comments_ui(tmp_path, write=write, comment_writer=lambda creds: writer)
    ui.writer = writer
    seed_comment(ui, "comment0001", "Alice", "How often should I water it?",
                 "2026-06-03T10:00:00Z", category="question")
    ui.rid = seed_reply(ui, "comment0001", "Water it every two weeks.")
    return ui


def post_path(ui: Any, rid: int | None = None, video_id: str = V1) -> str:
    return f"/videos/{video_id}/comments/replies/{rid or ui.rid}/post"


def reply_row(ui: Any, rid: int | None = None) -> Any:
    with Store(ui.config.catalog_db) as store:
        return store.get_comment_reply(rid or ui.rid)


def post(ui: Any, text: str = "Water it every two weeks.", rid: int | None = None, **extra: Any) -> Any:
    data = {"text": text, "confirm": "yes", **extra}
    return ui.post(post_path(ui, rid), data=data, follow_redirects=False)


def alerts(html: str) -> str:
    return " ".join(re.findall(r'<div class="flash [a-z]+" role="[a-z]+">([^<]*)', html))


# --- the confirmation page ---------------------------------------------------------------------
def test_confirm_page_shows_the_text_the_comment_and_the_video(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    resp = ui.client.get(post_path(ui))
    assert resp.status_code == 200
    html = resp.text
    assert "Water it every two weeks.</textarea>" in html
    assert 'name="text"' in html and 'maxlength="10000"' in html
    assert "Alice" in html and "How often should I water it?" in html
    assert "Cactus repotting guide" in html
    assert "Post to YouTube" in html and "Cancel" in html
    assert "50 of the" in html
    assert "<script>" not in html and " style=" not in html
    assert ui.writer.calls == []


def test_confirm_page_keeps_the_page_it_came_from(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    html = ui.client.get(post_path(ui) + "?next=/comments%3Fc%3Dquestion").text
    assert 'name="next" value="/comments?c=question"' in html
    assert 'href="/comments?c=question">Cancel' in html
    html = ui.client.get(post_path(ui) + "?next=https://example.com/").text
    assert f'href="/videos/{V1}#comments">Cancel' in html


def test_confirm_page_for_an_unknown_draft_or_video(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    resp = ui.client.get(post_path(ui, 999), follow_redirects=True)
    assert "This draft does not exist." in alerts(resp.text)
    assert ui.client.get(post_path(ui, video_id="nosuchvideo")).status_code == 404
    other = ui.client.get(post_path(ui, video_id=V2), follow_redirects=True)  # the draft is on V1
    assert "This draft does not exist." in alerts(other.text)


def test_post_buttons_and_hover_texts(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    html = ui.client.get("/comments").text
    assert 'title="Post this reply on YouTube under your channel, after you check it">Post reply</a>' in html
    assert f'/videos/{V1}/comments/replies/{ui.rid}/post?next=/comments"' in html
    assert 'title="Tell Yezatas you already posted this on YouTube yourself. The draft is put away as done."' in html
    assert "I posted it" in html and "Dismiss" in html
    assert "Yezatas posts a reply only after you check its exact text and confirm." in html
    assert "you post replies on YouTube yourself" not in html
    section = page(ui).split('id="comments"')[1]
    assert f'/videos/{V1}/comments/replies/{ui.rid}/post"' in section and "Post reply" in section
    assert "Nothing is posted without that confirmation." in section


# --- posting -----------------------------------------------------------------------------------
def test_post_sends_the_text_stores_the_result_and_shows_posted(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    resp = post(ui, next="/comments")
    assert resp.status_code == 303 and resp.headers["location"] == "/comments"
    assert ui.writer.calls == [("reply", "comment0001", "Water it every two weeks.")]
    row = reply_row(ui)
    assert row.status == "posted" and row.posted_comment_id == "UgzNewReply.AAA"
    assert row.posted_at.startswith("20")
    with Store(ui.config.catalog_db) as store:
        today = quota_day(datetime.now(timezone.utc))
        ops = [(e["operation"], e["units"]) for e in store.quota_entries(today)]
    assert ("comments.insert", 50) in ops
    html = ui.client.get("/comments").text
    assert "Posted your reply to Alice on YouTube (50 quota units)." in alerts(html)
    assert f'id="posted-{ui.rid}"' in html and 'class="pill done">Posted<' in html
    assert "lc=UgzNewReply.AAA" in html and "v=" + V1 in html
    assert f'id="draft-{ui.rid}"' not in html
    video_card = page(ui).split('id="comments"')[1]
    assert "Posted" in video_card and "lc=UgzNewReply.AAA" in video_card
    assert f"/comments/replies/{ui.rid}/post" not in video_card


def test_post_records_it_in_history(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    post(ui)
    html = ui.client.get("/history").text
    assert 'id="posted-replies"' in html and "Replies and comments posted" in html
    assert "Water it every two weeks." in html and "Answering Alice" in html
    assert "Cactus repotting guide" in html and "lc=UgzNewReply.AAA" in html


def test_the_edited_text_is_what_is_sent(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    post(ui, text="  Every ten days works better for me.  ")
    assert ui.writer.calls == [("reply", "comment0001", "Every ten days works better for me.")]
    row = reply_row(ui)
    assert row.text == "Every ten days works better for me."


def test_an_empty_reply_is_refused_and_the_draft_stays_open(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    resp = post(ui, text="   ")
    assert resp.status_code == 200
    assert "The reply is empty." in alerts(resp.text) and "Post to YouTube" in resp.text
    assert ui.writer.calls == [] and reply_row(ui).status == "open"


def test_a_reply_over_10000_characters_is_refused_and_kept_in_the_box(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    text = "a" * 10_001
    resp = post(ui, text=text)
    assert resp.status_code == 200
    assert "10,001 characters" in alerts(resp.text) and text in resp.text
    assert ui.writer.calls == [] and reply_row(ui).status == "open"
    ok = post(ui, text="b" * 10_000)
    assert ok.status_code == 303 and len(ui.writer.calls[0][2]) == 10_000


def test_posting_without_the_confirm_field_sends_nothing(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    resp = ui.post(post_path(ui), data={"text": "Hello"}, follow_redirects=False)
    assert resp.status_code == 200 and ui.writer.calls == [] and reply_row(ui).status == "open"


def test_a_second_post_of_the_same_draft_is_refused(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    assert post(ui).status_code == 303
    again = post(ui, next="/comments")
    assert again.status_code == 303
    assert len(ui.writer.calls) == 1
    html = ui.client.get("/comments").text
    assert "already posted" in html
    assert ui.client.get(post_path(ui), follow_redirects=True).status_code == 200
    with Store(ui.config.catalog_db) as store:
        units = sum(e["units"] for e in store.quota_entries(quota_day(datetime.now(timezone.utc))))
    assert units == 50


def test_posted_draft_cannot_be_marked_used_or_dismissed(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    post(ui)
    for action in ("used", "dismissed"):
        path = f"/videos/{V1}/comments/replies/{ui.rid}/{'dismiss' if action == 'dismissed' else action}"
        ui.post(path, follow_redirects=False)
        assert reply_row(ui).status == "posted"


def test_a_put_away_draft_cannot_be_posted(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    nf(ui, f"/videos/{V1}/comments/replies/{ui.rid}/dismiss")
    resp = post(ui)
    assert resp.status_code == 303 and ui.writer.calls == []
    assert "put away" in alerts(ui.client.get(resp.headers["location"]).text)


# --- permission --------------------------------------------------------------------------------
def test_without_the_permission_the_reconnect_path_shows(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path, write=False)
    html = ui.client.get(post_path(ui)).text
    assert 'id="post-no-write"' in html and 'action="/changes/allow-edits"' in html
    assert "Allow Yezatas to edit your videos" in html and "Post to YouTube" not in html
    assert "Water it every two weeks." in html
    resp = post(ui)
    assert resp.status_code == 200 and 'id="post-no-write"' in resp.text
    assert ui.writer.calls == [] and reply_row(ui).status == "open"


def test_a_scope_error_from_youtube_shows_the_reconnect_path(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path, error=http_error(403, "insufficientPermissions", "Insufficient Permission"))
    resp = post(ui)
    assert resp.status_code == 200
    assert 'id="post-no-write"' in resp.text and "permission to post comments" in alerts(resp.text)
    assert len(ui.writer.calls) == 1 and reply_row(ui).status == "open"


@pytest.mark.parametrize("error, fragment", [
    (http_error(403, "quotaExceeded", "Quota"), "daily quota for this app is used up"),
    (http_error(403, "commentsDisabled", "Disabled"), "Comments are turned off on this video"),
    (http_error(404, "commentNotFound", "Gone"), "gone from YouTube"),
    (ConnectionError("down"), "Could not reach YouTube"),
    (http_error(400, "processingFailure", "Bad text"), "YouTube would not post the reply (HTTP 400, processingFailure)"),
])
def test_each_error_is_a_plain_flash_and_the_draft_stays_open(
        tmp_path: Path, error: Exception, fragment: str) -> None:
    ui = make_post_ui(tmp_path, error=error)
    resp = post(ui, text="My edited words")
    assert resp.status_code == 200
    assert fragment in alerts(resp.text)
    assert "My edited words</textarea>" in resp.text and "Post to YouTube" in resp.text
    assert reply_row(ui).status == "open" and reply_row(ui).posted_comment_id == ""
    assert "Traceback" not in resp.text


def test_the_daily_budget_stops_a_post_before_any_call(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    with Store(ui.config.catalog_db) as store:
        store.add_quota_entry("videos.update", ui.config.daily_quota_budget - 10, 1,
                              quota_day=quota_day(datetime.now(timezone.utc)))
    page_html = ui.client.get(post_path(ui)).text
    assert 'id="post-no-quota"' in page_html and "Post to YouTube" not in page_html
    resp = post(ui)
    assert resp.status_code == 200 and "budget has no room" in alerts(resp.text)
    assert ui.writer.calls == [] and reply_row(ui).status == "open"


# --- the pinned-comment draft ------------------------------------------------------------------
def test_the_pinned_draft_posts_as_a_new_comment_and_says_to_pin_it_in_studio(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    pid = seed_reply(ui, "", "Ask your cactus questions below.")
    section = page(ui).split('id="comments"')[1]
    assert f'/videos/{V1}/comments/replies/{pid}/post"' in section and "Post comment" in section
    assert "Post this comment on YouTube under your channel, after you check it" in section
    confirm = ui.client.get(post_path(ui, pid)).text
    assert "pin it there" in confirm and "Ask your cactus questions below.</textarea>" in confirm
    resp = post(ui, text="Ask your cactus questions below!", rid=pid)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#comments"
    assert ui.writer.calls == [("thread", V1, "Ask your cactus questions below!")]
    row = reply_row(ui, pid)
    assert row.status == "posted" and row.posted_comment_id == "UgzNewThread"
    with Store(ui.config.catalog_db) as store:
        ops = [e["operation"] for e in store.quota_entries(quota_day(datetime.now(timezone.utc)))]
    assert ops == ["commentThreads.insert"]
    html = page(ui)
    assert "pin it there" in alerts(html)
    assert 'id="pinned-posted"' in html and "Ask your cactus questions below!" in html
    assert "lc=UgzNewThread" in html and 'id="pinned-draft"' not in html
    assert reply_row(ui).status == "open"  # the reply draft is untouched


# --- the sidebar and the Today page ------------------------------------------------------------
def test_posted_drafts_do_not_count_as_waiting(tmp_path: Path) -> None:
    ui = make_post_ui(tmp_path)
    assert "1 question to answer" in ui.client.get("/comments").text
    with Store(ui.config.catalog_db) as store:
        assert [c.comment_id for c in store.comments_to_answer()] == ["comment0001"]
    post(ui)
    assert "question to answer" not in ui.client.get("/comments").text
    with Store(ui.config.catalog_db) as store:
        assert store.comments_to_answer() == [] and store.list_open_comment_replies() == []
