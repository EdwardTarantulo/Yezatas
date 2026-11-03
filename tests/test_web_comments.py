"""The Comments card on the video page: fetch, sort, drafts, templates and the pinned draft.

Fakes only: a fake comment reader and a fake assist function wrapping the real comments.assist
with a scripted model. No network, no real model, nothing is ever posted to YouTube.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient
from webhelpers import fake_creds, http_error, make_config, make_ui, make_video, seed_catalog
from yezatas import agent, comments
from yezatas.agent import ModelReply
from yezatas.quota import QuotaExceeded
from yezatas.store import Comment, Store
from yezatas.web import app as web_app
from yezatas.web.app import NO_PACKAGING_KEY
from yezatas.web.views import build_comments_view

V1, V2 = "vid00000001", "vid00000002"
NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


def reply(data: Any) -> ModelReply:
    return ModelReply(data if isinstance(data, str) else json.dumps(data), (), {}, 3000, 200, 150)


def item(comment_id: str, author: str, text: str, published: str, likes: int = 0) -> dict[str, Any]:
    return {"snippet": {"topLevelComment": {"id": comment_id, "snippet": {
        "authorDisplayName": author, "textOriginal": text, "publishedAt": published,
        "likeCount": likes}}, "totalReplyCount": 0}}


class FakeCommentReader:
    """Returns scripted commentThreads pages; records every call."""

    def __init__(self, pages: list[dict[str, Any]] | None = None,
                 error: Exception | None = None, gate: threading.Event | None = None) -> None:
        self.pages = list(pages) if pages is not None else [{"items": []}]
        self.error = error
        self.gate = gate
        self.entered = threading.Event()
        self.calls: list[tuple[str, Any]] = []

    def comment_threads(self, video_id: str, page_token: str | None) -> dict[str, Any]:
        self.calls.append((video_id, page_token))
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10), "the test never opened the gate"
        if self.error is not None:
            raise self.error
        return self.pages.pop(0) if self.pages else {"items": []}


def good_answer(**override: Any) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "categories": [{"ref": "C1", "category": "question"}, {"ref": "C2", "category": "praise"}],
        "replies": [{"ref": "C1", "text": "Thanks for asking. Water when the soil is dry."},
                    {"ref": "C2", "text": "Glad you liked it."}],
        "pinned": "Ask your cactus questions below."}
    answer.update(override)
    return answer


class FakeAssist:
    """The injected assist function: the real comments.assist with a scripted model."""

    def __init__(self, script: list[Any] | None = None, gate: threading.Event | None = None,
                 error: Exception | None = None) -> None:
        self.client = FakeModelClient(script if script is not None else [reply(good_answer())])
        self.gate = gate
        self.error = error
        self.entered = threading.Event()
        self.calls: list[str] = []

    def __call__(self, store: Store, video_id: str, now: datetime, report: Any) -> Any:
        self.calls.append(video_id)
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10), "the test never opened the gate"
        if self.error is not None:
            raise self.error
        return comments.assist(store, self.client, video_id, now=now, report=report)


def make_comments_ui(tmp_path: Path, reader: Any = "default", assist: Any = "default",
                     key: bool = True, write: bool = True, **kwargs: Any) -> Any:
    reader = FakeCommentReader() if reader == "default" else reader
    assist_fn = FakeAssist() if assist == "default" else assist
    loader = (lambda cfg: fake_creds()) if write else (lambda cfg: None)
    ui = make_ui(tmp_path, comment_reader=lambda creds: reader, comments_assist=assist_fn,
                 packaging_key=lambda cfg: key, write_credentials_loader=loader, **kwargs)
    seed_catalog(ui.config, [
        make_video(1, video_id=V1, title="Cactus repotting guide",
                   description="How to repot a cactus.", tags=["cactus"],
                   privacy_status="public"),
        make_video(2, video_id=V2, title="Second video", description="Plain.", tags=[],
                   privacy_status="public")])
    ui.reader = reader
    ui.assist_fn = assist_fn
    return ui


def nf(ui: Any, path: str, **kwargs: Any) -> Any:
    kwargs.setdefault("follow_redirects", False)
    return ui.post(path, **kwargs)


def page(ui: Any, video_id: str = V1, query: str = "") -> str:
    resp = ui.client.get(f"/videos/{video_id}" + query)
    assert resp.status_code == 200
    return resp.text


def card(html: str) -> str:
    return html.split('id="comments"')[1].split("</section>")[0]


def seed_comment(ui: Any, comment_id: str, author: str, text: str, published: str,
                 likes: int = 0, category: str = "", video_id: str = V1) -> None:
    with Store(ui.config.catalog_db) as store:
        store.upsert_comments([Comment(comment_id=comment_id, video_id=video_id, author=author,
                                       text=text, published_at=published, like_count=likes,
                                       category=category)], now=NOW)


def seed_reply(ui: Any, comment_id: str, text: str, video_id: str = V1) -> int:
    with Store(ui.config.catalog_db) as store:
        return store.add_comment_reply(video_id, comment_id, text, "muse-spark-1.3", [], now=NOW)


def seed_two_sorted(ui: Any) -> None:
    seed_comment(ui, "comment0002", "Bob", "Great video, loved it.", "2026-06-02T10:00:00Z",
                 category="praise")
    seed_comment(ui, "comment0001", "Alice", "How often should I water it?", "2026-06-03T10:00:00Z",
                 likes=1500, category="question")


def run_fetch(ui: Any, video_id: str = V1) -> Any:
    resp = nf(ui, f"/videos/{video_id}/comments/fetch")
    ui.web.comments_job.wait()
    return resp


def run_assist(ui: Any, video_id: str = V1) -> Any:
    resp = nf(ui, f"/videos/{video_id}/comments/assist")
    ui.web.assist_job.wait()
    return resp


# --- the card --------------------------------------------------------------------------------------
def test_card_after_adcheck_with_fetch_and_sort(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    html = page(ui)
    assert html.index('id="edit"') < html.index('id="comments"') < html.index('id="h-tools"') < html.index('id="adcheck"')
    section = card(html)
    assert "Comments" in section
    assert f'action="/videos/{V1}/comments/fetch"' in section
    assert f'action="/videos/{V1}/comments/assist"' in section
    assert "Sort and draft replies" in section
    assert 'id="comments-empty"' in section
    assert ui.reader.calls == [] and ui.assist_fn.calls == []


def test_without_permission_the_line_and_allow_edits_button_show(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path, write=False)
    section = card(page(ui))
    assert "Reading comments needs the YouTube permission that pushing uses. Nothing is posted." in section
    assert 'id="comments-no-write"' in section
    assert 'action="/changes/allow-edits"' in section and 'name="csrf"' in section
    assert "Allow Yezatas to edit your videos" in section
    assert f'action="/videos/{V1}/comments/fetch"' not in section


def test_fetch_stores_comments_and_shows_author_date_likes_text(tmp_path: Path) -> None:
    reader = FakeCommentReader([{"items": [
        item("comment0001", "Alice", "How often should I water it?", "2026-06-03T10:00:00Z", 1500),
        item("comment0002", "Bob", "Great video.", "2026-06-02T10:00:00Z", 3)]}])
    ui = make_comments_ui(tmp_path, reader=reader)
    resp = run_fetch(ui)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#comments"
    assert reader.calls == [(V1, None)]
    with Store(ui.config.catalog_db) as store:
        stored = store.list_comments(V1)
    assert [c.comment_id for c in stored] == ["comment0001", "comment0002"]
    section = card(page(ui))
    assert "Alice" in section and "Jun 3, 2026" in section and "1.5K likes" in section
    assert "How often should I water it?" in section and "Great video." in section
    assert "All (2)" in section and "Not sorted (2)" in section and "Question (0)" in section


def test_fetch_without_permission_flashes_and_reads_nothing(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path, write=False)
    resp = nf(ui, f"/videos/{V1}/comments/fetch")
    assert resp.status_code == 303
    assert ui.reader.calls == [] and ui.web.comments_job.snapshot().status == "idle"
    assert "Reading comments needs the YouTube permission" in page(ui)


def test_category_chips_counts_and_filter(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    seed_two_sorted(ui)
    seed_comment(ui, "comment0003", "Carol", "Loose text here.", "2026-06-01T10:00:00Z")
    section = card(page(ui))
    assert "All (3)" in section and "Question (1)" in section
    assert "Praise (1)" in section and "Not sorted (1)" in section
    filtered = card(page(ui, query="?c=question"))
    assert "How often should I water it?" in filtered and "Great video" not in filtered
    assert "Loose text here" not in filtered
    unsorted = card(page(ui, query="?c=not_sorted"))
    assert "Loose text here" in unsorted and "How often" not in unsorted
    assert "How often" in card(page(ui, query="?c=bogus"))


def test_newest_hundred_cap(tmp_path: Path) -> None:
    items = [item(f"comment{i:04d}", f"User{i}", f"Note number {i}", f"2026-06-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00Z") for i in range(105)]
    ui = make_comments_ui(tmp_path, reader=FakeCommentReader([{"items": items}]))
    run_fetch(ui)
    with Store(ui.config.catalog_db) as store:
        assert len(store.list_comments(V1)) == 105
    section = card(page(ui))
    assert section.count('class="cm-item"') == 100 and "All (105)" in section


def test_view_builder_counts_filter_pinned_and_link() -> None:
    made = [
        Comment(comment_id="c2", video_id=V1, author="Bob", text="Nice watering tip.", published_at="2026-06-02T10:00:00Z", category="praise"),
        Comment(comment_id="c1", video_id=V1, author="Alice", text="Question about watering?", published_at="2026-06-03T10:00:00Z", like_count=1500, category="question"),
        Comment(comment_id="c3", video_id=V1, author="Carol", text="Plain words.", published_at="2026-06-01T10:00:00Z"),
    ]
    from yezatas.store import CommentReply, ReplyTemplate
    replies = [CommentReply(5, V1, "", "Pinned words.", "m", "", "open", ()), CommentReply(6, V1, "c1", "Draft words.", "m", "", "open", ())]
    templates = [ReplyTemplate(1, "Watering", "Hi {name}, water rarely.", "", "")]
    view = build_comments_view(V1, made, replies, templates, {"c": "question"})
    assert view.total == 3 and view.filtered_total == 1 and view.filter_key == "question"
    assert view.pinned is not None and view.pinned.text == "Pinned words."
    assert [(c.label, c.count) for c in view.chips] == [("All", 3), ("Question", 1), ("Feedback", 0), ("Praise", 1), ("Spam", 0), ("Other", 0), ("Not sorted", 1)]
    row = view.rows[0]
    assert row.comment.comment_id == "c1" and row.date_text == "Jun 3, 2026"
    assert row.likes_text == "1.5K" and row.draft is not None
    assert row.youtube_url == f"https://www.youtube.com/watch?v={V1}&lc=c1"
    assert [t.text for t in row.templates] == ["Hi Alice, water rarely."]
    praise = build_comments_view(V1, made, replies, templates, {"c": "praise"})
    assert praise.rows[0].templates == ()


def test_templates_filled_under_question_only(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    seed_two_sorted(ui)
    with Store(ui.config.catalog_db) as store:
        store.save_reply_template("Watering help", "Hi {name}, water when dry.", now=NOW)
    section = card(page(ui))
    question = section.split('id="comment-comment0001"')[1].split('id="comment-comment0002"')[0]
    assert "Hi Alice, water when dry." in question and "Watering help" in question
    assert "data-copy-target=" in question
    praise = section.split('id="comment-comment0002"')[1]
    assert "Hi Bob" not in praise and "cm-template" not in praise


def test_pinned_draft_on_top_with_copy(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    seed_two_sorted(ui)
    seed_reply(ui, "", "Please read the guide first.")
    section = card(page(ui))
    assert 'id="pinned-draft"' in section and "Please read the guide first." in section
    assert section.index('id="pinned-draft"') < section.index('id="comments-list"')
    assert 'data-copy-target="pinned-text"' in section


def test_draft_copy_mark_used_and_dismiss(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    seed_two_sorted(ui)
    rid = seed_reply(ui, "comment0001", "Thanks Alice, try weekly.")
    section = card(page(ui))
    assert "Thanks Alice, try weekly." in section
    assert f'data-copy-target="draft-text-{rid}"' in section
    assert f'action="/videos/{V1}/comments/replies/{rid}/used"' in section
    resp = nf(ui, f"/videos/{V1}/comments/replies/{rid}/used")
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#comments"
    with Store(ui.config.catalog_db) as store:
        assert store.list_comment_replies(V1)[0].status == "used"
    assert "Thanks Alice, try weekly." not in card(page(ui))
    rid2 = seed_reply(ui, "comment0001", "Second draft words.")
    resp = nf(ui, f"/videos/{V1}/comments/replies/{rid2}/dismiss")
    assert resp.status_code == 303
    with Store(ui.config.catalog_db) as store:
        assert store.list_comment_replies(V1)[1].status == "dismissed"


def test_open_on_youtube_link(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    seed_two_sorted(ui)
    section = card(page(ui))
    assert f'href="https://www.youtube.com/watch?v={V1}&amp;lc=comment0001"' in section
    assert 'target="_blank" rel="noopener noreferrer">Open on YouTube</a>' in section


def test_assist_result_shows_once_with_counts_drafts_notes_cost(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    seed_two_sorted(ui)
    resp = run_assist(ui)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#comments"
    assert ui.assist_fn.calls == [V1]
    with Store(ui.config.catalog_db) as store:
        assert store.list_comments(V1)[0].category == "question"
        assert len(store.list_comment_replies(V1, status="open")) == 2
    section = card(page(ui))
    assert 'id="assist-result"' in section
    assert "Question: 1" in section and "Praise: 1" in section
    assert "2 drafts made." in section
    assert "does not answer a question or feedback" in section
    assert "muse-spark-1.3" in section and "3,150 tokens" in section and "cost $" in section
    assert "Thanks for asking. Water when the soil is dry." in section
    assert 'id="pinned-draft"' in section
    resp = nf(ui, "/comments/assist/done", data={"video_id": V1})
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#comments"
    assert 'id="assist-result"' not in card(page(ui))


def test_no_key_and_no_comments_disable_assist(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path, key=False)
    seed_two_sorted(ui)
    section = card(page(ui))
    assert 'id="assist-no-key"' in section and "no Meta API key" in section
    assert "button primary\" disabled>" in section
    resp = nf(ui, f"/videos/{V1}/comments/assist")
    assert resp.status_code == 303 and ui.assist_fn.calls == []
    assert NO_PACKAGING_KEY in page(ui)
    ui2 = make_comments_ui(tmp_path / "second")
    section2 = card(page(ui2))
    assert 'id="assist-no-comments"' in section2
    resp = nf(ui2, f"/videos/{V1}/comments/assist")
    assert resp.status_code == 303 and ui2.assist_fn.calls == []
    assert "There are no comments to sort yet." in page(ui2)


def test_comments_disabled_and_quota_errors_show_plain(tmp_path: Path) -> None:
    reader = FakeCommentReader(error=http_error(403, "commentsDisabled"))
    ui = make_comments_ui(tmp_path, reader=reader)
    run_fetch(ui)
    snap = ui.web.comments_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "comments"
    assert "Comments are turned off on this video." in card(page(ui))
    assert web_app.map_comments_error(comments.CommentsError("Plain words.")) == ("comments", "Plain words.")
    assert web_app.map_comments_error(QuotaExceeded("need 1 units but only 0 remaining for today"))[0] == "comments"
    kind, text = web_app.map_comments_error(RuntimeError("boom"))
    assert kind == "comments" and "RuntimeError" in text and "boom" not in text


def test_quota_exceeded_shows_plain(tmp_path: Path) -> None:
    reader = FakeCommentReader([{"items": [item("comment0001", "Alice", "Hello there friend.", "2026-06-03T10:00:00Z")]}])
    ui = make_comments_ui(tmp_path, reader=reader, config=make_config(tmp_path, daily_quota_budget=1))
    run_fetch(ui)
    assert ui.web.comments_job.snapshot().status == "done"
    run_fetch(ui)
    snap = ui.web.comments_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "comments"
    assert "remaining" in card(page(ui))


def test_unknown_video_and_unknown_reply(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    for path in ("/videos/missing0001/comments/fetch", "/videos/missing0001/comments/assist", "/videos/missing0001/comments/replies/1/used"):
        resp = nf(ui, path)
        assert resp.status_code == 404 and "Video not found" in resp.text, path
    assert ui.reader.calls == [] and ui.assist_fn.calls == []
    resp = nf(ui, f"/videos/{V1}/comments/replies/99999/used")
    assert resp.status_code == 303
    assert "This draft does not exist." in page(ui)


def test_csrf_required(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    seed_two_sorted(ui)
    rid = seed_reply(ui, "comment0001", "Draft words here.")
    for path in (f"/videos/{V1}/comments/fetch", f"/videos/{V1}/comments/assist", f"/videos/{V1}/comments/replies/{rid}/used", f"/videos/{V1}/comments/replies/{rid}/dismiss", "/comments/assist/done", "/comments/fetch/done"):
        assert ui.client.post(path, data={}, follow_redirects=False).status_code == 403, path
        assert ui.client.post(path, data={"csrf": "wrong"}, follow_redirects=False).status_code == 403, path
    assert ui.reader.calls == [] and ui.assist_fn.calls == []
    with Store(ui.config.catalog_db) as store:
        assert store.list_comment_replies(V1)[0].status == "open"


def test_api_status_and_js_poll_both_jobs(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    status = ui.client.get("/api/status").json()
    assert status["comments"] == {"status": "idle", "message": "", "error_kind": ""}
    assert status["assist"] == {"status": "idle", "message": "", "error_kind": ""}
    js = (Path(web_app.__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
    assert '"comments"' in js and '"assist"' in js
    assert "data-copy-target" in js and "navigator.clipboard" in js and "writeText" in js


def test_evil_text_is_escaped(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    evil = "<script>alert(1)</script>"
    seed_comment(ui, "comment0009", evil, evil, "2026-06-03T10:00:00Z", category="question")
    seed_reply(ui, "comment0009", evil)
    section = card(page(ui))
    assert "<script>alert(1)" not in section
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in section


def test_running_fetch_polls_and_other_video_is_busy(tmp_path: Path) -> None:
    gate = threading.Event()
    reader = FakeCommentReader([{"items": []}], gate=gate)
    ui = make_comments_ui(tmp_path, reader=reader)
    try:
        nf(ui, f"/videos/{V1}/comments/fetch")
        assert reader.entered.wait(10)
        html = page(ui)
        assert 'id="comments-msg"' in card(html) and 'data-poll="1"' in html
        assert ui.client.get("/api/status").json()["comments"]["status"] == "running"
        other = card(page(ui, V2))
        assert 'id="comments-busy"' in other
    finally:
        gate.set()
        ui.web.comments_job.wait()


def test_timeouts_constants_and_message(tmp_path: Path) -> None:
    assert web_app.COMMENTS_TIMEOUT_SECONDS == 120 and "2 minutes" in web_app.COMMENTS_TIMEOUT_MESSAGE
    assert web_app.ASSIST_TIMEOUT_SECONDS == 180 and "3 minutes" in web_app.ASSIST_TIMEOUT_MESSAGE
    now = [0.0]
    gate = threading.Event()
    reader = FakeCommentReader([{"items": []}], gate=gate)
    ui = make_comments_ui(tmp_path, reader=reader, clock=lambda: now[0])
    try:
        nf(ui, f"/videos/{V1}/comments/fetch")
        assert reader.entered.wait(10)
        now[0] = 121.0
        snap = ui.web.comments_job.snapshot()
        assert snap.status == "failed" and snap.message == web_app.COMMENTS_TIMEOUT_MESSAGE
    finally:
        gate.set()
        ui.web.comments_job.wait()


def test_default_assist_without_key_fails_auth(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, packaging_key=lambda cfg: True, comment_reader=lambda creds: FakeCommentReader(), write_credentials_loader=lambda cfg: fake_creds())
    seed_catalog(ui.config, [make_video(1, video_id=V1, title="Cactus guide", description="How to repot.", privacy_status="public")])
    seed_comment(ui, "comment0001", "Alice", "How often watering?", "2026-06-03T10:00:00Z")
    resp = nf(ui, f"/videos/{V1}/comments/assist")
    ui.web.assist_job.wait()
    assert resp.status_code == 303
    snap = ui.web.assist_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "auth" and snap.message == NO_PACKAGING_KEY


