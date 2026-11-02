"""Channel-wide comments page /comments."""

from __future__ import annotations

import re
import threading
from pathlib import Path

import pytest

from yezatas import agent
from yezatas.store import Comment, Store
from test_web_comments import V1, V2, NOW, FakeAssist, FakeCommentReader, good_answer, item, make_comments_ui, nf, reply, seed_comment, seed_reply


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


def test_nav_has_comments_after_copilot(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    html = ui.client.get("/").text
    items = re.findall(r'<a class="nav-item[^"]*" href="([^"]+)"', html)
    assert items.index("/comments") == items.index("/copilot") + 1
    assert items.index("/replies") == items.index("/comments") + 1
    html2 = ui.client.get("/comments").text
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html2) == ["/comments"]


def test_empty_page(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    resp = ui.client.get("/comments")
    assert resp.status_code == 200
    html = resp.text
    assert 'id="comments-empty"' in html
    assert 'id="fetch-button"' in html
    assert "Yezatas only reads comments: you post replies on YouTube yourself." in html
    assert "about 2 units" in html


def test_fetch(tmp_path: Path) -> None:
    reader = FakeCommentReader(pages=[{"items": [item("comment0001", "Alice", "Hello", "2026-06-02T10:00:00Z")]}, {"items": [item("comment0002", "Bob", "Hi", "2026-06-03T10:00:00Z")]}])
    ui = make_comments_ui(tmp_path, reader=reader)
    resp = nf(ui, "/comments/fetch")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/comments"
    ui.web.channel_comments_job.wait()
    assert len(reader.calls) == 2
    html = ui.client.get("/comments").text
    assert "Cactus repotting guide" in html
    assert "Second video" in html
    assert 'id="fetch-result"' in html
    assert "Fetched 2 comments from 2 videos." in html
    # dismiss
    resp2 = nf(ui, "/comments/channel/fetch/done")
    assert resp2.status_code == 303
    html2 = ui.client.get("/comments").text
    assert 'id="fetch-result"' not in html2


def test_fetch_without_write(tmp_path: Path) -> None:
    reader = FakeCommentReader(pages=[{"items": [item("comment0001", "Alice", "Hello", "2026-06-02T10:00:00Z")]}])
    ui = make_comments_ui(tmp_path, write=False, reader=reader)
    html = ui.client.get("/comments").text
    assert 'id="comments-no-write"' in html
    resp = nf(ui, "/comments/fetch")
    assert resp.status_code == 303
    assert reader.calls == []


def test_sort(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    seed_comment(ui, "comment0001", "Alice", "How often?", "2026-06-02T10:00:00Z", category="")
    seed_comment(ui, "comment0002", "Bob", "Nice", "2026-06-02T11:00:00Z", category="")
    resp = nf(ui, "/comments/sort")
    assert resp.status_code == 303
    ui.web.channel_sort_job.wait()
    assert ui.assist_fn.calls == [V1]
    html = ui.client.get("/comments").text
    assert 'id="sort-result"' in html
    assert 'id="sort-drafts"' in html


def test_sort_without_key(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path, key=False)
    seed_comment(ui, "comment0001", "Alice", "How often?", "2026-06-02T10:00:00Z", category="")
    html = ui.client.get("/comments").text
    assert 'id="sort-no-key"' in html
    resp = nf(ui, "/comments/sort")
    assert resp.status_code == 303
    assert ui.assist_fn.calls == []


def test_sort_nothing_unsorted(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    seed_comment(ui, "comment0001", "Alice", "Great", "2026-06-02T10:00:00Z", category="praise")
    nf(ui, "/comments/sort")
    html = ui.client.get("/comments").text
    assert "Every stored comment is sorted already." in html
    assert ui.assist_fn.calls == []


def test_mark_draft_used_from_comments(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    seed_comment(ui, "comment0001", "Alice", "How often?", "2026-06-02T10:00:00Z", category="question", video_id=V2)
    rid = seed_reply(ui, "comment0001", "Thanks for asking.", video_id=V2)
    # used with valid next
    resp = nf(ui, f"/videos/{V2}/comments/replies/{rid}/used", data={"next": "/comments?c=question"})
    assert resp.status_code == 303
    assert resp.headers["location"] == "/comments?c=question"
    html = ui.client.get("/comments").text
    assert "Thanks for asking." not in html
    assert 'id="comment-comment0001"' in html
    # seed again for bad next test
    rid2 = seed_reply(ui, "comment0001", "Another draft", video_id=V2)
    for bad in ["https://example.com/", "//example.com"]:
        resp2 = nf(ui, f"/videos/{V2}/comments/replies/{rid2}/used", data={"next": bad})
        assert resp2.headers["location"] == f"/videos/{V2}#comments"
    # dismiss with next
    resp3 = nf(ui, f"/videos/{V2}/comments/replies/{rid2}/dismiss", data={"next": "/comments"})
    assert resp3.status_code == 303
    assert resp3.headers["location"] == "/comments"


def test_filter_by_category_and_video(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    seed_comment(ui, "comment0001", "Alice", "How often?", "2026-06-02T10:00:00Z", category="question", video_id=V1)
    seed_comment(ui, "comment0002", "Bob", "Question two", "2026-06-02T11:00:00Z", category="question", video_id=V2)
    seed_comment(ui, "comment0003", "Carol", "Nice!", "2026-06-02T12:00:00Z", category="praise", video_id=V2)
    html = ui.client.get(f"/comments?c=question&v={V2}").text
    assert 'id="comment-comment0002"' in html
    assert 'id="comment-comment0001"' not in html
    assert 'id="comment-comment0003"' not in html


def test_sort_shows_without_write_permission(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path, write=False)
    seed_comment(ui, "comment0001", "Alice", "How often?", "2026-06-02T10:00:00Z", category="")
    html = ui.client.get("/comments").text
    assert 'id="sort-button"' in html
    assert 'id="comments-no-write"' in html


def test_rail_lists_videos_and_filters(tmp_path: Path) -> None:
    ui = make_comments_ui(tmp_path)
    seed_comment(ui, "comment0001", "Alice", "Hello", "2026-06-02T10:00:00Z", category="question", video_id=V1)
    seed_comment(ui, "comment0002", "Bob", "Hi", "2026-06-02T11:00:00Z", category="question", video_id=V2)
    html = ui.client.get("/comments").text
    assert 'id="h-by-video"' in html
    assert "Cactus repotting guide" in html
    assert "Second video" in html
    assert "2 videos" in html
    html2 = ui.client.get(f"/comments?v={V1}").text
    assert 'href="/comments?v=vid00000001"' in html2
    assert 'aria-current="true"' in html2
    assert "All videos" in html2
