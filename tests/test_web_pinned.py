"""The Pinned comment card on the video page: its states from the fetched comments, the suggestion job with a
scripted model, Copy, "I pinned one already", the fix rail and the tile, and the no-key path. No network."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient
from webhelpers import make_ui, make_video, seed_catalog
from yezatas import agent, pinnedcomment
from yezatas.agent import ModelReply
from yezatas.store import Comment, PinnedComment, Store

V1 = "vid00000001"


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


class FakePinned:
    """The injected suggestion: the real one with a scripted model."""

    def __init__(self, *texts: str) -> None:
        texts = texts or ("Which cactus should I repot next?",)
        self.client = FakeModelClient([ModelReply(json.dumps({"text": t}), (), {}, 800, 0, 60) for t in texts])
        self.calls: list[str] = []

    def __call__(self, store: Store, video_id: str, now: datetime, report: Any) -> PinnedComment:
        self.calls.append(video_id)
        return pinnedcomment.suggest_pinned(store, self.client, video_id, now=now, report=report)


def make_pin_ui(tmp_path: Path, fn: Any = None, key: bool = True) -> Any:
    fn = fn or FakePinned()
    ui = make_ui(tmp_path, pinned=fn, packaging_key=lambda cfg: key)
    seed_catalog(ui.config, [make_video(1, video_id=V1)])
    ui.fn = fn
    return ui


def add_comments(ui: Any, *, own: bool) -> None:
    with Store(ui.config.catalog_db) as store:
        store.upsert_comments([Comment("c1", V1, author="viewer", text="Nice"),
                               Comment("c2", V1, author="me", text="Pinned", by_owner=own)])


def page(ui: Any) -> str:
    resp = ui.client.get(f"/videos/{V1}")
    assert resp.status_code == 200
    return resp.text


def card(ui: Any) -> str:
    return page(ui).split('id="pinned-comment"')[1].split("</section>")[0]


def tile(html: str) -> str:
    found = re.search(r'id="fact-pinned"([^>]*)><span class="vfact-k">Pinned comment</span>'
                      r'<span class="vfact-v">([^<]*)</span>', html)
    assert found
    return found.group(2) + found.group(1)


def test_comments_never_fetched_cost_nothing(tmp_path: Path) -> None:
    ui = make_pin_ui(tmp_path)
    part = card(ui)
    assert ">Not fetched</span>" in part and "has not fetched this video" in part
    assert ">Suggest a pinned comment</button>" in part and ">I pinned one already</button>" in part
    html = page(ui)
    assert 'href="#check-pinned_comment"' not in html and tile(html) == "Not fetched"
    assert " style=" not in part


def test_fetched_without_one_from_you_fails_and_links_from_the_rail(tmp_path: Path) -> None:
    ui = make_pin_ui(tmp_path)
    add_comments(ui, own=False)
    html = page(ui)
    assert ">None found</span>" in card(ui)
    assert 'href="#check-pinned_comment"' in html and 'href="#pinned-comment">Pinned comment</a>' in html
    assert tile(html) == 'None found data-tone="warn"'


def test_a_comment_from_the_channel_passes(tmp_path: Path) -> None:
    ui = make_pin_ui(tmp_path)
    add_comments(ui, own=True)
    html = page(ui)
    assert ">Yours found</span>" in card(ui) and 'href="#check-pinned_comment"' not in html
    assert tile(html) == "Yes"


def test_suggest_then_suggest_another(tmp_path: Path) -> None:
    ui = make_pin_ui(tmp_path, FakePinned("Which cactus should I repot next?", "What should I repot next?"))
    resp = ui.post(f"/videos/{V1}/pinned/suggest", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#pinned-comment"
    ui.web.pinned_job.wait()
    part = card(ui)
    assert 'id="pin-suggestion-text">Which cactus should I repot next?</p>' in part
    assert 'data-copy-target="pin-suggestion-text"' in part and ">Suggest another</button>" in part
    assert "Suggested " in part
    ui.post(f"/videos/{V1}/pinned/suggest")
    ui.web.pinned_job.wait()
    assert 'id="pin-suggestion-text">What should I repot next?</p>' in card(ui)
    assert ui.fn.calls == [V1, V1]


def test_i_pinned_one_already_makes_the_check_pass(tmp_path: Path) -> None:
    ui = make_pin_ui(tmp_path)
    add_comments(ui, own=False)
    resp = ui.post(f"/videos/{V1}/pinned/mark", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#pinned-comment"
    part = card(ui)
    assert ">Marked pinned</span>" in part and "I pinned one already" not in part
    html = page(ui)
    assert 'href="#check-pinned_comment"' not in html and tile(html) == "Yes"


def test_a_refused_suggestion_shows_its_error(tmp_path: Path) -> None:
    ui = make_pin_ui(tmp_path, FakePinned("Buy it at https://shop.invalid/x"))
    ui.post(f"/videos/{V1}/pinned/suggest")
    ui.web.pinned_job.wait()
    part = card(ui)
    assert 'id="pinned-error"' in part and "https://shop.invalid/x" in part
    assert 'id="pin-suggestion-text"' not in part


def test_without_a_key_there_is_no_suggest_button(tmp_path: Path) -> None:
    ui = make_pin_ui(tmp_path, key=False)
    part = card(ui)
    assert 'id="pinned-no-key"' in part and "/pinned/suggest" not in part
    ui.post(f"/videos/{V1}/pinned/suggest")
    assert ui.fn.calls == []


def test_an_unknown_video_is_neither_suggested_nor_marked(tmp_path: Path) -> None:
    ui = make_pin_ui(tmp_path)
    ui.post("/videos/nope/pinned/suggest")
    ui.post("/videos/nope/pinned/mark")
    assert ui.fn.calls == []
    with Store(ui.config.catalog_db) as store:
        assert store.pinned_comment("nope") is None


def test_category_and_pinned_comment_sit_in_details_above_tools(tmp_path: Path) -> None:
    ui = make_pin_ui(tmp_path)
    html = page(ui)
    details = html.split('<section class="vdetails" id="details"')[1].split('id="h-tools"')[0]
    assert html.index('id="h-details">Video details</h2>') < html.index('id="h-tools"')
    assert html.index('id="h-edit"') < html.index('id="h-details"')
    assert '<details class="tool" id="detail-category" data-keep>' in html
    assert '<details class="tool" id="detail-pinned" data-keep>' in html
    assert html.index('id="detail-category"') < html.index('id="category"') < html.index('id="detail-pinned"')
    assert html.index('id="detail-pinned"') < html.index('id="pinned-comment"') < html.index('id="h-tools"')
    assert '<span class="tool-name">Pinned comment</span>' in details
    assert re.search(r'<span class="tool-name">Pinned comment</span>.*?<span class="pill unverified">Not fetched</span>'
                     r'<span class="row-act">Check pinned comment</span></summary>', html, re.S)
    assert '<a href="#h-details">Video details</a>' in html
    assert 'href="#category"' in html and 'href="#pinned-comment"' in html  # the tiles still lead there


def test_a_found_comment_from_the_channel_is_shown(tmp_path: Path) -> None:
    ui = make_pin_ui(tmp_path)
    with Store(ui.config.catalog_db) as store:
        store.upsert_comments([
            Comment("c1", V1, author="viewer", text="Nice", published_at="2026-10-02T10:00:00Z"),
            Comment("c2", V1, author="me", text="Old note", by_owner=True, published_at="2026-09-01T10:00:00Z"),
            Comment("c3", V1, author="me", text="Ask me <anything>", by_owner=True, like_count=1234,
                    published_at="2026-10-01T10:00:00Z"),
        ])
    part = card(ui)
    assert ">Yours found</span>" in part
    found = part.split('id="pinned-found"')[1].split("</div>")[0]
    assert '<p class="pin-text" id="pin-found-text">Ask me &lt;anything&gt;</p>' in found
    assert "Oct 1, 2026 &middot; 1.2K likes" in found and "Old note" not in found
    assert 'href="https://www.youtube.com/watch?v=vid00000001&amp;lc=c3"' in found and ">Open on YouTube</a>" in found


def test_no_comment_from_the_channel_shows_none(tmp_path: Path) -> None:
    ui = make_pin_ui(tmp_path)
    add_comments(ui, own=False)
    assert 'id="pinned-found"' not in card(ui)
