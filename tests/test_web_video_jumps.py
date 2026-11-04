"""The video page's jump links: every one has a place to land, and the landing clears the sticky bars."""

from __future__ import annotations

import re
from pathlib import Path

from webhelpers import make_ui, make_video, seed_catalog
from yezatas.web import app as web_app

STATIC = Path(web_app.__file__).parent / "static"


def test_every_jump_link_on_the_video_page_has_a_target(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    seed_catalog(ui.config, [make_video(1, video_id="vid00000001", title="Cactus", description="", tags=[])])
    html = ui.client.get("/videos/vid00000001").text
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    links = set(re.findall(r'href="#([^"]+)"', html))
    assert {"f-description", "f-tags", "tool-translations", "pinned-comment", "category"} <= links
    assert sorted(links - ids) == []


def test_the_landing_clears_the_sticky_bars_and_shows_a_field_label() -> None:
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert "html:has(.vnav) { scroll-padding-top: 120px; }" in css
    assert '#edit textarea[id^="f-"]' in css


def test_a_jump_opens_its_fold_and_focuses_a_field_without_scrolling() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function goTo(id)" in js and "row.open = true" in js
    assert "target.focus({ preventScroll: true })" in js
