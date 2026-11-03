"""Every page's heading is the label of the sidebar item it lights up, so the menu and the pages agree."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from webhelpers import make_ui, make_video, seed_catalog

PAGES = ["/", "/comments", "/videos", "/analytics", "/seo", "/checks", "/bulk", "/copilot", "/translations",
         "/links", "/affiliate", "/replies", "/outliers", "/search-gaps", "/competitors", "/strategy",
         "/changes", "/history", "/agent", "/configuration"]


@pytest.fixture(scope="module")
def ui(tmp_path_factory: pytest.TempPathFactory):
    ui = make_ui(tmp_path_factory.mktemp("names"))
    seed_catalog(ui.config, [make_video(n) for n in range(1, 6)])
    return ui


@pytest.mark.parametrize("path", PAGES)
def test_heading_is_the_sidebar_label(ui, path: str) -> None:
    html = ui.client.get(path).text
    active = re.search(r'<a class="nav-item active" href="([^"]+)"[^>]*>.*?<span>([^<]+)</span>', html)
    heading = re.search(r'<h1 class="page-title">([^<]+)</h1>', html)
    assert active is not None and heading is not None, path
    assert active.group(1) == path
    assert heading.group(1) == active.group(2)


def test_bulk_edit_has_its_own_item_and_a_button_on_videos(ui) -> None:
    html = ui.client.get("/bulk").text
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html) == ["/bulk"]
    videos = ui.client.get("/videos").text
    assert 'id="bulk-link" href="/bulk"' in videos and 'id="chip-fix"' in videos
