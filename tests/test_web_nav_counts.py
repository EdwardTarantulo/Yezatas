"""The sidebar's attention counts: which items get one, their tone and text, and that they follow the data."""

from __future__ import annotations

import re
from pathlib import Path

from webhelpers import make_ui, make_video, seed_catalog
from yezatas.seo import score_catalog
from yezatas.store import Comment, Store
from yezatas.web.nav_counts import build_nav_counts
from yezatas.web.views import band_of

BROKEN = "https://shop.invalid/old"


def counts(html: str) -> dict[str, tuple[str, str, str]]:
    """nav key -> (shown number, tone class, title)."""
    found = re.findall(r'<span class="nav-count( warn)?" id="nav-(?:count-)?([a-z]+)" title="([^"]*)">([^<]*)<', html)
    return {key: (shown, "warn" if warn else "info", title) for warn, key, title, shown in found}


def test_nothing_waiting_means_no_counts() -> None:
    assert build_nav_counts() == {}


def test_texts_and_the_cap() -> None:
    nav = build_nav_counts(questions=1, poor_videos=2, stale_translations=1, broken_links=150, drafts=1)
    assert nav["comments"].text == "1 question to answer" and nav["comments"].tone == "info"
    assert nav["content"].text == "2 videos score under 50" and nav["content"].tone == "warn"
    assert nav["translations"].text == "1 translation out of date"
    assert nav["links"].shown == "99+" and nav["links"].text == "150 broken links"
    assert nav["changes"].text == "1 draft to review"


def test_the_sidebar_counts_follow_the_data(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    videos = [make_video(n) for n in range(1, 13)]
    videos[0] = make_video(1, description=f"Shop: {BROKEN}")
    seed_catalog(ui.config, videos)
    with Store(ui.config.catalog_db) as store:
        store.save_link_check(BROKEN, "broken", 404)
        store.upsert_comments([Comment("q1", videos[1].video_id, author="viewer", text="How deep?",
                                       category="question")])
    poor = sum(1 for card in score_catalog(videos, {}, {}, {}).values() if band_of(card.score) == "poor")
    nav = counts(ui.client.get("/videos").text)
    assert nav["comments"] == ("1", "info", "1 question to answer")
    assert nav["links"] == ("1", "warn", "1 broken link")
    if poor:
        assert nav["content"] == (str(poor), "warn", f"{poor} video{'' if poor == 1 else 's'} "
                                  f"score{'s' if poor == 1 else ''} under 50")
    else:
        assert "content" not in nav
    assert "translations" not in nav and "changes" not in nav
    with Store(ui.config.catalog_db) as store:
        store.upsert_comments([Comment("q2", videos[2].video_id, author="viewer", text="Which soil?",
                                       category="question")])
    assert counts(ui.client.get("/videos").text)["comments"][0] == "2"


def test_a_page_with_no_catalog_has_no_counts(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    assert counts(ui.client.get("/configuration").text) == {}
