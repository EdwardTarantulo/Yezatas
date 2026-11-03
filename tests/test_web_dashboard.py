"""The shell (top bar, sidebar) and the dashboard. Assertions are on structure, not on SEO checks."""

from __future__ import annotations

import re
from pathlib import Path

from markupsafe import escape

from webhelpers import make_ui, make_video, seed_catalog
from yezatas.seo import fix_priority, score_catalog
from yezatas.web.views import band_of, format_count, format_date

XSS_TITLE = "<script>alert(1)</script>"


def seeded_ui(tmp_path: Path, count: int = 40, **first: object):
    ui = make_ui(tmp_path)
    videos = [make_video(n) for n in range(1, count + 1)]
    if first:
        videos[0] = make_video(1, **first)
    seed_catalog(ui.config, videos)
    return ui, videos


def fix_rows(html: str) -> list[str]:
    return re.findall(r'class="fix-row" href="/videos/([^"]+)"', html)


def test_pages_redirect_to_setup_before_any_pull(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    for path in ("/", "/seo", "/videos", "/videos?sort=score", "/videos/vid000001"):
        resp = ui.client.get(path, follow_redirects=False)
        assert resp.status_code == 303 and resp.headers["location"] == "/configuration", path


def test_dashboard_renders_four_cards_and_the_heading(tmp_path: Path) -> None:
    ui, _ = seeded_ui(tmp_path)
    resp = ui.client.get("/seo")
    assert resp.status_code == 200
    html = resp.text
    assert "SEO overview" in html
    for card_id, title in (("card-seo", "SEO score"), ("card-fix", "Fix next"),
                           ("card-problems", "Most common problems"), ("card-catalog", "Catalog")):
        assert f'id="{card_id}"' in html and f">{title}<" in html
    assert 'href="/videos?sort=fix"' in html and ">See all<" in html
    assert 'href="/configuration#step-pull"' in html and ">Pull again<" in html


def test_seo_card_shows_the_average_over_public_videos(tmp_path: Path) -> None:
    ui, videos = seeded_ui(tmp_path)
    cards = score_catalog(videos)
    public = [cards[v.video_id].score for v in videos if v.privacy_status == "public"]
    expected = int(sum(public) / len(public) + 0.5)
    html = ui.client.get("/seo").text
    assert re.search(rf'<span class="avg">{expected}</span><span class="out-of"> / 100', html)
    assert f"Average over {len(public)} public videos scored." in html
    rows = re.findall(r'class="band-label">([^<]+?) <span class="muted">[^<]*</span></span>'
                      r'.*?<span class="band-count">(\d+)</span>', html, flags=re.S)
    assert [r[0] for r in rows] == ["Good", "OK", "Needs work"]
    assert sum(int(r[1]) for r in rows) == len(public)
    expected_counts = [sum(1 for s in public if band_of(s) == k) for k in ("good", "ok", "poor")]
    assert [int(r[1]) for r in rows] == expected_counts


def test_fix_next_follows_fix_priority_and_is_capped(tmp_path: Path) -> None:
    ui, videos = seeded_ui(tmp_path)
    cards = score_catalog(videos)
    ranked = sorted((v for v in videos if cards[v.video_id].failed),
                    key=lambda v: fix_priority(v, cards[v.video_id]), reverse=True)
    html = ui.client.get("/seo").text
    shown = fix_rows(html)
    assert 0 < len(shown) <= 8
    priorities = [fix_priority(next(v for v in videos if v.video_id == i), cards[i]) for i in shown]
    assert priorities == sorted(priorities, reverse=True)
    # nothing outside the top 8 may outrank what is shown
    cutoff = min(priorities)
    shown_set = set(shown)
    assert all(fix_priority(v, cards[v.video_id]) <= cutoff
               for v in ranked if v.video_id not in shown_set)
    # each row shows views, a score badge and the biggest failed check
    row = html.split('class="fix-row"')[1]
    assert " views" in row and 'class="badge ' in row
    first = cards[shown[0]]
    assert str(escape(first.failed[0].title)) in row
    assert "To fix: " + str(escape(first.failed[0].title)) in row
    assert f"+{first.missing} points possible" in row


def test_common_problems_link_to_the_filtered_list(tmp_path: Path) -> None:
    ui, videos = seeded_ui(tmp_path)
    cards = score_catalog(videos)
    counts: dict[str, int] = {}
    for v in videos:
        if v.privacy_status != "public":
            continue
        for c in cards[v.video_id].checks:
            if c.applicable and not c.passed:
                counts[c.check_id] = counts.get(c.check_id, 0) + 1
    html = ui.client.get("/seo").text
    found = re.findall(r'class="problem-row" href="/videos\?check=([^"]+)">.*?'
                       r'class="problem-count">(\d+) video', html, flags=re.S)
    assert 0 < len(found) <= 6
    for check_id, count in found:
        assert counts[check_id] == int(count)
    top = sorted(counts.values(), reverse=True)[: len(found)]
    assert sorted((int(c) for _, c in found), reverse=True) == top


def test_catalog_card_counts_pull_time_and_quota(tmp_path: Path) -> None:
    ui, videos = seeded_ui(tmp_path)
    html = ui.client.get("/seo").text
    pub = sum(v.privacy_status == "public" for v in videos)
    unl = sum(v.privacy_status == "unlisted" for v in videos)
    prv = sum(v.privacy_status == "private" for v in videos)
    assert f'<span class="avg">{len(videos)}</span>' in html
    assert f"{pub} public" in html and f"{unl} unlisted" in html and f"{prv} private" in html
    assert f"{len(videos)} videos were pulled on " in html
    budget = ui.config.daily_quota_budget
    assert f"Quota today: 0 used, {budget:,} left of {budget:,}." in html


def test_setup_has_ai_data_anchor(tmp_path: Path) -> None:
    ui, _ = seeded_ui(tmp_path)
    html = ui.client.get("/configuration").text
    assert 'id="ai-data"' in html
    assert "on free credits Meta may use it to improve its products." in html



def test_thumbnail_and_placeholder(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    seed_catalog(ui.config, [
        make_video(1, view_count=900_000),
        make_video(2, view_count=800_000, thumbnails={}),
        make_video(3, view_count=700_000,
                   thumbnails={"default": {"url": "https://i.ytimg.com/vi/x/default.jpg"}}),
    ])
    html = ui.client.get("/videos").text
    assert "https://i.ytimg.com/vi/vid000001/mqdefault.jpg" in html
    assert "https://i.ytimg.com/vi/x/default.jpg" in html  # falls back to "default"
    assert 'class="thumb empty sm"' in html


def test_titles_are_escaped_on_every_page(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    videos = [make_video(n) for n in range(2, 12)]
    videos.append(make_video(1, title=XSS_TITLE, view_count=5_000_000, privacy_status="public",
                             description=XSS_TITLE, tags=[XSS_TITLE]))
    seed_catalog(ui.config, videos)
    for path in ("/", "/videos", "/videos?q=alert", "/videos/vid000001"):
        html = ui.client.get(path).text
        assert XSS_TITLE not in html, path
        assert "<script>alert" not in html, path
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in ui.client.get("/videos").text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in ui.client.get("/videos/vid000001").text
    # a search term is echoed back escaped too
    echoed = ui.client.get("/videos", params={"q": XSS_TITLE}).text
    assert XSS_TITLE not in echoed and "&lt;script&gt;" in echoed


def test_shell_marks_the_active_nav_item(tmp_path: Path) -> None:
    ui, _ = seeded_ui(tmp_path)

    def active(path: str) -> list[str]:
        html = ui.client.get(path).text
        return re.findall(r'class="nav-item active" href="([^"]+)"', html)

    assert active("/") == ["/"]
    assert active("/videos") == ["/videos"]
    assert active("/videos/vid000001") == ["/videos"]
    assert active("/configuration") == ["/configuration"]
    html = ui.client.get("/seo").text
    assert 'aria-current="page"' in html
    assert 'action="/videos" method="get"' in html and 'name="q"' in html
    assert "YouTube Studio" not in html  # we only borrow the style, not the name


def test_search_box_keeps_the_current_query(tmp_path: Path) -> None:
    ui, _ = seeded_ui(tmp_path)
    html = ui.client.get("/videos", params={"q": "cactus tips"}).text
    assert 'name="q" value="cactus tips"' in html


def test_pages_have_no_inline_scripts_or_styles(tmp_path: Path) -> None:
    ui, _ = seeded_ui(tmp_path)
    for path in ("/", "/seo", "/videos", "/videos/vid000001", "/configuration"):
        html = ui.client.get(path).text
        assert " style=" not in html, path
        assert not re.search(r"<script(?![^>]*\bsrc=)", html), path
        assert "<style" not in html, path


def test_formatting_helpers() -> None:
    assert [format_count(n) for n in (0, 999, 1000, 1500, 21_100, 999_999, 1_200_000)] == [
        "0", "999", "1K", "1.5K", "21.1K", "999K", "1.2M"]
    assert format_date("2026-10-30T10:00:00Z") == "Oct 30, 2026"
    assert format_date("") == "" and format_date("garbage") == ""
    assert (band_of(80), band_of(79), band_of(50), band_of(49)) == ("good", "ok", "ok", "poor")


def test_setup_page_uses_the_shell_and_keeps_its_hooks(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    html = ui.client.get("/configuration").text
    assert 'class="topbar"' in html and 'class="sidebar"' in html
    assert 'id="step-project"' in html and 'name="csrf"' in html
    assert "data-slider" in html and "data-slide-next" in html and 'class="gpanel' in html


def test_error_pages_have_no_shell(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, signed_in_browser=False)
    html = ui.client.get("/configuration").text
    assert 'class="topbar"' not in html and 'class="sidebar"' not in html


def test_fix_next_says_how_it_is_ranked(tmp_path: Path) -> None:
    ui, _ = seeded_ui(tmp_path)
    html = ui.client.get("/seo").text
    assert "Ranked by views &times; points missing (100 minus the SEO score)" in html
    assert "names the check that costs that video the most points" in html
