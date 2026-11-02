"""The content table (/videos) and the video page (/videos/{id}). Structure, not specific checks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from markupsafe import escape

from webhelpers import make_ui, make_video, seed_catalog
from yezatas.seo import CATEGORIES, fix_priority, score_catalog
from yezatas.store import Video
from yezatas.web.views import PAGE_SIZE


def seeded(tmp_path: Path, count: int = 30, videos: list[Video] | None = None) -> Any:
    ui = make_ui(tmp_path)
    videos = videos if videos is not None else [make_video(n) for n in range(1, count + 1)]
    seed_catalog(ui.config, videos)
    ui.videos = {v.video_id: v for v in videos}  # type: ignore[attr-defined]
    ui.cards = score_catalog(videos)  # type: ignore[attr-defined]
    return ui


def row_ids(html: str) -> list[str]:
    return re.findall(r'class="vtitle" href="/videos/([^"]+)"', html)


def listed(ui: Any, query: str = "") -> list[str]:
    resp = ui.client.get("/videos" + query)
    assert resp.status_code == 200, query
    return row_ids(resp.text)


def test_content_page_renders_rows_with_all_columns(tmp_path: Path) -> None:
    ui = seeded(tmp_path)
    html = ui.client.get("/videos").text
    assert "<h1 class=\"page-title\">Videos</h1>" in html
    assert f"{len(ui.videos)} videos" in html
    ids = row_ids(html)
    assert sorted(ids) == sorted(ui.videos)
    first = ui.videos[ids[0]]
    row = html.split(f'class="vtitle" href="/videos/{first.video_id}"')[1].split("</tr>")[0]
    assert str(escape(first.title)) in row
    assert 'class="badge ' in row
    # a description snippet of at most ~90 characters
    snippets = re.findall(r'class="vdesc">([^<]*)<', html)
    assert snippets and all(len(s) <= 91 for s in snippets)
    assert re.search(r"[A-Z][a-z]{2} \d{1,2}, 2026", html)


def test_default_sort_is_newest_and_fix_next_is_a_chip(tmp_path: Path) -> None:
    ui = seeded(tmp_path)
    for query in ("", "?sort=nonsense", "?sort=date", "?sort=newest"):
        dates = [ui.videos[i].published_at for i in listed(ui, query)]
        assert dates == sorted(dates, reverse=True), query
    ids = listed(ui, "?sort=fix&dir=asc")  # a direction means nothing for Fix next
    prio = [fix_priority(ui.videos[i], ui.cards[i]) for i in ids]
    assert prio == sorted(prio, reverse=True)
    html = ui.client.get("/videos").text
    assert 'aria-label="Sort by"' not in html
    assert re.search(r'class="chip" id="chip-fix" href="/videos\?sort=fix"[^>]*>Fix next first<', html)
    on = ui.client.get("/videos?sort=fix").text
    assert re.search(r'class="chip active" id="chip-fix" href="/videos" aria-current="true"', on)


def test_column_titles_sort_and_a_second_click_reverses(tmp_path: Path) -> None:
    ui = seeded(tmp_path)
    head = ui.client.get("/videos").text.split("<thead>")[1].split("</thead>")[0]
    assert re.search(r'<th class="col-date" scope="col" aria-sort="descending"><a class="th-sort active" '
                     r'href="/videos\?dir=asc" title="Order by Date, oldest first">Date<', head)
    assert 'href="/videos?sort=title" title="Order by Video, A to Z">Video<' in head
    assert 'href="/videos?sort=views" title="Order by Views, most first">Views<' in head
    assert 'href="/videos?sort=score" title="Order by SEO score, lowest first">SEO score<' in head
    assert head.count("aria-sort=") == 1
    titles = [ui.videos[i].title.casefold() for i in listed(ui, "?sort=title")]
    assert titles == sorted(titles)
    back = [ui.videos[i].title.casefold() for i in listed(ui, "?sort=title&dir=desc")]
    assert back == sorted(back, reverse=True)
    views = [ui.videos[i].view_count for i in listed(ui, "?sort=views&dir=asc")]
    assert views == sorted(views)
    scores = [ui.cards[i].score for i in listed(ui, "?sort=score&dir=desc")]
    assert scores == sorted(scores, reverse=True)
    oldest = [ui.videos[i].published_at for i in listed(ui, "?dir=asc")]
    assert oldest == sorted(oldest)
    flipped = ui.client.get("/videos?sort=views").text.split("<thead>")[1].split("</thead>")[0]
    assert re.search(r'aria-sort="descending"><a class="th-sort active" href="/videos\?sort=views&amp;dir=asc"', flipped)
    assert listed(ui, "?sort=views&dir=sideways") == listed(ui, "?sort=views")


def test_sort_score_views_newest(tmp_path: Path) -> None:
    ui = seeded(tmp_path)
    scores = [ui.cards[i].score for i in listed(ui, "?sort=score")]
    assert scores == sorted(scores)
    views = [ui.videos[i].view_count for i in listed(ui, "?sort=views")]
    assert views == sorted(views, reverse=True)
    dates = [ui.videos[i].published_at for i in listed(ui, "?sort=newest")]
    assert dates == sorted(dates, reverse=True)
    assert len(set(listed(ui, "?sort=score"))) == len(ui.videos)


def test_search_matches_title_description_and_tags_case_insensitively(tmp_path: Path) -> None:
    videos = [
        make_video(1, title="Zebra plant care", description="plain", tags=["a"]),
        make_video(2, title="Plain title", description="Sunlight and the ALOE vera", tags=["a"]),
        make_video(3, title="Plain again", description="plain", tags=["Haworthia", "b"]),
        make_video(4, title="Nothing here", description="plain", tags=["a"]),
    ]
    ui = seeded(tmp_path, videos=videos)
    assert listed(ui, "?q=zebra") == ["vid000001"]
    assert listed(ui, "?q=aloe%20VERA") == ["vid000002"]
    assert listed(ui, "?q=HAWORTHIA") == ["vid000003"]
    assert sorted(listed(ui, "?q=plain")) == ["vid000001", "vid000002", "vid000003", "vid000004"]
    assert listed(ui, "?q=%20%20") == listed(ui, "")  # blank search = no filter
    html = ui.client.get("/videos?q=nomatchatall").text
    assert "No videos match" in html and "0 videos" in html and 'id="no-videos"' in html


def test_visibility_and_band_filters(tmp_path: Path) -> None:
    ui = seeded(tmp_path, 40)
    for vis in ("public", "unlisted", "private"):
        ids = listed(ui, f"?visibility={vis}")
        assert ids and all(ui.videos[i].privacy_status == vis for i in ids)
        assert len(ids) == sum(v.privacy_status == vis for v in ui.videos.values())
    assert len(listed(ui, "?visibility=all")) == 40
    assert len(listed(ui, "?visibility=bogus")) == 40
    bounds = {"good": (80, 100), "ok": (50, 79), "poor": (0, 49)}
    seen = 0
    for band, (lo, hi) in bounds.items():
        ids = listed(ui, f"?band={band}")
        assert all(lo <= ui.cards[i].score <= hi for i in ids)
        seen += len(ids)
    assert seen == 40
    assert len(listed(ui, "?band=bogus")) == 40
    both = listed(ui, "?visibility=public&band=poor")
    assert all(ui.videos[i].privacy_status == "public" and ui.cards[i].score < 50 for i in both)


def test_check_filter_lists_videos_failing_that_check(tmp_path: Path) -> None:
    ui = seeded(tmp_path, 40)
    failing: dict[str, set[str]] = {}
    for vid, card in ui.cards.items():
        for c in card.checks:
            if c.applicable and not c.passed:
                failing.setdefault(c.check_id, set()).add(vid)
    assert failing, "the seeded catalog should fail at least one check"
    for check_id, expected in list(failing.items())[:3]:
        assert set(listed(ui, f"?check={check_id}")) == expected
    html = ui.client.get("/videos?check=no_such_check").text
    assert "No videos match" in html


def test_check_chip_shows_title_and_clears(tmp_path: Path) -> None:
    ui = seeded(tmp_path, 40)
    check_id = next(c.check_id for card in ui.cards.values() for c in card.failed)
    title = next(c.title for card in ui.cards.values() for c in card.checks
                 if c.check_id == check_id)
    html = ui.client.get(f"/videos?check={check_id}&sort=views").text
    assert str(escape(title)) in html
    assert 'href="/videos?sort=views"' in html  # "Clear" keeps the other params


def test_filter_chips_are_links_that_mark_the_active_one(tmp_path: Path) -> None:
    ui = seeded(tmp_path)
    html = ui.client.get("/videos?visibility=public&sort=views").text
    assert re.search(r'class="chip active" href="/videos\?sort=views&amp;visibility=public"', html)
    assert 'href="/videos?sort=views&amp;visibility=private"' in html
    assert re.search(r'class="th-sort active" href="/videos\?sort=views&amp;dir=asc&amp;visibility=public"'
                     r'[^>]*>Views<', html)


def test_pagination_with_120_videos(tmp_path: Path) -> None:
    ui = seeded(tmp_path, 120)
    page1 = ui.client.get("/videos?sort=views").text
    assert "120 videos" in page1 and "Page 1 of 3" in page1
    ids1, ids2, ids3 = (listed(ui, f"?sort=views&page={p}") for p in (1, 2, 3))
    assert (len(ids1), len(ids2), len(ids3)) == (PAGE_SIZE, PAGE_SIZE, 20)
    assert len(set(ids1 + ids2 + ids3)) == 120
    assert re.search(r'rel="next" href="/videos\?sort=views&amp;page=2"', page1)
    assert 'rel="prev"' not in page1
    page2 = ui.client.get("/videos?sort=views&page=2").text
    assert 'rel="prev" href="/videos?sort=views"' in page2
    assert 'rel="next" href="/videos?sort=views&amp;page=3"' in page2
    page3 = ui.client.get("/videos?sort=views&page=3").text
    assert 'rel="next"' not in page3 and "Page 3 of 3" in page3


def test_pagination_keeps_filters_in_links(tmp_path: Path) -> None:
    ui = seeded(tmp_path, 160)
    html = ui.client.get("/videos?visibility=public&band=poor&q=cactus&sort=score").text
    # whatever the number of pages, every link that is built keeps the other params
    for href in re.findall(r'rel="(?:next|prev)" href="([^"]+)"', html):
        assert "visibility=public" in href and "q=cactus" in href and "sort=score" in href


def test_bad_page_parameters_never_500(tmp_path: Path) -> None:
    ui = seeded(tmp_path, 120)
    first = listed(ui, "?sort=views&page=1")
    for bad in ("abc", "", "0", "-5", "1.5", "%20", "9" * 5000):
        assert listed(ui, f"?sort=views&page={bad}") == first, bad
    assert listed(ui, "?sort=views&page=999") == listed(ui, "?sort=views&page=3")
    for query in ("?q=" + "x" * 5000, "?sort=%FF", "?check=" + "y" * 500, "?band=&visibility="):
        assert ui.client.get("/videos" + query).status_code == 200, query[:30]


def test_video_page_header_and_links(tmp_path: Path) -> None:
    ui = seeded(tmp_path, videos=[make_video(7, view_count=21_100, like_count=1500,
                                              comment_count=12,
                                              published_at="2026-10-30T10:00:00Z")])
    html = ui.client.get("/videos/vid000007").text
    assert 'href="/videos"' in html and "<span>Videos</span>" in html
    facts = html[html.index('id="video-facts"'):html.index("</ul>", html.index('id="video-facts"'))]
    for key, value in (("views", "21.1K"), ("likes", "1.5K"), ("comments", "12")):
        assert re.search(rf'id="fact-{key}"><span class="vfact-k">[^<]+</span><span class="vfact-v">{value}</span>', facts)
    assert "Oct 30, 2026" in html
    assert re.search(r'<a class="button secondary" href="https://studio.youtube.com/video/vid000007/edit" '
                     r'target="_blank" rel="noopener noreferrer">Edit in YouTube Studio</a>', html)
    assert re.search(r'href="https://www.youtube.com/watch\?v=vid000007" target="_blank" '
                     r'rel="noopener noreferrer">Watch on YouTube</a>', html)
    score = ui.cards["vid000007"].score
    assert re.search(rf'id="video-score">.*?<span class="avg">{score}</span>', html, flags=re.S)


def test_video_page_groups_checks_and_shows_fixes(tmp_path: Path) -> None:
    ui = seeded(tmp_path, 40)
    # pick a video that has at least one failed check with a fix text
    vid = next(i for i, card in ui.cards.items() if any(c.fix for c in card.failed))
    card = ui.cards[vid]
    html = ui.client.get(f"/videos/{vid}").text
    groups = re.findall(r'id="group-(\w+)"', html)
    expected = [k for k in CATEGORIES if any(c.category == k for c in card.checks)]
    assert groups == expected
    for c in card.checks:
        assert f'id="check-{c.check_id}"' in html
        assert str(escape(c.title)) in html
    for c in card.failed:
        if c.fix:
            assert str(escape(c.fix)) in html
            assert f"{c.points}/{c.max_points}" in html
    # inside each group: failed first, then passed, then not applicable
    order = {"fail": 0, "part": 0, "pass": 1, "na": 2}
    for section in html.split('<section class="card check-group"')[1:]:
        states = [order[s] for s in re.findall(r'<li class="check (\w+)"', section)]
        assert states == sorted(states)
    # passed checks show no fix box
    for item in re.findall(r'<li class="check pass".*?</li>', html, flags=re.S):
        assert "check-fix" not in item


def test_unknown_video_is_a_404_page(tmp_path: Path) -> None:
    ui = seeded(tmp_path)
    for path in ("/videos/nope", "/videos/%22%3E%3Cscript%3Ealert(1)%3C",
                 "/videos/vid000001%20"):
        resp = ui.client.get(path)
        assert resp.status_code == 404, path
        assert resp.headers["content-type"].startswith("text/html")
        assert "Video not found" in resp.text and 'href="/videos"' in resp.text
        assert "Back to Content" in resp.text
        assert "<script>alert" not in resp.text
    assert 'class="topbar"' in ui.client.get("/videos/nope").text  # shown inside the shell


def test_video_page_menu_and_section_names(tmp_path: Path) -> None:
    ui = seeded(tmp_path, 40)
    vid = next(i for i, card in ui.cards.items() if card.failed)
    html = ui.client.get(f"/videos/{vid}").text
    start = html.index('<nav class="vnav"')
    nav = html[start:html.index("</nav>", start)]
    assert re.findall(r'href="(#[a-z-]+)"', nav) == ["#edit", "#h-tools", "#checks"]
    assert ">Tools</a>" in nav and ">Checklist" in nav and "What to fix" not in nav
    assert '<h2 id="h-tools">Tools</h2>' in html and "AI tools" not in html
    assert '<span class="tool-name">Checklist</span>' in html and "All checks" not in html
    assert '<span class="tool-name">Thumbnail</span>' in html and ">Packaging<" not in html


def test_tags_box_shows_every_tag_with_counts(tmp_path: Path) -> None:
    ui = seeded(tmp_path, 40)
    vid = next(i for i, v in ui.videos.items() if len(v.tags) >= 2)
    tags = ui.videos[vid].tags
    html = ui.client.get(f"/videos/{vid}").text
    box = re.search(r'<textarea id="f-tags" name="tags" rows="(\d+)"[^>]*>([^<]*)</textarea>', html)
    assert box is not None and int(box.group(1)) >= 2
    assert box.group(2) == ", ".join(tags)
    hint = re.search(r'<p class="hint" id="c-tags">(\d+) tags &middot; (\d+) of 500 characters</p>', html)
    assert hint is not None and int(hint.group(1)) == len(tags)
    assert int(hint.group(2)) == sum(len(t) + (2 if " " in t else 0) for t in tags) + len(tags) - 1


def test_video_page_puts_the_edit_form_first_and_folds_the_tools(tmp_path: Path) -> None:
    ui = seeded(tmp_path, 40)
    vid = next(i for i, card in ui.cards.items() if card.failed)
    html = ui.client.get(f"/videos/{vid}").text
    marks = ['id="video-head"', 'id="what-to-fix"', 'id="edit"', 'id="h-tools"', 'id="checks"']
    assert [html.index(m) for m in marks] == sorted(html.index(m) for m in marks)
    tids = ["tool-suggest", "tool-translations", "tool-transcript", "tool-subtitles", "tool-packaging", "tool-shorts", "tool-adcheck", "tool-comments"]
    last = -1
    for tid in tids:
        needle = f'<details class="tool" id="{tid}" data-keep>'
        assert needle in html
        pos = html.index(needle)
        assert pos > last
        last = pos
    for inner in ["suggest", "translations", "transcript", "subtitles", "packaging", "shorts", "adcheck", "comments"]:
        assert f'id="{inner}"' in html
    assert '<details class="all-checks" id="checks" data-keep>' in html


def test_what_to_fix_lists_the_failed_checks_with_points(tmp_path: Path) -> None:
    ui = seeded(tmp_path, 40)
    vid = next(i for i, card in ui.cards.items() if card.failed)
    card = ui.cards[vid]
    html = ui.client.get(f"/videos/{vid}").text
    part = html[html.index('id="what-to-fix"'):html.index('id="edit"')]
    for c in card.failed:
        assert f'href="#check-{c.check_id}"' in part
        assert f"+{c.max_points - c.points}" in part
    assert f"+{card.missing} points possible." in part
    passing = [vid2 for vid2, card2 in ui.cards.items() if not card2.failed]
    if passing:
        html2 = ui.client.get(f"/videos/{passing[0]}").text
        assert "Every check passes. Nothing to fix." in html2


def test_csp_allows_only_the_thumbnail_host_for_images(tmp_path: Path) -> None:
    ui = seeded(tmp_path)
    for path in ("/", "/videos", "/videos/vid000001", "/setup"):
        csp = ui.client.get(path).headers["content-security-policy"]
        assert "img-src 'self' https://i.ytimg.com;" in csp
        assert "script-src 'self';" in csp and "style-src 'self';" in csp


def test_what_to_fix_links_each_fix_to_where_it_is_made(tmp_path: Path) -> None:
    ui = seeded(tmp_path, 40)
    vid = next(i for i, card in ui.cards.items() if any(c.check_id == "localizations" for c in card.failed))
    html = ui.client.get(f"/videos/{vid}").text
    part = html[html.index('id="what-to-fix"'):html.index('id="edit"')]
    assert 'class="vfix-go" href="#tool-translations">Translations</a>' in part
    failed = ui.cards[vid].failed
    if len(failed) > 4:
        assert f"Show {len(failed) - 4} more" in part


def test_quick_info_tiles_come_from_the_stored_pull(tmp_path: Path) -> None:
    raw = {"snippet": {"categoryId": "26"}, "contentDetails": {"caption": "true", "definition": "hd"}}
    video = make_video(8, duration="PT1H2M3S", default_language="en", raw=raw,
                       localizations={"en": {}, "fr": {}, "de": {}},
                       description="0:00 Intro\n1:00 Soil\n2:00 Pot\n#cactus #garden")
    ui = seeded(tmp_path, videos=[video])
    html = ui.client.get("/videos/vid000008").text
    tiles = dict(re.findall(r'id="fact-([a-z]+)"[^>]*><span class="vfact-k">[^<]+</span>'
                            r'<span class="vfact-v">([^<]*)</span>', html))
    assert tiles == {"views": tiles["views"], "likes": "8", "comments": "3", "duration": "1:02:03",
                     "category": "Howto &amp; Style", "language": "English", "chapters": "3",
                     "hashtags": "2", "captions": "Yes", "translations": "2", "quality": "HD", "pinned": "Not fetched"}


def test_quick_info_marks_tiles_whose_check_fails(tmp_path: Path) -> None:
    ui = seeded(tmp_path, videos=[make_video(9, description="No tags here.")])
    html = ui.client.get("/videos/vid000009").text
    assert 'id="fact-hashtags" data-tone="warn"' in html
    assert 'id="fact-translations" data-tone="warn"' in html
    assert 'id="fact-language" data-tone="warn"' in html
    assert 'id="fact-views" data-tone' not in html
    for key in ("category", "captions", "quality"):
        assert re.search(rf'id="fact-{key}"><span class="vfact-k">[^<]+</span><span class="vfact-v">Not known</span>', html)


def test_the_video_id_sits_beside_the_date_with_a_copy_button(tmp_path: Path) -> None:
    ui = seeded(tmp_path, videos=[make_video(7, published_at="2026-10-30T10:00:00Z")])
    html = ui.client.get("/videos/vid000007").text
    meta = html.split('class="muted vmeta"')[1].split("</p>")[0]
    assert meta.index("Published Oct 30, 2026") < meta.index('<code id="video-id">vid000007</code>')
    assert 'data-copy-target="video-id" aria-label="Copy the video ID"' in meta


def test_quick_info_tiles_link_to_where_they_are_fixed(tmp_path: Path) -> None:
    ui = seeded(tmp_path, videos=[make_video(9, description="No tags here.")])
    html = ui.client.get("/videos/vid000009").text
    facts = html[html.index('id="video-facts"'):html.index("</ul>", html.index('id="video-facts"'))]
    tiles = dict(re.findall(r'<li class="vfact[^"]*" id="fact-([a-z]+)"[^>]*>(.*?)</li>', facts))
    links = {"category": "#category", "pinned": "#pinned-comment", "chapters": "#tool-transcript",
             "hashtags": "#f-description", "captions": "#tool-subtitles", "translations": "#tool-translations"}
    for key, target in links.items():
        assert f'<a class="vfact-go" href="{target}"' in tiles[key], key
        assert f'id="fact-{key}"' in facts and f'class="vfact vfact-linked" id="fact-{key}"' in facts
    for key in ("views", "likes", "comments", "duration", "language", "quality"):
        assert "vfact-go" not in tiles[key] and "vfact-fix" not in tiles[key], key
    assert '<span class="vfact-fix">Fix</span>' in tiles["hashtags"]  # no hashtags costs points
    assert "vfact-fix" not in tiles["captions"]  # captions are not a scored check
    for key in links:
        if 'id="fact-' + key + '" data-tone="warn"' in facts:
            assert "vfact-fix" in tiles[key], key
    for target in links.values():
        assert f'id="{target[1:]}"' in html, target  # every link lands on something on the page
