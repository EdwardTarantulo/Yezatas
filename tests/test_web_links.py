"""The Links page: the inventory table, filters, paging, escaping, the Replace-everywhere link into Bulk edit,
the check job and its summary line, and the nav item. The visit function is a fake; nothing touches the
network, and nothing here ever calls YouTube."""

from __future__ import annotations

import html as htmllib
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from webhelpers import Ui, make_push_ui, make_ui, make_video, seed_catalog
from yezatas import links as links_mod
from yezatas.links import LinkOutcome, outcome_for_status
from yezatas.store import Store
from yezatas.web import app as web_app
from yezatas.web.views import (LINKS_PAGE_SIZE, build_links_view, links_check_line, shorten_link)

NOW = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
SHOP = "https://shop.example.invalid/cactus?ref=a&x=1"
DOCS = "https://docs.example.invalid/guide"
GONE = "https://gone.example.invalid/page"
MOVED = "https://moved.example.invalid/old"
BLOCKS = "https://blocks.example.invalid/"
NEVER = "https://never.example.invalid/"
YT_OTHER = "https://www.youtube.com/@someone"


def vid(n: int) -> str:
    return f"vid{n:08d}"


def catalog() -> list[Any]:
    """Four public videos, a private one and an unlisted one; made-up links."""
    return [
        make_video(1, video_id=vid(1), title="Cactus basics", privacy_status="public", view_count=5000,
                   description=f"Shop {SHOP} and docs {DOCS}\nMore: {GONE}"),
        make_video(2, video_id=vid(2), title="Succulent <care> & more", privacy_status="public",
                   view_count=1200, description=f"Docs {DOCS}. Moved {MOVED} Blocks {BLOCKS}"),
        make_video(3, video_id=vid(3), title="Watering tips", privacy_status="public",
                   view_count=300, description=f"Never {NEVER} {YT_OTHER}"),
        make_video(4, video_id=vid(4), title="Hidden draft", privacy_status="private",
                   view_count=0, description="Nothing to link here."),
        make_video(5, video_id=vid(5), title="Series part two", privacy_status="public",
                   view_count=90, description=(f"Part one https://youtu.be/{vid(4)}?t=5 "
                                               f"and a stranger https://youtu.be/vid99999999 "
                                               f"and https://youtu.be/{vid(1)}")),
    ]


def make_links_ui(tmp_path: Path, fetch: Any = None, videos: list[Any] | None = None,
                  seeded: bool = True, **kwargs: Any) -> Ui:
    ui = make_ui(tmp_path, link_fetch=fetch or FakeFetch(), **kwargs)
    if seeded:
        seed_catalog(ui.config, videos if videos is not None else catalog())
    return ui


class FakeFetch:
    """A made-up web: urls not in `table` answer 200. Records every url it was asked for."""

    def __init__(self, table: dict[str, Any] | None = None, gate: threading.Event | None = None) -> None:
        self.table = table if table is not None else {}
        self.gate = gate
        self.entered = threading.Event()
        self.urls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, url: str) -> LinkOutcome:
        with self._lock:
            self.urls.append(url)
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10), "the test never opened the gate"
        answer = self.table.get(url, LinkOutcome("ok", 200))
        if isinstance(answer, BaseException):
            raise answer
        return answer


MIXED = {
    GONE: outcome_for_status(GONE, 404),
    MOVED: LinkOutcome("redirected", 200, "https://new.example.invalid/x", "Moves to https://new.example.invalid/x"),
    BLOCKS: outcome_for_status(BLOCKS, 403),
}


def save_check(ui: Ui, url: str, status: str, http: int | None = None, final: str = "",
               message: str = "") -> None:
    with Store(ui.config.catalog_db) as store:
        store.save_link_check(url, status, http, final, message, now=NOW)


def page(ui: Ui, query: str = "") -> str:
    resp = ui.client.get("/links" + query)
    assert resp.status_code == 200
    return resp.text


def run_check(ui: Ui) -> Any:
    resp = ui.post("/links/check", follow_redirects=False)
    ui.web.links_job.wait()
    return resp


def table_rows(html: str) -> list[str]:
    if 'id="links-table"' not in html:
        return []
    return html.split("<tbody>")[1].split("</tbody>")[0].split("<tr>")[1:]


def row_urls(html: str) -> list[str]:
    return [htmllib.unescape(m.group(1)) for m in
            (re.search(r'class="lk-url" href="[^"]*" title="([^"]*)"', r) for r in table_rows(html)) if m]


def line(html: str) -> str:
    m = re.search(r'id="links-line">([^<]*)<', html)
    assert m
    return htmllib.unescape(m.group(1))


# --- no pull, empty catalog ----------------------------------------------------------------------
def test_without_a_pull_everything_goes_to_setup(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path, seeded=False)
    resp = ui.client.get("/links", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/setup"
    resp = ui.post("/links/check", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/setup"
    assert ui.web.links_job.snapshot().status == "idle"


def test_a_catalog_without_links_says_so_and_cannot_be_checked(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path, videos=[make_video(1, video_id=vid(1), description="No links.")])
    html = page(ui)
    assert line(html) == "No links in your descriptions."
    assert 'id="no-links"' in html and "links-table" not in html
    assert re.search(r'<button type="submit" class="button primary" disabled>Check links</button>', html)


# --- the filled page -----------------------------------------------------------------------------
def test_the_table_is_ranked_by_reach_with_the_summary_line(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path)
    html = page(ui)
    # reach: DOCS 5000+1200, SHOP 5000, GONE 5000, MOVED 1200, BLOCKS 1200, NEVER 300, YT_OTHER 300, own video links
    urls = row_urls(html)
    assert urls[0] == DOCS
    assert urls[1:3] == sorted([SHOP, GONE])
    assert urls.index(MOVED) < urls.index(NEVER)
    assert line(html) == ("10 links in 4 videos; 0 broken, 0 redirected, 2 to your private or unknown "
                          "videos, 6 not checked.")
    assert "Yezatas visits each external link once from your computer (no YouTube quota)" in html
    assert "YouTube links are checked against your catalog." in html


def test_link_cells_open_in_a_new_tab_and_carry_the_full_url(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path)
    html = page(ui)
    row = next(r for r in table_rows(html) if "shop.example.invalid" in r)
    assert 'target="_blank" rel="noopener noreferrer"' in row
    assert f'title="{htmllib.escape(SHOP)}"' in row
    assert f'href="{htmllib.escape(SHOP)}"' in row
    assert ">shop.example.invalid/cactus?ref=a&amp;x=1<" in row  # no scheme in the text
    assert '<span class="lk-domain muted">shop.example.invalid</span>' in row
    assert "<summary>1 video</summary>" in row
    docs = next(r for r in table_rows(html) if "docs.example.invalid" in r)
    assert "<summary>2 videos</summary>" in docs
    assert docs.index("Cactus basics") < docs.index("Succulent")  # most viewed first
    assert f'<a href="/videos/{vid(1)}">Cactus basics</a>' in docs
    assert 'title="6,200 views in total"' in docs and ">6.2K<" in docs


def test_a_www_link_without_scheme_still_opens(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path, videos=[make_video(1, video_id=vid(1), description="See www.plain.example.invalid/a")])
    row = table_rows(page(ui))[0]
    assert 'href="https://www.plain.example.invalid/a"' in row and ">plain.example.invalid/a<" in row


def test_long_links_are_shortened_but_keep_the_full_text_in_the_title(tmp_path: Path) -> None:
    long = "https://long.example.invalid/" + "a" * 120
    ui = make_links_ui(tmp_path, videos=[make_video(1, video_id=vid(1), description=long)])
    row = table_rows(page(ui))[0]
    shown = re.search(r'rel="noopener noreferrer">([^<]*)<', row)
    assert shown and len(shown.group(1)) == 60 and shown.group(1).endswith("…")
    assert f'title="{long}"' in row
    assert shorten_link("https://www.a.example.invalid/") == "a.example.invalid/"


def test_own_video_links_show_the_catalog_findings(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path)
    html = page(ui)
    private = next(r for r in table_rows(html) if f"youtu.be/{vid(4)}" in r)
    private = htmllib.unescape(private)
    assert "Your private video" in private and 'Links to your private video "Hidden draft".' in private
    stranger = next(r for r in table_rows(html) if "vid99999999" in r)
    assert "Not in your catalog" in stranger
    assert "Not one of your videos, or deleted: Yezatas can't tell." in htmllib.unescape(stranger)
    own = next(r for r in table_rows(html) if f"youtu.be/{vid(1)}" in r)
    assert "Your public video" in own
    other = next(r for r in table_rows(html) if "@someone" in r)
    assert "Not visited" in other and "YouTube links are checked against your catalog." in other


def test_check_statuses_show_with_their_sentence_and_time(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path)
    save_check(ui, GONE, "broken", 404, "", "Not found (404).")
    save_check(ui, MOVED, "redirected", 200, "https://new.example.invalid/x", "Moves to https://new.example.invalid/x")
    save_check(ui, BLOCKS, "error", 403, "", "The site answered 403; it may block checks. Open it yourself.")
    save_check(ui, DOCS, "ok", 200)
    html = page(ui)
    when = web_app._local_time(NOW.isoformat())
    gone = next(r for r in table_rows(html) if "gone.example.invalid" in r)
    assert '<span class="pill problem">Broken</span>' in gone and "Not found (404)." in gone
    assert f"Checked {when}" in gone
    assert 'class="pill warn">Moves<' in next(r for r in table_rows(html) if "moved.example.invalid" in r)
    assert "Moves to https://new.example.invalid/x" in html
    assert "Could not be checked" in next(r for r in table_rows(html) if "blocks.example.invalid" in r)
    assert 'class="pill done">Working<' in next(r for r in table_rows(html) if "docs.example.invalid/guide" in r)
    assert "1 broken, 1 redirected, 1 could not be checked, 2 to your private or unknown videos, 2 not checked." in line(html)


# --- filters and paging ------------------------------------------------------------------------------
def test_filters_show_the_right_links_and_counts(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path)
    save_check(ui, GONE, "broken", 404, "", "Not found (404).")
    save_check(ui, MOVED, "redirected", 200, "https://new.example.invalid/x", "Moves to x")
    save_check(ui, BLOCKS, "error", 403, "", "Open it yourself.")
    save_check(ui, DOCS, "ok", 200)
    assert row_urls(page(ui, "?show=broken")) == [GONE]
    assert row_urls(page(ui, "?show=redirected")) == [MOVED]
    assert row_urls(page(ui, "?show=trouble")) == [BLOCKS]
    assert sorted(row_urls(page(ui, "?show=unchecked"))) == sorted([SHOP, NEVER])
    own = row_urls(page(ui, "?show=own"))
    assert sorted(own) == sorted([f"https://youtu.be/{vid(4)}?t=5", "https://youtu.be/vid99999999"])
    html = page(ui, "?show=broken")
    chips = re.findall(r'<a class="chip( active)?" href="([^"]+)"[^>]*>([^<]+)</a>', html)
    assert [(a.strip(), h, t) for a, h, t in chips] == [
        ("", "/links", "All (10)"), ("active", "/links?show=broken", "Broken (1)"),
        ("", "/links?show=redirected", "Redirected (1)"),
        ("", "/links?show=trouble", "Could not be checked (1)"),
        ("", "/links?show=own", "Own private or unknown videos (2)"),
        ("", "/links?show=unchecked", "Not checked (2)")]
    assert '<span class="result-count"' not in html and ">1 link<" in html


def test_an_unknown_filter_or_page_falls_back(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path)
    assert len(row_urls(page(ui, "?show=nonsense&page=abc"))) == 10
    assert len(row_urls(page(ui, "?page=-4"))) == 10
    html = page(ui, "?show=broken")
    assert 'id="no-links"' in html and "No links match" in html and "links-table" not in html


def test_paging_shows_at_most_200_rows_a_page(tmp_path: Path) -> None:
    assert LINKS_PAGE_SIZE == 200
    text = " ".join(f"https://l{n:03d}.example.invalid/" for n in range(205))
    ui = make_links_ui(tmp_path, videos=[make_video(1, video_id=vid(1), description=text)])
    first = page(ui)
    assert len(table_rows(first)) == 200
    assert "205 links &middot; showing 1-200" in first and "Page 1 of 2" in first
    assert 'rel="next" href="/links?page=2"' in first and 'aria-disabled="true">Previous' in first
    second = page(ui, "?page=2")
    assert len(table_rows(second)) == 5 and "showing 201-205" in second
    assert 'rel="prev" href="/links"' in second
    assert row_urls(second) == sorted(row_urls(second))  # same reach: ordered by url
    assert row_urls(page(ui, "?page=99")) == row_urls(second)  # clamped
    save_check(ui, "https://l204.example.invalid/", "broken", 404, "", "Not found (404).")
    broken = page(ui, "?show=broken&page=7")
    assert len(table_rows(broken)) == 1 and "Page" not in broken


def test_the_view_builder_never_raises_on_odd_params() -> None:
    view = build_links_view([], [], {"show": "x" * 500, "page": "9" * 40})
    assert view.rows == () and view.page == 1 and view.show == "all" and view.total == 0


# --- escaping -----------------------------------------------------------------------------------------
def test_hostile_links_and_titles_are_escaped(tmp_path: Path) -> None:
    hostile_url = 'https://evil.example.invalid/"onmouseover="alert(1)<script>alert(2)</script>'
    title = "<img src=x onerror=alert(3)>"
    ui = make_links_ui(tmp_path, videos=[make_video(
        1, video_id=vid(1), title=title, description=f"Click {hostile_url}")])
    save_check(ui, hostile_url, "error", None, "", "<b>bad</b> sentence")
    html = page(ui)
    for raw in ('"onmouseover="alert', "<script>alert(2)", "<img src=x", "<b>bad</b>"):
        assert raw not in html
    assert "&lt;img src=x onerror=alert(3)&gt;" in html
    assert "&lt;b&gt;bad&lt;/b&gt; sentence" in html
    assert row_urls(html) == [hostile_url]


# --- Replace everywhere ------------------------------------------------------------------------------
def replace_href(html: str, url: str) -> str:
    row = next(r for r in table_rows(html) if f'title="{htmllib.escape(url)}"' in r)
    return htmllib.unescape(re.search(r'<a class="button secondary" href="([^"]+)"', row).group(1))


def test_replace_everywhere_prefills_the_bulk_form_without_previewing(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path)
    href = replace_href(page(ui), SHOP)
    parts = urlsplit(href)
    assert parts.path == "/bulk"
    assert parse_qs(parts.query) == {"kind": ["replace"], "find": [SHOP], "field": ["description"],
                                     "visibility": ["all"]}
    bulk = ui.client.get(href)
    assert bulk.status_code == 200
    body = bulk.text
    assert f'id="bulk-find" name="find" value="{htmllib.escape(SHOP)}"' in body
    assert re.search(r'name="field" value="description" checked', body)
    assert not re.search(r'name="field" value="title" checked', body)
    assert re.search(r'name="visibility" value="all" checked', body)
    assert 'id="bulk-preview"' not in body  # nothing is previewed or drafted until the creator asks
    with Store(ui.config.catalog_db) as store:
        assert store.list_drafts() == []


def test_the_prefilled_link_can_be_previewed_with_a_new_link(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path)
    href = replace_href(page(ui), GONE) + "&replace_with=" + "https%3A%2F%2Fnew.example.invalid%2F&preview=1"
    body = ui.client.get(href).text
    assert 'id="bulk-preview"' in body and "Cactus basics" in body
    with Store(ui.config.catalog_db) as store:
        assert store.list_drafts() == []  # a preview only


def test_the_plain_bulk_page_is_unchanged(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path)
    body = ui.client.get("/bulk").text
    assert 'id="bulk-find" name="find" value=""' in body
    assert re.search(r'name="visibility" value="public" checked', body)


# --- the check job -----------------------------------------------------------------------------------
def test_every_post_needs_the_csrf_token(tmp_path: Path) -> None:
    fetch = FakeFetch()
    ui = make_links_ui(tmp_path, fetch)
    assert ui.client.post("/links/check", follow_redirects=False).status_code == 403
    assert ui.client.post("/links/check", data={"csrf": "wrong"}, follow_redirects=False).status_code == 403
    assert fetch.urls == [] and ui.web.links_job.snapshot().status == "idle"


def test_a_check_visits_only_external_links_saves_every_status_and_shows_a_summary(tmp_path: Path) -> None:
    fetch = FakeFetch(MIXED)
    ui = make_links_ui(tmp_path, fetch)
    resp = run_check(ui)
    assert resp.status_code == 303 and resp.headers["location"] == "/links"
    assert sorted(fetch.urls) == sorted([SHOP, DOCS, GONE, MOVED, BLOCKS, NEVER])  # no YouTube links
    with Store(ui.config.catalog_db) as store:
        saved = store.latest_link_checks()
    assert {u: c.status for u, c in saved.items()} == {
        SHOP: "ok", DOCS: "ok", GONE: "broken", MOVED: "redirected", BLOCKS: "error", NEVER: "ok",
        YT_OTHER: "skipped", f"https://youtu.be/{vid(4)}?t=5": "skipped",
        "https://youtu.be/vid99999999": "skipped", f"https://youtu.be/{vid(1)}": "skipped"}
    assert saved[GONE].http_status == 404 and saved[MOVED].final_url == "https://new.example.invalid/x"

    html = page(ui)
    summary = re.search(r'id="links-summary">([^<]*)<', html)
    assert summary and summary.group(1) == "Checked 6 links: 1 broken, 1 redirected, 1 could not be checked."
    assert "id=\"links-summary\"" not in page(ui)  # shown once
    html = page(ui, "?show=broken")
    assert row_urls(html) == [GONE] and "Not found (404)." in html
    assert "1 broken, 1 redirected, 1 could not be checked, 2 to your private or unknown videos, 0 not checked." in line(html)


def test_a_clean_check_says_all_answered(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path)
    run_check(ui)
    assert 'id="links-summary">Checked 6 links: all answered.<' in page(ui)


def test_the_page_shows_progress_while_the_job_runs_and_a_second_start_is_refused(tmp_path: Path) -> None:
    gate = threading.Event()
    fetch = FakeFetch(gate=gate)
    ui = make_links_ui(tmp_path, fetch)
    try:
        assert ui.post("/links/check", follow_redirects=False).status_code == 303
        assert fetch.entered.wait(10)
        html = page(ui)
        assert 'id="links-msg"' in html and 'data-poll="1"' in html
        assert 'action="/links/check"' not in html  # no second start button while running
        status = ui.client.get("/api/status").json()
        assert status["links"]["status"] == "running"
        ui.post("/links/check", follow_redirects=False)
        assert "A link check is already running." in page(ui)
    finally:
        gate.set()
        ui.web.links_job.wait()
    assert ui.client.get("/api/status").json()["links"]["status"] == "done"


def test_a_failing_run_shows_a_plain_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ui = make_links_ui(tmp_path)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(links_mod, "check_links", boom)
    run_check(ui)
    html = page(ui)
    assert 'id="links-failed">The link check stopped unexpectedly (RuntimeError).<' in html
    assert "boom" not in html
    assert 'id="links-failed"' not in page(ui)


def test_a_fetch_that_raises_is_a_result_not_a_failed_job(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path, FakeFetch({GONE: RuntimeError("boom")}))
    run_check(ui)
    html = page(ui)
    assert ui.web.links_job.snapshot().status == "done"
    assert "Checked 6 links: 1 could not be checked." in html


def test_the_check_line_covers_the_cut_and_the_empty_case() -> None:
    from yezatas.links import LinkCheckSummary
    cut = LinkCheckSummary({"ok": 297, "broken": 1, "skipped": 4}, 298, 4, 12)
    assert links_check_line(cut) == "Checked 298 links: 1 broken. 12 links with the least reach were left out."
    assert links_check_line(LinkCheckSummary({"ok": 1}, 1, 0, 1)) == (
        "Checked 1 link: all answered. 1 link with the least reach was left out.")
    assert links_check_line(LinkCheckSummary({"skipped": 2}, 0, 2, 0)) == "Checked 0 links."


# --- the shell ----------------------------------------------------------------------------------------
def test_the_nav_has_a_links_item_after_content(tmp_path: Path) -> None:
    ui = make_links_ui(tmp_path)
    html = page(ui)
    items = re.findall(r'<a class="nav-item[^"]*" href="([^"]+)"', html)
    assert items.index("/links") == items.index("/videos") + 1
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html) == ["/links"]
    assert re.findall(r'class="nav-item active" href="([^"]+)"', ui.client.get("/videos").text) == ["/videos"]
    assert not re.search(r"<script(?![^>]*\bsrc=)", html) and "style=" not in html


def test_nothing_on_this_page_touches_the_youtube_writer(tmp_path: Path) -> None:
    fetch = FakeFetch(MIXED)
    ui, writer = make_push_ui(tmp_path, link_fetch=fetch)
    seed_catalog(ui.config, catalog())
    page(ui)
    run_check(ui)
    page(ui, "?show=broken")
    ui.client.get(replace_href(page(ui), SHOP))
    assert fetch.urls  # the check did run
    assert writer.calls == [] and writer.updates == []
