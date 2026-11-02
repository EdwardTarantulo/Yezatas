"""The Today page (/): the four cards, the SEO overview at /seo, the sidebar, and the Analytics refresh.
The Analytics reader and the sign-in are fakes; nothing touches the network."""

import re
from pathlib import Path
from typing import Any

from test_channelstats import FakeReader, daily_answer
from webhelpers import http_error, make_ui, make_video, seed_catalog, write_token
from yezatas import auth, channelstats
from yezatas.store import Comment, Store


def make_today_ui(tmp_path: Path, *, connected: bool = True, reader: Any = None, seed: bool = True):
    r = reader or FakeReader({("day", ""): daily_answer([])})
    ui = make_ui(tmp_path,
                 analytics_credentials_loader=lambda cfg: object() if connected else None,
                 channel_reader=lambda creds: r,
                 analytics_sign_in=lambda cfg: None)
    if connected:
        write_token(ui.config, [auth.READONLY, auth.ANALYTICS])
    if seed:
        seed_catalog(ui.config, [make_video(n) for n in range(1, 9)])
    return ui, r


def section(html: str, section_id: str) -> str:
    return html.split(f'id="{section_id}"')[1].split("</section>")[0]


def test_home_is_today_with_its_cards(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path)
    html = ui.client.get("/").text
    assert '<h1 class="page-title">Today</h1>' in html
    for card_id in ("today-latest", "today-comments", "today-week", "today-fix"):
        assert f'id="{card_id}"' in html, card_id
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html) == ["/"]
    assert 'action="/setup/pull"' in html and 'name="back" value="/"' in html and ">Pull again<" in html
    assert " style=" not in html and "<style" not in html


def test_fix_card_lists_up_to_five_videos(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path)
    card = section(ui.client.get("/").text, "today-fix")
    assert ">Videos to fix<" in card and 'href="/videos?sort=fix"' in card
    rows = re.findall(r'class="td-vrow" href="(/videos/[^"]+)"', card)
    assert len(rows) == 5 and len(set(rows)) == 5
    assert card.count("points possible") == 5


def test_seo_overview_is_the_old_dashboard(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path)
    html = ui.client.get("/seo").text
    assert '<h1 class="page-title">SEO overview</h1>' in html and 'id="card-seo"' in html
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html) == ["/seo"]


def test_sidebar_groups_daily_pages_first(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path)
    html = ui.client.get("/").text
    items = re.findall(r'<a class="nav-item[^"]*" href="([^"]+)"', html)
    assert items == ["/", "/comments", "/videos", "/analytics",
                     "/seo", "/checks", "/bulk", "/copilot", "/translations", "/links", "/affiliate", "/replies",
                     "/outliers", "/search-gaps", "/competitors", "/strategy",
                     "/changes", "/history", "/agent", "/setup"]
    labels = re.findall(r'<p class="nav-label">([^<]+)</p>', html)
    assert labels == ["Daily", "Improve", "Research", "Review"]


def test_latest_videos_newest_public_first_with_views_since_the_last_pull(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path)
    newer = [make_video(n) for n in range(1, 9)]
    newer[7] = make_video(8, view_count=make_video(8).view_count + 250)
    seed_catalog(ui.config, newer)
    latest = section(ui.client.get("/").text, "today-latest")
    assert re.findall(r'class="td-vrow" href="/videos/([^"]+)"', latest) == ["vid000008", "vid000005", "vid000004"]
    assert '<span class="td-gain-n">+250</span>' in latest
    assert '<span class="td-gain-n">+0</span>' in latest


def test_comments_card_counts_questions_without_a_reply(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path)
    with Store(ui.config.catalog_db) as store:
        store.upsert_comments([
            Comment("cmt1", "vid000001", author="@mila", text="Which soil mix?", published_at="2026-06-01T10:00:00Z"),
            Comment("cmt2", "vid000001", author="@ben", text="Pot size?", published_at="2026-06-02T10:00:00Z"),
            Comment("cmt3", "vid000001", author="@joe", text="Great video", published_at="2026-06-03T10:00:00Z"),
        ])
        store.set_comment_categories({"cmt1": "question", "cmt2": "question", "cmt3": "praise"})
        store.add_comment_reply("vid000001", "cmt1", "A 2:1 mix of grit and soil.", "m", [])
    card = section(ui.client.get("/").text, "today-comments")
    assert '<span class="td-count">2</span>' in card
    assert "2 questions without a reply" in card
    assert card.index("Pot size?") < card.index("Which soil mix?")  # newest first
    assert "Great video" not in card
    assert card.count("Draft ready") == 1 and card.count("No draft") == 1
    assert 'href="/comments?c=question"' in card


def test_week_card_asks_to_connect_without_the_analytics_sign_in(tmp_path: Path) -> None:
    ui, reader = make_today_ui(tmp_path, connected=False)
    card = section(ui.client.get("/").text, "today-week")
    assert 'id="today-week-connect"' in card and ">Connect Analytics<" in card
    assert 'action="/analytics/refresh"' not in card
    html = ui.post("/analytics/refresh", data={"back": "/"}).text
    assert "needs the YouTube Analytics sign-in" in html
    assert reader.calls == []


def test_refresh_reads_the_week_and_shows_it(tmp_path: Path) -> None:
    ui, reader = make_today_ui(tmp_path)
    card = section(ui.client.get("/").text, "today-week")
    assert 'id="today-week-empty"' in card and 'action="/analytics/refresh"' in card
    resp = ui.post("/analytics/refresh", data={"back": "/"}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/"
    ui.web.channel_stats_job.wait()
    assert channelstats.load_stats(ui.config.home) is not None
    card = section(ui.client.get("/").text, "today-week")
    assert "data up to" in card and ">Views<" in card and ">Watch time<" in card and ">Subscribers<" in card
    assert card.count('class="h-') == 7
    assert reader.calls[0][3] == "day" and reader.calls[1][3] == "insightTrafficSourceType"


def test_refresh_goes_back_only_to_known_pages(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path)
    resp = ui.post("/analytics/refresh", data={"back": "https://example.com/"}, follow_redirects=False)
    assert resp.headers["location"] == "/"
    ui.web.channel_stats_job.wait()


def test_refresh_says_when_the_analytics_api_is_off(tmp_path: Path) -> None:
    class Off:
        calls: list = []

        def query(self, *args: Any) -> dict:
            raise http_error(403, "accessNotConfigured", "YouTube Analytics API has not been used in project 1 before")

    ui, _ = make_today_ui(tmp_path, reader=Off())
    ui.post("/analytics/refresh", data={"back": "/"})
    ui.web.channel_stats_job.wait()
    card = section(ui.client.get("/").text, "today-week")
    assert 'id="today-week-failed"' in card and "YouTube Analytics API is not enabled" in card
    ui.post("/analytics/done", data={"back": "/"})
    assert 'id="today-week-failed"' not in section(ui.client.get("/").text, "today-week")


def test_connect_needs_the_client_file(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path, connected=False)
    html = ui.post("/analytics/connect", data={"back": "/"}).text
    assert "Add the OAuth client file first" in html


def test_status_reports_the_channel_stats_job(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path)
    assert ui.client.get("/api/status").json()["channel_stats"]["status"] == "idle"
