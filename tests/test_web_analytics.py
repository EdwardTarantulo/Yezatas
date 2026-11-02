"""The Analytics page (/analytics): the shared week card, countries, traffic sources and the
impressions note. The Analytics reader is a fake; nothing touches the network."""

import re
from pathlib import Path

from test_web_today import make_today_ui, section
from yezatas.channelstats import ChannelStats, Day, country_name, save_stats, source_label
from yezatas.web.analytics_view import ShareRow, build_analytics, share_rows


def saved_stats(home: Path) -> ChannelStats:
    days = tuple(Day(f"2026-06-{d:02d}", 10 * d, 60, 2, 1) for d in range(4, 18))
    stats = ChannelStats("2026-06-20T12:00:00+00:00", "2026-06-17", days,
                         (("YT_SEARCH", 4100), ("RELATED_VIDEO", 3300), ("EXT_URL", 600)), (), 3,
                         (("US", 5000), ("DE", 2500), ("GB", 2500)))
    save_stats(home, stats)
    return stats


def test_share_rows_percent_of_all_and_bars_against_the_biggest() -> None:
    rows = share_rows((("US", 5000), ("DE", 2500), ("GB", 2500)), country_name, top=2)
    assert rows == (ShareRow("United States", "5K", 50, "w-100"), ShareRow("Germany", "2.5K", 25, "w-50"))
    assert share_rows((), source_label) == ()
    empty = build_analytics(None)
    assert (empty.week, empty.countries, empty.sources) == (None, (), ())


def test_page_before_any_fetch(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path)
    html = ui.client.get("/analytics").text
    assert '<h1 class="page-title">Analytics</h1>' in html
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html) == ["/analytics"]
    week = section(html, "analytics-week")
    assert 'id="analytics-week-empty"' in week and 'name="back" value="/analytics"' in week
    assert 'id="analytics-countries"' not in html and 'id="analytics-sources"' not in html
    assert "doesn't share impressions and click-through rate" in section(html, "analytics-ctr")
    assert " style=" not in html and "<style" not in html


def test_page_shows_the_saved_numbers(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path)
    saved_stats(ui.config.home)
    html = ui.client.get("/analytics").text
    week = section(html, "analytics-week")
    assert ">Views<" in week and "data up to Jun 17, 2026" in week
    assert "Where views came from (28 days)" not in week  # the sources card below shows them all
    countries = section(html, "analytics-countries")
    assert re.findall(r'class="an-label">([^<]+)<', countries) == ["United States", "Germany", "United Kingdom"]
    assert '<span class="an-num">50%</span>' in countries
    sources = section(html, "analytics-sources")
    assert re.findall(r'class="an-label">([^<]+)<', sources) == ["YouTube search", "Suggested videos", "External sites"]
    today = section(ui.client.get("/").text, "today-week")
    assert "Where views came from (28 days)" in today


def test_refresh_from_the_page_comes_back_to_it(tmp_path: Path) -> None:
    ui, reader = make_today_ui(tmp_path)
    resp = ui.post("/analytics/refresh", data={"back": "/analytics"}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/analytics"
    ui.web.channel_stats_job.wait()
    assert [c[3] for c in reader.calls[:3]] == ["day", "insightTrafficSourceType", "country"]


def test_not_connected_offers_the_sign_in(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path, connected=False)
    week = section(ui.client.get("/analytics").text, "analytics-week")
    assert 'id="analytics-week-connect"' in week and ">Connect Analytics<" in week
