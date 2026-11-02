"""The Analytics card of the video page: sign-in state, the refresh job filtered to the video, the numbers
from code. The Analytics reader is a fake; nothing touches the network."""

import re
from pathlib import Path

from test_web_today import make_today_ui, section
from yezatas.channelstats import Day
from yezatas.videostats import VideoStats, load_video_stats
from yezatas.web.video_analytics_view import build_video_analytics

VID = "vid000001"  # make_video(1): 5:30 long


class VideoReader:
    """Answers the three per-video queries; the daily row is the newest day asked for."""

    def __init__(self) -> None:
        self.calls = []

    def query(self, start_date, end_date, metrics, dimensions, filters, sort):
        self.calls.append((dimensions, filters))
        if dimensions == "day":
            names = ["day", "views", "estimatedMinutesWatched", "subscribersGained", "subscribersLost"]
            return {"columnHeaders": [{"name": n} for n in names], "rows": [[end_date, 120, 330, 3, 1]]}
        rows = {"insightTrafficSourceType": [["YT_SEARCH", 90], ["RELATED_VIDEO", 30]],
                "country": [["US", 100], ["DE", 20]]}.get(dimensions, [])
        return {"columnHeaders": [{"name": dimensions}, {"name": "views"}], "rows": rows}


def card(ui) -> str:
    return section(ui.client.get(f"/videos/{VID}").text, "video-analytics")


def test_not_connected_points_to_the_analytics_page(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path, connected=False)
    part = card(ui)
    assert 'id="video-analytics-connect"' in part and 'href="/analytics"' in part
    assert "/analytics/refresh" not in part
    assert "click-through rate are not in YouTube's API" in part


def test_connected_but_not_read_yet(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path)
    part = card(ui)
    assert 'id="video-analytics-empty"' in part
    assert f'action="/videos/{VID}/analytics/refresh"' in part and ">Refresh</button>" in part


def test_refresh_reads_three_queries_for_this_video_and_shows_them(tmp_path: Path) -> None:
    reader = VideoReader()
    ui, _ = make_today_ui(tmp_path, reader=reader)
    resp = ui.post(f"/videos/{VID}/analytics/refresh", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{VID}#video-analytics"
    ui.web.video_stats_job.wait()
    assert reader.calls == [("day", f"video=={VID}"), ("insightTrafficSourceType", f"video=={VID}"),
                            ("country", f"video=={VID}")]
    assert load_video_stats(ui.config.home, VID) is not None
    part = card(ui)
    values = re.findall(r'<span class="va-stat-k">([^<]+)</span><span class="va-stat-v">([^<]+)</span>', part)
    assert values == [("Views", "120"), ("Watch time", "5.5 h"), ("Average view", "2:45"), ("Subscribers", "+2")]
    assert "50% of the video" in part  # 165 s of a 330 s video
    assert part.count('<span class="h-') == 28 and part.count('class="h-100"') == 1
    assert re.findall(r'class="an-label">([^<]+)<', part) == [
        "YouTube search", "Suggested videos", "United States", "Germany"]
    assert '<span class="an-num">75%</span>' in part and "Read " in part
    assert " style=" not in part


def test_refresh_for_an_unknown_video_goes_back_to_the_list(tmp_path: Path) -> None:
    reader = VideoReader()
    ui, _ = make_today_ui(tmp_path, reader=reader)
    resp = ui.post("/videos/nope/analytics/refresh", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/videos"
    assert reader.calls == []


def test_the_view_without_views_says_so() -> None:
    days = tuple(Day(f"2026-06-{d:02d}", 0, 0, 0, 0) for d in range(1, 29))
    view = build_video_analytics(VideoStats(VID, "2026-06-30T12:00:00+00:00", "2026-06-28", days, (), (), 3), 330)
    assert view is not None and view.stats[2].value == "No views" and view.stats[2].note == ""
    assert set(view.bars) == {"h-0"} and (view.first_day, view.last_day) == ("Jun 1", "Jun 28")
    assert view.up_to == "Jun 28, 2026" and view.sources == () and view.countries == ()
    assert build_video_analytics(None, 330) is None


def stored(ui, hours_ago: float) -> None:
    from datetime import datetime, timedelta, timezone

    from yezatas.videostats import save_video_stats
    when = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    days = tuple(Day(f"2026-06-{d:02d}", 1, 1, 0, 0) for d in range(1, 29))
    save_video_stats(ui.config.home, VideoStats(VID, when, "2026-06-28", days, (), (), 3))


def test_the_card_asks_to_read_itself_when_nothing_is_stored(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path)
    part = card(ui)
    assert re.search(rf'<form class="va-refresh" method="post" action="/videos/{VID}/analytics/refresh" '
                     r'data-auto-refresh>', part)
    assert '<input type="hidden" name="auto" value="1">' in part
    assert '<button type="submit" class="link-button">Refresh</button>' in part


def test_a_fresh_read_is_not_repeated_and_an_old_one_is(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path)
    stored(ui, 1)
    fresh = card(ui)
    assert "data-auto-refresh" not in fresh and 'name="auto"' not in fresh and 'id="video-analytics-when"' in fresh
    stored(ui, 13)
    assert "data-auto-refresh" in card(ui)


def test_not_connected_never_reads_itself(tmp_path: Path) -> None:
    ui, _ = make_today_ui(tmp_path, connected=False)
    assert "data-auto-refresh" not in card(ui)


def test_the_read_on_opening_comes_back_without_the_jump(tmp_path: Path) -> None:
    reader = VideoReader()
    ui, _ = make_today_ui(tmp_path, reader=reader)
    resp = ui.post(f"/videos/{VID}/analytics/refresh", data={"auto": "1"}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{VID}"
    ui.web.video_stats_job.wait()
    assert len(reader.calls) == 3
    assert "data-auto-refresh" not in card(ui)  # just read: not due again


def test_the_page_script_submits_a_due_form_once() -> None:
    js = (Path(__file__).resolve().parents[1] / "src" / "yezatas" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'document.querySelector("form[data-auto-refresh]")' in js and "autoForm.submit()" in js
