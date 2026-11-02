"""One video's Analytics numbers: the three filtered queries, the sums, save and load. The reader is faked."""

from datetime import date, datetime, timezone

import pytest

from yezatas.videostats import (VIDEO_DAYS, VideoStatsError, fetch_video_stats, load_video_stats,
                                save_video_stats, stats_path)

NOW = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)  # newest day read: 2026-06-17
VID = "abcDEF12345"


class FakeReader:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def query(self, start_date, end_date, metrics, dimensions, filters, sort):
        self.calls.append((start_date, end_date, metrics, dimensions, filters, sort))
        return self.answers.get(dimensions, {})


def answer(names, rows):
    return {"columnHeaders": [{"name": n} for n in names], "rows": rows}


def reader():
    return FakeReader({
        "day": answer(["day", "views", "estimatedMinutesWatched", "subscribersGained", "subscribersLost"],
                      [["2026-06-17", 100, 300, 5, 1], ["2026-06-01", 20, 30, 0, 2], ["junk", 9, 9, 9, 9]]),
        "insightTrafficSourceType": answer(["insightTrafficSourceType", "views"],
                                           [["RELATED_VIDEO", 70], ["YT_SEARCH", 50], ["EXT_URL", 0]]),
        "country": answer(["country", "views"], [["us", 60], ["DE", 40]]),
    })


def test_three_queries_filtered_to_the_video_over_28_days():
    r = reader()
    stats = fetch_video_stats(r, VID, now=NOW)
    assert len(r.calls) == 3 and stats.calls == 3
    assert {c[4] for c in r.calls} == {f"video=={VID}"}
    assert {(c[0], c[1]) for c in r.calls} == {("2026-05-21", "2026-06-17")}
    assert [c[3] for c in r.calls] == ["day", "insightTrafficSourceType", "country"]
    assert len(stats.days) == VIDEO_DAYS and stats.days[0].date == "2026-05-21"
    assert stats.days[-1].views == 100 and stats.end_date == "2026-06-17"


def test_sums_and_the_average_come_from_code():
    stats = fetch_video_stats(reader(), VID, now=NOW)
    assert stats.views == 120 and stats.minutes == 330 and stats.subscribers == 2
    assert stats.average_seconds == 165  # 330 minutes * 60 / 120 views
    assert stats.sources == (("RELATED_VIDEO", 70), ("YT_SEARCH", 50))
    assert stats.countries == (("US", 60), ("DE", 40))


def test_no_views_means_no_average():
    stats = fetch_video_stats(FakeReader({}), VID, now=NOW)
    assert stats.views == 0 and stats.average_seconds is None
    assert stats.sources == () and stats.countries == ()


def test_save_and_load_round_trip(tmp_path):
    stats = fetch_video_stats(reader(), VID, now=NOW)
    path = save_video_stats(tmp_path, stats)
    assert path == tmp_path / "analytics" / "videos" / f"{VID}.json"
    assert load_video_stats(tmp_path, VID) == stats
    assert load_video_stats(tmp_path, "otherVideo1") is None


def test_load_refuses_broken_or_foreign_files(tmp_path):
    stats = fetch_video_stats(reader(), VID, now=NOW)
    path = save_video_stats(tmp_path, stats)
    for text in ("not json", "[]", '{"video_id": "someoneElse"}',
                 path.read_text(encoding="utf-8").replace('"calls": 3', '"calls": true')):
        path.write_text(text, encoding="utf-8")
        assert load_video_stats(tmp_path, VID) is None


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "a b", "x" * 65])
def test_unsafe_video_ids_are_refused(tmp_path, bad):
    with pytest.raises(VideoStatsError):
        stats_path(tmp_path, bad)
    with pytest.raises(VideoStatsError):
        fetch_video_stats(reader(), bad, now=NOW)
    assert load_video_stats(tmp_path, bad) is None
