from datetime import date, datetime, timezone

from yezatas.channelstats import (
    DAYS,
    ChannelStats,
    Day,
    SameAge,
    change,
    compare_latest,
    fetch_days,
    fetch_sources,
    fetch_stats,
    last_day,
    load_stats,
    newest_uploads,
    save_stats,
    source_label,
    source_shares,
    table,
    week_totals,
)
from yezatas.store import Video

NOW = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)
END = date(2026, 6, 17)


class FakeReader:
    """Answers by (dimensions, filters); records every query."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def query(self, start_date, end_date, metrics, dimensions, filters, sort):
        self.calls.append((start_date, end_date, metrics, dimensions, filters, sort))
        return self.answers.get((dimensions, filters), {})


def daily_answer(rows):
    names = ["day", "views", "estimatedMinutesWatched", "subscribersGained", "subscribersLost"]
    return {"columnHeaders": [{"name": n} for n in names], "rows": rows}


def views_answer(per_day):
    return {"columnHeaders": [{"name": "day"}, {"name": "views"}],
            "rows": [[d, v] for d, v in per_day.items()]}


def vid(video_id, published, duration="PT8M", privacy="public"):
    return Video(video_id=video_id, published_at=published, duration=duration, privacy_status=privacy)


def test_table_reads_named_columns_in_order_and_skips_bad_rows():
    resp = {"columnHeaders": [{"name": "views"}, {"name": "day"}], "rows": [[5, "2026-06-01"], ["x"], "bad"]}
    assert table(resp, ("day", "views")) == [("2026-06-01", 5)]
    assert table(resp, ("day", "likes")) == []
    assert table({}, ("day",)) == []


def test_last_day_is_three_days_back():
    assert last_day(NOW) == END


def test_fetch_days_fills_every_date_and_splits_the_weeks():
    reader = FakeReader({("day", ""): daily_answer([
        ["2026-06-17", 70, 300, 9, 1],
        ["2026-06-11", 30, 100, 2, 0],
        ["2026-06-04", 10, 50, 1, 1],
        ["2026-06-05", "n/a", 5, 0, 0],
    ])})
    days = fetch_days(reader, END)
    assert len(days) == DAYS
    assert days[0].date == "2026-06-04" and days[-1].date == "2026-06-17"
    assert days[-1] == Day("2026-06-17", 70, 300, 9, 1)
    assert days[1] == Day("2026-06-05", 0, 5, 0, 0)
    assert days[2] == Day("2026-06-06", 0, 0, 0, 0)
    this, before = week_totals(days)
    assert (this.views, this.minutes, this.subscribers) == (100, 400, 10)
    assert (before.views, before.minutes, before.subscribers) == (10, 55, 0)
    assert reader.calls == [("2026-06-04", "2026-06-17", "views,estimatedMinutesWatched,subscribersGained,"
                             "subscribersLost", "day", "", "day")]


def test_change_is_a_whole_percent_or_none():
    assert change(112, 100) == 12
    assert change(91, 100) == -9
    assert change(5, 0) is None


def test_sources_sorted_labelled_and_shared():
    reader = FakeReader({("insightTrafficSourceType", ""): {
        "columnHeaders": [{"name": "insightTrafficSourceType"}, {"name": "views"}],
        "rows": [["EXT_URL", 10], ["YT_SEARCH", 60], ["RELATED_VIDEO", 30], ["PLAYLIST", 0], ["NEW_THING", "x"]],
    }})
    sources = fetch_sources(reader, END)
    assert sources == (("YT_SEARCH", 60), ("RELATED_VIDEO", 30), ("EXT_URL", 10))
    assert reader.calls[0][:2] == ("2026-05-21", "2026-06-17")
    assert source_shares(sources, top=2) == [("YouTube search", 60), ("Suggested videos", 30)]
    assert source_shares(()) == []
    assert source_label("SOME_NEW_PLACE") == "Some new place"


def test_newest_uploads_are_public_and_newest_first():
    videos = [vid("a", "2026-06-01T10:00:00Z"), vid("b", "2026-06-10T10:00:00Z"),
              vid("c", "2026-06-12T10:00:00Z", privacy="private"), vid("d", ""),
              vid("e", "2026-06-05T10:00:00Z")]
    assert [v.video_id for v in newest_uploads(videos, count=2)] == ["b", "e"]


def test_compare_latest_same_age_median_too_new_and_few_earlier():
    # new1: 2 days of data -> too new. new2: 8 days old, compared with 3 earlier long videos.
    # short1: a Short with no earlier Shorts -> few earlier. The Short never counts for long videos.
    videos = [
        vid("new1", "2026-06-16T08:00:00Z"),
        vid("new2", "2026-06-10T08:00:00Z"),
        vid("short1", "2026-06-09T08:00:00Z", duration="PT40S"),
        vid("old1", "2026-05-01T08:00:00Z"),
        vid("old2", "2026-04-01T08:00:00Z"),
        vid("old3", "2026-03-01T08:00:00Z"),
    ]
    reader = FakeReader({
        ("day", "video==new2"): views_answer({"2026-06-10": 100, "2026-06-17": 80}),
        ("day", "video==old1"): views_answer({"2026-05-01": 50, "2026-05-08": 10, "2026-05-09": 999}),
        ("day", "video==old2"): views_answer({"2026-04-02": 100}),
        ("day", "video==old3"): views_answer({"2026-03-01": 20}),
        ("day", "video==short1"): views_answer({"2026-06-09": 7}),
    })
    latest, calls = compare_latest(reader, videos, END)
    assert [s.video_id for s in latest] == ["new1", "new2", "short1"]
    assert latest[0] == SameAge("new1", "too_new", 2, 0, None, 0)
    assert latest[1] == SameAge("new2", "compared", 8, 180, 60, 3)
    assert latest[1].ratio == 3.0 and latest[1].verdict == "above"
    assert latest[2] == SameAge("short1", "few_earlier", 9, 7, None, 0)
    assert latest[2].ratio is None and latest[2].verdict == ""
    assert calls == 5
    assert ("2026-05-01", "2026-05-28", "views", "day", "video==old1", "day") in reader.calls


def test_verdicts_at_the_edges():
    assert SameAge("v", "compared", 5, 125, 100, 3).verdict == "above"
    assert SameAge("v", "compared", 5, 124, 100, 3).verdict == "usual"
    assert SameAge("v", "compared", 5, 80, 100, 3).verdict == "below"
    assert SameAge("v", "compared", 5, 80, 0, 3).ratio is None


def test_fetch_stats_saves_and_loads_back(tmp_path):
    videos = [vid("new1", "2026-06-16T08:00:00Z")]
    reader = FakeReader({("day", ""): daily_answer([["2026-06-17", 70, 300, 9, 1]])})
    said = []
    stats = fetch_stats(reader, videos, now=NOW, report=said.append)
    assert stats.end_date == "2026-06-17" and stats.calls == 2
    assert stats.fetched_at == NOW.isoformat()
    assert stats.latest == (SameAge("new1", "too_new", 2, 0, None, 0),)
    assert said == ["Reading the channel's daily numbers", "Reading where views came from",
                    "Comparing the newest uploads"]
    path = save_stats(tmp_path, stats)
    assert path == tmp_path / "analytics" / "channel.json"
    assert load_stats(tmp_path) == stats


def test_load_stats_none_when_missing_or_broken(tmp_path):
    assert load_stats(tmp_path) is None
    (tmp_path / "analytics").mkdir()
    (tmp_path / "analytics" / "channel.json").write_text("{not json", encoding="utf-8")
    assert load_stats(tmp_path) is None
    (tmp_path / "analytics" / "channel.json").write_text('{"fetched_at": 1}', encoding="utf-8")
    assert load_stats(tmp_path) is None


def test_load_stats_keeps_a_compared_upload(tmp_path):
    stats = ChannelStats("2026-06-20T12:00:00+00:00", "2026-06-17",
                         (Day("2026-06-17", 1, 2, 3, 4),), (("YT_SEARCH", 5),),
                         (SameAge("v", "compared", 8, 180, 60, 3),), 7)
    save_stats(tmp_path, stats)
    assert load_stats(tmp_path) == stats
