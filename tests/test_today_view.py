from datetime import datetime, timezone

from yezatas.channelstats import ChannelStats, Day, SameAge
from yezatas.seo import score_catalog
from yezatas.store import Comment, Video
from yezatas.web.today_view import (
    CommentRow,
    Stat,
    age_text,
    build_comments,
    build_latest,
    build_today,
    build_week,
    gain_text,
    shorten,
    vs_label,
)
from yezatas.web.views import make_entries

NOW = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)


def video(video_id, published, views=0, privacy="public", title="A video"):
    return Video(video_id=video_id, title=title, published_at=published, privacy_status=privacy,
                 view_count=views, duration="PT8M")


def entries(videos):
    return make_entries(videos, score_catalog(videos))


def test_age_text_steps():
    assert age_text("2026-06-20T01:00:00Z", NOW) == "Published today"
    assert age_text("2026-06-19T23:00:00Z", NOW) == "Published yesterday"
    assert age_text("2026-06-18T10:00:00Z", NOW) == "Published 2 days ago"
    assert age_text("2026-06-07T10:00:00Z", NOW) == "Published 13 days ago"
    assert age_text("2026-06-06T10:00:00Z", NOW) == "Published 2 weeks ago"
    assert age_text("2026-04-22T10:00:00Z", NOW) == "Published 8 weeks ago"
    assert age_text("2026-04-21T10:00:00Z", NOW) == "Published Apr 21, 2026"
    assert age_text("", NOW) == ""


def test_gain_text():
    assert gain_text(8100, 4860) == "+3,240"
    assert gain_text(10, 10) == "+0"
    assert gain_text(90, 102) == "-12"
    assert gain_text(500, None) == ""


def test_vs_label():
    assert vs_label(None) == ("", "")
    assert vs_label(SameAge("v", "too_new", 2, 0, None, 0)) == ("Too new to compare", "new")
    assert vs_label(SameAge("v", "few_earlier", 9, 7, None, 1)) == ("", "")
    assert vs_label(SameAge("v", "compared", 8, 180, 100, 10)) == ("1.8× your usual", "above")
    assert vs_label(SameAge("v", "compared", 8, 60, 100, 10)) == ("0.6× your usual", "below")
    assert vs_label(SameAge("v", "compared", 8, 100, 100, 10)) == ("Usual", "usual")
    assert vs_label(SameAge("v", "compared", 8, 100, 0, 10)) == ("", "")


def test_shorten():
    assert shorten("  Which   soil\nmix? ") == "Which soil mix?"
    assert shorten("one two three four", limit=10) == "one two…"
    assert shorten("abcdefghijklmnop", limit=5) == "abcde…"


def test_build_latest_newest_public_first_with_gain_and_comparison():
    vids = [video("old", "2026-05-01T10:00:00Z", 21400, title="Old one"),
            video("new", "2026-06-18T10:00:00Z", 8100, title="Repotting cactus tips #7"),
            video("mid", "2026-06-11T10:00:00Z", 6200),
            video("hidden", "2026-06-19T10:00:00Z", 5, privacy="private")]
    stats = ChannelStats("2026-06-20T12:00:00+00:00", "2026-06-17", (), (),
                         (SameAge("new", "too_new", 0, 0, None, 0),
                          SameAge("mid", "compared", 7, 180, 100, 10)), 3)
    rows = build_latest(entries(vids), {"new": ("2026-06-19T10:00:00+00:00", 4860)}, stats, NOW)
    assert [r.video_id for r in rows] == ["new", "mid", "old"]
    first = rows[0]
    assert (first.title, first.age_text, first.views_text, first.gain_text) == (
        "Repotting cactus tips #7", "Published 2 days ago", "8.1K", "+3,240")
    assert (first.vs_text, first.vs_class) == ("Too new to compare", "new")
    assert (rows[1].gain_text, rows[1].vs_text, rows[1].vs_class) == ("", "1.8× your usual", "above")
    assert (rows[2].vs_text, rows[2].vs_class) == ("", "")
    assert build_latest(entries(vids), {}, None, NOW)[0].vs_text == ""


def test_build_comments_counts_new_and_marks_drafts():
    vids = {"v1": video("v1", "2026-06-01T10:00:00Z", title="Cactus soil")}
    found = [Comment(f"c{n}", "v1", author=f"@viewer{n}", text=f"Question {n}?",
                     published_at=f"2026-06-{20 - n:02d}T10:00:00Z") for n in range(5)]
    found.append(Comment("c9", "gone", author="@x", text="Old video?", published_at=""))
    rows, total, new = build_comments(found, {"c1"}, vids, NOW)
    assert total == 6 and new == 1
    assert [r.comment_id for r in rows] == ["c0", "c1", "c2"]
    assert rows[1] == CommentRow("c1", "v1", "@viewer1", "Cactus soil", "Question 1?", True)
    assert rows[0].has_draft is False


def test_build_week_stats_bars_and_sources():
    days = tuple(Day(f"2026-06-{d:02d}", v, m, g, l) for d, v, m, g, l in [
        (4, 10, 60, 1, 0), (5, 10, 60, 1, 0), (6, 10, 60, 1, 0), (7, 10, 60, 1, 0),
        (8, 10, 60, 1, 0), (9, 10, 60, 1, 0), (10, 40, 60, 4, 0),
        (11, 0, 0, 0, 0), (12, 20, 120, 2, 1), (13, 40, 120, 2, 0), (14, 40, 120, 1, 0),
        (15, 1, 120, 0, 0), (16, 80, 120, 0, 0), (17, 39, 120, 0, 0)])
    stats = ChannelStats("2026-06-20T12:00:00+00:00", "2026-06-17", days,
                         (("YT_SEARCH", 60), ("RELATED_VIDEO", 30), ("EXT_URL", 10)), (), 2)
    week = build_week(stats)
    assert week is not None and week.up_to == "Jun 17, 2026"
    assert week.stats == (Stat("Views", "220", "+120%", "up"),
                          Stat("Watch time", "12 h", "+71%", "up"),
                          Stat("Subscribers", "+4", "-60%", "down"))
    assert week.bars == (0, 25, 50, 50, 5, 100, 50)
    assert week.sources == (("YouTube search", 60), ("Suggested videos", 30), ("External sites", 10))
    assert build_week(None) is None
    assert build_week(ChannelStats("x", "2026-06-17", (), (), (), 0)) is None


def test_build_today_puts_it_together():
    vids = [video("new", "2026-06-18T10:00:00Z", 100)]
    fix = entries(vids)[0]
    today = build_today(entries(vids), {}, [], set(), {}, None, fix, NOW)
    assert [r.video_id for r in today.latest] == ["new"]
    assert (today.comments, today.comment_total, today.comment_new) == ((), 0, 0)
    assert today.week is None and today.fix is fix
