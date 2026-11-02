"""The Analytics card of the video page: one video's last 28 days, from the saved YouTube Analytics read.
Display values only; every number comes from code."""

from __future__ import annotations

from dataclasses import dataclass

from ..channelstats import country_name, source_label
from ..videostats import TOP_ROWS, VideoStats
from .analytics_view import ShareRow, share_rows
from .views import format_date, format_duration


@dataclass(frozen=True)
class VideoStat:
    label: str
    value: str
    note: str = ""


@dataclass(frozen=True)
class VideoAnalyticsView:
    up_to: str  # the last day of data, like "Jun 17, 2026"
    stats: tuple[VideoStat, ...]
    bars: tuple[str, ...]  # one height class per day (h-0 ... h-100), against the day with most views
    first_day: str  # like "May 21"
    last_day: str
    sources: tuple[ShareRow, ...]
    countries: tuple[ShareRow, ...]


def _hours(minutes: int) -> str:
    hours = minutes / 60
    return f"{hours:.1f} h" if hours < 10 else f"{round(hours):,} h"


def _height(views: int, top: int) -> str:
    if top <= 0 or views <= 0:
        return "h-0"
    return f"h-{min(100, max(5, round(views * 100 / top / 5) * 5))}"


def build_video_analytics(stats: VideoStats | None, duration_seconds: int | None) -> VideoAnalyticsView | None:
    if stats is None or not stats.days:
        return None
    average = stats.average_seconds
    if average is None:
        average_text, note = "No views", ""
    else:
        average_text = format_duration(average) or "0:00"
        note = f"{round(average * 100 / duration_seconds)}% of the video" if duration_seconds else ""
    top = max(d.views for d in stats.days)
    return VideoAnalyticsView(
        up_to=format_date(stats.end_date),
        stats=(VideoStat("Views", f"{stats.views:,}"), VideoStat("Watch time", _hours(stats.minutes)),
               VideoStat("Average view", average_text, note),
               VideoStat("Subscribers", f"{stats.subscribers:+,}")),
        bars=tuple(_height(d.views, top) for d in stats.days),
        first_day=format_date(stats.days[0].date).rsplit(",", 1)[0],
        last_day=format_date(stats.days[-1].date).rsplit(",", 1)[0],
        sources=share_rows(stats.sources, source_label, TOP_ROWS),
        countries=share_rows(stats.countries, country_name, TOP_ROWS),
    )
