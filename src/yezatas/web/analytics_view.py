"""The Analytics page (/analytics): the channel's week, where viewers watch from and where views came
from, from the saved YouTube Analytics fetch. Display values only; every number comes from code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..channelstats import TOP_COUNTRIES, ChannelStats, country_name, source_label
from .today_view import WeekView, build_week
from .views import bar_class, format_count


@dataclass(frozen=True)
class ShareRow:
    label: str
    views_text: str  # like "4.1K"
    percent: int  # of all views in the list, whole percent
    bar: str  # a width class (w-0 ... w-100), relative to the biggest row shown


@dataclass(frozen=True)
class AnalyticsView:
    week: WeekView | None
    countries: tuple[ShareRow, ...]
    sources: tuple[ShareRow, ...]


def share_rows(pairs: Sequence[tuple[str, int]], label: Callable[[str], str],
               top: int | None = None) -> tuple[ShareRow, ...]:
    total = sum(v for _, v in pairs)
    if total <= 0:
        return ()
    picked = list(pairs)[:top] if top is not None else list(pairs)
    biggest = max(v for _, v in picked)
    return tuple(ShareRow(label(code), format_count(v), round(v * 100 / total), bar_class(v, biggest))
                 for code, v in picked)


def build_analytics(stats: ChannelStats | None) -> AnalyticsView:
    if stats is None:
        return AnalyticsView(None, (), ())
    return AnalyticsView(build_week(stats), share_rows(stats.countries, country_name, TOP_COUNTRIES),
                         share_rows(stats.sources, source_label))
