"""One video's numbers for its page: the last 28 days of YouTube Analytics (read-only) filtered to that
video, in three queries (daily numbers, where views came from, where viewers watch from). Code does every
number; saved as JSON under home/analytics/videos; nothing is written to YouTube."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .channelstats import DAILY_METRICS, ChannelReader, Day, _is_int_strict, last_day, table
from .outliers import _utc

VIDEO_DAYS = 28
TOP_ROWS = 5  # sources and countries shown
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class VideoStatsError(ValueError):
    """The video id cannot be used for a file name or a filter."""


@dataclass(frozen=True)
class VideoStats:
    video_id: str
    fetched_at: str
    end_date: str
    days: tuple[Day, ...]  # VIDEO_DAYS days, oldest first, zero for a day without data
    sources: tuple[tuple[str, int], ...]  # (traffic source code, views), most views first
    countries: tuple[tuple[str, int], ...]  # (country code, views), most views first
    calls: int

    @property
    def views(self) -> int:
        return sum(d.views for d in self.days)

    @property
    def minutes(self) -> int:
        return sum(d.minutes for d in self.days)

    @property
    def subscribers(self) -> int:
        return sum(d.gained - d.lost for d in self.days)

    @property
    def average_seconds(self) -> int | None:
        """Watch time divided by views; None without views."""
        return round(self.minutes * 60 / self.views) if self.views > 0 else None


def _check_id(video_id: str) -> str:
    if not _SAFE_ID.match(video_id or ""):
        raise VideoStatsError("That video id cannot be read from YouTube Analytics.")
    return video_id


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _pairs(response: dict[str, Any], dimension: str) -> tuple[tuple[str, int], ...]:
    pairs: list[tuple[str, int]] = []
    for code, views in table(response, (dimension, "views")):
        if isinstance(code, str) and code.strip() and _int(views) > 0:
            pairs.append((code.strip(), _int(views)))
    pairs.sort(key=lambda p: (-p[1], p[0]))
    return tuple(pairs)


def fetch_video_stats(reader: ChannelReader, video_id: str, *, now: datetime | None = None,
                      report: Callable[[str], None] | None = None) -> VideoStats:
    _check_id(video_id)
    now = now or datetime.now(timezone.utc)
    end = last_day(now)
    start = end - timedelta(days=VIDEO_DAYS - 1)
    first, last = start.isoformat(), end.isoformat()
    only = f"video=={video_id}"
    if report is not None:
        report("Reading this video's daily numbers")
    rows = table(reader.query(first, last, DAILY_METRICS, "day", only, "day"),
                 ("day", "views", "estimatedMinutesWatched", "subscribersGained", "subscribersLost"))
    by_day = {row[0]: row[1:] for row in rows if isinstance(row[0], str)}
    days = []
    for i in range(VIDEO_DAYS):
        iso = (start + timedelta(days=i)).isoformat()
        values = by_day.get(iso, (0, 0, 0, 0))
        days.append(Day(iso, *(max(0, _int(v)) for v in values)))
    if report is not None:
        report("Reading where this video's views came from")
    sources = _pairs(reader.query(first, last, "views", "insightTrafficSourceType", only, "-views"),
                     "insightTrafficSourceType")
    if report is not None:
        report("Reading where this video's viewers watch from")
    countries = tuple((c.upper(), v) for c, v in
                      _pairs(reader.query(first, last, "views", "country", only, "-views"), "country"))
    return VideoStats(video_id, _utc(now).isoformat(), last, tuple(days), sources, countries, 3)


def stats_path(home: Path, video_id: str) -> Path:
    return Path(home) / "analytics" / "videos" / f"{_check_id(video_id)}.json"


def save_video_stats(home: Path, stats: VideoStats) -> Path:
    dest = stats_path(home, stats.video_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video_id": stats.video_id, "fetched_at": stats.fetched_at, "end_date": stats.end_date,
        "days": [[d.date, d.views, d.minutes, d.gained, d.lost] for d in stats.days],
        "sources": [[c, v] for c, v in stats.sources],
        "countries": [[c, v] for c, v in stats.countries],
        "calls": stats.calls,
    }
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix="video.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        os.replace(tmp_name, dest)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return dest


def _pair_list(raw: Any) -> tuple[tuple[str, int], ...] | None:
    if not isinstance(raw, list):
        return None
    out = []
    for item in raw:
        if not (isinstance(item, list) and len(item) == 2 and isinstance(item[0], str)
                and _is_int_strict(item[1])):
            return None
        out.append((item[0], item[1]))
    return tuple(out)


def load_video_stats(home: Path, video_id: str) -> VideoStats | None:
    """The saved numbers of one video; None when there are none or the file is not what Yezatas wrote."""
    try:
        data = json.loads(stats_path(home, video_id).read_text(encoding="utf-8"))
    except (OSError, ValueError, VideoStatsError):
        return None
    if not isinstance(data, dict) or data.get("video_id") != video_id:
        return None
    fetched_at, end_date, calls = data.get("fetched_at"), data.get("end_date"), data.get("calls")
    if not isinstance(fetched_at, str) or not isinstance(end_date, str) or not _is_int_strict(calls):
        return None
    try:
        date.fromisoformat(end_date)
    except ValueError:
        return None
    days_raw = data.get("days")
    if not isinstance(days_raw, list):
        return None
    days = []
    for item in days_raw:
        if not (isinstance(item, list) and len(item) == 5 and isinstance(item[0], str)
                and all(_is_int_strict(v) for v in item[1:])):
            return None
        days.append(Day(*item))
    sources, countries = _pair_list(data.get("sources")), _pair_list(data.get("countries"))
    if sources is None or countries is None:
        return None
    return VideoStats(video_id, fetched_at, end_date, tuple(days), sources, countries, calls)
