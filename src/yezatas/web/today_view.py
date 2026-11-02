"""The Today page (/): how the newest uploads are doing, the comments to answer, the channel's
week and one thing to fix. Display values only; every number comes from the store or the saved Analytics
fetch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping, Sequence

from ..channelstats import ChannelStats, SameAge, WEEK_DAYS, change, newest_uploads, source_shares, week_totals
from ..outliers import _parse_time, _utc
from ..store import Comment, Video
from .views import Entry, format_count, format_date

COMMENT_ROWS = 3
COMMENT_CHARS = 200


@dataclass(frozen=True)
class LatestRow:
    video_id: str
    title: str
    thumb: str
    age_text: str  # "Published 2 days ago"
    views_text: str  # total views, like "21.1K"
    gain_text: str  # views since the last pull, like "+3,240"; "" when there is no earlier pull
    vs_text: str  # "1.8× your usual", "Usual", "Too new to compare" or ""
    vs_class: str  # "above" | "usual" | "below" | "new" | ""


@dataclass(frozen=True)
class CommentRow:
    comment_id: str
    video_id: str
    author: str
    video_title: str
    text: str  # at most COMMENT_CHARS characters, cut at a space with "…"
    has_draft: bool


@dataclass(frozen=True)
class Stat:
    label: str
    value: str
    delta_text: str  # "+12%" or "" when there is no earlier week to compare with
    delta_class: str  # "up" | "down" | ""


@dataclass(frozen=True)
class WeekView:
    up_to: str  # the last day of data, like "Oct 31, 2026"
    stats: tuple[Stat, ...]  # Views, Watch time, Subscribers
    bars: tuple[int, ...]  # the last 7 days' views as heights in percent, multiples of 5
    sources: tuple[tuple[str, int], ...]  # (label, percent), the top 3


@dataclass(frozen=True)
class TodayView:
    latest: tuple[LatestRow, ...]
    comments: tuple[CommentRow, ...]
    comment_total: int
    comment_new: int  # published in the 24 hours before `now`
    week: WeekView | None
    fixes: tuple[Entry, ...]  # up to FIXES videos, in the SEO overview's Fix next order (views x points)


def age_text(published_at: str, now: datetime) -> str:
    dt = _parse_time(published_at)
    if dt is None:
        return ""
    now_utc = _utc(now)
    days = (now_utc.date() - dt.date()).days
    if days <= 0:
        return "Published today"
    if days == 1:
        return "Published yesterday"
    if days <= 13:
        return f"Published {days} days ago"
    if days <= 59:
        return f"Published {days // 7} weeks ago"
    return "Published " + format_date(published_at)


def gain_text(current: int, previous: int | None) -> str:
    if previous is None:
        return ""
    return f"{current - previous:+,}"


def vs_label(same: SameAge | None) -> tuple[str, str]:
    if same is None:
        return ("", "")
    if same.status == "too_new":
        return ("Too new to compare", "new")
    if same.status == "compared":
        verdict = same.verdict
        if verdict == "usual":
            return ("Usual", "usual")
        if verdict in ("above", "below"):
            # ratio is guaranteed not None when verdict is above/below
            return (f"{same.ratio:.1f}× your usual", verdict)
    return ("", "")


def shorten(text: str, limit: int = COMMENT_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    cut = cut.rstrip()
    return cut + "…"


def build_latest(
    entries: Sequence[Entry],
    previous: Mapping[str, tuple[str, int]],
    stats: ChannelStats | None,
    now: datetime,
) -> tuple[LatestRow, ...]:
    videos = [e.video for e in entries]
    newest = newest_uploads(videos)
    entry_by_id: dict[str, Entry] = {e.video.video_id: e for e in entries}
    same_by_id: dict[str, SameAge] = {}
    if stats is not None:
        for s in stats.latest:
            same_by_id[s.video_id] = s
    rows: list[LatestRow] = []
    for v in newest:
        e = entry_by_id.get(v.video_id)
        if e is None:
            continue
        prev = previous.get(v.video_id)
        prev_count = prev[1] if prev is not None else None
        g = gain_text(v.view_count, prev_count)
        vs_t, vs_c = vs_label(same_by_id.get(v.video_id))
        rows.append(
            LatestRow(
                video_id=v.video_id,
                title=v.title,
                thumb=e.thumb,
                age_text=age_text(v.published_at, now),
                views_text=e.views_text,
                gain_text=g,
                vs_text=vs_t,
                vs_class=vs_c,
            )
        )
    return tuple(rows)


def build_comments(
    to_answer: Sequence[Comment],
    draft_ids: frozenset[str] | set[str],
    videos: Mapping[str, Video],
    now: datetime,
) -> tuple[tuple[CommentRow, ...], int, int]:
    total = len(to_answer)
    now_utc = _utc(now)
    new = 0
    for c in to_answer:
        dt = _parse_time(c.published_at)
        if dt is None:
            continue
        # within 24 hours before now: 0 <= now - dt <= 24h
        delta = now_utc - dt
        if timedelta(0) <= delta <= timedelta(hours=24):
            new += 1
    rows: list[CommentRow] = []
    for c in list(to_answer)[:COMMENT_ROWS]:
        vid = videos.get(c.video_id)
        title = vid.title if vid is not None else ""
        rows.append(
            CommentRow(
                comment_id=c.comment_id,
                video_id=c.video_id,
                author=c.author,
                video_title=title,
                text=shorten(c.text),
                has_draft=c.comment_id in draft_ids,
            )
        )
    return (tuple(rows), total, new)


def build_week(stats: ChannelStats | None) -> WeekView | None:
    if stats is None or not stats.days:
        return None
    this, before = week_totals(stats.days)

    def delta_str(this_val: int, before_val: int) -> tuple[str, str]:
        p = change(this_val, before_val)
        if p is None:
            return ("", "")
        text = f"{p:+d}%"
        if p > 0:
            cls = "up"
        elif p < 0:
            cls = "down"
        else:
            cls = ""
        return (text, cls)

    views_delta, views_cls = delta_str(this.views, before.views)
    minutes_delta, minutes_cls = delta_str(this.minutes, before.minutes)
    subs_delta, subs_cls = delta_str(this.subscribers, before.subscribers)

    stat_views = Stat("Views", f"{this.views:,}", views_delta, views_cls)
    stat_watch = Stat("Watch time", f"{round(this.minutes / 60):,} h", minutes_delta, minutes_cls)
    stat_subs = Stat("Subscribers", f"{this.subscribers:+,}", subs_delta, subs_cls)

    last_week_days = stats.days[-WEEK_DAYS:]
    view_counts = [d.views for d in last_week_days]
    m = max(view_counts) if view_counts else 0
    bars: list[int] = []
    for v in view_counts:
        if m > 0:
            h = round(v * 20 / m) * 5
        else:
            h = 0
        if v > 0 and h < 5:
            h = 5
        bars.append(h)

    up_to = format_date(stats.end_date)
    sources = tuple(source_shares(stats.sources))

    return WeekView(
        up_to=up_to,
        stats=(stat_views, stat_watch, stat_subs),
        bars=tuple(bars),
        sources=sources,
    )


FIXES = 5  # videos in the "Videos to fix" card


def build_today(
    entries: Sequence[Entry],
    previous: Mapping[str, tuple[str, int]],
    to_answer: Sequence[Comment],
    draft_ids: frozenset[str] | set[str],
    videos: Mapping[str, Video],
    stats: ChannelStats | None,
    fixes: Sequence[Entry],
    now: datetime,
) -> TodayView:
    latest = build_latest(entries, previous, stats, now)
    comments, comment_total, comment_new = build_comments(to_answer, draft_ids, videos, now)
    week = build_week(stats)
    return TodayView(
        latest=latest,
        comments=comments,
        comment_total=comment_total,
        comment_new=comment_new,
        week=week,
        fixes=tuple(fixes[:FIXES]),
    )
