"""Search gaps on the page: the gaps table and the last ranking, as plain strings for the template."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
from urllib.parse import quote

from ..searchgaps import GapRanking, SearchGap
from .views import format_cost


@dataclass(frozen=True)
class GapRow:
    term: str
    views: str
    video_title: str
    video_url: str
    missing: str


def _video(gap: SearchGap) -> tuple[str, str]:
    if gap.video_id == "":
        return "No video has these words", ""
    return gap.title, "/videos/" + quote(gap.video_id, safe="")


def build_gap_rows(gaps: Sequence[SearchGap]) -> tuple[GapRow, ...]:
    """One row per gap for the gaps table."""
    rows: list[GapRow] = []
    for gap in gaps:
        views = f"{gap.views:,}"
        missing = ", ".join(gap.missing)
        video_title, video_url = _video(gap)
        rows.append(
            GapRow(term=gap.term, views=views, video_title=video_title, video_url=video_url, missing=missing)
        )
    return tuple(rows)


@dataclass(frozen=True)
class PickRow:
    label: str
    term: str
    views: str
    video_title: str
    video_url: str
    missing: str
    priority: str
    reason: str


@dataclass(frozen=True)
class RankingView:
    status: str
    error: str
    picks: tuple[PickRow, ...]
    notes: tuple[str, ...]
    when: str
    model: str
    tokens: str
    cost: str


def build_ranking_view(ranking: GapRanking | None, local_time: Callable[[str], str]) -> RankingView | None:
    """Turn a saved ranking into what the template shows."""
    if ranking is None:
        return None
    picks: list[PickRow] = []
    for ranked in ranking.picks:
        gap = ranked.gap
        views = f"{gap.views:,}"
        missing = ", ".join(gap.missing)
        video_title, video_url = _video(gap)
        picks.append(
            PickRow(
                label=ranked.label,
                term=gap.term,
                views=views,
                video_title=video_title,
                video_url=video_url,
                missing=missing,
                priority=ranked.priority,
                reason=ranked.reason,
            )
        )
    notes: tuple[str, ...] = tuple(ranking.problems)
    has_tokens = (ranking.input_tokens + ranking.output_tokens) != 0
    tokens = ""
    if has_tokens:
        tokens = f"{ranking.input_tokens:,} in, {ranking.output_tokens:,} out"
    cost = ""
    if ranking.status == "done" or (ranking.status == "error" and has_tokens):
        cost = format_cost(ranking.cost_usd)
    when = local_time(ranking.created_at)
    return RankingView(
        status=ranking.status,
        error=ranking.error,
        picks=tuple(picks),
        notes=notes,
        when=when,
        model=ranking.model,
        tokens=tokens,
        cost=cost,
    )
