from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
from urllib.parse import quote

from ..competitors import CompetitorChannel, CompetitorInsight, channel_outliers, channel_scores
from ..outliers import ratio_label
from .views import format_cost, format_date


@dataclass(frozen=True)
class CompetitorVideoRow:
    title: str
    url: str
    published: str
    views: str
    ratio: str


@dataclass(frozen=True)
class CompetitorCard:
    channel_id: str
    title: str
    url: str
    handle: str
    when: str
    video_count: int
    scored_count: int
    rows: tuple[CompetitorVideoRow, ...]


@dataclass(frozen=True)
class InsightRef:
    label: str
    title: str
    url: str
    channel: str


@dataclass(frozen=True)
class InsightPattern:
    text: str
    kind: str
    refs: tuple[InsightRef, ...]


@dataclass(frozen=True)
class InsightWhy:
    ref: InsightRef
    why: str


@dataclass(frozen=True)
class InsightView:
    status: str
    error: str
    patterns: tuple[InsightPattern, ...]
    whys: tuple[InsightWhy, ...]
    ideas: tuple[str, ...]
    notes: tuple[str, ...]
    when: str
    model: str
    cost: str
    tokens: str


def watch_url(video_id: str) -> str:
    return "https://www.youtube.com/watch?v=" + quote(video_id, safe="")


def channel_url(channel_id: str) -> str:
    return "https://www.youtube.com/channel/" + quote(channel_id, safe="")


def build_competitor_cards(
    channels: Sequence[CompetitorChannel],
    local_time: Callable[[str], str],
) -> list[CompetitorCard]:
    cards: list[CompetitorCard] = []
    for channel in channels:
        scores = channel_scores(channel)
        scored_count = sum(1 for s in scores if s.status == "scored")
        outliers = channel_outliers(channel)
        rows: list[CompetitorVideoRow] = []
        for o in outliers:
            rows.append(
                CompetitorVideoRow(
                    title=o.video.title,
                    url=watch_url(o.video.video_id),
                    published=format_date(o.video.published_at),
                    views=f"{o.score.views:,}",
                    ratio=ratio_label(o.score.ratio),
                )
            )
        cards.append(
            CompetitorCard(
                channel_id=channel.channel_id,
                title=channel.title,
                url=channel_url(channel.channel_id),
                handle=channel.handle,
                when=local_time(channel.fetched_at),
                video_count=len(channel.videos),
                scored_count=scored_count,
                rows=tuple(rows),
            )
        )
    return cards


def build_insight_view(
    insight: CompetitorInsight | None,
    local_time: Callable[[str], str],
) -> InsightView | None:
    if insight is None:
        return None

    def make_ref(label: str) -> InsightRef | None:
        entry = insight.labels.get(label)
        if entry is None:
            return None
        return InsightRef(
            label=label,
            title=str(entry.get("title", "")),
            url=watch_url(str(entry.get("video_id", ""))),
            channel=str(entry.get("channel_title", "")),
        )

    patterns: list[InsightPattern] = []
    for p in insight.patterns:
        text = str(p.get("text", ""))
        kind = str(p.get("kind", ""))
        evidence = p.get("evidence", []) or []
        refs: list[InsightRef] = []
        for label in evidence:
            r = make_ref(str(label))
            if r is not None:
                refs.append(r)
        patterns.append(InsightPattern(text=text, kind=kind, refs=tuple(refs)))

    whys: list[InsightWhy] = []
    for entry in insight.per_video:
        label = str(entry.get("label", ""))
        r = make_ref(label)
        if r is None:
            continue
        whys.append(InsightWhy(ref=r, why=str(entry.get("why", ""))))

    notes: tuple[str, ...] = tuple(insight.problems)

    has_tokens = (insight.input_tokens + insight.output_tokens) != 0
    tokens = ""
    if has_tokens:
        tokens = f"{insight.input_tokens:,} in, {insight.output_tokens:,} out"

    cost = ""
    if insight.status == "done" or (insight.status == "error" and has_tokens):
        cost = format_cost(insight.cost_usd)

    when = local_time(insight.created_at)

    return InsightView(
        status=insight.status,
        error=insight.error,
        patterns=tuple(patterns),
        whys=tuple(whys),
        ideas=tuple(insight.ideas),
        notes=notes,
        when=when,
        model=insight.model,
        cost=cost,
        tokens=tokens,
    )
