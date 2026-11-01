"""What the thumbnail swap page and History's Thumbnails section show (F3). No YouTube call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ..thumbswap import ThumbnailSwap


@dataclass(frozen=True)
class SwapRow:
    swap: ThumbnailSwap
    title: str
    when: str
    label: str
    backup_url: str
    restore_url: str


def swap_label(swap: ThumbnailSwap) -> str:
    """Human label for a swap."""
    if swap.kind == "replace":
        return "Replaced with a variant idea"
    if swap.kind == "restore":
        return "Put an old thumbnail back"
    return swap.kind


def build_swap_rows(
    swaps: Sequence[ThumbnailSwap],
    titles: Mapping[str, str],
    when: Callable[[str], str],
) -> list[SwapRow]:
    """Build rows for History's Thumbnails section in the given order."""
    rows: list[SwapRow] = []
    for swap in swaps:
        title = titles.get(swap.video_id, swap.video_id)
        rows.append(
            SwapRow(
                swap=swap,
                title=title,
                when=when(swap.at),
                label=swap_label(swap),
                backup_url=f"/history/thumbnails/{swap.swap_id}/backup.jpg",
                restore_url=f"/history/thumbnails/{swap.swap_id}/restore",
            )
        )
    return rows
