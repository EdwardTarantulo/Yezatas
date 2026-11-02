"""The sidebar's attention counts: a number beside each menu item that has something waiting."""

from __future__ import annotations

from dataclasses import dataclass

from .views import OK_FROM


@dataclass(frozen=True)
class NavCount:
    count: int
    tone: str  # "info": work waiting for you; "warn": something costing points or broken
    noun: str  # what is counted, after the number: "questions to answer"

    @property
    def shown(self) -> str:
        return "99+" if self.count > 99 else str(self.count)

    @property
    def text(self) -> str:
        return f"{self.count} {self.noun}"


def _noun(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def build_nav_counts(*, questions: int = 0, poor_videos: int = 0, stale_translations: int = 0,
                     broken_links: int = 0, drafts: int = 0) -> dict[str, NavCount]:
    """nav key -> its count, only for the items with something waiting."""
    found = {
        "comments": (questions, "info", _noun(questions, "question to answer", "questions to answer")),
        "content": (poor_videos, "warn", _noun(poor_videos, f"video scores under {OK_FROM}",
                                               f"videos score under {OK_FROM}")),
        "translations": (stale_translations, "warn", _noun(stale_translations, "translation out of date",
                                                           "translations out of date")),
        "links": (broken_links, "warn", _noun(broken_links, "broken link", "broken links")),
        "changes": (drafts, "info", _noun(drafts, "draft to review", "drafts to review")),
    }
    return {key: NavCount(n, tone, noun) for key, (n, tone, noun) in found.items() if n > 0}
