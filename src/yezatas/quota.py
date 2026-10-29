"""YouTube Data API v3 quota accounting. Nothing here talks to YouTube."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

from .store import Store

QUOTA_TZ = ZoneInfo("America/Los_Angeles")

COSTS: dict[str, int] = {
    "videos.list": 1,
    "channels.list": 1,
    "playlistItems.list": 1,
    "videos.update": 50,
    "thumbnails.set": 50,
    "captions.list": 50,
    "captions.download": 200,
    "captions.insert": 400,
    "captions.update": 450,
}


class UnknownOperation(KeyError):
    """The operation has no entry in the cost table."""


class QuotaExceeded(RuntimeError):
    """The requested charge does not fit in the remaining daily budget."""


def cost_of(operation: str) -> int:
    try:
        return COSTS[operation]
    except KeyError:
        raise UnknownOperation(f"unknown YouTube operation: {operation!r}") from None


def quota_day(now_utc: datetime) -> str:
    """The quota day (Pacific date, ISO) that the given aware datetime falls in."""
    if now_utc.tzinfo is None or now_utc.utcoffset() is None:
        raise ValueError("quota_day needs a timezone-aware datetime")
    return now_utc.astimezone(QUOTA_TZ).date().isoformat()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CostPreview:
    lines: tuple[tuple[str, int, int], ...]  # (operation, count, units)
    total: int
    used: int
    remaining: int
    fits: bool


class QuotaGuard:
    def __init__(
        self,
        store: Store,
        budget: int,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        self._store = store
        self.budget = budget
        self._clock = clock

    def _now(self) -> datetime:
        now = self._clock()
        quota_day(now)  # validates awareness
        return now

    def used(self) -> int:
        return self._store.quota_used(quota_day(self._now()))

    def remaining(self) -> int:
        return self.budget - self.used()

    def _price(self, plan: Iterable[tuple[str, int]]) -> tuple[tuple[str, int, int], ...]:
        lines = []
        for operation, count in plan:
            if count < 1:
                raise ValueError(f"count must be >= 1, got {count} for {operation}")
            lines.append((operation, count, cost_of(operation) * count))
        return tuple(lines)

    def preview(self, plan: Iterable[tuple[str, int]]) -> CostPreview:
        lines = self._price(plan)
        total = sum(units for _, _, units in lines)
        used = self.used()
        remaining = self.budget - used
        return CostPreview(lines, total, used, remaining, total <= remaining)

    def charge(self, operation: str, count: int = 1, note: str = "") -> int:
        return self._charge_lines(self._price([(operation, count)]), note)

    def charge_plan(self, plan: Iterable[tuple[str, int]], note: str = "") -> int:
        return self._charge_lines(self._price(plan), note)

    def _charge_lines(self, lines: tuple[tuple[str, int, int], ...], note: str) -> int:
        now = self._now()
        day = quota_day(now)
        total = sum(units for _, _, units in lines)
        remaining = self.budget - self._store.quota_used(day)
        if total > remaining:
            raise QuotaExceeded(f"need {total} units but only {remaining} remaining for {day}")
        self._store.add_quota_entries(
            [(op, units, count, note) for op, count, units in lines], quota_day=day, now=now
        )
        return total
