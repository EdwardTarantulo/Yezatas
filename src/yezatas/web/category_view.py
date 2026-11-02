"""The Category card of the video page: the video's YouTube category and Muse Spark's view of it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..seo import CATEGORY_NAMES, CheckResult
from ..store import CategoryCheck


@dataclass(frozen=True)
class CategoryView:
    current: str  # the category name now, or "Not set"
    state: str  # "unchecked" | "fits" | "confirmed" | "differs" | "changed"
    chip: str
    tone: str  # a pill class: "done", "warn" or "unverified"
    suggested: str  # the category Muse Spark named ("" before a finished check)
    reason: str
    when: str  # when it was checked, local time ("" before any check)
    error: str  # why the newest check failed ("" when it did not)
    can_confirm: bool  # "It's correct" is offered


def build_category_view(current_id: str, record: CategoryCheck | None, result: CheckResult | None,
                        local_time: Callable[[str], str]) -> CategoryView:
    current = CATEGORY_NAMES.get(current_id, f"Category {current_id}") if current_id else "Not set"
    when = local_time(record.checked_at) if record is not None else ""
    if record is None or record.status != "done" or result is None:
        error = record.error if record is not None and record.status == "failed" else ""
        return CategoryView(current, "unchecked", "Not checked", "unverified", "", "", when, error, False)
    suggested = CATEGORY_NAMES.get(record.suggested_id, record.suggested_id)
    if record.suggested_id == current_id:
        return CategoryView(current, "fits", "Fits", "done", suggested, record.reason, when, "", False)
    if not result.applicable:
        return CategoryView(current, "changed", "Check again", "warn", suggested, record.reason, when, "", False)
    if result.passed:
        return CategoryView(current, "confirmed", "Marked correct", "done", suggested, record.reason, when, "", False)
    return CategoryView(current, "differs", f"Muse Spark suggests {suggested}", "warn", suggested, record.reason,
                        when, "", True)
