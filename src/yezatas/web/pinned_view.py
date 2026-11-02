"""The Pinned comment card of the video page: whether a comment from the channel is there, and Muse Spark's
suggestion for one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from urllib.parse import urlencode

from ..seo import CheckResult
from ..store import Comment, PinnedComment
from .views import format_count, format_date

DETAILS: dict[str, str] = {
    "marked": "You marked that a pinned comment is up on this video.",
    "own": "A comment from your channel is on this video. YouTube's API does not say which comment is pinned, "
           "so make sure yours is.",
    "none": "No comment from your channel was found on this video. YouTube's API does not say which comment is "
            "pinned, so Yezatas looks for one written by you; if yours is pinned already, mark it below.",
    "unknown": "Yezatas has not fetched this video's comments yet, so it cannot tell whether one from you is "
               "there. Fetch them in the Comments tool, or mark it below if yours is pinned already.",
}
CHIPS: dict[str, tuple[str, str]] = {
    "marked": ("Marked pinned", "done"), "own": ("Yours found", "done"),
    "none": ("None found", "warn"), "unknown": ("Not fetched", "unverified"),
}


@dataclass(frozen=True)
class OwnComment:
    """The newest fetched comment written by the channel itself."""
    text: str
    date_text: str
    likes_text: str
    youtube_url: str


@dataclass(frozen=True)
class PinnedView:
    state: str  # "marked" | "own" | "none" | "unknown"
    chip: str
    tone: str  # a pill class
    detail: str
    suggestion: str  # Muse Spark's newest kept suggestion ("" when none)
    when: str  # when it was suggested, local time ("" when none)
    error: str  # why the newest suggestion failed ("" when it did not)
    marked: bool
    own: OwnComment | None = None  # the channel's newest comment on the video, when one was fetched


def own_comment(comments: list[Comment]) -> OwnComment | None:
    """The newest comment by the channel among `comments`, or None."""
    mine = [c for c in comments if c.by_owner]
    if not mine:
        return None
    c = max(mine, key=lambda c: (c.published_at or "", c.comment_id))
    return OwnComment(c.text, format_date(c.published_at), format_count(c.like_count),
                      "https://www.youtube.com/watch?" + urlencode({"v": c.video_id, "lc": c.comment_id}))


def build_pinned_view(record: PinnedComment | None, result: CheckResult | None,
                      local_time: Callable[[str], str], comments: list[Comment] | None = None) -> PinnedView:
    marked = bool(record is not None and record.marked_at)
    if marked:
        state = "marked"
    elif result is None or not result.applicable:
        state = "unknown"
    elif result.passed:
        state = "own"
    else:
        state = "none"
    chip, tone = CHIPS[state]
    suggestion = record.suggestion if record is not None else ""
    when = local_time(record.suggested_at) if record is not None and record.suggested_at else ""
    return PinnedView(state, chip, tone, DETAILS[state], suggestion, when,
                      record.error if record is not None else "", marked, own_comment(comments or []))
