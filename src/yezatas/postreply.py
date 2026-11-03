"""Post a reply draft (or the pinned-comment draft) to YouTube, only after the creator confirmed it.

Nothing here runs unless the creator saw the exact text and pressed the confirm button. The text
sent is the text they confirmed (they may have edited it), checked first: not empty, at most
10,000 characters. A reply answers a top-level comment (comments.insert); the pinned-comment
draft, which has no comment to answer, becomes a new top-level comment (commentThreads.insert).
Both cost 50 quota units, charged before the call. The API cannot pin a comment: the creator
pins it in YouTube Studio.

Posting twice is refused: the draft must still be open, checked under a lock while the call is
made, and the store marks it posted only if it is still open.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

from .quota import QuotaExceeded, QuotaGuard
from .store import Store
from .youtube import CommentWriter, api_error_info

MAX_CHARS = 10_000  # YouTube's limit for one comment
REPLY_OPERATION = "comments.insert"
PINNED_OPERATION = "commentThreads.insert"

_QUOTA_REASONS = {"quotaExceeded", "rateLimitExceeded", "dailyLimitExceeded",
                  "userRateLimitExceeded", "RATE_LIMIT_EXCEEDED", "RESOURCE_EXHAUSTED"}
_SCOPE_REASONS = {"insufficientPermissions", "ACCESS_TOKEN_SCOPE_INSUFFICIENT", "authError"}
_GONE_REASONS = {"commentNotFound", "parentCommentIsDeleted", "videoNotFound"}
SCOPE_MESSAGE = ("Yezatas does not have YouTube's permission to post comments yet. Allow it "
                 "again below. Nothing was posted.")
_POST_LOCK = threading.Lock()


class PostReplyError(Exception):
    """A post that did not happen; the message is a plain sentence.

    `kind` is one of: invalid, missing, closed, already, scope, quota, deleted, disabled,
    network, api. For every kind but missing, already and closed the draft stays open.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class PostedReply:
    """What one successful post did."""

    reply_id: int  # the draft's id in Yezatas
    video_id: str
    youtube_id: str  # the new comment's id on YouTube ("" if YouTube sent none back)
    text: str
    pinned_draft: bool  # a new top-level comment, which the creator still has to pin in Studio
    units: int
    url: str  # the new comment on YouTube, "" without an id


def posted_url(video_id: str, youtube_id: str) -> str:
    """The link to a comment or reply on YouTube; empty when the id is unknown."""
    if not youtube_id:
        return ""
    return "https://www.youtube.com/watch?" + urlencode({"v": video_id, "lc": youtube_id})


def check_text(text: str) -> str:
    """The text to post, trimmed; PostReplyError("invalid") when empty or over the limit."""
    clean = text.strip()
    if not clean:
        raise PostReplyError("invalid", "The reply is empty. Write something, or cancel.")
    if len(clean) > MAX_CHARS:
        raise PostReplyError(
            "invalid", f"The reply is {len(clean):,} characters. YouTube allows at most "
                       f"{MAX_CHARS:,}. Shorten it.")
    return clean


def operation_for(pinned_draft: bool) -> str:
    return PINNED_OPERATION if pinned_draft else REPLY_OPERATION


def api_failure(exc: BaseException, pinned_draft: bool) -> PostReplyError:
    """A plain-English PostReplyError for a failed YouTube call."""
    info = api_error_info(exc)
    reasons = set(info.reasons)
    what = "comment" if pinned_draft else "reply"
    if info.status == 403 and "commentsDisabled" in reasons:
        return PostReplyError("disabled", f"Comments are turned off on this video, so the {what} "
                                          "was not posted.")
    if reasons & _QUOTA_REASONS:
        return PostReplyError("quota", "YouTube's daily quota for this app is used up. Nothing "
                                       "was posted. Try again tomorrow.")
    if info.status == 401 or reasons & _SCOPE_REASONS or (
            info.status == 403 and "scope" in info.message.lower()):
        return PostReplyError("scope", SCOPE_MESSAGE)
    if info.status == 404 or reasons & _GONE_REASONS:
        if pinned_draft:
            return PostReplyError("deleted", "YouTube could not find this video, so the comment "
                                             "was not posted.")
        return PostReplyError("deleted", "The comment you answer is gone from YouTube (it may have "
                                         "been deleted), so the reply was not posted. Dismiss "
                                         "this draft.")
    if info.status is None and not info.reasons and not info.message:
        if isinstance(exc, OSError) or type(exc).__name__ == "ServerNotFoundError":
            return PostReplyError("network", f"Could not reach YouTube. The {what} was probably "
                                             "not posted: check YouTube before you try again.")
        return PostReplyError("api", f"The {what} could not be sent ({type(exc).__name__}). "
                                     "Check YouTube before you try again.")
    detail = f"HTTP {info.status}" if info.status else "no HTTP status"
    if info.reason:
        detail += f", {info.reason}"
    said = f" It said: {info.message[:200]}" if info.message else ""
    return PostReplyError("api", f"YouTube would not post the {what} ({detail}).{said}")


def _new_comment_id(response: Any, pinned_draft: bool) -> str:
    """The id of the comment YouTube created, from the response of either insert call."""
    if not isinstance(response, dict):
        return ""
    if pinned_draft:
        snippet = response.get("snippet")
        top = snippet.get("topLevelComment") if isinstance(snippet, dict) else None
        found = top.get("id") if isinstance(top, dict) else None
        if isinstance(found, str) and found:
            return found
    found = response.get("id")
    return found if isinstance(found, str) else ""


def post_reply(store: Store, writer: CommentWriter, guard: QuotaGuard, reply_id: int, text: str,
               *, now: datetime | None = None) -> PostedReply:
    """Post draft `reply_id` as `text` (the creator's confirmed, possibly edited, words).

    Raises PostReplyError; the draft stays open unless the kind is missing, already or closed.
    """
    clean = check_text(text)
    with _POST_LOCK:
        draft = store.get_comment_reply(reply_id)
        if draft is None:
            raise PostReplyError("missing", "This draft does not exist.")
        if draft.status == "posted":
            raise PostReplyError("already", "This reply was already posted. Nothing was sent.")
        if draft.status != "open":
            raise PostReplyError("closed", "This draft was put away, so it cannot be posted. "
                                           "Nothing was sent.")
        pinned = draft.comment_id == ""
        operation = operation_for(pinned)
        try:
            units = guard.charge(operation, note="post comment" if pinned else "post reply")
        except QuotaExceeded as exc:
            raise PostReplyError(
                "quota", f"Yezatas's daily YouTube budget has no room for a post ({exc}). "
                         "Nothing was posted.") from exc
        try:
            if pinned:
                response = writer.insert_top_level(draft.video_id, clean)
            else:
                response = writer.insert_reply(draft.comment_id, clean)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a plain sentence
            raise api_failure(exc, pinned) from exc
        youtube_id = _new_comment_id(response, pinned)
        store.mark_comment_reply_posted(draft.id, clean, youtube_id, now=now)
    return PostedReply(draft.id, draft.video_id, youtube_id, clean, pinned, units,
                       posted_url(draft.video_id, youtube_id))
