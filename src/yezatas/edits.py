"""Draft lifecycle: save an edit, approve it, rebase it, restore an older text.

Drafts never touch YouTube. A draft remembers its base (the catalog text it was made from)
so the changes page can show a diff and a push can notice that the video changed meanwhile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Sequence

from .store import DraftRow, Store, Video, _iso
from .validate import Issue, has_errors, validate_draft


class DraftError(ValueError):
    """An edit step was refused; the message is a plain sentence for the creator."""


@dataclass(frozen=True)
class SaveResult:
    saved: bool  # a draft now exists for the video
    discarded: bool  # the text equals the base, so any draft was removed
    issues: tuple[Issue, ...]
    draft: DraftRow | None


def _normalize_tags(tags: Sequence[str]) -> tuple[str, ...]:
    kept: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        tag = tag.strip()
        if tag and tag.casefold() not in seen:
            seen.add(tag.casefold())
            kept.append(tag)
    return tuple(kept)


def _text_of(video: Video) -> tuple[str, str, tuple[str, ...]]:
    return video.title, video.description, tuple(video.tags)


def save_draft(
    store: Store,
    video_id: str,
    title: str,
    description: str,
    tags: Sequence[str],
    *,
    source: str = "manual",
    note: str = "",
    now: datetime | None = None,
) -> SaveResult:
    """Save an edit as a draft (even with validation errors) or drop it if nothing changed."""
    video = store.get_video(video_id)
    if video is None:
        raise DraftError("This video is not in your catalog. Pull your videos first.")
    title = title.strip()
    description = description.replace("\r\n", "\n").replace("\r", "\n")
    new_tags = _normalize_tags(tags)
    existing = store.get_draft(video_id)
    if existing is not None:
        base = (existing.base_title, existing.base_description, existing.base_tags)
    else:
        base = _text_of(video)
    if (title, description, new_tags) == base:
        store.delete_draft(video_id)
        return SaveResult(False, True, (), None)
    stamp = _iso(now)
    row = DraftRow(
        video_id=video_id,
        title=title,
        description=description,
        tags=new_tags,
        base_title=base[0],
        base_description=base[1],
        base_tags=base[2],
        source=source,
        note=note,
        created_at=existing.created_at if existing else stamp,
        updated_at=stamp,
        approved_at=None,
    )
    store.put_draft(row)
    return SaveResult(True, False, issues_for(row), row)


def issues_for(row: DraftRow) -> tuple[Issue, ...]:
    """Validation issues of the draft text, compared with its base."""
    return tuple(validate_draft(row.draft, original=row.base))


def is_stale(row: DraftRow, video: Video) -> bool:
    """True when the catalog text no longer matches the text the draft was made from."""
    return (row.base_title, row.base_description, row.base_tags) != _text_of(video)


def approve(store: Store, video_id: str, now: datetime | None = None) -> DraftRow:
    """Mark a draft as approved by the creator; refused when it is stale or has errors."""
    row = store.get_draft(video_id)
    if row is None:
        raise DraftError("There is no draft for this video.")
    video = store.get_video(video_id)
    if video is None or is_stale(row, video):
        raise DraftError(
            "The video's text changed after this draft was made. "
            "Update the draft to the current text and review it again."
        )
    if has_errors(issues_for(row)):
        raise DraftError("Fix the problems marked as errors before approving this draft.")
    stamp = _iso(now)
    store.set_draft_approval(video_id, stamp)
    return replace(row, approved_at=stamp)


def unapprove(store: Store, video_id: str) -> None:
    store.set_draft_approval(video_id, None)


def discard(store: Store, video_id: str) -> bool:
    """Delete the draft. True when there was one."""
    return store.delete_draft(video_id)


def rebase(store: Store, video_id: str, now: datetime | None = None) -> DraftRow | None:
    """Make the draft's base the catalog's current text; the draft is dropped if it now equals it.

    Returns the updated draft, or None when it was dropped. Approval is cleared either way.
    """
    row = store.get_draft(video_id)
    if row is None:
        raise DraftError("There is no draft for this video.")
    video = store.get_video(video_id)
    if video is None:
        raise DraftError("This video is not in your catalog. Pull your videos first.")
    base = _text_of(video)
    if (row.title, row.description, row.tags) == base:
        store.delete_draft(video_id)
        return None
    new_row = replace(
        row,
        base_title=base[0],
        base_description=base[1],
        base_tags=base[2],
        updated_at=_iso(now),
        approved_at=None,
    )
    store.put_draft(new_row)
    return new_row


def restore_draft(store: Store, push_id: int, now: datetime | None = None) -> SaveResult:
    """Turn the backup of a push (the text from before it) into a draft to review and approve."""
    record = store.get_push(push_id)
    if record is None:
        raise DraftError("This push record does not exist.")
    before = record.before
    if not isinstance(before.get("title"), str) or not before["title"].strip():
        raise DraftError("This push record has no saved text to restore.")
    if store.get_draft(record.video_id) is not None:
        raise DraftError(
            "This video already has a draft. Push or discard it first, then restore this text."
        )
    description = before.get("description")
    tags = before.get("tags")
    return save_draft(
        store,
        record.video_id,
        before["title"],
        description if isinstance(description, str) else "",
        [t for t in tags if isinstance(t, str)] if isinstance(tags, list) else [],
        source="restore",
        note=f"Restore of the text from before push #{push_id}",
        now=now,
    )


def fingerprint(row: DraftRow) -> str:
    """Short hash of the draft text and base; changes whenever either changes."""
    canonical = json.dumps(
        {
            "video_id": row.video_id,
            "title": row.title,
            "description": row.description,
            "tags": list(row.tags),
            "base_title": row.base_title,
            "base_description": row.base_description,
            "base_tags": list(row.base_tags),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def pending(store: Store) -> list[tuple[DraftRow, Video]]:
    """All drafts with their videos, most viewed first."""
    pairs = []
    for row in store.list_drafts():
        video = store.get_video(row.video_id)
        if video is not None:
            pairs.append((row, video))
    pairs.sort(key=lambda pair: (-pair[1].view_count, pair[0].video_id))
    return pairs
