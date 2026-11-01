"""Bulk edits without a model: find and replace, or a footer block, over many videos at once.

Only drafts are made. `plan_bulk` is a pure read and writes nothing, so the creator can preview
every change. `apply_bulk` plans again (the catalog may have changed since the preview) and
drafts only the approved ids that are still in the plan and not blocked. Nothing here talks to
YouTube or to a model; the drafts go through the normal review, approval and push steps.

A video that already has a draft is blocked, so a bulk edit never mixes with another edit. A new
text with an error-level validation issue is blocked too; warnings are shown but do not block.
Matching is plain text and case-sensitive (no regex). Line breaks are compared as "\\n", the way
drafts store them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

from .edits import DraftError, save_draft
from .store import Store, Video
from .validate import Draft, Issue, find_urls, has_errors, validate_draft

KINDS = ("replace", "footer")
FIELDS = ("title", "description")
VISIBILITIES = ("public", "all")
FIND_LIMIT = 500
FOOTER_LIMIT = 1500
MARKER_LIMIT = 500
MAX_BULK = 500  # videos per apply

BLOCKED_HAS_DRAFT = "This video already has a draft: push or discard it first."


@dataclass(frozen=True)
class BulkSpec:
    kind: str  # "replace" | "footer"
    find: str = ""  # replace: the exact text to find (case-sensitive, plain text, no regex)
    replace_with: str = ""  # replace: may be "" to delete
    fields: tuple[str, ...] = ("description",)  # replace: any of "title", "description"
    footer: str = ""  # footer: the block to put at the end of the description
    footer_marker: str = ""  # footer: optional first line of an existing block to replace from
    visibility: str = "public"  # "public" | "all"
    published_from: str = ""  # optional ISO date (inclusive), compared with published_at
    published_to: str = ""  # optional ISO date (inclusive)
    video_ids: tuple[str, ...] = ()  # optional explicit subset (from the preview's checkboxes)


@dataclass(frozen=True)
class BulkChange:
    video_id: str
    title: str  # the new title (unchanged when only the description changes)
    description: str  # the new description
    count: int  # replacements made (replace) or 1 (footer)
    issues: tuple[Issue, ...]  # validate_draft(new, original) issues
    blocked: str  # "" or why it can't be drafted


@dataclass(frozen=True)
class BulkItem:
    video_id: str
    status: str  # "drafted" | "blocked" | "not_in_plan"
    message: str


@dataclass(frozen=True)
class BulkResult:
    drafted: int
    blocked: int
    not_in_plan: int
    items: tuple[BulkItem, ...]  # one per approved id, in the order given (duplicates removed)


def _newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _parse_date(value: str, label: str) -> str:
    """The ISO date (YYYY-MM-DD) of `value`, or "" when empty."""
    value = value.strip()
    if not value:
        return ""
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError:
        raise ValueError(f"The {label} date \"{value}\" is not a date like 2026-01-31.") from None


def _check_spec(spec: BulkSpec) -> tuple[str, str, str]:
    """Refuse a spec that cannot be run; returns the cleaned (published_from, published_to, footer)."""
    if spec.kind not in KINDS:
        raise ValueError("Choose what to do: find and replace, or a footer block.")
    if spec.visibility not in VISIBILITIES:
        raise ValueError("Choose which videos to change: public only, or all.")
    start = _parse_date(spec.published_from, "published from")
    end = _parse_date(spec.published_to, "published to")
    if start and end and start > end:
        raise ValueError("The \"published from\" date is after the \"published to\" date.")
    footer = ""
    if spec.kind == "replace":
        if not spec.fields or any(f not in FIELDS for f in spec.fields):
            raise ValueError("Choose the title, the description or both to change.")
        if not spec.find:
            raise ValueError("Enter the text to find.")
        if len(spec.find) > FIND_LIMIT:
            raise ValueError(f"The text to find is {len(spec.find)} characters; the limit is {FIND_LIMIT}.")
        if spec.find == spec.replace_with:
            raise ValueError("The text to find and the replacement are the same, so nothing would change.")
        if "title" in spec.fields and ("\n" in spec.replace_with or "\r" in spec.replace_with):
            raise ValueError("A title cannot contain a line break.")
    else:
        footer = _newlines(spec.footer).strip()
        if not footer:
            raise ValueError("Enter the footer text.")
        if len(footer) > FOOTER_LIMIT:
            raise ValueError(f"The footer is {len(footer)} characters; the limit is {FOOTER_LIMIT}.")
        if "\n" in spec.footer_marker.strip() or len(spec.footer_marker) > MARKER_LIMIT:
            raise ValueError(
                f"The marker must be one line of at most {MARKER_LIMIT} characters.")
    return start, end, footer


check_spec = _check_spec


def _in_scope(video: Video, spec: BulkSpec, start: str, end: str, ids: frozenset[str]) -> bool:
    if ids and video.video_id not in ids:
        return False
    if spec.visibility == "public" and video.privacy_status != "public":
        return False
    if start or end:
        day = video.published_at[:10]
        if not day or (start and day < start) or (end and day > end):
            return False
    return True


def _footer_text(description: str, footer: str, marker: str) -> str:
    """The description with the footer at its end (replacing the block that starts at the last marker line)."""
    marker = marker.strip()
    if marker:
        lines = description.split("\n")
        for index in range(len(lines) - 1, -1, -1):
            if lines[index].strip() == marker:
                return "\n".join(lines[:index] + [footer])
    body = description.rstrip()
    if body == footer or body.endswith("\n" + footer):
        return description
    return f"{body}\n\n{footer}" if body else footer


footer_text = _footer_text


def _changed(video: Video, spec: BulkSpec, footer: str) -> tuple[str, str, int] | None:
    """(new title, new description, count) for the video, or None when the edit would not touch it."""
    title, description = video.title, _newlines(video.description)
    if spec.kind == "footer":
        new = _footer_text(description, footer, spec.footer_marker)
        if new.rstrip() == description.rstrip():
            return None
        return title, new, 1
    find, replacement = _newlines(spec.find), _newlines(spec.replace_with)
    count = 0
    if "title" in spec.fields and find in title:
        count += title.count(find)
        title = title.replace(find, replacement).strip()
    if "description" in spec.fields and find in description:
        count += description.count(find)
        description = description.replace(find, replacement)
    return (title, description, count) if count else None


def _asked_links(spec: BulkSpec, original: str, new: str) -> set[str]:
    """Original links that the find & replace turned into another link, as the creator asked."""
    if spec.kind != "replace":
        return set()
    find, replacement = _newlines(spec.find), _newlines(spec.replace_with)
    new_urls = set(find_urls(new))
    return {url for url in find_urls(original)
            if find in url and url.replace(find, replacement) in new_urls}


def plan_bulk(store: Store, spec: BulkSpec) -> list[BulkChange]:
    """What the edit would do to each video it touches (most viewed first). Writes nothing."""
    start, end, footer = _check_spec(spec)
    ids = frozenset(spec.video_ids)
    plan: list[BulkChange] = []
    for video in store.list_videos():
        if not _in_scope(video, spec, start, end, ids):
            continue
        result = _changed(video, spec, footer)
        if result is None:
            continue
        title, description, count = result
        asked = _asked_links(spec, video.description, description)
        issues = tuple(
            i for i in validate_draft(
                Draft(title, description, tuple(video.tags)),
                original=Draft(video.title, video.description, tuple(video.tags)),
            )
            if not (i.code == "link_removed" and any(f'"{url}"' in i.message for url in asked))
        )
        if store.get_draft(video.video_id) is not None:
            blocked = BLOCKED_HAS_DRAFT
        elif has_errors(issues):
            blocked = " ".join(i.message for i in issues if i.severity == "error")
        else:
            blocked = ""
        plan.append(BulkChange(video.video_id, title, description, count, issues, blocked))
    return plan


def _quote(text: str) -> str:
    text = text.replace("\r", "").replace("\n", "\\n")
    return '"' + (text if len(text) <= 60 else text[:57] + "...") + '"'


def describe(spec: BulkSpec) -> str:
    """One line for the draft's note."""
    if spec.kind == "footer":
        return "Bulk: footer block"
    where = "" if spec.fields == ("description",) else " in " + " and ".join(
        f for f in FIELDS if f in spec.fields)
    if spec.replace_with == "":
        return f"Bulk: remove {_quote(spec.find)}{where}"
    return f"Bulk: replace {_quote(spec.find)} with {_quote(spec.replace_with)}{where}"


def apply_bulk(
    store: Store, spec: BulkSpec, approved_ids: Sequence[str], now: datetime | None = None
) -> BulkResult:
    """Make drafts for the approved ids that are still in a fresh plan and not blocked."""
    ids = list(dict.fromkeys(approved_ids))
    if len(ids) > MAX_BULK:
        raise ValueError(
            f"That is {len(ids)} videos; at most {MAX_BULK} can be changed at once. "
            "Narrow the dates or choose fewer videos."
        )
    plan = {change.video_id: change for change in plan_bulk(store, spec)}
    note = describe(spec)
    items: list[BulkItem] = []
    for video_id in ids:
        change = plan.get(video_id)
        if change is None:
            items.append(BulkItem(video_id, "not_in_plan",
                                  "This edit no longer applies to this video."))
            continue
        if change.blocked:
            items.append(BulkItem(video_id, "blocked", change.blocked))
            continue
        video = store.get_video(video_id)
        try:
            if video is None:
                raise DraftError("This video is not in your catalog. Pull your videos first.")
            saved = save_draft(store, video_id, change.title, change.description, video.tags,
                               source="bulk", note=note, now=now)
        except DraftError as error:
            items.append(BulkItem(video_id, "blocked", str(error)))
            continue
        if saved.saved:
            items.append(BulkItem(video_id, "drafted", "Draft made."))
        else:
            items.append(BulkItem(video_id, "blocked", "This edit changes nothing."))
    return BulkResult(
        drafted=sum(i.status == "drafted" for i in items),
        blocked=sum(i.status == "blocked" for i in items),
        not_in_plan=sum(i.status == "not_in_plan" for i in items),
        items=tuple(items),
    )
