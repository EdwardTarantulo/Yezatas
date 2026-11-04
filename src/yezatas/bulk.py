"""Bulk edits without a model: find and replace, or a footer block, over many videos at once.

Only drafts are made. `plan_bulk` is a pure read and writes nothing, so the creator can preview
every change. `apply_bulk` plans again (the catalog may have changed since the preview) and
drafts only the approved ids that are still in the plan and not blocked. Nothing here talks to
YouTube or to a model; the drafts go through the normal review, approval and push steps.

A video that already has a draft is blocked, so a bulk edit never mixes with another edit. A new
text with an error-level validation issue is blocked too; warnings are shown but do not block.
Matching is plain text and case-sensitive (no regex). Line breaks are compared as "\\n", the way
drafts store them.

The step helpers at the end (`Step`, `step_from_dict`, `apply_step`) are the small edits the bulk edit
agent (`bulkagent.py`) applies by code to many videos; they are pure and need no model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence

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

# The steps of the bulk edit agent: small edits that code applies the same way to many videos.
STEP_KINDS = ("replace", "footer", "insert_after_first_paragraph", "remove_line", "add_first_line",
              "title_prefix", "title_suffix", "add_tags", "remove_tags")
_STEP_KEYS: dict[str, tuple[str, ...]] = {
    "replace": ("find", "replace_with", "fields"),
    "footer": ("footer", "marker"),
    "insert_after_first_paragraph": ("text",),
    "remove_line": ("contains",),
    "add_first_line": ("text",),
    "title_prefix": ("text",),
    "title_suffix": ("text",),
    "add_tags": ("tags",),
    "remove_tags": ("tags",),
}
LINE_LIMIT = 200  # one added line
TITLE_TEXT_LIMIT = 100
TAGS_PER_STEP = 50


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


# --- steps: one small edit applied the same way to many videos --------------------------------------
@dataclass(frozen=True)
class Step:
    """One edit by code. Which fields matter depends on `kind` (see `STEP_KINDS`)."""

    kind: str
    find: str = ""  # replace
    replace_with: str = ""  # replace
    fields: tuple[str, ...] = ("description",)  # replace
    footer: str = ""  # footer
    marker: str = ""  # footer
    text: str = ""  # insert_after_first_paragraph, add_first_line, title_prefix, title_suffix
    contains: str = ""  # remove_line
    tags: tuple[str, ...] = ()  # add_tags, remove_tags


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == ()


def _string(data: dict[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"The {label} must be text.")
    return _newlines(value)


def step_from_dict(data: Any) -> Step:
    """A step from decoded JSON, checked; raises ValueError with a plain sentence."""
    if not isinstance(data, dict):
        raise ValueError('A step must be an object with a "kind".')
    kind = data.get("kind")
    if not isinstance(kind, str) or kind not in STEP_KINDS:
        raise ValueError(f'A step needs a "kind", one of: {", ".join(STEP_KINDS)}.')
    stray = sorted(k for k, v in data.items()
                   if k != "kind" and k not in _STEP_KEYS[kind] and not _empty(v))
    if stray:
        raise ValueError(f'The step "{kind}" does not take: {", ".join(stray)}. '
                         f'It takes: {", ".join(_STEP_KEYS[kind])}.')
    if kind == "replace":
        find = _string(data, "find", "text to find")
        replace_with = _string(data, "replace_with", "replacement")
        if not find.strip():
            raise ValueError("Enter the text to find.")
        raw_fields = data.get("fields") or ["description"]
        if (not isinstance(raw_fields, (list, tuple)) or not all(isinstance(f, str) for f in raw_fields)
                or any(f not in FIELDS for f in raw_fields)):
            raise ValueError("Choose the title, the description or both to change.")
        fields = tuple(f for f in FIELDS if f in raw_fields)
        check_spec(BulkSpec(kind="replace", find=find, replace_with=replace_with, fields=fields))
        return Step("replace", find=find, replace_with=replace_with, fields=fields)
    if kind == "footer":
        footer = _string(data, "footer", "footer").strip()
        marker = _string(data, "marker", "footer marker").strip()
        check_spec(BulkSpec(kind="footer", footer=footer, footer_marker=marker))
        return Step("footer", footer=footer, marker=marker)
    if kind == "remove_line":
        contains = _string(data, "contains", "text in the line to remove").strip()
        if not contains:
            raise ValueError("Enter the text in the line to remove.")
        if len(contains) > FIND_LIMIT:
            raise ValueError(f"The text in the line to remove is {len(contains)} characters; "
                             f"the limit is {FIND_LIMIT}.")
        return Step("remove_line", contains=contains)
    if kind in ("add_tags", "remove_tags"):
        raw_tags = data.get("tags")
        if (not isinstance(raw_tags, (list, tuple)) or not raw_tags
                or not all(isinstance(t, str) for t in raw_tags)):
            raise ValueError('"tags" must be a list of text with at least one tag.')
        if len(raw_tags) > TAGS_PER_STEP:
            raise ValueError(f"At most {TAGS_PER_STEP} tags per step.")
        tags = tuple(dict.fromkeys(t.strip() for t in raw_tags if t.strip()))
        if not tags:
            raise ValueError('"tags" must be a list of text with at least one tag.')
        if kind == "add_tags" and any(c in t for t in tags for c in "<>,"):
            raise ValueError("A tag cannot contain <, > or a comma.")
        return Step(kind, tags=tags)
    text = _string(data, "text", "text to add").strip()
    if not text:
        raise ValueError("Enter the text to add.")
    if kind in ("add_first_line", "title_prefix", "title_suffix") and "\n" in text:
        raise ValueError("This text must be one line.")
    limit = {"add_first_line": LINE_LIMIT, "title_prefix": TITLE_TEXT_LIMIT,
             "title_suffix": TITLE_TEXT_LIMIT}.get(kind, FOOTER_LIMIT)
    if len(text) > limit:
        raise ValueError(f"The text is {len(text)} characters; the limit is {limit}.")
    return Step(kind, text=text)


def apply_step(step: Step, title: str, description: str, tags: Sequence[str]
               ) -> tuple[str, str, tuple[str, ...]]:
    """The title, description and tags after one step. Pure, and a step that already took effect
    changes nothing."""
    description = _newlines(description)
    kept = tuple(tags)
    kind = step.kind
    if kind == "replace":
        find, replacement = _newlines(step.find), _newlines(step.replace_with)
        if "title" in step.fields and find in title:
            title = title.replace(find, replacement).strip()
        if "description" in step.fields and find in description:
            description = description.replace(find, replacement)
    elif kind == "footer":
        new = _footer_text(description, step.footer, step.marker)
        if new.rstrip() != description.rstrip():
            description = new
    elif kind == "insert_after_first_paragraph":
        if step.text not in description:
            index = description.find("\n\n")
            if index >= 0:
                description = description[:index] + "\n\n" + step.text + description[index:]
            else:
                body = description.rstrip()
                description = (body + "\n\n" + step.text) if body else step.text
    elif kind == "remove_line":
        lines = description.split("\n")
        left = [line for line in lines if step.contains not in line]
        if left != lines:
            collapsed: list[str] = []
            for line in left:
                if line.strip() == "" and collapsed and collapsed[-1].strip() == "":
                    continue  # one blank line where the removed line was
                collapsed.append(line if line.strip() else "")
            description = "\n".join(collapsed)
    elif kind == "add_first_line":
        if description.split("\n", 1)[0].strip() != step.text:
            description = step.text + "\n" + description if description else step.text
    elif kind == "title_prefix":
        if not title.startswith(step.text):
            title = f"{step.text} {title}".strip()
    elif kind == "title_suffix":
        if not title.endswith(step.text):
            title = f"{title} {step.text}".strip()
    elif kind == "add_tags":
        have = {t.casefold() for t in kept}
        kept += tuple(t for t in step.tags if t.casefold() not in have)
    elif kind == "remove_tags":
        gone = {t.casefold() for t in step.tags}
        kept = tuple(t for t in kept if t.casefold() not in gone)
    else:
        raise ValueError(f'Unknown step "{kind}".')
    return title, description, kept
