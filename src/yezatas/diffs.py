"""Word, line and tag diffs for the draft review screen.

Pure, deterministic Python with no I/O. The review screen shows a draft
against the current YouTube text before the creator approves the push.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from .validate import Draft

if TYPE_CHECKING:
    from .store import DraftRow

_WORD = re.compile(r"\S+|\s+")


@dataclass(frozen=True)
class Segment:
    kind: str  # "same" | "added" | "removed"
    text: str


@dataclass(frozen=True)
class Line:
    kind: str  # "same" | "added" | "removed" | "gap"
    text: str  # for "gap": "1 unchanged line" or "N unchanged lines"
    old_no: int | None  # 1-based line number in the old text (None for added and gap)
    new_no: int | None  # 1-based line number in the new text (None for removed and gap)


@dataclass(frozen=True)
class TextDiff:
    changed: bool
    lines: tuple[Line, ...]
    added: int  # number of added lines
    removed: int  # number of removed lines


@dataclass(frozen=True)
class TagsDiff:
    changed: bool  # any difference, including order
    kept: tuple[str, ...]
    added: tuple[str, ...]
    removed: tuple[str, ...]
    reordered: bool  # same tags (same multiset) in a different order


@dataclass(frozen=True)
class LocalizationsDiff:
    changed: bool  # any language added, removed or with a different title or description
    added: tuple[str, ...]  # languages, sorted
    removed: tuple[str, ...]
    changed_langs: tuple[str, ...]
    kept: tuple[str, ...]  # in both and identical


@dataclass(frozen=True)
class DraftDiff:
    title: tuple[Segment, ...]  # word-level diff of the title
    title_changed: bool
    description: TextDiff
    tags: TagsDiff
    changed: bool  # title_changed or description.changed or tags.changed (and, from diff_row, the rest)
    language: tuple[str | None, str | None] | None = None  # (old, new) when the draft changes it
    localizations: LocalizationsDiff = field(
        default_factory=lambda: LocalizationsDiff(False, (), (), (), ())
    )


def _merge(segments: list[Segment]) -> tuple[Segment, ...]:
    """Merge neighbouring segments of the same kind."""
    merged: list[Segment] = []
    for segment in segments:
        if merged and merged[-1].kind == segment.kind:
            merged[-1] = Segment(segment.kind, merged[-1].text + segment.text)
        else:
            merged.append(segment)
    return tuple(merged)


def diff_words(old: str, new: str) -> tuple[Segment, ...]:
    """Word-level diff keeping whitespace so both sides round-trip."""
    if old == new:
        return () if not old else (Segment("same", old),)
    old_tokens = _WORD.findall(old)
    new_tokens = _WORD.findall(new)
    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens)
    segments: list[Segment] = []
    for tag, a1, a2, b1, b2 in matcher.get_opcodes():
        if tag == "equal":
            segments.append(Segment("same", "".join(old_tokens[a1:a2])))
        elif tag == "delete":
            segments.append(Segment("removed", "".join(old_tokens[a1:a2])))
        elif tag == "insert":
            segments.append(Segment("added", "".join(new_tokens[b1:b2])))
        else:
            segments.append(Segment("removed", "".join(old_tokens[a1:a2])))
            segments.append(Segment("added", "".join(new_tokens[b1:b2])))
    merged = _merge([s for s in segments if s.text])
    # Mostly rewritten: old words interleaved with new ones are hard to read, so show the old
    # text as one removed run followed by the new text.
    same = sum(len(s.text) for s in merged if s.kind == "same" and s.text.strip())
    if 2 * same < max(len(old), len(new)):
        return tuple(
            Segment(kind, text) for kind, text in (("removed", old), ("added", new)) if text
        )
    return merged


def _gap_text(hidden: int) -> str:
    """Gap label for the number of hidden unchanged lines."""
    return "1 unchanged line" if hidden == 1 else f"{hidden} unchanged lines"


def diff_text(old: str, new: str, context: int = 2) -> TextDiff:
    """Line diff with context and gaps for long unchanged runs."""
    old_lines = [] if old == "" else old.splitlines()
    new_lines = [] if new == "" else new.splitlines()
    opcodes = difflib.SequenceMatcher(None, old_lines, new_lines).get_opcodes()
    if all(tag == "equal" for tag, *_ in opcodes):
        return TextDiff(False, (), 0, 0)
    added = sum(b2 - b1 for tag, _, _, b1, b2 in opcodes if tag in ("insert", "replace"))
    removed = sum(a2 - a1 for tag, a1, a2, *_ in opcodes if tag in ("delete", "replace"))
    out: list[Line] = []
    last = len(opcodes) - 1
    for index, (tag, a1, a2, b1, b2) in enumerate(opcodes):
        if tag == "equal":
            size = a2 - a1
            if index == 0:
                keep = min(size, context)
                hidden = size - keep
                if hidden:
                    out.append(Line("gap", _gap_text(hidden), None, None))
                for k in range(size - keep, size):
                    out.append(Line("same", old_lines[a1 + k], a1 + k + 1, b1 + k + 1))
            elif index == last:
                keep = min(size, context)
                for k in range(keep):
                    out.append(Line("same", old_lines[a1 + k], a1 + k + 1, b1 + k + 1))
                if size - keep:
                    out.append(Line("gap", _gap_text(size - keep), None, None))
            elif size > 2 * context:
                for k in range(context):
                    out.append(Line("same", old_lines[a1 + k], a1 + k + 1, b1 + k + 1))
                out.append(Line("gap", _gap_text(size - 2 * context), None, None))
                for k in range(size - context, size):
                    out.append(Line("same", old_lines[a1 + k], a1 + k + 1, b1 + k + 1))
            else:
                for k in range(size):
                    out.append(Line("same", old_lines[a1 + k], a1 + k + 1, b1 + k + 1))
        elif tag == "delete":
            for k in range(a1, a2):
                out.append(Line("removed", old_lines[k], k + 1, None))
        elif tag == "insert":
            for k in range(b1, b2):
                out.append(Line("added", new_lines[k], None, k + 1))
        else:
            for k in range(a1, a2):
                out.append(Line("removed", old_lines[k], k + 1, None))
            for k in range(b1, b2):
                out.append(Line("added", new_lines[k], None, k + 1))
    return TextDiff(True, tuple(out), added, removed)


def diff_tags(old: Sequence[str], new: Sequence[str]) -> TagsDiff:
    """Tag diff as a multiset; kept/added follow the new order."""
    old_list = list(old)
    new_list = list(new)
    remaining = Counter(old_list)
    kept: list[str] = []
    added: list[str] = []
    for tag in new_list:
        if remaining[tag] > 0:
            kept.append(tag)
            remaining[tag] -= 1
        else:
            added.append(tag)
    kept_remaining = Counter(kept)
    removed: list[str] = []
    for tag in old_list:
        if kept_remaining[tag] > 0:
            kept_remaining[tag] -= 1
        else:
            removed.append(tag)
    changed = old_list != new_list
    reordered = changed and Counter(old_list) == Counter(new_list)
    return TagsDiff(changed, tuple(kept), tuple(added), tuple(removed), reordered)


def diff_draft(original: Draft, draft: Draft) -> DraftDiff:
    """Diff a whole draft: title words, description lines, tags."""
    title = diff_words(original.title, draft.title)
    title_changed = original.title != draft.title
    description = diff_text(original.description, draft.description)
    tags = diff_tags(original.tags, draft.tags)
    changed = title_changed or description.changed or tags.changed
    return DraftDiff(title, title_changed, description, tags, changed)


def _pair(entry: object) -> tuple[str, str]:
    """(title, description) of one translation; other keys do not count."""
    if not isinstance(entry, Mapping):
        return "", ""
    title, description = entry.get("title"), entry.get("description")
    return (title if isinstance(title, str) else "", description if isinstance(description, str) else "")


def diff_localizations(base: Mapping, new: Mapping | None) -> LocalizationsDiff:
    """Which languages a translations map adds, removes or changes; None (not set) changes nothing."""
    if new is None:
        return LocalizationsDiff(False, (), (), (), tuple(sorted(base)))
    added = tuple(sorted(lang for lang in new if lang not in base))
    removed = tuple(sorted(lang for lang in base if lang not in new))
    both = sorted(lang for lang in new if lang in base)
    changed_langs = tuple(lang for lang in both if _pair(base[lang]) != _pair(new[lang]))
    kept = tuple(lang for lang in both if _pair(base[lang]) == _pair(new[lang]))
    return LocalizationsDiff(bool(added or removed or changed_langs), added, removed, changed_langs, kept)


def diff_row(row: "DraftRow") -> DraftDiff:
    """Diff a stored draft against its base: text, language and translations."""
    diff = diff_draft(row.base, row.draft)
    language = None
    if row.default_language is not None and row.default_language != row.base_default_language:
        language = (row.base_default_language, row.default_language)
    localizations = diff_localizations(row.base_localizations, row.localizations)
    return replace(
        diff,
        language=language,
        localizations=localizations,
        changed=diff.changed or language is not None or localizations.changed,
    )
