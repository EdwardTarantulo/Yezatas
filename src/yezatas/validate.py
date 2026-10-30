"""Validator for proposed title, description and tags edits before pushing.

Pure, deterministic Python with no network or YouTube calls. Every proposed
edit must pass `validate_draft` before it can be pushed: "error" issues block
the push, "warning" issues are shown but do not block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .seo import _BROKEN_LINK, _HASHTAG, _URL, parse_chapters, tags_char_count

TITLE_LIMIT = 100
DESCRIPTION_BYTE_LIMIT = 5000
HASHTAGS_LIMIT = 60
TAGS_CHAR_LIMIT = 500
CHAPTERS_MIN_COUNT = 3
CHAPTER_MIN_GAP = 10


@dataclass(frozen=True)
class Draft:
    title: str
    description: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Issue:
    field: str  # "title" | "description" | "tags"
    code: str  # stable id, e.g. "title_empty"
    severity: str  # "error" blocks the push, "warning" does not
    message: str  # one plain sentence for the creator


def _chapters_problem(stamps: list[int]) -> str:
    """Why the timestamps are not valid chapters, or "" when they are valid."""
    if len(stamps) < CHAPTERS_MIN_COUNT:
        return f"Found {len(stamps)} timestamps; chapters need at least {CHAPTERS_MIN_COUNT}."
    if stamps[0] != 0:
        return "The first timestamp is not 0:00."
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        return "The timestamps must go up in time."
    if any(b - a < CHAPTER_MIN_GAP for a, b in zip(stamps, stamps[1:])):
        return (f"Each chapter must start at least {CHAPTER_MIN_GAP} seconds "
                "after the previous one.")
    return ""


def _check_title(draft: Draft) -> list[Issue]:
    """Title errors, in stable order."""
    issues = []
    if not draft.title.strip():
        issues.append(Issue("title", "title_empty", "error", "The title is empty."))
    if len(draft.title) > TITLE_LIMIT:
        issues.append(Issue("title", "title_too_long", "error",
                            f"The title is {len(draft.title)} characters; "
                            f"YouTube allows {TITLE_LIMIT}."))
    if "<" in draft.title or ">" in draft.title:
        issues.append(Issue("title", "title_angle_brackets", "error",
                            "The title contains < or >; YouTube rejects them."))
    return issues


def _check_description(draft: Draft) -> list[Issue]:
    """Description errors then the chapters warning, in stable order."""
    issues = []
    size = len(draft.description.encode("utf-8"))
    if size > DESCRIPTION_BYTE_LIMIT:
        issues.append(Issue("description", "description_too_long", "error",
                            f"The description is {size} bytes; "
                            f"YouTube allows {DESCRIPTION_BYTE_LIMIT:,} bytes."))
    if "<" in draft.description or ">" in draft.description:
        issues.append(Issue("description", "description_angle_brackets", "error",
                            "The description contains < or >; YouTube rejects them."))
    match = _BROKEN_LINK.search(draft.description)
    if match:
        issues.append(Issue("description", "description_placeholder_link", "error",
                            f"The description has a placeholder link "
                            f"\"{match.group(0).strip()}\"; replace it with the real "
                            "link or delete it."))
    count = len(_HASHTAG.findall(draft.description))
    if count > HASHTAGS_LIMIT:
        issues.append(Issue("description", "description_too_many_hashtags", "error",
                            f"The description has {count} hashtags; YouTube ignores all "
                            f"of them when there are more than {HASHTAGS_LIMIT}."))
    stamps = parse_chapters(draft.description)
    if len(stamps) >= 2:
        problem = _chapters_problem(stamps)
        if problem:
            issues.append(Issue("description", "chapters_invalid", "warning", problem))
    return issues


def _check_tags(draft: Draft) -> list[Issue]:
    """Tag errors, in stable order: the total first, then one per bad tag."""
    issues = []
    count = tags_char_count(draft.tags)
    if count > TAGS_CHAR_LIMIT:
        issues.append(Issue("tags", "tags_too_long", "error",
                            f"The tags use {count} of {TAGS_CHAR_LIMIT} characters; "
                            "remove some tags to fit."))
    for tag in draft.tags:
        if not tag.strip():
            issues.append(Issue("tags", "tag_invalid", "error",
                                "One tag is empty; delete it."))
        elif "<" in tag or ">" in tag or "," in tag:
            issues.append(Issue("tags", "tag_invalid", "error",
                                f"Tag \"{tag}\" contains <, > or a comma; "
                                "remove those characters."))
    return issues


def _urls(text: str) -> list[str]:
    """URLs in order, without sentence punctuation stuck to the end ("(see x.com/a)." -> x.com/a)."""
    return [url.rstrip(".,;:!?)]}'\"") for url in _URL.findall(text)]


def _check_links_removed(draft: Draft, original: Draft) -> list[Issue]:
    """One warning per original URL that is missing from the draft."""
    current = set(_urls(draft.description))
    issues, seen = [], set()
    for url in _urls(original.description):
        if url not in seen:
            seen.add(url)
            if url not in current:
                issues.append(Issue("description", "link_removed", "warning",
                                    f"The link \"{url}\" was in the original description "
                                    "but is missing now."))
    return issues


def validate_draft(draft: Draft, original: Draft | None = None) -> list[Issue]:
    """Check a proposed edit; the original-only checks run when given."""
    title_issues = _check_title(draft)
    description_issues = _check_description(draft)
    tag_issues = _check_tags(draft)
    if original is not None:
        description_issues.extend(_check_links_removed(draft, original))
        if (not _chapters_problem(parse_chapters(original.description))
                and parse_chapters(draft.description)
                != parse_chapters(original.description)):
            description_issues.append(Issue("description", "chapters_changed", "warning",
                                            "The chapter timestamps differ from the "
                                            "original; make sure they still match "
                                            "the video."))
        if (draft.title == original.title
                and draft.description == original.description
                and draft.tags == original.tags):
            title_issues.append(Issue("title", "no_change", "warning",
                                      "Nothing changed; the title, description and tags "
                                      "are identical to the original."))
    return title_issues + description_issues + tag_issues


def has_errors(issues: Iterable[Issue]) -> bool:
    """True when at least one issue blocks the push."""
    return any(issue.severity == "error" for issue in issues)
