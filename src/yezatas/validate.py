"""Validator for proposed title, description and tags edits before pushing.

Pure, deterministic Python with no network or YouTube calls. Every proposed
edit must pass `validate_draft` before it can be pushed: "error" issues block
the push, "warning" issues are shown but do not block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

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
    field: str  # "title" | "description" | "tags" | "localizations" | "localizations.<language>"
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
    """One warning per original URL that is missing from the draft.

    An http:// link kept as https:// (the secure address of the same page) is not missing.
    """
    current = set(_urls(draft.description))
    issues, seen = [], set()
    for url in _urls(original.description):
        if url not in seen:
            seen.add(url)
            secure = "https://" + url[len("http://"):] if url.startswith("http://") else url
            if url not in current and secure not in current:
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


def validate_localizations(
    default_language: str | None,
    localizations: Mapping[str, Any] | None,
    base_default_language: str | None,
    base_localizations: Mapping[str, Any] | None = None,
) -> list[Issue]:
    """Errors in a translations map {language: {"title", "description"}}, in stable order.

    The video language is the draft's own, else the one the draft was made from. Translations
    that are exactly as on YouTube (`base_localizations`) are not judged: YouTube already took
    them, and Studio even lists the video's own language among them.
    """
    if not localizations:
        return []
    language = default_language or base_default_language
    issues = []
    if not language:
        issues.append(Issue("localizations", "localizations_language_missing", "error",
                            "Set the video language first; YouTube only takes translations "
                            "for a video with a language."))
    for lang in sorted(localizations):
        field = f"localizations.{lang}"
        entry = localizations[lang]
        entry = entry if isinstance(entry, Mapping) else {}
        title = entry.get("title") if isinstance(entry.get("title"), str) else ""
        description = entry.get("description") if isinstance(entry.get("description"), str) else ""
        kept = (base_localizations or {}).get(lang)
        if (isinstance(kept, Mapping) and kept.get("title") == title
                and kept.get("description") == description):
            continue
        if lang == language:
            issues.append(Issue(field, "localization_is_default", "error",
                                f"The video language is {lang}, so it cannot also have a "
                                f"{lang} translation."))
        if not title.strip():
            issues.append(Issue(field, "localization_title_empty", "error",
                                f"The {lang} translation has an empty title."))
        if len(title) > TITLE_LIMIT:
            issues.append(Issue(field, "localization_title_too_long", "error",
                                f"The {lang} title is {len(title)} characters; "
                                f"YouTube allows {TITLE_LIMIT}."))
        size = len(description.encode("utf-8"))
        if size > DESCRIPTION_BYTE_LIMIT:
            issues.append(Issue(field, "localization_description_too_long", "error",
                                f"The {lang} description is {size} bytes; "
                                f"YouTube allows {DESCRIPTION_BYTE_LIMIT:,} bytes."))
        if "<" in title or ">" in title:
            issues.append(Issue(field, "localization_title_angle_brackets", "error",
                                f"The {lang} title contains < or >; YouTube rejects them."))
        if "<" in description or ">" in description:
            issues.append(Issue(field, "localization_description_angle_brackets", "error",
                                f"The {lang} description contains < or >; YouTube rejects them."))
    return issues


def has_errors(issues: Iterable[Issue]) -> bool:
    """True when at least one issue blocks the push."""
    return any(issue.severity == "error" for issue in issues)
