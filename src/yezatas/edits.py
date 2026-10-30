"""Draft lifecycle: save an edit, approve it, rebase it, restore an older text.

Drafts never touch YouTube. A draft remembers its base (the catalog text it was made from)
so the changes page can show a diff and a push can notice that the video changed meanwhile.
A draft can also set the video language and the translations (`localizations`); like the
text, those are only proposals until pushed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Mapping, Sequence

from .store import DraftRow, Store, Video, _iso
from .validate import Issue, has_errors, validate_draft, validate_localizations

# YouTube language codes: "es", "pt-BR", "zh-Hans", "es-419".
_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,4})?$")  # also es-419


class DraftError(ValueError):
    """An edit step was refused; the message is a plain sentence for the creator."""


@dataclass(frozen=True)
class SaveResult:
    saved: bool  # a draft now exists for the video
    discarded: bool  # the draft equals the base, so any draft was removed
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


def _newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def plain_localizations(value: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    """A translations map reduced to {language: {"title", "description"}}; other keys are ignored."""
    plain: dict[str, dict[str, str]] = {}
    for language, loc in (value or {}).items():
        if isinstance(loc, Mapping):
            title, description = loc.get("title"), loc.get("description")
            plain[language] = {
                "title": title if isinstance(title, str) else "",
                "description": description if isinstance(description, str) else "",
            }
    return plain


def _check_language(language: str) -> None:
    if not isinstance(language, str) or not _LANGUAGE_CODE.fullmatch(language):
        raise DraftError(
            f"\"{language}\" is not a language code YouTube accepts (for example en, es or zh-Hans)."
        )


def _video_and_draft(store: Store, video_id: str) -> tuple[Video, DraftRow | None]:
    video = store.get_video(video_id)
    if video is None:
        raise DraftError("This video is not in your catalog. Pull your videos first.")
    return video, store.get_draft(video_id)


def _draft_text(video: Video, existing: DraftRow | None) -> tuple[str, str, tuple[str, ...]]:
    if existing is not None:
        return existing.title, existing.description, existing.tags
    return _text_of(video)


def _store_draft(
    store: Store,
    video: Video,
    existing: DraftRow | None,
    text: tuple[str, str, tuple[str, ...]],
    default_language: str | None,
    localizations: dict[str, dict[str, str]] | None,
    *,
    source: str,
    note: str,
    now: datetime | None,
) -> SaveResult:
    """Write a draft (keeping the base of an existing one), or drop it when nothing differs from the base."""
    if existing is not None:
        base = (existing.base_title, existing.base_description, existing.base_tags)
        base_language = existing.base_default_language
        base_localizations = existing.base_localizations
    else:
        base = _text_of(video)
        base_language = video.default_language
        base_localizations = plain_localizations(video.localizations)
    if (text == base
            and (default_language is None or default_language == base_language)
            and (localizations is None or localizations == base_localizations)):
        store.delete_draft(video.video_id)
        return SaveResult(False, True, (), None)
    stamp = _iso(now)
    row = DraftRow(
        video_id=video.video_id,
        title=text[0],
        description=text[1],
        tags=text[2],
        base_title=base[0],
        base_description=base[1],
        base_tags=base[2],
        source=source,
        note=note,
        created_at=existing.created_at if existing else stamp,
        updated_at=stamp,
        approved_at=None,
        default_language=default_language,
        localizations=localizations,
        base_default_language=base_language,
        base_localizations=base_localizations,
    )
    store.put_draft(row)
    return SaveResult(True, False, issues_for(row), row)


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
    """Save an edit as a draft (even with validation errors) or drop it if nothing changed.

    The draft's language and translations, if it has any, are kept as they are.
    """
    video, existing = _video_and_draft(store, video_id)
    text = (title.strip(), _newlines(description), _normalize_tags(tags))
    return _store_draft(
        store, video, existing, text,
        existing.default_language if existing else None,
        existing.localizations if existing else None,
        source=source, note=note, now=now,
    )


def set_language(
    store: Store,
    video_id: str,
    language: str,
    *,
    source: str = "manual",
    now: datetime | None = None,
) -> SaveResult:
    """Set the language of the video's title and description in the draft (made from the current text)."""
    _check_language(language)
    video, existing = _video_and_draft(store, video_id)
    return _store_draft(
        store, video, existing, _draft_text(video, existing), language,
        existing.localizations if existing else None,
        source=source, note=existing.note if existing else "", now=now,
    )


def _current_localizations(video: Video, existing: DraftRow | None) -> dict[str, dict[str, str]]:
    """A copy of the translations map the draft would send now: its own, else the one it was made from."""
    if existing is None:
        return plain_localizations(video.localizations)
    own = existing.localizations if existing.localizations is not None else existing.base_localizations
    return {language: dict(loc) for language, loc in own.items()}


def add_translations(
    store: Store,
    video_id: str,
    source_language: str,
    translations: Mapping[str, tuple[str, str]],
    *,
    source: str = "translation",
    now: datetime | None = None,
) -> SaveResult:
    """Put translations {language: (title, description)} into the draft, next to the ones it has.

    `source_language` is the language of the text that was translated: it becomes the draft's
    video language unless the draft (or, if it has none, the video) already has it.
    """
    _check_language(source_language)
    for language in translations:
        _check_language(language)
        if language == source_language:
            raise DraftError(
                f"The text is already in {language}, so it cannot also be a translation into {language}."
            )
    video, existing = _video_and_draft(store, video_id)
    if existing is not None:
        language_now = existing.default_language
        current = language_now if language_now is not None else existing.base_default_language
    else:
        language_now = None
        current = video.default_language
    merged = _current_localizations(video, existing)
    for language, (title, description) in translations.items():
        merged[language] = {"title": title.strip(), "description": _newlines(description)}
    return _store_draft(
        store, video, existing, _draft_text(video, existing),
        # "en" text on an "en-GB" video keeps the video's own code
        language_now if current is not None and (
            current == source_language or current.split("-", 1)[0] == source_language)
        else source_language,
        merged,
        source=source, note=existing.note if existing else "", now=now,
    )


def remove_translation(
    store: Store, video_id: str, language: str, *, now: datetime | None = None
) -> SaveResult:
    """Take one language out of the draft's translations (a draft is made if there is none)."""
    video, existing = _video_and_draft(store, video_id)
    merged = _current_localizations(video, existing)
    if language not in merged:
        raise DraftError(f"There is no {language} translation to remove.")
    del merged[language]
    return _store_draft(
        store, video, existing, _draft_text(video, existing),
        existing.default_language if existing else None, merged,
        source=existing.source if existing else "manual",
        note=existing.note if existing else "", now=now,
    )


def _changes_i18n(row: DraftRow) -> bool:
    """True when the draft changes the video language or the translations."""
    return ((row.default_language is not None and row.default_language != row.base_default_language)
            or (row.localizations is not None and row.localizations != row.base_localizations))


def issues_for(row: DraftRow) -> tuple[Issue, ...]:
    """Validation issues of the draft text and translations, compared with their base."""
    issues = validate_draft(row.draft, original=row.base)
    if _changes_i18n(row):  # a translation-only draft is a change, not "nothing changed"
        issues = [issue for issue in issues if issue.code != "no_change"]
    issues.extend(
        validate_localizations(row.default_language, row.localizations, row.base_default_language,
                               row.base_localizations)
    )
    return tuple(issues)


def is_stale(row: DraftRow, video: Video) -> bool:
    """True when the catalog no longer matches the text, language and translations the draft was made from."""
    return (
        (row.base_title, row.base_description, row.base_tags) != _text_of(video)
        or row.base_default_language != video.default_language
        or row.base_localizations != plain_localizations(video.localizations)
    )


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
    """Make the draft's base the catalog's current text, language and translations.

    The draft keeps its own text, language and translations; it is dropped if they now equal the
    base. Returns the updated draft, or None when it was dropped. Approval is cleared either way.
    """
    row = store.get_draft(video_id)
    if row is None:
        raise DraftError("There is no draft for this video.")
    video = store.get_video(video_id)
    if video is None:
        raise DraftError("This video is not in your catalog. Pull your videos first.")
    base = _text_of(video)
    base_localizations = plain_localizations(video.localizations)
    if ((row.title, row.description, row.tags) == base
            and (row.default_language is None or row.default_language == video.default_language)
            and (row.localizations is None or row.localizations == base_localizations)):
        store.delete_draft(video_id)
        return None
    new_row = replace(
        row,
        base_title=base[0],
        base_description=base[1],
        base_tags=base[2],
        base_default_language=video.default_language,
        base_localizations=base_localizations,
        updated_at=_iso(now),
        approved_at=None,
    )
    store.put_draft(new_row)
    return new_row


def restore_draft(store: Store, push_id: int, now: datetime | None = None) -> SaveResult:
    """Turn the backup of a push (the text from before it) into a draft to review and approve.

    A backup that also holds the video language or the translations restores them too (when
    they differ from the video's now); older backups leave them alone.
    """
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
    video, _ = _video_and_draft(store, record.video_id)
    description = before.get("description")
    tags = before.get("tags")
    text = (
        before["title"].strip(),
        _newlines(description) if isinstance(description, str) else "",
        _normalize_tags([t for t in tags if isinstance(t, str)] if isinstance(tags, list) else []),
    )
    language = before.get("defaultLanguage")
    if not isinstance(language, str) or language == video.default_language:
        language = None
    localizations = before.get("localizations")
    localizations = plain_localizations(localizations) if isinstance(localizations, dict) else None
    if localizations == plain_localizations(video.localizations):
        localizations = None
    return _store_draft(
        store, video, None, text, language, localizations,
        source="restore", note=f"Restore of the text from before push #{push_id}", now=now,
    )


def fingerprint(row: DraftRow) -> str:
    """Short hash of the draft text, language, translations and base; changes whenever one changes."""
    canonical = json.dumps(
        {
            "video_id": row.video_id,
            "title": row.title,
            "description": row.description,
            "tags": list(row.tags),
            "base_title": row.base_title,
            "base_description": row.base_description,
            "base_tags": list(row.base_tags),
            "default_language": row.default_language,
            "localizations": row.localizations,
            "base_default_language": row.base_default_language,
            "base_localizations": row.base_localizations,
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
