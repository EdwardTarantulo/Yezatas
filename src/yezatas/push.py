"""Push approved drafts to YouTube, with a backup first and a check for Studio edits.

Only drafts the creator approved, and whose text is exactly what they reviewed, are pushed.
For each one the live text is fetched first: if it differs from the draft's base the video was
changed elsewhere (for example in YouTube Studio), so nothing is pushed and the catalog is
refreshed instead. Otherwise the live text is saved as a backup before videos.update is called.
A draft that sets the video language or the translations also sends them (the complete
translations map: YouTube replaces the whole map), and is checked against the live ones too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping

from .edits import fingerprint, is_stale, issues_for, plain_localizations
from .quota import QuotaExceeded, QuotaGuard
from .store import DraftRow, Store
from .validate import has_errors
from .youtube import MAX_RESULTS, YouTubeWriter, api_error_info

# videos.update deletes any writable snippet field that is left out, so these are copied over.
KEEP_SNIPPET_KEYS: tuple[str, ...] = ("categoryId", "defaultLanguage", "defaultAudioLanguage")

_STOP_REASONS = frozenset(
    {"quotaExceeded", "dailyLimitExceeded", "insufficientPermissions", "forbidden"}
)


@dataclass(frozen=True)
class PushItem:
    video_id: str
    status: str  # pushed | conflict | failed | skipped | missing | not_attempted
    message: str
    push_id: int | None = None


@dataclass(frozen=True)
class PushReport:
    items: tuple[PushItem, ...]
    units_used: int

    def count(self, status: str) -> int:
        return sum(1 for item in self.items if item.status == status)


def backup_of(
    live_snippet: dict[str, Any], live_localizations: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The writable part of YouTube's live snippet (and translations, when given): what a restore needs."""
    backup: dict[str, Any] = {
        "title": live_snippet.get("title", ""),
        "description": live_snippet.get("description", ""),
        "tags": list(live_snippet.get("tags") or []),
    }
    for key in KEEP_SNIPPET_KEYS:
        if key in live_snippet:
            backup[key] = live_snippet[key]
    if live_localizations is not None:
        backup["localizations"] = plain_localizations(live_localizations)
    return backup


def push_units(n: int) -> list[tuple[str, int]]:
    """The quota plan for pushing n drafts: batched live reads plus one update each."""
    if n < 1:
        return []
    return [("videos.list", math.ceil(n / MAX_RESULTS)), ("videos.update", n)]


def _is_stop(exc: BaseException) -> bool:
    """True when retrying other videos is pointless (auth, permission or quota problems)."""
    if isinstance(exc, QuotaExceeded):
        return True
    info = api_error_info(exc)
    return info.status in (401, 403) or any(r in _STOP_REASONS for r in info.reasons)


TITLE_TEST_REASON = "UPDATE_TITLE_NOT_ALLOWED_DURING_TEST_AND_COMPARE"
TITLE_TEST_MESSAGE = (
    "YouTube does not allow title changes while a title test (Test & Compare) is running on this "
    "video. Nothing changed on YouTube and your draft is kept. Stop the test in YouTube Studio or "
    "wait for it to finish, pull your videos again, then push. Trying again before that uses more "
    "quota and fails the same way."
)


def _error_message(action: str, exc: BaseException) -> str:
    if isinstance(exc, QuotaExceeded):
        return "Not enough YouTube quota left today to continue."
    info = api_error_info(exc)
    if TITLE_TEST_REASON in info.reasons:
        return TITLE_TEST_MESSAGE
    if info.status is None and not info.reasons and not info.message:
        return f"{action} failed ({type(exc).__name__})."
    detail = ", ".join(
        part for part in (f"HTTP {info.status}" if info.status else "", info.reason) if part
    )
    text = f"{action} failed" + (f" ({detail})" if detail else "")
    return f"{text}: {info.message}" if info.message else f"{text}."


def _skip_reason(row: DraftRow | None, seen: str, store: Store) -> str:
    """Why a draft cannot be pushed, or "" when it can."""
    if row is None:
        return "There is no draft for this video any more."
    if row.approved_at is None:
        return "The draft is not approved."
    if fingerprint(row) != seen:
        return "The draft changed after you reviewed it. Review and approve it again."
    video = store.get_video(row.video_id)
    if video is None or is_stale(row, video):
        return "The video's text changed after this draft was made. Update the draft first."
    if has_errors(issues_for(row)):
        return "The draft has errors. Fix them before pushing."
    return ""


def _live_text(snippet: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    return (
        snippet.get("title", ""),
        snippet.get("description", ""),
        tuple(snippet.get("tags") or ()),
    )


def push_drafts(
    writer: YouTubeWriter,
    store: Store,
    guard: QuotaGuard,
    approved: Mapping[str, str],
    report: Callable[[str], None] | None = None,
    now: datetime | None = None,
) -> PushReport:
    """Push the approved drafts. `approved` maps video id to the fingerprint the creator saw.

    Raises QuotaExceeded, before any YouTube call, when the plan does not fit the daily budget.
    """

    def say(message: str) -> None:
        if report is not None:
            report(message)

    results: dict[str, PushItem] = {}
    selected: list[DraftRow] = []
    for video_id, seen in approved.items():
        row = store.get_draft(video_id)
        reason = _skip_reason(row, seen, store)
        if reason:
            results[video_id] = PushItem(video_id, "skipped", reason)
        else:
            assert row is not None
            selected.append(row)

    units = 0
    live: dict[str, dict[str, Any]] = {}
    live_localizations: dict[str, dict[str, Any]] = {}
    run_id: int | None = None  # opened once there is something to push
    pushes_made = 0
    try:
        if selected:
            preview = guard.preview(push_units(len(selected)))
            if not preview.fits:
                raise QuotaExceeded(
                    f"Pushing {len(selected)} video(s) needs {preview.total} quota units "
                    f"but only {preview.remaining} are left today."
                )
            run_id = store.begin_push_run(now)  # before the first YouTube call
            stopped = False
            for start in range(0, len(selected), MAX_RESULTS):
                batch = selected[start : start + MAX_RESULTS]
                if stopped:
                    for row in batch:
                        results[row.video_id] = PushItem(
                            row.video_id, "not_attempted", "Stopped before reading this video."
                        )
                    continue
                say(f"Reading {len(batch)} video(s) from YouTube...")
                try:
                    units += guard.charge("videos.list", note="push")
                    response = writer.videos([row.video_id for row in batch], "snippet,localizations")
                except Exception as exc:  # noqa: BLE001 - reported per video, never raised
                    stopped = _is_stop(exc)
                    for row in batch:
                        results[row.video_id] = PushItem(
                            row.video_id, "failed", _error_message("Reading from YouTube", exc)
                        )
                    continue
                for item in response.get("items") or []:
                    if isinstance(item, dict) and isinstance(item.get("snippet"), dict):
                        live[item["id"]] = item["snippet"]
                        localized = item.get("localizations")
                        live_localizations[item["id"]] = localized if isinstance(localized, dict) else {}

        stop_message = ""
        for row in selected:
            video_id = row.video_id
            if video_id in results:
                continue
            if stop_message:
                results[video_id] = PushItem(video_id, "not_attempted", stop_message)
                continue
            snippet = live.get(video_id)
            if snippet is None:
                results[video_id] = PushItem(
                    video_id,
                    "missing",
                    "This video is not on YouTube any more (deleted or unavailable). Nothing was pushed.",
                )
                continue
            localized = live_localizations.get(video_id, {})
            sets_i18n = row.default_language is not None or row.localizations is not None
            # The translations go into the backup only when this push changes them, so restoring
            # an older text never rolls back translations added later.
            before = backup_of(snippet, localized if sets_i18n else None)
            if (_live_text(snippet) != (row.base_title, row.base_description, row.base_tags)
                    or (row.default_language is not None
                        and snippet.get("defaultLanguage") != row.base_default_language)
                    or (row.localizations is not None
                        and plain_localizations(localized) != row.base_localizations)):
                push_id = store.begin_push(video_id, row.source, before, now, run_id)
                pushes_made += 1
                message = (
                    "This video was changed on YouTube (for example in Studio) after your last pull. "
                    "Nothing was pushed. The catalog now shows the live text; check your draft against it."
                )
                store.finish_push(push_id, "conflict", {}, message, now)
                store.update_video_text(
                    video_id, before["title"], before["description"], before["tags"], now,
                    default_language=snippet.get("defaultLanguage"), localizations=localized,
                )
                store.set_draft_approval(video_id, None)
                results[video_id] = PushItem(video_id, "conflict", message, push_id)
                say(f"{video_id}: changed on YouTube, not pushed.")
                continue

            sent: dict[str, Any] = {key: snippet[key] for key in KEEP_SNIPPET_KEYS if key in snippet}
            sent.update(title=row.title, description=row.description, tags=list(row.tags))
            body: dict[str, Any] = {"id": video_id, "snippet": sent}
            part = "snippet"
            after = sent
            if sets_i18n:
                if row.default_language is not None:
                    sent["defaultLanguage"] = row.default_language
                sent_localizations = (row.localizations if row.localizations is not None
                                      else plain_localizations(localized))
                body["localizations"] = sent_localizations
                part = "snippet,localizations"
                after = {**sent, "localizations": sent_localizations}
            push_id = store.begin_push(video_id, row.source, before, now, run_id)  # backup first
            pushes_made += 1
            try:
                units += guard.charge("videos.update", note="push")
                say(f"Updating {video_id}...")
                response = writer.update_video(body, part)
            except Exception as exc:  # noqa: BLE001 - recorded and reported, never raised
                message = _error_message("YouTube update", exc)
                store.finish_push(push_id, "failed", {}, message, now)
                results[video_id] = PushItem(video_id, "failed", message, push_id)
                if _is_stop(exc):
                    stop_message = "Stopped after an error that would affect every video."
                continue
            returned = response.get("snippet") if isinstance(response, dict) else None
            returned = returned if isinstance(returned, dict) else {}
            store.finish_push(push_id, "pushed", after, "Pushed to YouTube.", now)
            extra: dict[str, Any] = {}
            if sets_i18n:
                returned_localizations = response.get("localizations") if isinstance(response, dict) else None
                extra = {
                    "default_language": returned.get("defaultLanguage", sent.get("defaultLanguage")),
                    "localizations": (returned_localizations if isinstance(returned_localizations, dict)
                                      else sent_localizations),
                }
            store.update_video_text(
                video_id,
                returned.get("title", sent["title"]),
                returned.get("description", sent["description"]),
                # The update response lists tags alphabetically, but YouTube keeps the order sent (a
                # later read returns it): storing the response's order made the next push conflict.
                sent["tags"],
                now,
                **extra,
            )
            store.delete_draft(video_id)
            results[video_id] = PushItem(video_id, "pushed", "Pushed to YouTube.", push_id)
            say(f"{video_id}: pushed.")
    finally:
        if run_id is not None:  # a stopped or failed push closes its run too
            store.finish_push_run(run_id, pushes_made, now)

    items = tuple(results[video_id] for video_id in approved if video_id in results)
    return PushReport(items, units)
