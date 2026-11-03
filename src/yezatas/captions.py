"""Captions that are already on YouTube: pick the best caption track of a video, download it as SRT
and turn it into a transcript, the same shape the audio transcript pipeline makes.

Every segment time is a time written in the SRT file YouTube sent. The SRT parser is strict: a
malformed time is an error, never a guess, so chapters and subtitles built on the transcript can
only use times that exist. Reading touches nothing on YouTube: listing tracks costs 50 quota units
and downloading one costs 200. Nothing here stores anything; the caller adds the record.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .store import TranscriptRecord
from .transcribe import ASR_LANGUAGES
from .youtube import CaptionReader, api_error_info

MODEL_NAME = "YouTube captions"
LIST_QUOTA = 50  # YouTube Data API units for captions.list
DOWNLOAD_QUOTA = 200  # ... and for captions.download

SCOPE_MESSAGE = "Sign in again to allow reading captions."
QUOTA_MESSAGE = ("YouTube's daily quota for this app is used up (reading captions costs "
                 f"{LIST_QUOTA + DOWNLOAD_QUOTA} units). Try again tomorrow.")
NO_CAPTIONS_MESSAGE = ("This video has no captions on YouTube that Yezatas can read. Transcribe "
                       "the video file instead.")

# Names for caption language codes (the transcription languages plus a few more).
LANGUAGE_NAMES: dict[str, str] = {
    **ASR_LANGUAGES, "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian", "fi": "Finnish",
    "da": "Danish", "no": "Norwegian", "cs": "Czech", "sk": "Slovak", "ro": "Romanian",
    "hu": "Hungarian", "el": "Greek", "he": "Hebrew", "bg": "Bulgarian", "hr": "Croatian",
    "sl": "Slovenian", "sr": "Serbian", "ms": "Malay", "fa": "Persian", "bn": "Bengali",
}

_QUOTA_REASONS = {"quotaExceeded", "rateLimitExceeded", "dailyLimitExceeded",
                  "userRateLimitExceeded", "RATE_LIMIT_EXCEEDED", "RESOURCE_EXHAUSTED"}
_SCOPE_REASONS = {"insufficientPermissions", "ACCESS_TOKEN_SCOPE_INSUFFICIENT",
                  "authError", "forbidden"}
_TIME = re.compile(r"(\d{2,}):([0-5]\d):([0-5]\d),(\d{3})")
_TAG = re.compile(r"<[^>]*>")


class CaptionsError(RuntimeError):
    """Reading the captions failed. `kind` is "none" (no readable track), "scope" (sign in
    again), "quota", "not_found", "network", "api" or "srt" (the file was malformed)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class CaptionChoice:
    """The caption track picked for a video, and why."""

    caption_id: str
    language: str  # the track's language code, e.g. "en"
    manual: bool  # True for a track someone uploaded, False for YouTube's automatic one
    label: str  # e.g. "English · uploaded by you" or "English · automatic"


@dataclass(frozen=True)
class CaptionsResult:
    record: TranscriptRecord  # not stored yet (id 0)
    choice: CaptionChoice
    message: str  # one line, e.g. "Read 212 lines of English captions (English · automatic)."
    cost: float  # USD: always 0, the captions already exist
    quota_units: int  # YouTube API quota this read used


def language_name(code: str) -> str:
    """"English" for "en" or "en-GB"; the code itself when the language is not known."""
    base = code.strip().lower().replace("_", "-").split("-")[0]
    return LANGUAGE_NAMES.get(base) or code.strip() or "Unknown language"


def _base(code: Any) -> str:
    return str(code or "").strip().lower().replace("_", "-").split("-")[0]


# --- choosing a track ----------------------------------------------------------------------------------
def choose_track(listing: Mapping[str, Any], video_language: str = "") -> CaptionChoice:
    """The best readable track in a captions.list answer.

    Order: a manual track in the video's language, any manual track, an automatic (ASR) track in
    the video's language, any automatic track. Within a group the API's own order decides, so the
    choice is the same every time. Forced tracks (foreign-language parts only), drafts and failed
    tracks are never chosen.
    """
    wanted = _base(video_language)
    manual: list[tuple[str, str]] = []
    auto: list[tuple[str, str]] = []
    items = listing.get("items") if isinstance(listing, Mapping) else None
    for item in items if isinstance(items, list) else []:
        snippet = item.get("snippet") if isinstance(item, Mapping) else None
        if not isinstance(snippet, Mapping) or not isinstance(item.get("id"), str):
            continue
        kind = str(snippet.get("trackKind") or "standard").lower()
        if (kind not in ("standard", "asr") or snippet.get("isDraft")
                or str(snippet.get("status") or "serving").lower() == "failed"):
            continue
        (auto if kind == "asr" else manual).append((item["id"], str(snippet.get("language") or "")))
    for group, is_manual in ((manual, True), (auto, False)):
        pick = next((t for t in group if wanted and _base(t[1]) == wanted), None) or (
            group[0] if group else None)
        if pick is not None:
            name = language_name(pick[1])
            return CaptionChoice(pick[0], pick[1], is_manual,
                                 f"{name} · " + ("uploaded by you" if is_manual else "automatic"))
    raise CaptionsError("none", NO_CAPTIONS_MESSAGE)


# --- reading the SRT -----------------------------------------------------------------------------------
def _time_ms(text: str) -> int | None:
    match = _TIME.fullmatch(text.strip())
    if match is None:
        return None
    hours, minutes, seconds, millis = (int(g) for g in match.groups())
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def parse_srt(text: str) -> list[dict[str, Any]]:
    """SRT text as segments `{"start_ms", "end_ms", "text"}`, in file order.

    Strict: every cue needs a `HH:MM:SS,mmm --> HH:MM:SS,mmm` line with valid numbers, a cue may
    not end before it starts, and cues may not start out of order. Any violation raises
    CaptionsError("srt") naming the cue. Simple markup (<i>, <font>) is dropped from the text and
    HTML entities are decoded; cues left with no text are skipped. A file with no cue is an error.
    """
    body = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").strip()
    segments: list[dict[str, Any]] = []
    previous_start = 0
    for number, block in enumerate(re.split(r"\n[ \t]*\n", body) if body else [], start=1):
        lines = block.split("\n")
        if len(lines) > 1 and lines[0].strip().isdigit():
            lines = lines[1:]  # the cue's counter
        if "-->" not in lines[0]:
            raise CaptionsError("srt", f"The caption file is malformed: cue {number} has no "
                                       "time line.")
        left, _, right = lines[0].partition("-->")
        start, end = _time_ms(left), _time_ms(right)
        if start is None or end is None:
            raise CaptionsError("srt", f"The caption file is malformed: cue {number} has a time "
                                       "that is not HH:MM:SS,mmm.")
        if end < start:
            raise CaptionsError("srt", f"The caption file is malformed: cue {number} ends "
                                       "before it starts.")
        if start < previous_start:
            raise CaptionsError("srt", f"The caption file is malformed: cue {number} starts "
                                       "before the cue before it.")
        previous_start = start
        words = " ".join(html.unescape(_TAG.sub("", " ".join(lines[1:]))).split())
        if words:
            segments.append({"start_ms": start, "end_ms": end, "text": words})
    if not segments:
        raise CaptionsError("srt", "The caption file has no text in it.")
    return segments


# --- the API -------------------------------------------------------------------------------------------
def _api_error(exc: Exception, doing: str) -> CaptionsError:
    """A plain-English CaptionsError for a failed YouTube call (`doing`: "list" or "download")."""
    info = api_error_info(exc)
    reasons = set(info.reasons)
    if reasons & _QUOTA_REASONS:
        return CaptionsError("quota", QUOTA_MESSAGE)
    if info.status == 401 or reasons & _SCOPE_REASONS or (
            "scope" in info.message.lower() and info.status == 403):
        return CaptionsError("scope", SCOPE_MESSAGE)
    if info.status == 404 or reasons & {"videoNotFound", "captionNotFound"}:
        return CaptionsError("not_found", "YouTube could not find this video or its captions.")
    if info.status is None:
        return CaptionsError("network", "Could not reach YouTube. Check your internet "
                                        "connection and try again.")
    what = "list the captions of" if doing == "list" else "download the captions of"
    return CaptionsError("api", f"YouTube would not {what} this video (HTTP {info.status}).")


def _as_text(raw: Any) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw).decode("utf-8-sig", errors="replace")
    if isinstance(raw, str):
        return raw
    raise CaptionsError("api", "YouTube sent captions in a shape Yezatas does not understand.")


def build_record(video_id: str, choice: CaptionChoice,
                 segments: list[dict[str, Any]]) -> TranscriptRecord:
    """The transcript record for downloaded captions (id 0: the caller stores it). Its duration
    is the end of the last cue."""
    language = _base(choice.language) or ""
    return TranscriptRecord(
        id=0, video_id=video_id, language=language, model=MODEL_NAME,
        source_name=f"YouTube captions ({language_name(choice.language)})",
        duration_ms=max(s["end_ms"] for s in segments), segments=segments, problems=[],
        created_at="")


def fetch_captions(api: CaptionReader, video_id: str, video_language: str = "") -> CaptionsResult:
    """Read the best caption track of a video from YouTube as an (unstored) transcript.

    Uses 250 quota units (50 to list, 200 to download). Errors are CaptionsError.
    """
    if not video_id.strip():
        raise ValueError("A video id is needed to read its captions.")
    try:
        listing = api.caption_tracks(video_id)
    except Exception as exc:  # noqa: BLE001 - mapped to a plain-English message
        raise _api_error(exc, "list") from exc
    choice = choose_track(listing, video_language)
    try:
        raw = api.caption_srt(choice.caption_id)
    except Exception as exc:  # noqa: BLE001
        raise _api_error(exc, "download") from exc
    segments = parse_srt(_as_text(raw))
    record = build_record(video_id, choice, segments)
    message = f"Read {len(segments)} caption line" + ("" if len(segments) == 1 else "s") + (
        f" ({choice.label}).")
    return CaptionsResult(record, choice, message, 0.0, LIST_QUOTA + DOWNLOAD_QUOTA)
