"""Shorts finder: moments of a long video that could stand alone as a vertical Short.

The transcript's segments are real cuts of the video's audio, so each has a true
start time. The model never writes a time: it only picks the first and last
segment numbers and the code looks up the times and checks everything (length,
overlap, titles). Nothing here talks to YouTube and nothing is changed; each
run is stored.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable

from . import chapters
from .store import ShortsFind, Store, TranscriptRecord
from .subtitle_translate import TextModel
from .translate import TranslateError

MAX_SHORTS = 5
MIN_SECONDS = 15
MAX_SECONDS = 60
TITLE_MAX = 60
WHY_MAX = 200
QUOTE_MAX = 160

SYSTEM_PROMPT = (
    "You are the Shorts finder in Yezatas. You get the transcript of ONE long video cut "
    "into numbered segments, each line [number] m:ss words. Find up to 5 moments that work "
    "on their own as a vertical Short: a clear hook in the first seconds, one complete idea, "
    "an ending that does not need the rest of the video. Rules: choose each moment ONLY by "
    "its first and last segment numbers and never write a time (the code looks up the times); "
    "each moment is 15 to 60 seconds long; moments do not overlap; the title is a short hook "
    "of at most 60 characters in the language of the transcript with no < or >; \"why\" is one "
    "sentence based only on what is said; invent nothing. If no moment works, answer <none/>. "
    "Answer only with blocks like "
    "<short><start>12</start><end>18</end><title>...</title><why>...</why></short>."
)


def _between_block(block: str, tag: str) -> str:
    start = block.find(f"<{tag}>")
    if start < 0:
        return ""
    start += len(tag) + 2
    end = block.find(f"</{tag}>", start)
    if end < 0:
        return ""
    return block[start:end]


def parse_answer(content: str | None) -> tuple[list[dict[str, str]], bool]:
    """Parse Shorts answer into list of dicts and whether <none/> was present."""
    has_none = bool(content and "<none/>" in content)
    if not content:
        return [], has_none
    blocks = re.findall(r"<short>(.*?)</short>", content, re.S)
    result: list[dict[str, str]] = []
    for b in blocks:
        result.append(
            {
                "start": _between_block(b, "start"),
                "end": _between_block(b, "end"),
                "title": _between_block(b, "title"),
                "why": _between_block(b, "why"),
            }
        )
    return result, has_none


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _clean_text(text: str) -> str:
    text = text.replace("<", "").replace(">", "")
    return _collapse(text).strip()


def _cut_at_word(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip()


def _make_quote(segments: list[dict], start: int, end: int) -> str:
    parts: list[str] = []
    for i in range(start, end + 1):
        t = _collapse(str(segments[i].get("text") or ""))
        if t:
            parts.append(t)
    quote = _collapse(" ".join(parts))
    if len(quote) > QUOTE_MAX:
        quote = _cut_at_word(quote, QUOTE_MAX) + " ..."
    return quote


def _save(
    store: Store,
    video_id: str,
    transcript_id: int,
    model: str,
    status: str,
    message: str,
    clips: list[dict],
    notes: list[str],
    error: str,
    input_tokens: int,
    output_tokens: int,
    now: datetime | None,
) -> ShortsFind:
    rec = ShortsFind(
        id=0,
        video_id=video_id,
        transcript_id=transcript_id,
        created_at="",
        model=model,
        status=status,
        message=message,
        clips=clips,
        notes=notes,
        error=error,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    store.save_shorts_find(rec, now)
    stored = store.latest_shorts_find(video_id)
    assert stored is not None
    return stored


def find_shorts(
    store: Store,
    client: TextModel,
    video_id: str,
    *,
    now: datetime | None = None,
    report: Callable[[str], None] | None = None,
) -> ShortsFind:
    """Find Shorts moments for a video via the model.

    Raises ValueError if the video is unknown, has no transcript, or the
    transcript has no text. Stores one ShortsFind row for every model run
    (found, none, or failed) and never changes the video or transcript.
    """
    video = store.get_video(video_id)
    if video is None:
        raise ValueError(
            "This video is not in your catalog. Pull your videos first."
        )
    record: TranscriptRecord | None = store.latest_transcript(video_id)
    if record is None:
        raise ValueError("Transcribe the video first.")
    has_text = any(str(s.get("text") or "").strip() for s in record.segments)
    if not has_text:
        raise ValueError("The transcript has no text to find Shorts in.")

    say = report or (lambda line: None)
    say("Reading the transcript")

    lines, offered = chapters._prompt_lines(record.segments)

    draft = store.get_draft(video_id)
    title = draft.title if draft else video.title
    user_content = (
        f"Video title: {title}\n\n"
        "Transcript segments (segment number, start time, words spoken):\n"
        + "\n".join(lines)
        + "\n\nNow find the Shorts."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    say(f"Asking {client.model}")

    try:
        completion = client.complete(messages)
    except TranslateError as exc:
        say("Checking the answer")
        err = str(exc)
        return _save(
            store,
            video_id,
            record.id,
            client.model,
            "failed",
            err,
            [],
            [],
            err,
            0,
            0,
            now,
        )

    say("Checking the answer")

    content = completion.content
    input_tokens = completion.input_tokens
    output_tokens = completion.output_tokens

    parsed, has_none = parse_answer(content)
    if not content or (not parsed and not has_none):
        err = (
            "The model's answer had no Shorts in the expected shape. "
            "Try again."
        )
        return _save(
            store,
            video_id,
            record.id,
            client.model,
            "failed",
            err,
            [],
            [],
            err,
            input_tokens,
            output_tokens,
            now,
        )

    clips: list[dict] = []
    notes: list[str] = []
    kept_intervals: list[tuple[int, int]] = []

    for item in parsed:
        raw_start = item.get("start", "")
        raw_end = item.get("end", "")

        if len(clips) >= MAX_SHORTS:
            notes.append("A moment was dropped: there were more than 5.")
            continue

        s_str = raw_start.strip()
        e_str = raw_end.strip()
        if not s_str.isdigit() or not e_str.isdigit():
            notes.append(
                "A moment was dropped: its segment numbers are not in "
                "the transcript."
            )
            continue
        s = int(s_str)
        e = int(e_str)
        if s not in offered or s > e or e >= len(record.segments):
            notes.append(
                "A moment was dropped: its segment numbers are not in "
                "the transcript."
            )
            continue

        start_ms = int(record.segments[s]["start_ms"])
        end_ms = int(record.segments[e]["end_ms"])
        seconds = round((end_ms - start_ms) / 1000)
        if seconds < MIN_SECONDS or seconds > MAX_SECONDS:
            notes.append(
                f"A moment was dropped: it was {seconds} seconds long "
                "(a Short here is 15 to 60)."
            )
            continue

        overlap = False
        for ks, ke in kept_intervals:
            if start_ms < ke and end_ms > ks:
                overlap = True
                break
        if overlap:
            notes.append("A moment was dropped: it overlaps another one.")
            continue

        title_text = _clean_text(item.get("title", ""))
        title_text = _cut_at_word(title_text, TITLE_MAX)
        if not title_text:
            notes.append("A moment was dropped: it had no title.")
            continue

        why_text = _clean_text(item.get("why", ""))
        why_text = _cut_at_word(why_text, WHY_MAX)

        quote = _make_quote(record.segments, s, e)

        clips.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "title": title_text,
                "why": why_text,
                "quote": quote,
            }
        )
        kept_intervals.append((start_ms, end_ms))

    clips.sort(key=lambda c: c["start_ms"])

    if clips:
        status = "found"
        if len(clips) == 1:
            message = "Found 1 moment that could work as a Short."
        else:
            message = f"Found {len(clips)} moments that could work as Shorts."
    else:
        status = "none"
        message = "No moment in this video works as a Short on its own."

    return _save(
        store,
        video_id,
        record.id,
        client.model,
        status,
        message,
        clips,
        notes,
        "",
        input_tokens,
        output_tokens,
        now,
    )
