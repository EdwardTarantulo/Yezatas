"""Chapters from a transcript: the model picks segment numbers, code turns them into times.

The transcript's segments are real cuts of the video's audio, so each has a true start time. The
model never writes a time: it only chooses which numbered segments start a chapter, and the
chapter time is that segment's own start. Whatever the model returns is repaired by code (never
invented): unknown or repeated segments and chapters too close together are dropped, titles are
cleaned. The result becomes a description DRAFT; nothing here talks to YouTube, and a description
that already has the creator's own timestamps is never touched.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable

from . import edits
from .agent import ModelClient, ModelError
from .seo import CHAPTER_MIN_GAP, CHAPTERS_MIN_COUNT, CHAPTERS_MIN_DURATION, parse_chapters, \
    parse_duration
from .store import Store, TranscriptRecord
from .validate import DESCRIPTION_BYTE_LIMIT

MAX_CHAPTERS = 12
TITLE_MAX = 50
PROMPT_CHAR_LIMIT = 60_000  # about; longer transcripts are merged and cut to stay near it
MAX_PROMPT_LINES = 200
# A first pick this close to the start moves to 0:00; a later one would put its title on the wrong
# part of the video, so the model is asked again instead.
FIRST_MOVE_MAX = 30
HEADING = "Chapters:"

CHAPTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chapters": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "segment": {"type": "integer",
                            "description": "The number of the segment where the chapter starts."},
                "title": {"type": "string", "description": "A short title for viewers."},
            },
            "required": ["segment", "title"],
            "additionalProperties": False,
        }},
        "summary": {"type": "string", "description": "One sentence: how the video is divided."},
    },
    "required": ["chapters", "summary"],
    "additionalProperties": False,
}

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "chapters", "strict": True, "schema": CHAPTER_SCHEMA},
}

SYSTEM_PROMPT = f"""\
You are the chapter assistant in Yezatas, a tool that helps a YouTube creator. You are given the \
transcript of ONE video, cut into numbered segments. Each line starts with the segment number in \
square brackets, then the minute and second when that segment starts, then the words spoken. \
Decide where the chapters of the video start.

Rules
- You choose chapters ONLY by segment number. Never write a time yourself: the code looks up the \
time of the segment you choose.
- Give {CHAPTERS_MIN_COUNT} to {MAX_CHAPTERS} chapters for viewers, in the order of the video, at \
least {CHAPTER_MIN_GAP} seconds apart. The first chapter starts at segment 0.
- Each title is short: 2 to 6 words and at most {TITLE_MAX} characters, in the language of the \
transcript, with no < or > characters, no times and no numbering.
- Base every title only on what the segments say. Do not invent topics, names or claims.

Answer
Your final answer is JSON: the chapters (each with the segment number and the title) and a \
one-sentence summary.
"""


@dataclass(frozen=True)
class Chapter:
    start_s: int  # seconds from the start of the video, taken from a transcript segment
    title: str


@dataclass(frozen=True)
class ChaptersResult:
    status: str  # "drafted" | "no_change" | "failed"
    message: str  # one or two sentences for the UI
    chapters: tuple[Chapter, ...] = ()
    run_id: int = 0  # the agent_runs row; 0 when the request was refused before a run started


# --- formatting and inserting --------------------------------------------------------------------
def clock(seconds: int) -> str:
    """Seconds as `m:ss`, or `h:mm:ss` from one hour on."""
    seconds = max(int(seconds), 0)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def chapter_lines(chapters: Iterable[Chapter]) -> list[str]:
    """One description line per chapter, like `0:00 Intro` or `1:02:03 End`."""
    return [f"{clock(c.start_s)} {c.title}" for c in chapters]


def _newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def insert_chapters(description: str, chapters: Iterable[Chapter]) -> str:
    """The description with a "Chapters:" block after its first paragraph.

    The first paragraph is the text before the first blank line. The block is separated from the
    text around it by blank lines; with no blank line it goes at the end. Nothing else changes.
    """
    block = "\n".join([HEADING, *chapter_lines(chapters)])
    text = _newlines(description)
    if not text.strip():
        return block
    first = len(text) - len(text.lstrip())  # skip blank lines before the first paragraph
    blank = re.compile(r"\n[ \t]*\n").search(text, first)
    if blank is None:
        return text.rstrip("\n") + "\n\n" + block + ("\n" if text.endswith("\n") else "")
    return text[:blank.start()] + "\n\n" + block + text[blank.start():]


# --- the transcript as a prompt ------------------------------------------------------------------
def _segment_text(segment: dict[str, Any]) -> str:
    return " ".join(str(segment.get("text") or "").split())


def _prompt_lines(segments: list[dict[str, Any]], limit: int = PROMPT_CHAR_LIMIT
                  ) -> tuple[list[str], set[int]]:
    """The numbered lines `[12] 3:05 text` and the set of segment numbers they offer.

    Empty segments are skipped (but keep their numbers). When the transcript is longer than
    `limit` characters, neighbouring segments are merged in groups (the text of the group under
    the first one's number and time) so there are at most `MAX_PROMPT_LINES` lines, and each
    line's text is cut at a word to fit its share of the limit. Merged-away numbers are not offered.
    """
    items = [(i, int(s["start_ms"]) // 1000, _segment_text(s)) for i, s in enumerate(segments)]
    items = [item for item in items if item[2]]

    def render(group: int, budget: int | None) -> list[str]:
        lines = []
        for at in range(0, len(items), group):
            chunk = items[at:at + group]
            text = " ".join(text for _, _, text in chunk)
            if budget is not None and len(text) > budget:
                text = text[:budget].rsplit(" ", 1)[0] + " ..."
            lines.append(f"[{chunk[0][0]}] {clock(chunk[0][1])} {text}")
        return lines

    group, lines = 1, render(1, None)
    if sum(len(line) + 1 for line in lines) > limit:
        group = max(1, -(-len(items) // MAX_PROMPT_LINES))
        count = -(-len(items) // group)
        lines = render(group, max(limit // count - 20, 40))
    return lines, {items[at][0] for at in range(0, len(items), group)}


def _user_prompt(title: str, language: str, lines: list[str]) -> str:
    head = f"Video title: {title}\n"
    if language:
        head += f"Transcript language code: {language}\n"
    return (head + "\nTranscript segments (segment number, start time, words spoken):\n"
            + "\n".join(lines) + "\n\nNow choose the chapters.")


# --- reading and repairing the answer ------------------------------------------------------------
class _BadAnswer(ValueError):
    """The model's answer cannot be read at all."""


_LEADING_TIME = re.compile(r"^[\s\[(]*\d{1,2}(?::\d{2}){1,2}[\])]?\s*[-–—:.)]*\s*")
_LEADING_NUMBER = re.compile(r"^\s*\d{1,2}[.)]\s+")
_LEADING_MARK = re.compile(r"^[\s\-–—•*]+")


def clean_title(raw: str) -> str:
    """A chapter title ready for a description line: no brackets, times or numbering, <= 50 chars."""
    title = " ".join(raw.replace("<", "").replace(">", "").split())
    for pattern in (_LEADING_TIME, _LEADING_NUMBER, _LEADING_MARK):
        title = pattern.sub("", title)
    title = title.strip()
    if len(title) > TITLE_MAX:
        cut = title[:TITLE_MAX]
        if title[TITLE_MAX] != " " and " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        title = cut.rstrip(" ,;:-–—")
    return title.strip()


def parse_answer(content: str | None) -> tuple[list[tuple[Any, Any]], str]:
    """(entries as (segment, title) pairs, summary) from the model's JSON; `_BadAnswer` if unusable."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        try:
            data = json.loads(text[start:end + 1]) if 0 <= start < end else None
        except ValueError:
            data = None
        if data is None:
            raise _BadAnswer("The model's answer was not valid JSON.") from None
    items = data.get("chapters") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise _BadAnswer("The model's answer did not match the expected format.")
    entries = [(item.get("segment"), item.get("title")) for item in items if isinstance(item, dict)]
    summary = data.get("summary")
    return entries, " ".join(summary.split()) if isinstance(summary, str) else ""


def _repair(entries: list[tuple[Any, Any]], starts: dict[int, int], offered: set[int],
            end_s: int | None = None) -> tuple[list[tuple[int, Chapter]], list[str], bool]:
    """Turn the model's picks into valid chapters: [(segment, Chapter)], problems, first_moved.

    Times come only from `starts` (segment number -> whole seconds). A first chapter within
    `FIRST_MOVE_MAX` seconds of the start is moved to segment 0, which is 0:00 (YouTube needs
    chapters to begin there); a later one makes the whole answer unusable (no chapters, a problem),
    since its title would then name the wrong part. A chapter less than `CHAPTER_MIN_GAP` seconds
    before `end_s` (the video's end) is dropped too. Anything unusable is dropped.
    """
    problems: list[str] = []
    picked: list[tuple[int, str]] = []
    seen: set[int] = set()
    valid = offered | {0}
    for segment, raw in entries:
        label = segment if isinstance(segment, int) and not isinstance(segment, bool) else "?"
        title = clean_title(raw) if isinstance(raw, str) else ""
        if label == "?":
            problems.append("A chapter had no segment number.")
        elif segment not in valid:
            problems.append(f"Segment {segment} is not one of the segments listed.")
        elif segment in seen:
            problems.append(f"Segment {segment} was chosen twice.")
        elif not title:
            problems.append(f"The chapter at segment {segment} has no usable title.")
        else:
            seen.add(segment)
            picked.append((segment, title))
    picked.sort(key=lambda pair: pair[0])
    first_moved = bool(picked) and picked[0][0] != 0
    if first_moved and starts[picked[0][0]] > FIRST_MOVE_MAX:
        problems.append(f"The first chapter must start at segment 0, not segment {picked[0][0]}.")
        return [], problems, False
    if first_moved:
        picked[0] = (0, picked[0][1])
    kept: list[tuple[int, Chapter]] = []
    for segment, title in picked:
        seconds = 0 if segment == 0 else starts[segment]
        if kept and seconds - kept[-1][1].start_s < CHAPTER_MIN_GAP:
            problems.append(f"Segment {segment} starts less than {CHAPTER_MIN_GAP} seconds after "
                            "the chapter before it.")
            continue
        if end_s is not None and kept and seconds > end_s - CHAPTER_MIN_GAP:
            problems.append(f"Segment {segment} starts less than {CHAPTER_MIN_GAP} seconds before "
                            "the end of the video.")
            continue
        kept.append((segment, Chapter(seconds, title)))
    return kept[:MAX_CHAPTERS], problems, first_moved


@dataclass
class _Tally:
    input: int = 0
    cached: int = 0
    output: int = 0

    def complete(self, client: ModelClient, messages: list[dict[str, Any]]) -> Any:
        reply = client.complete(messages, response_format=RESPONSE_FORMAT)
        self.input += reply.input_tokens
        self.cached += reply.cached_tokens
        self.output += reply.output_tokens
        return reply

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.input, self.cached, self.output)


def _ask(client: ModelClient, messages: list[dict[str, Any]], tally: _Tally, starts: dict[int, int],
         offered: set[int], end_s: int, log: list[dict[str, Any]], round_no: int
         ) -> tuple[Any, list[tuple[int, Chapter]], list[str], str]:
    """One model call and its repair: (reply, chapters, problems, summary)."""
    reply = tally.complete(client, messages)
    try:
        entries, summary = parse_answer(reply.content)
    except _BadAnswer as exc:
        log.append({"type": "problems", "round": round_no, "problems": [str(exc)]})
        return reply, [], [str(exc)], ""
    chapters, problems, moved = _repair(entries, starts, offered, end_s)
    log.append({"type": "answer", "round": round_no, "returned": len(entries),
                "kept": len(chapters), "first_moved_to_start": moved})
    if problems:
        log.append({"type": "problems", "round": round_no, "problems": problems})
    return reply, chapters, problems, summary


def _converse(store: Store, client: ModelClient, video_id: str, title: str, description: str,
              tags: tuple[str, ...], record: TranscriptRecord, tally: _Tally,
              log: list[dict[str, Any]], say: Callable[[str], None], now: datetime | None
              ) -> tuple[str, str, list[Chapter], str]:
    """The model conversation and the draft: (status, message, chapters, summary)."""
    segments = record.segments
    starts = {i: int(s["start_ms"]) // 1000 for i, s in enumerate(segments)}
    end_s = record.duration_ms // 1000 or max((int(s["end_ms"]) for s in segments), default=0) // 1000
    lines, offered = _prompt_lines(segments)
    log.append({"type": "prompt", "segments": len(segments), "offered": len(offered)})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(title, record.language, lines)},
    ]
    say("Choosing the chapters")
    reply, chapters, problems, summary = _ask(client, messages, tally, starts, offered, end_s, log, 1)
    if len(chapters) < CHAPTERS_MIN_COUNT:
        say("Fixing problems the checks found")
        found = "\n".join(f"- {p}" for p in problems) or "- Too few chapters were given."
        messages.append(reply.message)
        messages.append({"role": "user", "content": (
            f"Your answer could not be used: only {len(chapters)} valid chapters remained, and "
            f"{CHAPTERS_MIN_COUNT} to {MAX_CHAPTERS} are needed.\n{found}\n"
            f"Choose again. Use segment numbers from the transcript, at least {CHAPTER_MIN_GAP} "
            "seconds apart, each with a different number, and reply with only the JSON.")})
        reply, chapters, problems, summary = _ask(client, messages, tally, starts, offered, end_s, log, 2)
        if len(chapters) < CHAPTERS_MIN_COUNT:
            return ("failed", f"The model could not pick at least {CHAPTERS_MIN_COUNT} usable "
                    "chapters, so nothing was drafted. Try again, or write the chapters yourself.",
                    [], summary)
    final = [chapter for _, chapter in chapters]
    new_description = insert_chapters(description, final)
    if len(new_description.encode("utf-8")) > DESCRIPTION_BYTE_LIMIT:
        return ("failed", "The chapters do not fit in the description (YouTube allows "
                f"{DESCRIPTION_BYTE_LIMIT:,} bytes). Shorten the description and try again.",
                final, summary)
    say("Saving the draft")
    saved = edits.save_draft(store, video_id, title, new_description, tags, source="chapters",
                             note=f"Chapters from the transcript ({client.model})", now=now)
    if saved.discarded:
        return "no_change", "The description already matches; nothing to change.", final, summary
    return ("drafted", f"Saved {len(final)} chapters as a draft for you to review.", final,
            summary)


def suggest_chapters(store: Store, client: ModelClient, video_id: str, *,
                     now: datetime | None = None,
                     report: Callable[[str], None] | None = None) -> ChaptersResult:
    """Ask the model where the chapters start and put them in the description draft.

    The model picks transcript segment numbers; the times come from the segments. Refused (status
    "failed", no run recorded) when the video or its transcript is missing, the video is shorter
    than `CHAPTERS_MIN_DURATION`, or the description (the draft's if there is one) already has
    timestamps, which Yezatas never rewrites. The result is a draft only; nothing reaches YouTube.
    """
    def refuse(message: str) -> ChaptersResult:
        return ChaptersResult("failed", message, (), 0)

    video = store.get_video(video_id)
    if video is None:
        return refuse("This video is not in your catalog. Pull your videos first.")
    record = store.latest_transcript(video_id)
    if record is None:
        return refuse("Transcribe the video first.")
    seconds = parse_duration(video.duration)
    if seconds is None:
        seconds = record.duration_ms // 1000
    if seconds < CHAPTERS_MIN_DURATION:
        return refuse(f"Chapters only help on videos of {CHAPTERS_MIN_DURATION // 60} minutes or "
                      "more; this one is shorter.")
    draft = store.get_draft(video_id)
    title, description, tags = ((draft.title, draft.description, draft.tags) if draft
                                else (video.title, video.description, tuple(video.tags)))
    if parse_chapters(description):
        return refuse("The description already has timestamps. Yezatas never rewrites your own "
                      "chapters; remove them first if you want new ones.")
    if not any(_segment_text(s) for s in record.segments):
        return refuse("The transcript has no text to make chapters from.")

    say = report or (lambda line: None)
    say("Reading the transcript")
    run_id = store.begin_agent_run(video_id, client.model, now)
    log: list[dict[str, Any]] = []
    tally = _Tally()
    chapters: list[Chapter] = []
    summary = ""
    try:
        status, message, chapters, summary = _converse(store, client, video_id, title, description,
                                                       tags, record, tally, log, say, now)
    except ModelError as exc:
        log.append({"type": "error", "kind": exc.kind, "message": str(exc)})
        status, message = "failed", str(exc)
    except BaseException:
        store.finish_agent_run(run_id, "failed", "The run stopped with an unexpected error.", log,
                               {}, tally.as_tuple(), now)
        raise
    result = {"chapters": [{"start_s": c.start_s, "title": c.title} for c in chapters],
              "summary": summary}
    store.finish_agent_run(run_id, status, summary or message, log, result, tally.as_tuple(), now)
    return ChaptersResult(status, message, tuple(chapters), run_id)
