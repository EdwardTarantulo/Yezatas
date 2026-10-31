"""SRT subtitles from a transcript.

A transcript segment is a real cut of the audio, so its start and end are true. Subtitles keep
those times. Only a segment too long for the screen is split into several cues, and then the
time inside it is divided by an estimate (see `to_srt`).
"""

from __future__ import annotations

import textwrap
from typing import Any, Iterable, Mapping

MAX_LINE = 42  # characters per subtitle line (the usual broadcast guideline)
MAX_LINES = 2  # lines per cue


def srt_time(ms: int) -> str:
    """Milliseconds as an SRT time, `HH:MM:SS,mmm`."""
    ms = max(int(ms), 0)
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _cue_texts(text: str, max_line: int, max_lines: int) -> list[str]:
    """Split text into cues of at most `max_lines` lines of `max_line` characters, at word breaks."""
    lines = textwrap.wrap(text, width=max_line, break_long_words=True, break_on_hyphens=False)
    return ["\n".join(lines[i:i + max_lines]) for i in range(0, len(lines), max_lines)]


def to_srt(segments: Iterable[Mapping[str, Any]], *, max_line: int = MAX_LINE,
           max_lines: int = MAX_LINES) -> str:
    """The segments `{"start_ms", "end_ms", "text"}` as SRT text.

    Each segment with text becomes one cue with the segment's own start and end. A segment whose
    text does not fit in `max_lines` lines of `max_line` characters is split into several cues at
    word boundaries. The split points inside such a segment are an ESTIMATE: its time is divided
    in proportion to the characters of each cue (the audio does not say where each word falls).
    The first cue still starts, and the last one ends, exactly at the segment's own times.
    Segments with no text make no cue. Cues are numbered from 1.
    """
    if max_line < 1 or max_lines < 1:
        raise ValueError("max_line and max_lines must be at least 1.")
    cues: list[str] = []
    for segment in segments:
        text = " ".join(str(segment.get("text") or "").split())
        if not text:
            continue
        start = int(segment["start_ms"])
        end = max(int(segment["end_ms"]), start)
        pieces = _cue_texts(text, max_line, max_lines)
        weights = [len(piece.replace("\n", " ")) for piece in pieces]
        total = sum(weights)
        done = 0
        for piece, weight in zip(pieces, weights):
            cue_start = start + round((end - start) * done / total)
            done += weight
            cue_end = start + round((end - start) * done / total)
            cues.append(f"{len(cues) + 1}\n{srt_time(cue_start)} --> {srt_time(cue_end)}\n{piece}\n")
    return "\n".join(cues)
