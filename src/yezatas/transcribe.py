"""Transcripts: ffmpeg pulls the audio out of a video file on the creator's own computer, the audio
is cut at pauses, and Muse Voice Transcribe turns each piece into text.

Every segment time is a cut point in the real audio. Nothing here estimates, shifts or invents a
time, so chapters and subtitles built on a transcript can only use times that exist.
The Meta API key is only ever sent in the Authorization header: never logged, never in an error
message, a repr or the database. Nothing here touches YouTube.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from . import agent
from .store import Store, TranscriptRecord

ASR_MODEL = "muse-voice-transcribe-1.0"
ASR_PRICE_PER_HOUR = 0.18  # USD per hour of audio

# Languages Muse Voice Transcribe accepts (not Lithuanian).
ASR_LANGUAGES: dict[str, str] = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "ar": "Arabic",
    "hi": "Hindi", "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "id": "Indonesian",
    "vi": "Vietnamese", "th": "Thai", "ru": "Russian", "uk": "Ukrainian", "sv": "Swedish",
}

VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mp3", ".wav",
                                   ".m4a", ".flac", ".aac")

SAMPLE_RATE = 16000
FFMPEG_TIMEOUT = 3600  # seconds; a stuck ffmpeg must not hang the job for ever
RETRY_PAUSE = 2.0  # seconds before the one retry of a piece
FRAME_MS = 50  # loudness is measured in frames this long
QUIET_SPAN = 3  # frames averaged when looking for the quietest moment (150 ms)
QUIET_PULL_DB = 0.5  # dB per second away from the target length that a quieter moment must win by
SILENT_DB = -120.0  # what a frame of digital silence ("-inf") counts as
NO_FFMPEG_MESSAGE = "ffmpeg was not found. Install ffmpeg and restart Yezatas."
AUTH_MESSAGE = "The transcription API key was refused. Check it in Configuration."


class TranscribeError(RuntimeError):
    """Something in the transcript pipeline failed. `kind` is "file" (the media file), "ffmpeg",
    "auth", "rate_limit", "network", "api" or "answer" (the API answered in an unexpected shape)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class Piece:
    """One cut of the audio, in milliseconds from its start."""

    start_ms: int
    end_ms: int


def _pause(seconds: float) -> None:
    time.sleep(seconds)  # a function of its own so tests can patch it out


def clock(ms: int) -> str:
    """m:ss, or h:mm:ss from one hour on (milliseconds are cut off, not rounded)."""
    seconds = max(0, ms) // 1000
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


# --- ffmpeg and the media file -----------------------------------------------------------------------
def find_ffmpeg() -> str | None:
    """The ffmpeg program on this computer's PATH, or None."""
    return shutil.which("ffmpeg")


def check_media_path(raw: str) -> Path:
    """The media file a creator typed or pasted (quotes and spaces around it are ignored)."""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if not text:
        raise TranscribeError("file", "Type the path of the video file on this computer.")
    path = Path(text)
    if not path.exists():
        raise TranscribeError("file", f"There is no file at that path (looked for {path.name}).")
    if not path.is_file():
        raise TranscribeError("file", "That path is a folder, not a video file.")
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise TranscribeError(
            "file", f"{path.name} is not a video or audio file Yezatas can read. Use one of: "
                    f"{', '.join(VIDEO_SUFFIXES)}.")
    return path


def _run(runner: Callable[..., Any], argv: list[str]) -> Any:
    """Runs ffmpeg through `runner` (subprocess.run in real life); output is captured as text."""
    try:
        return runner(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                      check=False, timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise TranscribeError("ffmpeg", "ffmpeg took too long and was stopped.") from exc
    except OSError as exc:
        raise TranscribeError("ffmpeg", "ffmpeg could not be started.") from exc


def _last_line(text: Any) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[-1][:300] if lines else "no details"


def extract_audio(ffmpeg: str, media: Path, wav: Path,
                  runner: Callable[..., Any] = subprocess.run) -> None:
    """Writes the media's sound to `wav` as mono 16-bit 16 kHz PCM, the format the API wants."""
    argv = [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(media), "-vn", "-ac", "1",
            "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-map_metadata", "-1", str(wav)]
    done = _run(runner, argv)
    if done.returncode != 0:
        raise TranscribeError(
            "ffmpeg", f"ffmpeg could not read the audio of this file: {_last_line(done.stderr)}")
    if not wav.is_file():
        raise TranscribeError("ffmpeg", "ffmpeg could not read the audio of this file: "
                                        "it produced no sound.")


def detect_silences(ffmpeg: str, wav: Path,
                    runner: Callable[..., Any] = subprocess.run) -> list[tuple[int, int]]:
    """Pauses in the audio as (start_ms, end_ms), found by ffmpeg's silencedetect filter."""
    argv = [ffmpeg, "-nostdin", "-i", str(wav), "-af", "silencedetect=noise=-35dB:d=0.35",
            "-f", "null", "-"]
    done = _run(runner, argv)
    if done.returncode != 0:
        raise TranscribeError(
            "ffmpeg", f"ffmpeg could not look for pauses in the audio: {_last_line(done.stderr)}")
    silences: list[tuple[int, int]] = []
    start: int | None = None
    for line in str(done.stderr or "").splitlines():
        if "silence_start:" in line:
            value = _seconds_after(line, "silence_start:")
            start = None if value is None else max(0, value)
        elif "silence_end:" in line and start is not None:
            end = _seconds_after(line, "silence_end:")
            if end is not None and end >= start:
                silences.append((start, end))
            start = None
    return silences


def _seconds_after(line: str, label: str) -> int | None:
    """The number after `label` in a silencedetect line, in whole milliseconds."""
    rest = line.split(label, 1)[1].strip().split()
    try:
        return round(float(rest[0]) * 1000)
    except (IndexError, ValueError):
        return None


def loudness_profile(ffmpeg: str, wav: Path,
                     runner: Callable[..., Any] = subprocess.run) -> list[float]:
    """The loudness of every `FRAME_MS` frame of the audio, in dB (RMS), from ffmpeg's astats.

    Used to cut where speech pauses when a music bed under the voice hides real silence.
    """
    samples = SAMPLE_RATE * FRAME_MS // 1000
    argv = [ffmpeg, "-nostdin", "-i", str(wav), "-af",
            f"asetnsamples=n={samples}:p=0,astats=metadata=1:reset=1:measure_perchannel=none,"
            "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-",
            "-f", "null", "-"]
    done = _run(runner, argv)
    if done.returncode != 0:
        raise TranscribeError(
            "ffmpeg", f"ffmpeg could not measure the loudness of the audio: "
                      f"{_last_line(done.stderr)}")
    levels: list[float] = []
    for line in str(done.stdout or "").splitlines():
        if "RMS_level=" in line:
            value = line.split("RMS_level=", 1)[1].strip()
            try:
                level = float(value)
            except ValueError:
                level = SILENT_DB
            levels.append(SILENT_DB if level != level else max(level, SILENT_DB))  # nan, -inf
    return levels


def quietest_cut(levels: Sequence[float]) -> Callable[[int, int, int], int]:
    """A cut chooser for `plan_pieces`: the quietest moment between low and high (ms).

    Loudness is averaged over `QUIET_SPAN` frames; a moment further from `target` must be
    quieter by `QUIET_PULL_DB` per second to win. Deterministic: ties go to the earlier moment.
    """
    def choose(low: int, high: int, target: int) -> int:
        first, last = -(-low // FRAME_MS), high // FRAME_MS
        best, best_score = max(low, min(high, target)), None
        for frame in range(first, last + 1):
            window = levels[max(0, frame - QUIET_SPAN // 2):frame + QUIET_SPAN // 2 + 1]
            if not window:
                continue
            at = frame * FRAME_MS
            score = sum(window) / len(window) + QUIET_PULL_DB * abs(at - target) / 1000
            if best_score is None or score < best_score:
                best, best_score = at, score
        return best

    return choose


# --- cutting the audio -------------------------------------------------------------------------------
def plan_pieces(duration_ms: int, silences: Sequence[tuple[int, int]], *, min_ms: int = 4000,
                target_ms: int = 12000, max_ms: int = 30000,
                quietest: Callable[[int, int, int], int] | None = None) -> list[Piece]:
    """Where to cut `duration_ms` of audio: at the middle of pauses, pieces about `target_ms` long.

    Deterministic. Pieces cover 0..duration_ms in order with no gap or overlap. A piece is never
    shorter than `min_ms` (unless the whole audio is) and never longer than `max_ms`. With no
    pause in reach, `quietest(low, high, target)` picks the cut (see `quietest_cut`); without it
    the cut is a hard one at `max_ms`.
    """
    if not 0 < min_ms <= target_ms <= max_ms:
        raise ValueError("Piece lengths must satisfy 0 < min <= target <= max.")
    if duration_ms <= 0:
        return []
    middles = sorted({(a + b) // 2 for a, b in silences})
    cuts: list[int] = []
    start = 0
    # Keep cutting while the rest is too long for one piece, or long enough for a target piece
    # plus a remainder that is not too short.
    while duration_ms - start > max_ms or duration_ms - start >= target_ms + min_ms:
        remaining = duration_ms - start
        low = start + min_ms
        high = min(start + max_ms, duration_ms - min_ms)
        near = [m for m in middles if low <= m <= high]
        if near:
            cut = min(near, key=lambda m: (abs(m - (start + target_ms)), m))
        elif quietest is not None:
            cut = min(high, max(low, quietest(low, high, start + target_ms)))
        elif remaining <= max_ms:
            break  # no pause in reach and it fits in one piece
        else:
            cut = max(low, high)  # hard cut
        cuts.append(cut)
        start = cut
    edges = [0, *cuts, duration_ms]
    pieces = [Piece(a, b) for a, b in zip(edges, edges[1:])]
    if len(pieces) > 1 and pieces[-1].end_ms - pieces[-1].start_ms < min_ms:
        last = pieces.pop()  # only for odd settings (max < 2 * min)
        pieces[-1] = Piece(pieces[-1].start_ms, last.end_ms)
    return pieces


def wav_duration_ms(wav: Path) -> int:
    """Length of a WAV file in milliseconds."""
    try:
        with wave.open(str(wav), "rb") as fh:
            return fh.getnframes() * 1000 // fh.getframerate()
    except (wave.Error, EOFError, OSError) as exc:
        raise TranscribeError("ffmpeg", "The extracted audio could not be read.") from exc


def wav_slice(wav: Path, piece: Piece) -> bytes:
    """A complete WAV file (header and sound) holding just `piece` of `wav`."""
    with wave.open(str(wav), "rb") as src:
        rate = src.getframerate()
        first = min(src.getnframes(), max(0, piece.start_ms * rate // 1000))
        last = min(src.getnframes(), max(first, piece.end_ms * rate // 1000))
        src.setpos(first)
        frames = src.readframes(last - first)
        params = (src.getnchannels(), src.getsampwidth(), rate)
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setnchannels(params[0])
        dst.setsampwidth(params[1])
        dst.setframerate(params[2])
        dst.writeframes(frames)
    return out.getvalue()


# --- the transcription API -----------------------------------------------------------------------------
class AsrClient(Protocol):
    model: str

    def transcribe(self, wav_bytes: bytes, language: str,
                   keywords: Sequence[str] = ()) -> str: ...


class MetaAsrClient:
    """Muse Voice Transcribe on Meta's API. The key only goes in the Authorization header."""

    def __init__(self, api_key: str, model: str = ASR_MODEL, base_url: str = agent.MODEL_BASE_URL,
                 timeout: float = 120, client: Any = None) -> None:
        import httpx  # lazy: the module imports without it (it comes with the openai package)

        self.model = model
        self._httpx = httpx
        self._key = api_key
        self._url = base_url.rstrip("/") + "/asr/transcribe"
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    def __repr__(self) -> str:
        return f"MetaAsrClient(model={self.model!r})"

    def _clean(self, text: str) -> str:
        text = " ".join(text.split())[:200]
        return text.replace(self._key, "***") if self._key else text

    def transcribe(self, wav_bytes: bytes, language: str, keywords: Sequence[str] = ()) -> str:
        request: dict[str, Any] = {"model": self.model, "audioEncoding": "WAV",
                                   "languageBias": [language]}  # a list: a string is rejected
        words = [w for w in (k.strip() for k in keywords) if w]
        if words:
            request["keywords"] = words
        files = {
            "request": (None, json.dumps(request), "application/json"),
            "audio": ("audio.wav", wav_bytes, "audio/wav"),
        }
        try:
            response = self._client.post(
                self._url, files=files, headers={"Authorization": f"Bearer {self._key}"})
        except self._httpx.TimeoutException as exc:
            raise TranscribeError(
                "network", "The transcription service took too long to answer. Try again.") from exc
        except self._httpx.TransportError as exc:
            raise TranscribeError(
                "network", "Could not reach the transcription service. Check your internet "
                           "connection and try again.") from exc
        status = response.status_code
        if status in (401, 403):
            raise TranscribeError("auth", AUTH_MESSAGE)
        if status == 429:
            raise TranscribeError(
                "rate_limit", "The transcription service is limiting requests. Try again in a "
                              "minute.")
        if status != 200:
            raise TranscribeError("api", self._api_message(response))
        try:
            data = response.json()
        except ValueError:
            data = None
        text = data.get("transcript") if isinstance(data, dict) else None
        if not isinstance(text, str):
            raise TranscribeError("answer", "The transcription service sent no transcript text.")
        return text

    def _api_message(self, response: Any) -> str:
        head = f"The transcription service returned an error (HTTP {response.status_code})"
        try:
            data = response.json()
            detail = data["error"]["message"] if isinstance(data, dict) else ""
        except (ValueError, KeyError, TypeError):
            detail = ""
        detail = self._clean(detail) if isinstance(detail, str) else ""
        return f"{head}: {detail}" if detail else f"{head}."


# --- the run ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TranscribeResult:
    record: TranscriptRecord  # stored, with its id
    message: str  # one line, e.g. "Transcribed 9:41 of audio in 42 pieces."
    cost: float  # USD, from the audio's length


def _piece_text(client: AsrClient, wav: Path, piece: Piece, language: str,
                keywords: Sequence[str], stop: threading.Event) -> str | TranscribeError:
    """Runs in a worker thread: the client only, no Store access. Errors are returned, not raised.
    One retry after a network or rate-limit error."""
    error: TranscribeError | None = None
    for attempt in (1, 2):
        if stop.is_set():
            return TranscribeError("api", "Stopped before this piece was sent.")
        try:
            return client.transcribe(wav_slice(wav, piece), language, keywords)
        except TranscribeError as exc:
            error = exc
            if exc.kind == "auth":
                stop.set()
                return exc
            if exc.kind not in ("network", "rate_limit") or attempt == 2:
                return exc
            _pause(RETRY_PAUSE)
        except Exception as exc:  # noqa: BLE001 - a bug must not lose the other pieces
            return TranscribeError("api", f"The transcription failed unexpectedly "
                                          f"({type(exc).__name__}).")
    return error or TranscribeError("api", "The transcription failed.")


def transcribe_video(
    store: Store,
    client: AsrClient,
    video_id: str,
    media_path: str | Path,
    language: str,
    *,
    keywords: Sequence[str] = (),
    ffmpeg: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    workdir: str | Path | None = None,
    now: datetime | None = None,
    report: Callable[[str], None] | None = None,
    workers: int = 8,
) -> TranscribeResult:
    """Transcribe the video file on this computer and store the transcript.

    `language` is the language spoken (one of `ASR_LANGUAGES`). Pieces that fail become problems
    and keep their times with empty text; if more than half fail, nothing is stored.
    """
    def say(text: str) -> None:
        if report is not None:
            report(text)

    video = store.get_video(video_id)
    if video is None:
        raise ValueError("This video is not in your catalog. Pull your videos first.")
    media = check_media_path(str(media_path))
    if language not in ASR_LANGUAGES:
        raise ValueError(
            f"Transcription does not support the language {language!r}. Lithuanian and some "
            f"other languages are not supported. Supported: "
            f"{', '.join(ASR_LANGUAGES.values())}.")
    program = ffmpeg or find_ffmpeg()
    if not program:
        raise TranscribeError("ffmpeg", NO_FFMPEG_MESSAGE)

    cleanup = tempfile.TemporaryDirectory() if workdir is None else nullcontext(str(workdir))
    with cleanup as folder:
        wav = Path(folder) / "audio.wav"
        try:
            say("Reading the audio")
            extract_audio(program, media, wav, runner)
            duration_ms = wav_duration_ms(wav)
            if duration_ms <= 0:
                raise TranscribeError("ffmpeg", "This file has no sound to transcribe.")
            silences = detect_silences(program, wav, runner)
            levels = loudness_profile(program, wav, runner)
            pieces = plan_pieces(duration_ms, silences,
                                 quietest=quietest_cut(levels) if levels else None)
            say(f"Transcribing {len(pieces)} pieces" if len(pieces) != 1
                else "Transcribing 1 piece")
            texts = _transcribe_pieces(client, wav, pieces, language, keywords, workers, say)
        finally:
            if workdir is not None:  # the caller's folder stays; our file does not
                wav.unlink(missing_ok=True)

    problems: list[str] = []
    segments: list[dict[str, Any]] = []
    for piece, text in zip(pieces, texts):
        if isinstance(text, TranscribeError):
            problems.append(f"Could not transcribe {clock(piece.start_ms)}-{clock(piece.end_ms)}: "
                            f"{text}")
        segments.append({"start_ms": piece.start_ms, "end_ms": piece.end_ms,
                         "text": "" if isinstance(text, TranscribeError) else text.strip()})
    if len(problems) * 2 > len(pieces):
        raise TranscribeError(
            "api", f"{len(problems)} of {len(pieces)} pieces could not be transcribed, so "
                   f"nothing was saved. Last problem: {problems[-1]}")
    record = TranscriptRecord(
        id=0, video_id=video_id, language=language, model=client.model, source_name=media.name,
        duration_ms=duration_ms, segments=segments, problems=problems, created_at="")
    transcript_id = store.add_transcript(record, now)
    stored = store.get_transcript(transcript_id) or record
    message = f"Transcribed {clock(duration_ms)} of audio in {len(pieces)} piece" + (
        "" if len(pieces) == 1 else "s")
    message += (f" ({len(problems)} could not be transcribed)." if problems else ".")
    return TranscribeResult(stored, message, duration_ms / 3_600_000 * ASR_PRICE_PER_HOUR)


def _transcribe_pieces(client: AsrClient, wav: Path, pieces: Sequence[Piece], language: str,
                       keywords: Sequence[str], workers: int,
                       say: Callable[[str], None]) -> list[str | TranscribeError]:
    """The text (or the error) of every piece, in piece order. An "auth" error is raised."""
    stop = threading.Event()
    results: list[str | TranscribeError | None] = [None] * len(pieces)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pieces)))) as pool:
        futures = {pool.submit(_piece_text, client, wav, piece, language, keywords, stop): index
                   for index, piece in enumerate(pieces)}
        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            say(f"{done} of {len(pieces)}")
    for result in results:
        if isinstance(result, TranscribeError) and result.kind == "auth":
            raise result
    return [r if r is not None else TranscribeError("api", "No answer.") for r in results]
