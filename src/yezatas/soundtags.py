"""Sound captions for videos without speech: [Music], [Applause], [Hammer] ... with real times.

ffmpeg turns the video's sound into 32 kHz mono audio, PANNs (CNN14, trained on the 527 AudioSet
labels) rates every one-second window of it, and each window gets the name of its strongest label.
Neighbouring windows with the same name are joined, and a run shorter than two seconds is given
to its longer neighbour.

Every segment time is a window boundary of the real audio: whole seconds from its start, and the
end of the audio for the last one. Nothing here estimates or invents a time. A window where no
label is sure enough gets no caption. PANNs is optional (imported only when used) and its
checkpoint downloads on first use. Nothing here touches YouTube.
"""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import wave
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .store import TranscriptRecord
from .subtitles import to_srt
from .transcribe import (
    NO_FFMPEG_MESSAGE, TranscribeError, _last_line, _run, check_media_path, clock, find_ffmpeg,
    wav_duration_ms,
)

MODEL_NAME = "PANNs CNN14"
SAMPLE_RATE = 32000  # what the CNN14 model was trained on
WINDOW_MS = 1000  # one rating per second of audio
THRESHOLD = 0.3  # the strongest label must be at least this sure to caption a window
MIN_RUN_MS = 2000  # a run shorter than this is given to a neighbour
KEEP_PROBABILITY = 0.05  # the default tagger drops labels below this (keeps the answer small)
BATCH_WINDOWS = 32  # windows rated per model call
MISSING_MESSAGE = ("The sound tagger is not installed. See Configuration > Transcripts and "
                   "sounds.")

# AudioSet label -> the short word used in the caption. A label not listed here keeps its own
# AudioSet name (cut at the first comma, so "Bird vocalization, bird call, bird song" becomes
# "Bird vocalization"). Labels in IGNORED describe the room, not a sound, and are skipped.
LABEL_WORDS: dict[str, str] = {
    "Speech": "Speech", "Male speech, man speaking": "Speech",
    "Female speech, woman speaking": "Speech", "Child speech, kid speaking": "Speech",
    "Conversation": "Speech", "Narration, monologue": "Speech",
    "Music": "Music", "Musical instrument": "Music", "Background music": "Music",
    "Singing": "Singing", "Guitar": "Guitar", "Acoustic guitar": "Guitar",
    "Electric guitar": "Guitar", "Piano": "Piano", "Drum": "Drums", "Drum kit": "Drums",
    "Whistling": "Whistling",
    "Applause": "Applause", "Clapping": "Applause", "Cheering": "Cheering", "Crowd": "Crowd",
    "Laughter": "Laughter", "Giggle": "Laughter", "Chuckle, chortle": "Laughter",
    "Tap": "Tapping", "Knock": "Knocking", "Hammer": "Hammer", "Click": "Clicking",
    "Typing": "Typing", "Computer keyboard": "Typing", "Typewriter": "Typing",
    "Water": "Water", "Water tap, faucet": "Water", "Pour": "Water", "Liquid": "Water",
    "Splash, splatter": "Water", "Sink (filling or washing)": "Water", "Rain": "Rain",
    "Engine": "Engine", "Motor vehicle (road)": "Vehicle", "Vehicle": "Vehicle", "Car": "Car",
    "Drill": "Drill", "Power tool": "Power tool", "Sawing": "Sawing", "Sanding": "Sanding",
    "Mechanical fan": "Fan", "Vacuum cleaner": "Vacuum cleaner", "Sewing machine": "Sewing machine",
    "Printer": "Printer", "Walk, footsteps": "Footsteps", "Door": "Door", "Wind": "Wind",
    "Thunder": "Thunder", "Fire": "Fire", "Frying (food)": "Frying", "Boiling": "Boiling",
    "Dog": "Dog", "Bark": "Dog", "Cat": "Cat", "Meow": "Cat",
    "Bird": "Birds", "Bird vocalization, bird call, bird song": "Birds",
    "Noise": "Noise", "White noise": "Noise", "Static": "Noise", "Hum": "Hum", "Buzz": "Buzz",
    "Silence": "Silence",
}
IGNORED: frozenset[str] = frozenset({
    "Inside, small room", "Inside, large room or hall", "Inside, public space",
    "Outside, urban or manmade", "Outside, rural or natural", "Reverberation", "Echo",
    "Sound effect", "Environmental noise",
})

# What a tagger returns: one {AudioSet label: probability} per window, in time order.
Tagger = Callable[[Path], Sequence[Mapping[str, float]]]


class SoundTagError(TranscribeError):
    """The sound tagger could not run. `kind` is "missing" (PANNs is not installed), "tagger"
    (it failed or answered in an unexpected shape), or one of TranscribeError's kinds ("file",
    "ffmpeg") for the media file."""


@dataclass(frozen=True)
class SoundTagResult:
    record: TranscriptRecord  # not stored yet (id 0); segments are the sound captions
    message: str  # one line, e.g. "Found 6 sound captions in 2:10 of audio."
    cost: float  # USD: always 0, the model runs on this computer


def available() -> bool:
    """Whether the PANNs package is installed (it is not imported here)."""
    return importlib.util.find_spec("panns_inference") is not None


# --- from ratings to captions --------------------------------------------------------------------------
def label_word(label: str) -> str | None:
    """The caption word for an AudioSet label, or None for a label to skip."""
    name = label.strip()
    if not name or name in IGNORED:
        return None
    if name in LABEL_WORDS:
        return LABEL_WORDS[name]
    word = name.split(",")[0].strip()
    word = "".join(ch for ch in word if ch not in "[]<>").strip()
    return word or None


def window_words(windows: Sequence[Mapping[str, float]],
                 threshold: float = THRESHOLD) -> list[str | None]:
    """The caption word of every window: its most probable label that is at least `threshold`
    (skipped labels do not count), or None when there is none. Ties go to the label that sorts
    first, so the answer is the same every time."""
    if not 0 < threshold <= 1:
        raise ValueError("The threshold must be above 0 and at most 1.")
    words: list[str | None] = []
    for window in windows:
        ranked = sorted(((p, name) for name, p in window.items() if p >= threshold),
                        key=lambda pair: (-pair[0], pair[1]))
        words.append(next((w for w in (label_word(name) for _, name in ranked) if w), None))
    return words


def _join_runs(runs: list[list[Any]]) -> list[list[Any]]:
    joined: list[list[Any]] = []
    for word, start, end in runs:
        if joined and joined[-1][0] == word:
            joined[-1][2] = end
        else:
            joined.append([word, start, end])
    return joined


def plan_runs(words: Sequence[str | None], duration_ms: int, *,
              min_run_ms: int = MIN_RUN_MS) -> list[tuple[str | None, int, int]]:
    """Runs of the same word as (word, start_ms, end_ms) covering 0..duration_ms with no gap.

    Window `i` spans `i * WINDOW_MS` to the next whole second (the last one ends at the end of
    the audio). A run shorter than `min_run_ms` is merged into its longer neighbour (the earlier
    one on a tie), shortest first; a word of None (no caption) takes part like any other.
    """
    if len(words) != -(-max(duration_ms, 0) // WINDOW_MS):
        raise ValueError("There must be one word per second of audio.")
    runs = _join_runs([[w, i * WINDOW_MS, min((i + 1) * WINDOW_MS, duration_ms)]
                       for i, w in enumerate(words)])
    while len(runs) > 1:
        short = [i for i, r in enumerate(runs) if r[2] - r[1] < min_run_ms]
        if not short:
            break
        i = min(short, key=lambda k: (runs[k][2] - runs[k][1], k))
        if i == 0:
            target = 1
        elif i == len(runs) - 1:
            target = i - 1
        else:
            left, right = runs[i - 1], runs[i + 1]
            target = i - 1 if left[2] - left[1] >= right[2] - right[1] else i + 1
        runs[target][1] = min(runs[target][1], runs[i][1])
        runs[target][2] = max(runs[target][2], runs[i][2])
        del runs[i]
        runs = _join_runs(runs)
    return [(w, s, e) for w, s, e in runs]


def plan_segments(windows: Sequence[Mapping[str, float]], duration_ms: int, *,
                  threshold: float = THRESHOLD,
                  min_run_ms: int = MIN_RUN_MS) -> list[dict[str, Any]]:
    """Sound captions `{"start_ms", "end_ms", "text": "[Music]"}` for the per-window ratings of
    `duration_ms` of audio. Stretches with no sure label get no caption."""
    runs = plan_runs(window_words(windows, threshold), duration_ms, min_run_ms=min_run_ms)
    return [{"start_ms": start, "end_ms": end, "text": f"[{word}]"}
            for word, start, end in runs if word]


def srt_for(segments: Sequence[Mapping[str, Any]]) -> str:
    """The sound captions as an SRT file's text (the same writer the transcripts use)."""
    return to_srt(segments)


# --- the model and the audio -------------------------------------------------------------------------
def panns_tagger(device: str = "cpu") -> Tagger:
    """The real tagger: PANNs CNN14 over a 32 kHz mono 16-bit WAV, one rating per second.

    Labels below `KEEP_PROBABILITY` are left out. The last window is padded with silence for the
    model only; its caption time still ends where the audio ends. The checkpoint downloads on
    first use. Raises SoundTagError("missing") when PANNs is not installed.
    """
    if not available():
        raise SoundTagError("missing", MISSING_MESSAGE)
    try:
        import numpy as np
        from panns_inference import AudioTagging, labels
    except ImportError as exc:
        raise SoundTagError("missing", MISSING_MESSAGE) from exc
    try:
        model = AudioTagging(checkpoint_path=None, device=device)
    except Exception as exc:  # noqa: BLE001 - e.g. the checkpoint could not be downloaded
        raise SoundTagError("tagger", f"The sound tagger could not start "
                                      f"({type(exc).__name__}).") from exc
    window = SAMPLE_RATE * WINDOW_MS // 1000

    def run(wav: Path) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        try:
            with wave.open(str(wav), "rb") as fh:
                if (fh.getframerate(), fh.getnchannels(), fh.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                    raise SoundTagError("tagger", "The extracted audio is not 32 kHz mono.")
                while True:
                    raw = fh.readframes(window * BATCH_WINDOWS)
                    if not raw:
                        break
                    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                    pad = -len(audio) % window
                    if pad:
                        audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])
                    clipwise, _ = model.inference(audio.reshape(-1, window))
                    for row in clipwise:
                        out.append({labels[i]: float(row[i])
                                    for i in np.flatnonzero(row >= KEEP_PROBABILITY)})
        except SoundTagError:
            raise
        except (wave.Error, EOFError, OSError) as exc:
            raise SoundTagError("ffmpeg", "The extracted audio could not be read.") from exc
        except Exception as exc:  # noqa: BLE001 - a model failure must come out as plain English
            raise SoundTagError("tagger", f"The sound tagger failed ({type(exc).__name__}).") from exc
        return out

    return run


def extract_audio_32k(ffmpeg: str, media: Path, wav: Path,
                      runner: Callable[..., Any] = subprocess.run) -> None:
    """Writes the media's sound to `wav` as mono 16-bit 32 kHz PCM, the format PANNs wants."""
    argv = [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(media), "-vn", "-ac", "1",
            "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-map_metadata", "-1", str(wav)]
    done = _run(runner, argv)
    if done.returncode != 0:
        raise SoundTagError(
            "ffmpeg", f"ffmpeg could not read the audio of this file: {_last_line(done.stderr)}")
    if not wav.is_file():
        raise SoundTagError("ffmpeg", "ffmpeg could not read the audio of this file: "
                                      "it produced no sound.")


def tag_video(
    video_id: str,
    media_path: str | Path,
    *,
    tagger: Tagger | None = None,
    threshold: float = THRESHOLD,
    ffmpeg: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    workdir: str | Path | None = None,
    report: Callable[[str], None] | None = None,
) -> SoundTagResult:
    """Sound captions for the video file on this computer, as an (unstored) transcript record.

    `tagger(wav)` rates every second of the 32 kHz WAV (default: `panns_tagger()`); it must give
    one rating per second of audio. Raises SoundTagError("missing") before anything else when
    the default tagger is wanted but PANNs is not installed.
    """
    def say(text: str) -> None:
        if report is not None:
            report(text)

    if tagger is None:
        tagger = panns_tagger()
    media = check_media_path(str(media_path))
    program = ffmpeg or find_ffmpeg()
    if not program:
        raise SoundTagError("ffmpeg", NO_FFMPEG_MESSAGE)
    cleanup = tempfile.TemporaryDirectory() if workdir is None else nullcontext(str(workdir))
    with cleanup as folder:
        wav = Path(folder) / "sounds.wav"
        try:
            say("Reading the audio")
            extract_audio_32k(program, media, wav, runner)
            duration_ms = wav_duration_ms(wav)
            if duration_ms <= 0:
                raise SoundTagError("ffmpeg", "This file has no sound to tag.")
            say("Listening for sounds")
            windows = list(tagger(wav))
        finally:
            if workdir is not None:  # the caller's folder stays; our file does not
                wav.unlink(missing_ok=True)
    expected = -(-duration_ms // WINDOW_MS)
    if len(windows) != expected:
        raise SoundTagError("tagger", f"The sound tagger rated {len(windows)} seconds of a "
                                      f"{expected}-second audio.")
    segments = plan_segments(windows, duration_ms, threshold=threshold)
    record = TranscriptRecord(
        id=0, video_id=video_id, language="", model=MODEL_NAME, source_name=media.name,
        duration_ms=duration_ms, segments=segments, problems=[], created_at="")
    count = len(segments)
    message = (f"Found {count} sound caption" + ("" if count == 1 else "s")
               + f" in {clock(duration_ms)} of audio.")
    return SoundTagResult(record, message, 0.0)
