import io
import json
import random
import threading
import time
import wave
from array import array
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import httpx
import pytest

from yezatas import transcribe
from yezatas.store import Store, Video
from yezatas.transcribe import (
    ASR_LANGUAGES, ASR_MODEL, ASR_PRICE_PER_HOUR, MetaAsrClient, Piece, TranscribeError,
    check_media_path, clock, detect_silences, extract_audio, find_ffmpeg, loudness_profile,
    plan_pieces, quietest_cut,
    transcribe_video, wav_slice,
)

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
V1 = "vidA00001"
KEY = "sk-test-SECRET-0123456789"
RATE = 16000


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    pauses: list[float] = []
    monkeypatch.setattr(transcribe, "_pause", pauses.append)
    return pauses


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as s:
        pull = s.begin_pull("UC_fake", NOW)
        s.upsert_video(Video(video_id=V1, channel_id="UC_fake", title="Repotting a cactus",
                             description="How to repot.", tags=["cactus"],
                             privacy_status="public", published_at="2026-01-01T10:00:00Z",
                             duration="PT1M10S", default_language="en"), pull, NOW)
        yield s


@pytest.fixture
def media(tmp_path: Path) -> Path:
    path = tmp_path / "My Video.MP4"
    path.write_bytes(b"not really a video")
    return path


# --- fakes --------------------------------------------------------------------------------------------
def write_wav(path: Path, seconds: int, rate: int = RATE) -> None:
    """A mono 16-bit WAV in which the sample value of every frame is its whole second, so a slice
    tells the fake client where it starts."""
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        for second in range(seconds):
            fh.writeframes(array("h", [second]) * rate if second else b"\0\0" * rate)


def start_second(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes), "rb") as fh:
        frames = fh.readframes(1)
    return int(array("h", frames)[0]) if frames else -1


SILENCE_STDERR = """\
Input #0, wav, from 'audio.wav':
[silencedetect @ 0000021] silence_start: 12.5
[silencedetect @ 0000021] silence_end: 13.5 | silence_duration: 1
[silencedetect @ 0000021] silence_start: 30
[silencedetect @ 0000021] silence_end: 31.002 | silence_duration: 1.002
size=N/A time=00:01:10.00 bitrate=N/A speed= 800x
"""


class FakeRunner:
    """Stands in for subprocess.run: records argv, writes a real WAV for the extract step and
    answers the silence step with canned stderr."""

    def __init__(self, seconds: int = 70, stderr: str = SILENCE_STDERR, returncode: int = 0,
                 fail_stderr: str = "") -> None:
        self.seconds, self.stderr, self.returncode = seconds, stderr, returncode
        self.fail_stderr = fail_stderr
        self.calls: list[list[str]] = []
        self.wavs: list[Path] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> Any:
        self.calls.append(list(argv))
        assert kwargs.get("capture_output") and kwargs.get("text")
        if self.returncode:
            return SimpleNamespace(returncode=self.returncode, stdout="", stderr=self.fail_stderr)
        if "-vn" in argv:
            self.wavs.append(Path(argv[-1]))
            write_wav(Path(argv[-1]), self.seconds)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr=self.stderr)


class FakeAsr:
    """Text is `s<second the slice starts at>`; `fail` maps start second -> errors to raise."""

    model = ASR_MODEL

    def __init__(self, fail: dict[int, list[Exception]] | None = None, jitter: bool = False) -> None:
        self.fail = fail or {}
        self.jitter = jitter
        self.calls: list[tuple[int, str, tuple[str, ...]]] = []
        self.lock = threading.Lock()

    def transcribe(self, wav_bytes: bytes, language: str, keywords: Sequence[str] = ()) -> str:
        second = start_second(wav_bytes)
        with self.lock:
            self.calls.append((second, language, tuple(keywords)))
            queue = self.fail.get(second)
            error = queue.pop(0) if queue else None
        if self.jitter:
            time.sleep(0.02 if second < 25 else 0)  # early pieces finish last
        if error is not None:
            raise error
        return f"  s{second}  "


def run(store: Store, client: Any, media: Path, runner: FakeRunner | None = None, **kw: Any):
    runner = runner or FakeRunner()
    return transcribe_video(store, client, V1, media, kw.pop("language", "en"), ffmpeg="ffmpeg",
                            runner=runner, now=NOW, **kw), runner


# --- small helpers --------------------------------------------------------------------------------------
def test_constants_and_clock() -> None:
    assert ASR_MODEL == "muse-voice-transcribe-1.0" and ASR_PRICE_PER_HOUR == 0.18
    assert len(ASR_LANGUAGES) == 20 and "lt" not in ASR_LANGUAGES and "en" in ASR_LANGUAGES
    assert ".mp4" in transcribe.VIDEO_SUFFIXES
    assert clock(0) == "0:00" and clock(61_999) == "1:01" and clock(3_723_000) == "1:02:03"


def test_find_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcribe.shutil, "which", lambda name: "/bin/" + name)
    assert find_ffmpeg() == "/bin/ffmpeg"
    monkeypatch.setattr(transcribe.shutil, "which", lambda name: None)
    assert find_ffmpeg() is None


# --- media path ---------------------------------------------------------------------------------------------
def test_check_media_path_accepts_quotes_and_spaces(media: Path) -> None:
    assert check_media_path(f'  "{media}"  ') == media
    assert check_media_path(f"'{media}'") == media


def test_check_media_path_refusals(tmp_path: Path) -> None:
    (tmp_path / "folder.mp4").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    for raw in ("", '  ""  ', str(tmp_path / "missing.mp4"), str(tmp_path / "folder.mp4"),
                str(tmp_path / "notes.txt")):
        with pytest.raises(TranscribeError) as info:
            check_media_path(raw)
        assert info.value.kind == "file" and str(info.value).endswith(".")


# --- ffmpeg commands ---------------------------------------------------------------------------------------------
def test_extract_audio_argv(tmp_path: Path, media: Path) -> None:
    runner = FakeRunner()
    wav = tmp_path / "a.wav"
    extract_audio("ffmpeg", media, wav, runner)
    assert runner.calls == [[
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(media), "-vn", "-ac", "1",
        "-ar", "16000", "-c:a", "pcm_s16le", "-map_metadata", "-1", str(wav)]]
    assert wav.is_file()


def test_extract_audio_failure_names_last_stderr_line(tmp_path: Path, media: Path) -> None:
    runner = FakeRunner(returncode=1, fail_stderr="first\nOutput file does not contain any stream\n\n")
    with pytest.raises(TranscribeError) as info:
        extract_audio("ffmpeg", media, tmp_path / "a.wav", runner)
    assert info.value.kind == "ffmpeg"
    assert str(info.value) == ("ffmpeg could not read the audio of this file: "
                               "Output file does not contain any stream")


def test_extract_audio_runner_oserror(tmp_path: Path, media: Path) -> None:
    def broken(argv: list[str], **kw: Any) -> Any:
        raise FileNotFoundError("nope")

    with pytest.raises(TranscribeError) as info:
        extract_audio("ffmpeg", media, tmp_path / "a.wav", broken)
    assert info.value.kind == "ffmpeg"


def test_detect_silences_argv_and_parsing(tmp_path: Path) -> None:
    runner = FakeRunner()
    wav = tmp_path / "a.wav"
    assert detect_silences("ffmpeg", wav, runner) == [(12500, 13500), (30000, 31002)]
    assert runner.calls == [["ffmpeg", "-nostdin", "-i", str(wav), "-af",
                             "silencedetect=noise=-35dB:d=0.35", "-f", "null", "-"]]


def test_detect_silences_edge_cases(tmp_path: Path) -> None:
    stderr = ("[silencedetect @ 1] silence_start: -0.0013\n"
              "[silencedetect @ 1] silence_end: 0.9 | silence_duration: 0.9\n"
              "[silencedetect @ 1] silence_start: 50.25\n")  # runs to the end: no end, no pair
    assert detect_silences("ffmpeg", tmp_path / "a.wav", FakeRunner(stderr=stderr)) == [(0, 900)]
    assert detect_silences("ffmpeg", tmp_path / "a.wav", FakeRunner(stderr="")) == []
    with pytest.raises(TranscribeError) as info:
        detect_silences("ffmpeg", tmp_path / "a.wav",
                        FakeRunner(returncode=1, fail_stderr="bad filter"))
    assert info.value.kind == "ffmpeg"


# --- plan_pieces --------------------------------------------------------------------------------------------------
def check_plan(pieces: list[Piece], duration: int, *, min_ms: int = 4000, max_ms: int = 30000) -> None:
    assert pieces[0].start_ms == 0 and pieces[-1].end_ms == duration
    for a, b in zip(pieces, pieces[1:]):
        assert a.end_ms == b.start_ms  # no gap, no overlap
    for p in pieces:
        assert min_ms <= p.end_ms - p.start_ms <= max_ms


def test_plan_pieces_empty_and_short() -> None:
    assert plan_pieces(0, []) == []
    assert plan_pieces(2500, [(1000, 1500)]) == [Piece(0, 2500)]  # shorter than min: one piece
    assert plan_pieces(11_000, [(5000, 6000)]) == [Piece(0, 11_000)]


def test_plan_pieces_cuts_at_silence_middles_near_target() -> None:
    silences = [(5000, 5400), (11_800, 12_200), (20_000, 21_000)]
    pieces = plan_pieces(40_000, silences)
    assert pieces[0] == Piece(0, 12_000)  # the silence nearest 12 s, in its middle
    assert pieces[1].start_ms == 12_000
    check_plan(pieces, 40_000)


def test_plan_pieces_hard_cut_without_silence() -> None:
    pieces = plan_pieces(75_000, [])
    assert pieces == [Piece(0, 30_000), Piece(30_000, 60_000), Piece(60_000, 75_000)]
    check_plan(pieces, 75_000)


def test_plan_pieces_remainder_never_shorter_than_min() -> None:
    pieces = plan_pieces(61_000, [])  # 30 + 30 would leave 1 s
    check_plan(pieces, 61_000)
    assert pieces[-1].end_ms - pieces[-1].start_ms >= 4000
    # a silence 1 s before the end is too close to the end to cut at
    pieces = plan_pieces(30_500, [(29_500, 29_900)])
    check_plan(pieces, 30_500)


def test_plan_pieces_ignores_silences_too_close_to_the_start() -> None:
    pieces = plan_pieces(20_000, [(500, 700), (1000, 1200)])
    assert all(p.start_ms != 600 for p in pieces)
    check_plan(pieces, 20_000)


def test_plan_pieces_properties_and_determinism() -> None:
    rng = random.Random(7)
    for _ in range(200):
        duration = rng.randint(4000, 400_000)
        silences = sorted((s, s + rng.randint(350, 3000))
                          for s in (rng.randint(0, duration) for _ in range(rng.randint(0, 80))))
        first = plan_pieces(duration, silences)
        check_plan(first, duration)
        assert first == plan_pieces(duration, list(reversed(silences)))  # same input set, same plan
        assert first == plan_pieces(duration, silences)


def test_plan_pieces_rejects_bad_limits() -> None:
    with pytest.raises(ValueError):
        plan_pieces(10_000, [], min_ms=5000, target_ms=1000, max_ms=2000)


def test_loudness_profile_argv_and_parsing(tmp_path: Path) -> None:
    out = ("frame:0    pts:0       pts_time:0\nlavfi.astats.Overall.RMS_level=-inf\n"
           "frame:1    pts:800     pts_time:0.05\nlavfi.astats.Overall.RMS_level=-21.5\n"
           "frame:2    pts:1600    pts_time:0.1\nlavfi.astats.Overall.RMS_level=nan\n")
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    assert loudness_profile("ffmpeg", tmp_path / "a.wav", runner) == [-120.0, -21.5, -120.0]
    assert "asetnsamples=n=800:p=0" in calls[0][calls[0].index("-af") + 1]
    with pytest.raises(TranscribeError) as err:
        loudness_profile("ffmpeg", tmp_path / "a.wav",
                         lambda argv, **kw: SimpleNamespace(returncode=1, stdout="", stderr="bad"))
    assert err.value.kind == "ffmpeg"


def test_quietest_cut_finds_the_dip_and_prefers_the_target() -> None:
    levels = [-20.0] * 600  # 30 s of speech at -20 dB, 50 ms frames
    for frame in (100, 101, 102):  # a dip at 5.05 s
        levels[frame] = -40.0
    for frame in (239, 240, 241):  # an equally deep dip at 12 s
        levels[frame] = -40.0
    choose = quietest_cut(levels)
    assert choose(4000, 26000, 12000) == 12000  # same depth: the one at the target wins
    assert choose(4000, 10000, 12000) == 5050  # the only dip in reach
    flat = quietest_cut([-20.0] * 600)
    assert flat(4000, 26000, 12000) == 12000  # no dip: the target itself
    assert quietest_cut([])(4000, 26000, 12000) == 12000


def test_plan_pieces_uses_the_quietest_moment_without_silences() -> None:
    picks: list[tuple[int, int, int]] = []

    def quietest(low: int, high: int, target: int) -> int:
        picks.append((low, high, target))
        return target + 1500

    pieces = plan_pieces(60000, [], quietest=quietest)
    assert [p.start_ms for p in pieces] == [0, 13500, 27000, 40500, 54000]
    assert picks[0] == (4000, 30000, 12000)
    assert pieces[-1].end_ms == 60000
    # a chooser answering outside the allowed range is clamped into it
    assert plan_pieces(60000, [], quietest=lambda low, high, t: 10**9)[0].end_ms == 30000


# --- wav_slice ------------------------------------------------------------------------------------------------------
def test_wav_slice_is_a_valid_wav_of_the_right_length(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    write_wav(wav, 20)
    data = wav_slice(wav, Piece(3000, 8500))
    with wave.open(io.BytesIO(data), "rb") as fh:
        assert (fh.getnchannels(), fh.getsampwidth(), fh.getframerate()) == (1, 2, RATE)
        assert fh.getnframes() == 5500 * RATE // 1000
        assert array("h", fh.readframes(1))[0] == 3  # starts at second 3
    assert start_second(wav_slice(wav, Piece(0, 1000))) == 0


def test_wav_slice_clamps_to_the_file(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    write_wav(wav, 5)
    with wave.open(io.BytesIO(wav_slice(wav, Piece(4000, 9000))), "rb") as fh:
        assert fh.getnframes() == RATE


# --- transcribe_video: refusals --------------------------------------------------------------------------------------
def test_refuses_before_any_work(store: Store, media: Path, tmp_path: Path) -> None:
    runner, client = FakeRunner(), FakeAsr()
    with pytest.raises(ValueError, match="not in your catalog"):
        transcribe_video(store, client, "nope", media, "en", ffmpeg="ffmpeg", runner=runner)
    with pytest.raises(TranscribeError) as info:
        transcribe_video(store, client, V1, tmp_path / "missing.mp4", "en", ffmpeg="ffmpeg",
                         runner=runner)
    assert info.value.kind == "file"
    with pytest.raises(ValueError) as lang:
        transcribe_video(store, client, V1, media, "lt", ffmpeg="ffmpeg", runner=runner)
    assert "'lt'" in str(lang.value) and "Lithuanian" in str(lang.value)
    assert runner.calls == [] and client.calls == [] and store.latest_transcript(V1) is None


def test_refuses_without_ffmpeg(store: Store, media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcribe, "find_ffmpeg", lambda: None)
    runner = FakeRunner()
    with pytest.raises(TranscribeError) as info:
        transcribe_video(store, FakeAsr(), V1, media, "en", runner=runner)
    assert info.value.kind == "ffmpeg" and "Install ffmpeg and restart Yezatas" in str(info.value)
    assert runner.calls == []


# --- transcribe_video: the run ---------------------------------------------------------------------------------------
def test_run_stores_segments_with_real_cut_times(store: Store, media: Path) -> None:
    messages: list[str] = []
    client = FakeAsr(jitter=True)
    result, runner = run(store, client, media, keywords=["Yezatas"], report=messages.append)
    record = result.record
    stored = store.latest_transcript(V1)
    assert stored is not None and stored.id == record.id
    assert (record.language, record.model, record.source_name) == ("en", ASR_MODEL, "My Video.MP4")
    assert record.duration_ms == 70_000 and record.problems == []
    # pieces are in order, contiguous, and the text belongs to the piece that was sent
    segments = record.segments
    assert segments[0]["start_ms"] == 0 and segments[-1]["end_ms"] == 70_000
    for a, b in zip(segments, segments[1:]):
        assert a["end_ms"] == b["start_ms"]
    assert [s["text"] for s in segments] == [f"s{s['start_ms'] // 1000}" for s in segments]
    assert all(call[1:] == ("en", ("Yezatas",)) for call in client.calls)
    assert len(client.calls) == len(segments)
    # silences were at 12.5-13.5 s and 30-31.002 s: the first cut is at their middle
    assert segments[0]["end_ms"] == 13_000
    assert result.message == f"Transcribed 1:10 of audio in {len(segments)} pieces."
    assert result.cost == pytest.approx(70 / 3600 * 0.18)
    assert messages[0] == "Reading the audio" and messages[1] == f"Transcribing {len(segments)} pieces"
    assert messages[-1] == f"{len(segments)} of {len(segments)}"
    assert "silencedetect=noise=-35dB:d=0.35" in runner.calls[1]


def test_temp_dir_is_removed(store: Store, media: Path) -> None:
    result, runner = run(store, FakeAsr(), media)
    assert runner.wavs and not runner.wavs[0].exists() and not runner.wavs[0].parent.exists()


def test_temp_dir_is_removed_after_a_failure(store: Store, media: Path) -> None:
    client = FakeAsr(fail={0: [TranscribeError("auth", "refused")]})
    runner = FakeRunner()
    with pytest.raises(TranscribeError):
        run(store, client, media, runner)
    assert not runner.wavs[0].parent.exists()


def test_given_workdir_stays_but_our_file_goes(store: Store, media: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    run(store, FakeAsr(), media, workdir=work)
    assert work.is_dir() and list(work.iterdir()) == []


def test_ffmpeg_failure_stores_nothing(store: Store, media: Path) -> None:
    with pytest.raises(TranscribeError) as info:
        run(store, FakeAsr(), media, FakeRunner(returncode=1, fail_stderr="Invalid data"))
    assert info.value.kind == "ffmpeg" and store.latest_transcript(V1) is None


def test_retry_after_network_and_rate_limit(store: Store, media: Path, no_sleep: list[float]) -> None:
    client = FakeAsr(fail={0: [TranscribeError("network", "down")],
                           13: [TranscribeError("rate_limit", "slow down")]})
    result, _ = run(store, client, media)
    assert result.record.problems == []
    assert all(s["text"].startswith("s") for s in result.record.segments)
    assert len(no_sleep) == 2
    assert sum(1 for call in client.calls if call[0] == 0) == 2


def test_failed_piece_keeps_its_times(store: Store, media: Path) -> None:
    bad = [TranscribeError("api", "Malformed request"), TranscribeError("api", "again")]
    client = FakeAsr(fail={13: bad})  # second piece: no retry for an "api" error
    result, _ = run(store, client, media)
    record = result.record
    second = record.segments[1]
    assert second["start_ms"] == 13_000 and second["text"] == ""
    assert record.problems == [f"Could not transcribe 0:13-{clock(second['end_ms'])}: Malformed request"]
    assert [s["text"] for s in record.segments if s is not second]  # the others kept their text
    assert "(1 could not be transcribed)." in result.message
    assert len(record.segments) == len(plan_pieces(70_000, [(12_500, 13_500), (30_000, 31_002)]))


def test_piece_that_fails_twice_is_a_problem(store: Store, media: Path) -> None:
    client = FakeAsr(fail={0: [TranscribeError("network", "x"), TranscribeError("network", "y")]})
    result, _ = run(store, client, media)
    assert result.record.segments[0]["text"] == "" and len(result.record.problems) == 1
    assert result.record.problems[0].endswith(": y")


def test_unexpected_client_error_is_a_problem_not_a_crash(store: Store, media: Path) -> None:
    result, _ = run(store, FakeAsr(fail={0: [KeyError("secret internals")]}), media)
    assert "KeyError" in result.record.problems[0] and "secret internals" not in result.record.problems[0]


def test_empty_text_keeps_times(store: Store, media: Path) -> None:
    class Quiet(FakeAsr):
        def transcribe(self, wav_bytes: bytes, language: str, keywords: Sequence[str] = ()) -> str:
            return "   " if start_second(wav_bytes) == 13 else super().transcribe(wav_bytes, language)

    result, _ = run(store, Quiet(), media)
    assert result.record.segments[1] == {"start_ms": 13_000, "end_ms": result.record.segments[1]["end_ms"],
                                         "text": ""}
    assert result.record.problems == []  # silence is not a failure


def test_auth_error_stops_everything(store: Store, media: Path) -> None:
    client = FakeAsr(fail={0: [TranscribeError("auth", "The key was refused.")]})
    with pytest.raises(TranscribeError) as info:
        run(store, client, media, workers=1)
    assert info.value.kind == "auth"
    assert len(client.calls) == 1  # nothing was sent after the refusal
    assert store.latest_transcript(V1) is None


def test_more_than_half_failing_stores_nothing(store: Store, media: Path) -> None:
    pieces = plan_pieces(70_000, [(12_500, 13_500), (30_000, 31_002)])
    starts = [p.start_ms // 1000 for p in pieces]
    fail = {s: [TranscribeError("api", "boom")] for s in starts[: len(starts) // 2 + 1]}
    with pytest.raises(TranscribeError) as info:
        run(store, FakeAsr(fail=fail), media)
    assert info.value.kind == "api" and "nothing was saved" in str(info.value)
    assert store.latest_transcript(V1) is None


def test_exactly_half_failing_is_kept(store: Store, media: Path) -> None:
    pieces = plan_pieces(70_000, [(12_500, 13_500), (30_000, 31_002)])
    assert len(pieces) % 2 == 0
    fail = {p.start_ms // 1000: [TranscribeError("api", "boom")] for p in pieces[: len(pieces) // 2]}
    result, _ = run(store, FakeAsr(fail=fail), media)
    assert len(result.record.problems) == len(pieces) // 2


# --- MetaAsrClient ------------------------------------------------------------------------------------------------------
def parts_of(request: httpx.Request) -> dict[str, tuple[dict[str, str], bytes]]:
    """Multipart body -> {part name: (headers, content)}."""
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    parts: dict[str, tuple[dict[str, str], bytes]] = {}
    for chunk in request.content.split(b"--" + boundary)[1:-1]:
        head, _, body = chunk.partition(b"\r\n\r\n")
        headers = {}
        for line in head.decode().strip().splitlines():
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()
        name = headers["content-disposition"].split('name="')[1].split('"')[0]
        parts[name] = (headers, body[:-2] if body.endswith(b"\r\n") else body)
    return parts


def make_client(handler: Any, **kw: Any) -> MetaAsrClient:
    return MetaAsrClient(KEY, client=httpx.Client(transport=httpx.MockTransport(handler)), **kw)


def test_meta_client_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"sessionId": "s", "transcript": "hello there",
                                         "audioDurationMs": 1000, "turns": []})

    client = make_client(handler)
    assert client.transcribe(b"RIFFfake", "en", ["Yezatas", " ", "Kalanchoe"]) == "hello there"
    request = seen[0]
    assert request.method == "POST" and str(request.url) == "https://api.meta.ai/v1/asr/transcribe"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["content-type"].startswith("multipart/form-data")
    parts = parts_of(request)
    assert set(parts) == {"request", "audio"}
    assert parts["request"][0]["content-type"] == "application/json"
    body = json.loads(parts["request"][1])
    assert body == {"model": ASR_MODEL, "audioEncoding": "WAV", "languageBias": ["en"],
                    "keywords": ["Yezatas", "Kalanchoe"]}
    assert isinstance(body["languageBias"], list) and "mode" not in body
    assert parts["audio"][1] == b"RIFFfake"
    assert KEY.encode() not in request.content  # the key is only in the header


def test_meta_client_without_keywords_omits_them() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(parts_of(request)["request"][1]))
        return httpx.Response(200, json={"transcript": ""})

    assert make_client(handler, base_url="https://example.invalid/v9/").transcribe(b"x", "de") == ""
    assert "keywords" not in seen[0] and seen[0]["languageBias"] == ["de"]


@pytest.mark.parametrize("status, kind", [(401, "auth"), (403, "auth"), (429, "rate_limit"),
                                          (400, "api"), (500, "api"), (503, "api")])
def test_meta_client_error_mapping(status: int, kind: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"type": "bad", "message": "Malformed request"}})

    with pytest.raises(TranscribeError) as info:
        make_client(handler).transcribe(b"x", "en")
    assert info.value.kind == kind
    if kind == "api":
        assert f"HTTP {status}" in str(info.value) and "Malformed request" in str(info.value)


def test_meta_client_non_json_error_and_bad_answers() -> None:
    with pytest.raises(TranscribeError) as info:
        make_client(lambda r: httpx.Response(502, text="<html>bad gateway</html>")).transcribe(b"x", "en")
    assert info.value.kind == "api" and str(info.value).endswith("(HTTP 502).")
    for payload in ({"transcript": 5}, {"other": 1}, ["x"]):
        with pytest.raises(TranscribeError) as bad:
            make_client(lambda r, p=payload: httpx.Response(200, json=p)).transcribe(b"x", "en")
        assert bad.value.kind == "answer"
    with pytest.raises(TranscribeError) as text:
        make_client(lambda r: httpx.Response(200, text="not json")).transcribe(b"x", "en")
    assert text.value.kind == "answer"


def test_meta_client_network_errors() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    for handler in (refuse, slow):
        with pytest.raises(TranscribeError) as info:
            make_client(handler).transcribe(b"x", "en")
        assert info.value.kind == "network"


def test_the_key_never_appears_in_errors_or_repr() -> None:
    def leaky(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"type": "x", "message": f"bad key {KEY} given"}})

    client = make_client(leaky)
    with pytest.raises(TranscribeError) as info:
        client.transcribe(b"x", "en")
    assert KEY not in str(info.value) and "***" in str(info.value)
    assert KEY not in repr(client) and KEY not in str(client.__dict__.get("model"))
    for status in (401, 429):
        with pytest.raises(TranscribeError) as other:
            make_client(lambda r, s=status: httpx.Response(s, json={"error": {"message": KEY}})
                        ).transcribe(b"x", "en")
        assert KEY not in str(other.value)


def test_meta_client_through_transcribe_video(store: Store, media: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"transcript": f"text {calls['n']}"})

    result, _ = run(store, make_client(handler), media, workers=1)
    assert [s["text"] for s in result.record.segments] == [
        f"text {i}" for i in range(1, len(result.record.segments) + 1)]
