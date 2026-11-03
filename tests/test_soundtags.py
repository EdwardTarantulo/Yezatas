import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from yezatas import soundtags
from yezatas.soundtags import (
    MISSING_MESSAGE, MODEL_NAME, SoundTagError, available, label_word, panns_tagger,
    plan_runs, plan_segments, srt_for, tag_video, window_words,
)
from yezatas.transcribe import TranscribeError

V1 = "vidA00001"


def win(label: str = "", p: float = 0.9, **more: float) -> dict[str, float]:
    out = dict(more)
    if label:
        out[label] = p
    return out


def segs(windows: Sequence[Mapping[str, float]], duration_ms: int | None = None, **kw: Any):
    return plan_segments(windows, len(windows) * 1000 if duration_ms is None else duration_ms, **kw)


# --- labels and threshold -----------------------------------------------------------------------------
def test_label_words_come_from_the_curated_map() -> None:
    assert label_word("Tap") == "Tapping"
    assert label_word("Male speech, man speaking") == "Speech"
    assert label_word("Water tap, faucet") == "Water"


def test_unknown_labels_keep_their_audioset_name_cut_at_the_comma() -> None:
    assert label_word("Accordion") == "Accordion"
    assert label_word("Hiss, hissing sound") == "Hiss"
    assert label_word("Odd <b>[x]") == "Odd bx"  # brackets and angle brackets are dropped


def test_scene_labels_and_blank_names_are_skipped() -> None:
    assert label_word("Inside, small room") is None
    assert label_word("  ") is None


def test_window_takes_the_most_probable_label_at_or_above_the_threshold() -> None:
    windows = [win("Music", 0.31, Speech=0.30), win("Music", 0.29), win("Applause", 0.3)]
    assert window_words(windows) == ["Music", None, "Applause"]
    assert window_words(windows, threshold=0.2) == ["Music", "Music", "Applause"]


def test_skipped_labels_do_not_hide_the_next_label() -> None:
    windows = [{"Inside, small room": 0.95, "Hammer": 0.4}, {"Reverberation": 0.9}]
    assert window_words(windows) == ["Hammer", None]


def test_a_tie_goes_to_the_label_that_sorts_first() -> None:
    assert window_words([{"Tap": 0.5, "Applause": 0.5}]) == ["Applause"]


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_bad_thresholds_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        window_words([{}], bad)


# --- merging and short runs ---------------------------------------------------------------------------
def test_consecutive_windows_with_the_same_word_merge() -> None:
    windows = [win("Music")] * 4 + [win("Applause")] * 3
    assert segs(windows) == [
        {"start_ms": 0, "end_ms": 4000, "text": "[Music]"},
        {"start_ms": 4000, "end_ms": 7000, "text": "[Applause]"},
    ]


def test_labels_with_the_same_word_merge_too() -> None:
    assert len(segs([win("Pour")] * 2 + [win("Liquid")] * 2)) == 1  # both are "Water"
    assert len(segs([win("Tap")] * 2 + [win("Knock")] * 2)) == 2  # Tapping, Knocking


def test_a_short_run_goes_to_its_longer_neighbour() -> None:
    windows = [win("Music")] * 5 + [win("Hammer")] + [win("Applause")] * 3
    assert segs(windows) == [
        {"start_ms": 0, "end_ms": 6000, "text": "[Music]"},
        {"start_ms": 6000, "end_ms": 9000, "text": "[Applause]"},
    ]


def test_a_short_run_between_the_same_word_joins_them() -> None:
    windows = [win("Music")] * 3 + [win("Hammer")] + [win("Music")] * 3
    assert segs(windows) == [{"start_ms": 0, "end_ms": 7000, "text": "[Music]"}]


def test_a_tie_between_neighbours_goes_to_the_earlier_one() -> None:
    windows = [win("Music")] * 3 + [win("Hammer")] + [win("Applause")] * 3
    assert [s["text"] for s in segs(windows)] == ["[Music]", "[Applause]"]
    assert segs(windows)[0]["end_ms"] == 4000


def test_short_runs_at_either_end_are_absorbed() -> None:
    windows = [win("Hammer")] + [win("Music")] * 4 + [win("Hammer")]
    assert segs(windows) == [{"start_ms": 0, "end_ms": 6000, "text": "[Music]"}]


def test_a_short_unlabelled_gap_is_absorbed_but_a_long_one_stays_empty() -> None:
    short_gap = [win("Music")] * 3 + [{}] + [win("Music")] * 3
    assert segs(short_gap) == [{"start_ms": 0, "end_ms": 7000, "text": "[Music]"}]
    long_gap = [win("Music")] * 3 + [{}] * 3 + [win("Applause")] * 3
    assert segs(long_gap) == [
        {"start_ms": 0, "end_ms": 3000, "text": "[Music]"},
        {"start_ms": 6000, "end_ms": 9000, "text": "[Applause]"},
    ]


def test_a_video_shorter_than_the_minimum_run_keeps_its_only_run() -> None:
    assert segs([win("Music")]) == [{"start_ms": 0, "end_ms": 1000, "text": "[Music]"}]


def test_the_last_window_ends_where_the_audio_ends() -> None:
    windows = [win("Music")] * 3
    assert segs(windows, duration_ms=2400) == [{"start_ms": 0, "end_ms": 2400, "text": "[Music]"}]
    assert segs(windows, duration_ms=2400)[-1]["end_ms"] == 2400


def test_all_unlabelled_gives_no_captions() -> None:
    assert segs([{}] * 5) == []
    assert segs([]) == []


def test_runs_cover_the_audio_without_gaps_or_overlaps() -> None:
    words = ["a"] * 2 + [None] * 3 + ["b"] + ["c"] * 4 + ["a"] * 3
    runs = plan_runs(words, 12_500)
    assert runs[0][1] == 0 and runs[-1][2] == 12_500
    assert all(x[2] == y[1] for x, y in zip(runs, runs[1:]))
    assert all(e - s >= 2000 for _, s, e in runs)


def test_runs_need_one_word_per_second() -> None:
    with pytest.raises(ValueError):
        plan_runs(["a", "b"], 5000)


# --- SRT ----------------------------------------------------------------------------------------------
def test_srt_output_uses_the_segment_times() -> None:
    srt = srt_for(segs([win("Music")] * 3 + [win("Applause")] * 2))
    assert srt == ("1\n00:00:00,000 --> 00:00:03,000\n[Music]\n\n"
                   "2\n00:00:03,000 --> 00:00:05,000\n[Applause]\n")
    assert srt_for([]) == ""


# --- PANNs is optional --------------------------------------------------------------------------------
def test_available_reflects_whether_the_package_can_be_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(soundtags.importlib.util, "find_spec", lambda name: None)
    assert available() is False
    monkeypatch.setattr(soundtags.importlib.util, "find_spec", lambda name: object())
    assert available() is True


def test_missing_panns_is_a_plain_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(soundtags, "available", lambda: False)
    with pytest.raises(SoundTagError) as err:
        panns_tagger()
    assert err.value.kind == "missing"
    assert str(err.value) == MISSING_MESSAGE
    assert MISSING_MESSAGE == ("The sound tagger is not installed. See Configuration > "
                               "Transcripts and sounds.")
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    ran: list[Any] = []
    with pytest.raises(SoundTagError, match="not installed"):
        tag_video(V1, media, ffmpeg="ffmpeg", runner=lambda *a, **k: ran.append(a))
    assert ran == []  # nothing was run before the check


# --- the whole run, with a fake ffmpeg and a fake tagger ----------------------------------------------
def fake_runner(seconds: float, rate: int = 8000, code: int = 0):
    """Stands in for ffmpeg: writes a silent WAV of `seconds` to the last argument."""
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv)
        if code == 0:
            with wave.open(argv[-1], "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(rate)
                fh.writeframes(b"\0\0" * int(rate * seconds))
        return SimpleNamespace(returncode=code, stdout="", stderr="line one\nBad input\n")

    run.calls = calls  # type: ignore[attr-defined]
    return run


@pytest.fixture
def media(tmp_path: Path) -> Path:
    path = tmp_path / "Clip One.MP4"
    path.write_bytes(b"not really a video")
    return path


def test_tag_video_builds_an_unstored_record(media: Path) -> None:
    seen: list[Path] = []

    def tagger(wav: Path) -> list[dict[str, float]]:
        seen.append(wav)
        return [win("Music")] * 4 + [win("Applause")] * 2 + [win("Laughter", 0.2)]

    runner = fake_runner(6.5)
    result = tag_video(V1, media, tagger=tagger, ffmpeg="ffmpeg", runner=runner)
    argv = runner.calls[0]
    assert argv[argv.index("-ar") + 1] == "32000" and argv[argv.index("-ac") + 1] == "1"
    record = result.record
    assert record.id == 0 and record.video_id == V1
    assert record.model == MODEL_NAME and record.language == "" and record.problems == []
    assert record.source_name == "Clip One.MP4"
    assert record.duration_ms == 6500
    assert record.segments == [
        {"start_ms": 0, "end_ms": 4000, "text": "[Music]"},
        {"start_ms": 4000, "end_ms": 6500, "text": "[Applause]"},
    ]
    assert result.cost == 0.0
    assert result.message == "Found 2 sound captions in 0:06 of audio."
    assert not seen[0].exists()  # the temporary audio is gone


def test_tag_video_removes_only_its_own_file_from_a_given_folder(media: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    tag_video(V1, media, tagger=lambda wav: [win("Music")] * 3, ffmpeg="ffmpeg",
              runner=fake_runner(2.5), workdir=work)
    assert list(work.iterdir()) == []


def test_tag_video_checks_the_number_of_ratings(media: Path) -> None:
    with pytest.raises(SoundTagError) as err:
        tag_video(V1, media, tagger=lambda wav: [win("Music")] * 2, ffmpeg="ffmpeg",
                  runner=fake_runner(5))
    assert err.value.kind == "tagger"


def test_tag_video_reports_ffmpeg_failure_in_plain_english(media: Path) -> None:
    with pytest.raises(SoundTagError, match="Bad input") as err:
        tag_video(V1, media, tagger=lambda wav: [], ffmpeg="ffmpeg", runner=fake_runner(1, code=1))
    assert err.value.kind == "ffmpeg"


def test_tag_video_rejects_a_missing_file_and_missing_ffmpeg(tmp_path: Path, media: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(TranscribeError) as err:
        tag_video(V1, tmp_path / "nope.mp4", tagger=lambda wav: [])
    assert err.value.kind == "file"
    monkeypatch.setattr(soundtags, "find_ffmpeg", lambda: None)
    with pytest.raises(SoundTagError, match="ffmpeg was not found"):
        tag_video(V1, media, tagger=lambda wav: [])


def test_tag_video_says_what_it_is_doing(media: Path) -> None:
    said: list[str] = []
    tag_video(V1, media, tagger=lambda wav: [win("Music")] * 2, ffmpeg="ffmpeg",
              runner=fake_runner(1.5), report=said.append)
    assert said == ["Reading the audio", "Listening for sounds"]


def test_a_silent_file_with_no_labels_stores_no_captions(media: Path) -> None:
    result = tag_video(V1, media, tagger=lambda wav: [{}] * 3, ffmpeg="ffmpeg",
                       runner=fake_runner(3))
    assert result.record.segments == []
    assert result.message == "Found 0 sound captions in 0:03 of audio."
