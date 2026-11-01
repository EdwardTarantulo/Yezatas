"""Shorts finder core tests."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from yezatas import shorts
from yezatas.store import Store, TranscriptRecord, Video
from yezatas.translate import Completion, TranslateError

NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)
VID = "vid000001"
VID2 = "vid000002"
TITLE = "Cactus"


class FakeClient:
    model = "spark-open-test"

    def __init__(self, answers: list[Any]):
        self.answers = list(answers)
        self.messages_list: list[list[dict[str, str]]] = []
        self.idx = 0

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        self.messages_list.append(messages)
        ans = self.answers[self.idx]
        self.idx += 1
        if isinstance(ans, Exception):
            raise ans
        return Completion(ans, 900, 120)


def make_store(
    tmp_path: Path, vid: str = VID, title: str = TITLE, count: int = 12
) -> Store:
    store = Store(tmp_path / "c.sqlite")
    pull = store.begin_pull("chan0000001", now=NOW)
    store.upsert_video(
        Video(
            video_id=vid,
            title=title,
            description="desc",
            tags=[],
            privacy_status="public",
            duration="PT6M",
        ),
        pull,
        now=NOW,
    )
    store.finish_pull(pull, 1, now=NOW)
    segments = []
    for i in range(count):
        text = "" if i == 4 else f"Line {i} about the Cactus pot."
        segments.append(
            {"start_ms": i * 10_000, "end_ms": i * 10_000 + 10_000, "text": text}
        )
    rec = TranscriptRecord(
        0, vid, "en", "m", "f.mp4", count * 10_000, segments, [], ""
    )
    store.add_transcript(rec, now=NOW)
    return store


def test_user_message_and_system_prompt(tmp_path: Path):
    store = make_store(tmp_path)
    client = FakeClient(
        ["<short><start>2</start><end>5</end><title>T</title><why>W</why></short>"]
    )
    result = shorts.find_shorts(store, client, VID, now=NOW)
    msgs = client.messages_list[0]
    assert msgs[0]["content"] == shorts.SYSTEM_PROMPT
    assert "never write a time" in msgs[0]["content"].lower()
    user = msgs[1]["content"]
    assert "Video title: Cactus" in user
    assert "[3] 0:30" in user
    assert "[4]" not in user
    assert store.latest_shorts_find(VID) is not None
    store.close()


def test_good_answer(tmp_path: Path):
    store = make_store(tmp_path)
    client = FakeClient(
        [
            "<short><start>2</start><end>5</end><title>My Title</title>"
            "<why>Because it is good.</why></short>"
        ]
    )
    result = shorts.find_shorts(store, client, VID, now=NOW)
    assert result.status == "found"
    assert result.message == "Found 1 moment that could work as a Short."
    assert len(result.clips) == 1
    assert result.clips[0]["start_ms"] == 20_000
    assert result.clips[0]["end_ms"] == 60_000
    quote = result.clips[0]["quote"]
    assert "Line 2" in quote
    assert "Line 3" in quote
    assert "Line 5" in quote
    assert "Line 4" not in quote
    assert "moment" in result.message.lower()
    store.close()


def test_dropped_notes(tmp_path: Path):
    store = make_store(tmp_path)
    good = (
        "<short><start>0</start><end>2</end><title>Good One</title>"
        "<why>W</why></short>"
    )
    too_short = (
        "<short><start>1</start><end>1</end><title>Short</title>"
        "<why>W</why></short>"
    )
    too_long = (
        "<short><start>0</start><end>9</end><title>Long</title>"
        "<why>W</why></short>"
    )
    overlap = (
        "<short><start>1</start><end>3</end><title>Overlap</title>"
        "<why>W</why></short>"
    )
    unknown = (
        "<short><start>99</start><end>99</end><title>Unknown</title>"
        "<why>W</why></short>"
    )
    nonnum = (
        "<short><start>two</start><end>3</end><title>Bad</title>"
        "<why>W</why></short>"
    )
    empty_title = (
        "<short><start>6</start><end>8</end><title>   </title>"
        "<why>W</why></short>"
    )
    answer = good + too_short + too_long + overlap + unknown + nonnum + empty_title
    client = FakeClient([answer])
    result = shorts.find_shorts(store, client, VID, now=NOW)
    assert result.status == "found"
    assert len(result.clips) == 1
    notes_text = " ".join(result.notes)
    assert "10 seconds long" in notes_text
    assert "100 seconds long" in notes_text
    assert "overlaps another one" in notes_text
    assert "segment numbers are not in the transcript" in notes_text
    assert "had no title" in notes_text
    store.close()


def test_max_five_and_sorted(tmp_path: Path):
    store = make_store(tmp_path, count=16)
    pairs = [(13, 14), (0, 1), (2, 3), (5, 6), (7, 8), (9, 10), (11, 12)]
    blocks = []
    for s, e in pairs:
        blocks.append(
            f"<short><start>{s}</start><end>{e}</end><title>Title {s}</title>"
            "<why>W</why></short>"
        )
    answer = "".join(blocks)
    client = FakeClient([answer])
    result = shorts.find_shorts(store, client, VID, now=NOW)
    assert len(result.clips) == 5
    assert sum("more than 5" in n for n in result.notes) == 2
    starts = [c["start_ms"] for c in result.clips]
    assert starts == sorted(starts)
    store.close()


def test_none_and_failures(tmp_path: Path):
    store = make_store(tmp_path)
    client = FakeClient(["<none/>"])
    result = shorts.find_shorts(store, client, VID, now=NOW)
    assert result.status == "none"
    assert result.clips == []
    assert result.message == (
        "No moment in this video works as a Short on its own."
    )

    client2 = FakeClient(["garbage text no tags"])
    result2 = shorts.find_shorts(store, client2, VID, now=NOW)
    assert result2.status == "failed"
    err2 = "The model's answer had no Shorts in the expected shape. Try again."
    assert result2.error == err2
    assert result2.message == result2.error

    client3 = FakeClient(
        [TranslateError("api", "The translation service returned an error.")]
    )
    result3 = shorts.find_shorts(store, client3, VID, now=NOW)
    assert result3.status == "failed"
    assert result3.error == "The translation service returned an error."
    assert result3.message == result3.error
    store.close()


def test_unknown_and_no_transcript(tmp_path: Path):
    store = make_store(tmp_path)
    client = FakeClient(["<none/>"])
    with pytest.raises(ValueError, match="This video is not in your catalog"):
        shorts.find_shorts(store, client, "vid999999", now=NOW)
    assert store.latest_shorts_find("vid999999") is None

    store2 = Store(tmp_path / "c2.sqlite")
    pull = store2.begin_pull("chan0000001", now=NOW)
    store2.upsert_video(
        Video(
            video_id="vid000003",
            title="Mesa",
            description="",
            tags=[],
            privacy_status="public",
            duration="PT2M",
        ),
        pull,
        now=NOW,
    )
    store2.finish_pull(pull, 1, now=NOW)
    with pytest.raises(ValueError, match="Transcribe the video first"):
        shorts.find_shorts(store2, client, "vid000003", now=NOW)
    assert store2.latest_shorts_find("vid000003") is None
    store.close()
    store2.close()


def test_empty_transcript_text(tmp_path: Path):
    store = Store(tmp_path / "c.sqlite")
    pull = store.begin_pull("chan0000001", now=NOW)
    store.upsert_video(
        Video(
            video_id=VID,
            title=TITLE,
            description="desc",
            tags=[],
            privacy_status="public",
            duration="PT6M",
        ),
        pull,
        now=NOW,
    )
    store.finish_pull(pull, 1, now=NOW)
    segments = [
        {"start_ms": i * 10_000, "end_ms": i * 10_000 + 10_000, "text": ""}
        for i in range(6)
    ]
    rec = TranscriptRecord(0, VID, "en", "m", "f.mp4", 60_000, segments, [], "")
    store.add_transcript(rec, now=NOW)
    client = FakeClient(["<none/>"])
    with pytest.raises(
        ValueError, match="The transcript has no text to find Shorts in."
    ):
        shorts.find_shorts(store, client, VID, now=NOW)
    assert store.latest_shorts_find(VID) is None
    store.close()


def test_draft_title_used(tmp_path: Path):
    from yezatas import edits

    store = make_store(tmp_path)
    edits.save_draft(store, VID, "Mesa draft title", "desc", [])
    client = FakeClient(
        ["<short><start>0</start><end>2</end><title>T</title><why>W</why></short>"]
    )
    shorts.find_shorts(store, client, VID, now=NOW)
    user = client.messages_list[0][1]["content"]
    assert "Video title: Mesa draft title" in user
    store.close()


def test_title_clean_and_cut(tmp_path: Path):
    store = make_store(tmp_path)
    long_title = "<b> " + "word " * 20
    title = long_title.strip()
    answer = (
        f"<short><start>0</start><end>2</end><title>{title}</title>"
        "<why>W</why></short>"
    )
    client = FakeClient([answer])
    result = shorts.find_shorts(store, client, VID, now=NOW)
    assert result.status == "found"
    t = result.clips[0]["title"]
    assert "<" not in t and ">" not in t
    assert len(t) <= 60
    assert not t.endswith("wor")
    store.close()


def test_latest_returns_newest(tmp_path: Path):
    store = make_store(tmp_path)
    client = FakeClient(
        [
            "<short><start>0</start><end>2</end><title>A</title><why>W</why></short>",
            "<short><start>6</start><end>8</end><title>B</title><why>W</why></short>",
        ]
    )
    r1 = shorts.find_shorts(store, client, VID, now=NOW)
    r2 = shorts.find_shorts(store, client, VID, now=NOW)
    latest = store.latest_shorts_find(VID)
    assert latest.id == r2.id
    assert latest.clips[0]["title"] == "B"
    store.close()
