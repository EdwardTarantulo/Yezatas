import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient
from yezatas import chapters as chapters_module
from yezatas.agent import ModelError, ModelReply
from yezatas.chapters import (
    CHAPTER_SCHEMA, RESPONSE_FORMAT, Chapter, ChaptersResult, chapter_lines, clean_title,
    insert_chapters, suggest_chapters,
)
from yezatas.edits import save_draft
from yezatas.seo import parse_chapters
from yezatas.store import Store, TranscriptRecord, Video

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
VID = "vidA00001"
DESCRIPTION = "How to repot a cactus.\nSecond line of the opening.\n\nShop: https://shop.invalid/c\n"
TIME_IN_TEXT = re.compile(r"\b\d{1,2}(?::\d{2}){1,2}\b")


def video(**fields: Any) -> Video:
    base = Video(video_id=VID, channel_id="UC_fake", title="Cactus repotting guide",
                 description=DESCRIPTION, tags=["cactus", "repot"], privacy_status="public",
                 published_at="2026-01-01T10:00:00Z", duration="PT10M", default_language="en",
                 view_count=100)
    return replace(base, **fields)


def make_transcript(starts_s: list[int], *, language: str = "en", texts: list[str] | None = None,
                    duration_ms: int = 600_000) -> TranscriptRecord:
    segments = []
    for i, start in enumerate(starts_s):
        end = starts_s[i + 1] if i + 1 < len(starts_s) else duration_ms // 1000
        text = texts[i] if texts is not None else f"Segment {i} talks about topic {i}."
        segments.append({"start_ms": start * 1000, "end_ms": end * 1000, "text": text})
    return TranscriptRecord(0, VID, language, "voice-test", "clip.mp4", duration_ms, segments, [], "")


STARTS = [i * 30 for i in range(12)]  # 0:00, 0:30 ... 5:30


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as s:
        pull = s.begin_pull("UC_fake", NOW)
        s.upsert_video(video(), pull, NOW)
        s.add_transcript(make_transcript(STARTS), NOW)
        yield s


def answer(chapters: list[tuple[Any, Any]] | str, summary: str = "Divided by topic.",
           tokens: tuple[int, int, int] = (0, 0, 0)) -> ModelReply:
    if isinstance(chapters, str):
        content = chapters
    else:
        content = json.dumps({"chapters": [{"segment": s, "title": t} for s, t in chapters],
                              "summary": summary})
    return ModelReply(content, (), {"role": "assistant", "content": content}, *tokens)


GOOD = [(0, "Intro"), (3, "Choosing a pot"), (6, "Repotting"), (9, "Aftercare")]


def run(store: Store, script: list[Any], **kwargs: Any) -> tuple[ChaptersResult, FakeModelClient]:
    client = FakeModelClient(script)
    return suggest_chapters(store, client, VID, now=NOW, **kwargs), client


# --- formatting and insertion ---------------------------------------------------------------------
def test_chapter_lines_use_hours_only_when_needed() -> None:
    lines = chapter_lines([Chapter(0, "Intro"), Chapter(65, "Unboxing"), Chapter(3723, "End")])
    assert lines == ["0:00 Intro", "1:05 Unboxing", "1:02:03 End"]


def test_insert_after_first_paragraph() -> None:
    result = insert_chapters(DESCRIPTION, [Chapter(0, "Intro"), Chapter(70, "Soil")])
    assert result == ("How to repot a cactus.\nSecond line of the opening.\n\n"
                      "Chapters:\n0:00 Intro\n1:10 Soil\n\n"
                      "Shop: https://shop.invalid/c\n")


def test_insert_at_end_when_there_is_no_blank_line() -> None:
    chapters = [Chapter(0, "Intro")]
    assert insert_chapters("One paragraph only.", chapters) == "One paragraph only.\n\nChapters:\n0:00 Intro"
    assert insert_chapters("One paragraph.\n", chapters) == "One paragraph.\n\nChapters:\n0:00 Intro\n"


def test_insert_into_empty_description_and_after_leading_blank_lines() -> None:
    chapters = [Chapter(0, "Intro")]
    assert insert_chapters("", chapters) == "Chapters:\n0:00 Intro"
    assert insert_chapters("\n\nHello.\n\nBye.", chapters) == "\n\nHello.\n\nChapters:\n0:00 Intro\n\nBye."


def test_insert_keeps_everything_else_and_normalizes_newlines() -> None:
    result = insert_chapters("Top.\r\n\r\nLinks:\r\nhttps://a.invalid\r\n", [Chapter(0, "Intro")])
    assert result == "Top.\n\nChapters:\n0:00 Intro\n\nLinks:\nhttps://a.invalid\n"


def test_clean_title_rules() -> None:
    assert clean_title("  Setup <and> tools  ") == "Setup and tools"
    assert clean_title("0:00 Intro") == "Intro"
    assert clean_title("[1:05] - Unboxing") == "Unboxing"
    assert clean_title("2. Repotting") == "Repotting"
    assert clean_title("3 tips for soil") == "3 tips for soil"
    assert clean_title("<>") == ""
    long = "Everything you need to know about choosing the right soil mix for cacti"
    cut = clean_title(long)
    assert len(cut) <= 50 and long.startswith(cut) and not cut.endswith(" ")
    assert cut == "Everything you need to know about choosing the"
    assert len(clean_title("x" * 80)) == 50  # no space to cut at


def test_schema_is_strict() -> None:
    assert RESPONSE_FORMAT["json_schema"]["strict"] is True
    assert CHAPTER_SCHEMA["additionalProperties"] is False
    assert CHAPTER_SCHEMA["required"] == ["chapters", "summary"]
    item = CHAPTER_SCHEMA["properties"]["chapters"]["items"]
    assert item["required"] == ["segment", "title"] and item["additionalProperties"] is False


# --- the run ---------------------------------------------------------------------------------------
def test_happy_path_times_come_from_segments_and_make_a_draft(store: Store) -> None:
    result, client = run(store, [answer(GOOD, tokens=(1000, 200, 50))])

    assert result.status == "drafted" and result.run_id > 0
    assert result.chapters == (Chapter(0, "Intro"), Chapter(90, "Choosing a pot"),
                               Chapter(180, "Repotting"), Chapter(270, "Aftercare"))
    draft = store.get_draft(VID)
    assert draft is not None and draft.source == "chapters"
    assert draft.note == "Chapters from the transcript (muse-spark-1.3)"
    assert draft.description == insert_chapters(DESCRIPTION, result.chapters)
    assert "Chapters:\n0:00 Intro\n1:30 Choosing a pot\n3:00 Repotting\n4:30 Aftercare" in draft.description
    assert parse_chapters(draft.description) == [0, 90, 180, 270]
    assert (draft.title, draft.tags) == ("Cactus repotting guide", ("cactus", "repot"))
    assert store.get_video(VID).description == DESCRIPTION  # the catalog is untouched
    assert len(client.calls) == 1


def test_first_chapter_is_forced_to_segment_zero(store: Store) -> None:
    result, _ = run(store, [answer([(1, "Opening"), (5, "Middle"), (8, "End")])])
    assert result.status == "drafted"
    assert [c.start_s for c in result.chapters] == [0, 150, 240]
    assert result.chapters[0].title == "Opening"
    assert chapter_lines(result.chapters)[0] == "0:00 Opening"


def test_a_late_first_chapter_is_not_moved_but_asked_again(store: Store) -> None:
    result, client = run(store, [answer([(2, "Opening"), (5, "Middle"), (8, "End")]),
                                 answer(GOOD)])
    assert result.status == "drafted" and len(client.calls) == 2
    assert result.chapters[0] == Chapter(0, "Intro")
    assert "must start at segment 0, not segment 2" in client.calls[1]["messages"][-1]["content"]


def test_a_chapter_too_close_to_the_end_is_dropped() -> None:
    kept, problems, _ = chapters_module._repair(
        [(0, "Intro"), (1, "Middle"), (2, "Outro")], {0: 0, 1: 100, 2: 195}, {0, 1, 2}, end_s=200)
    assert [c.start_s for _, c in kept] == [0, 100]
    assert "before the end of the video" in " ".join(problems)


def test_unknown_duplicate_and_close_chapters_are_dropped() -> None:
    starts = [0, 5, 30, 60, 62, 120, 200]  # segment 1 and segment 4 are too close to the one before
    entries = [(0, "Intro"), (1, "Too close"), (2, "Soil"), (3, "Pot"), (4, "Also close"), (3, "Dup"),
               (99, "Unknown"), (-1, "Negative"), (True, "Bool"), ("5", "Text number"), (5, "Plant"),
               (6, ""), (None, "No number")]
    kept, problems, moved = chapters_module._repair(
        entries, {i: s for i, s in enumerate(starts)}, set(range(7)))
    assert [(seg, c.start_s, c.title) for seg, c in kept] == [
        (0, 0, "Intro"), (2, 30, "Soil"), (3, 60, "Pot"), (5, 120, "Plant")]
    assert not moved
    text = " ".join(problems)
    assert "Segment 1 starts less than 10 seconds" in text
    assert "Segment 4 starts less than 10 seconds" in text
    assert "Segment 3 was chosen twice" in text
    assert "Segment 99 is not one of the segments listed" in text
    assert "no usable title" in text and "no segment number" in text


def test_times_are_never_taken_from_the_model(store: Store) -> None:
    result, _ = run(store, [answer([(0, "0:00 Intro"), (3, "9:59 Soil"), (6, "7:30 Pot")])])
    assert [c.start_s for c in result.chapters] == [0, 90, 180]
    assert [c.title for c in result.chapters] == ["Intro", "Soil", "Pot"]


def test_at_most_twelve_chapters(store: Store) -> None:
    result, _ = run(store, [answer([(i, f"Part {i}") for i in range(12)] + [(0, "Again")])])
    assert len(result.chapters) == 12


def test_repair_round_asks_again_and_says_what_was_wrong(store: Store) -> None:
    first = answer([(0, "Intro"), (99, "Nowhere"), (3, "Soil"), (3, "Soil again")],
                   tokens=(1000, 100, 40))
    second = answer(GOOD, tokens=(1200, 600, 60))
    result, client = run(store, [first, second])

    assert result.status == "drafted" and len(result.chapters) == 4
    assert len(client.calls) == 2
    repair = client.calls[1]["messages"][-1]["content"]
    assert "Segment 99 is not one of the segments listed" in repair
    assert "Segment 3 was chosen twice" in repair
    assert "only 2 valid chapters" in repair
    assert client.calls[1]["messages"][-2]["role"] == "assistant"
    assert client.calls[1]["response_format"] == RESPONSE_FORMAT
    saved = store.get_agent_run(result.run_id)
    assert (saved.input_tokens, saved.cached_tokens, saved.output_tokens) == (2200, 700, 100)
    assert any(e["type"] == "problems" and e["round"] == 1 for e in saved.log)


def test_unreadable_first_answer_gets_the_repair_round(store: Store) -> None:
    result, client = run(store, [answer("I think the video has parts."), answer(GOOD)])
    assert result.status == "drafted" and len(client.calls) == 2
    assert "not valid JSON" in client.calls[1]["messages"][-1]["content"]


def test_answer_wrapped_in_a_code_fence_is_read(store: Store) -> None:
    fenced = "```json\n" + json.dumps({"chapters": [{"segment": s, "title": t} for s, t in GOOD],
                                      "summary": "x"}) + "\n```"
    result, client = run(store, [answer(fenced)])
    assert result.status == "drafted" and len(client.calls) == 1


def test_still_too_few_after_repair_fails_and_drafts_nothing(store: Store) -> None:
    result, client = run(store, [answer([(0, "Intro"), (1, "Too close")]),
                                 answer([(0, "Intro"), (50, "Nowhere")])])
    assert result.status == "failed" and result.chapters == ()
    assert "at least 3" in result.message
    assert len(client.calls) == 2
    assert store.get_draft(VID) is None
    saved = store.get_agent_run(result.run_id)
    assert saved.status == "failed" and saved.finished_at is not None


def test_model_error_is_a_failed_run(store: Store) -> None:
    result, _ = run(store, [ModelError("rate_limit", "The model is busy. Try again soon.")])
    assert result.status == "failed" and result.message == "The model is busy. Try again soon."
    saved = store.get_agent_run(result.run_id)
    assert saved.status == "failed"
    assert saved.log[-1] == {"type": "error", "kind": "rate_limit",
                             "message": "The model is busy. Try again soon."}
    assert store.get_draft(VID) is None


def test_unexpected_error_closes_the_run_and_raises(store: Store) -> None:
    client = FakeModelClient([RuntimeError("boom")])
    with pytest.raises(RuntimeError):
        suggest_chapters(store, client, VID, now=NOW)
    (saved,) = store.list_agent_runs(VID)
    assert saved.status == "failed" and saved.finished_at is not None


def test_run_is_recorded_like_the_suggestion_agent(store: Store) -> None:
    lines: list[str] = []
    result, _ = run(store, [answer(GOOD, summary="Four parts.", tokens=(900, 100, 70))],
                    report=lines.append)
    (saved,) = store.list_agent_runs(VID)
    assert saved.id == result.run_id and saved.model == "muse-spark-1.3"
    assert saved.status == "drafted" and saved.summary == "Four parts."
    assert (saved.input_tokens, saved.cached_tokens, saved.output_tokens) == (900, 100, 70)
    assert saved.result["chapters"][1] == {"start_s": 90, "title": "Choosing a pot"}
    assert saved.log[0]["type"] == "prompt" and saved.log[0]["segments"] == 12
    assert lines[0] == "Reading the transcript" and lines[-1] == "Saving the draft"


# --- refusals --------------------------------------------------------------------------------------
def refused(result: ChaptersResult, client: FakeModelClient, store: Store) -> None:
    assert result.status == "failed" and result.chapters == () and result.run_id == 0
    assert client.calls == [] and store.get_draft(VID) is None and store.list_agent_runs() == []


def test_no_transcript(tmp_path: Path) -> None:
    with Store(tmp_path / "n.sqlite") as s:
        s.upsert_video(video(), s.begin_pull("UC_fake", NOW), NOW)
        result, client = run(s, [])
        assert result == ChaptersResult("failed", "Transcribe the video first.", (), 0)
        refused(result, client, s)


def test_unknown_video(store: Store) -> None:
    client = FakeModelClient([])
    result = suggest_chapters(store, client, "nope", now=NOW)
    assert result.status == "failed" and "not in your catalog" in result.message and client.calls == []


def test_short_video_is_refused(tmp_path: Path) -> None:
    with Store(tmp_path / "s.sqlite") as s:
        s.upsert_video(video(duration="PT3M59S"), s.begin_pull("UC_fake", NOW), NOW)
        s.add_transcript(make_transcript(STARTS), NOW)
        result, client = run(s, [])
        assert "4 minutes or more" in result.message
        refused(result, client, s)


def test_unknown_catalog_duration_falls_back_to_the_transcript(tmp_path: Path) -> None:
    with Store(tmp_path / "d.sqlite") as s:
        s.upsert_video(video(duration=""), s.begin_pull("UC_fake", NOW), NOW)
        s.add_transcript(make_transcript(STARTS, duration_ms=200_000), NOW)
        result, _ = run(s, [])
        assert result.status == "failed" and "4 minutes" in result.message


def test_existing_timestamps_in_the_description_are_never_rewritten(tmp_path: Path) -> None:
    with Store(tmp_path / "t.sqlite") as s:
        own = "Intro text.\n\n0:00 Start\n1:10 Middle\n"
        s.upsert_video(video(description=own), s.begin_pull("UC_fake", NOW), NOW)
        s.add_transcript(make_transcript(STARTS), NOW)
        result, client = run(s, [])
        assert "already has timestamps" in result.message
        refused(result, client, s)


def test_existing_timestamps_in_the_draft_are_refused_too(store: Store) -> None:
    save_draft(store, VID, "Cactus repotting guide", "Intro.\n\n0:00 Start\n", ["cactus"], now=NOW)
    client = FakeModelClient([])
    result = suggest_chapters(store, client, VID, now=NOW)
    assert result.status == "failed" and "already has timestamps" in result.message
    assert client.calls == [] and store.list_agent_runs() == []


def test_transcript_without_text_is_refused(tmp_path: Path) -> None:
    with Store(tmp_path / "e.sqlite") as s:
        s.upsert_video(video(), s.begin_pull("UC_fake", NOW), NOW)
        s.add_transcript(make_transcript(STARTS, texts=[""] * 12), NOW)
        result, client = run(s, [])
        assert "no text" in result.message
        refused(result, client, s)


# --- drafts ----------------------------------------------------------------------------------------
def test_chapters_go_on_top_of_an_existing_draft(store: Store) -> None:
    save_draft(store, VID, "A better title for the cactus guide",
               "Fresh opening line.\n\nMy own links below.\n", ["cactus", "succulent"],
               source="manual", now=NOW)
    result, _ = run(store, [answer(GOOD)])
    assert result.status == "drafted"
    draft = store.get_draft(VID)
    assert draft.title == "A better title for the cactus guide"
    assert draft.tags == ("cactus", "succulent")
    assert draft.description == ("Fresh opening line.\n\nChapters:\n0:00 Intro\n1:30 Choosing a pot\n"
                                 "3:00 Repotting\n4:30 Aftercare\n\nMy own links below.\n")
    assert draft.source == "chapters"
    assert draft.base_description == DESCRIPTION and draft.approved_at is None


def test_over_the_byte_limit_fails_and_drafts_nothing(tmp_path: Path) -> None:
    with Store(tmp_path / "b.sqlite") as s:
        long = "é" * 2480  # 4,960 bytes: fine alone, too big with the chapter block
        s.upsert_video(video(description=long), s.begin_pull("UC_fake", NOW), NOW)
        s.add_transcript(make_transcript(STARTS), NOW)
        assert len(long.encode("utf-8")) <= 5000 < len(long.encode("utf-8")) + 70
        result, _ = run(s, [answer(GOOD)])
        assert result.status == "failed" and "do not fit in the description" in result.message
        assert s.get_draft(VID) is None
        assert s.get_agent_run(result.run_id).status == "failed"


# --- the prompt ------------------------------------------------------------------------------------
def prompt_of(call: dict[str, Any]) -> tuple[str, str]:
    system, user = call["messages"][0], call["messages"][1]
    assert (system["role"], user["role"]) == ("system", "user")
    return system["content"], user["content"]


def test_prompt_has_numbered_segments_and_no_other_times(tmp_path: Path) -> None:
    starts = [0, 45, 61, 125, 3600, 3725]
    texts = ["Hello and welcome.", "", "Choosing soil.", "Pots.", "Long break later.", "Goodbye."]
    with Store(tmp_path / "p.sqlite") as s:
        s.upsert_video(video(duration="PT1H5M"), s.begin_pull("UC_fake", NOW), NOW)
        s.add_transcript(make_transcript(starts, texts=texts, duration_ms=3_900_000), NOW)
        _, client = run(s, [answer([(0, "Intro"), (2, "Soil"), (3, "Pots")])])
    system, user = prompt_of(client.calls[0])
    assert 'Video title: Cactus repotting guide' in user
    assert "Transcript language code: en" in user
    assert "[0] 0:00 Hello and welcome." in user
    assert "[1]" not in user  # the empty segment is skipped, its number stays unused
    assert "[2] 1:01 Choosing soil." in user
    assert "[3] 2:05 Pots." in user
    assert "[4] 1:00:00 Long break later." in user
    assert "[5] 1:02:05 Goodbye." in user
    found = set(TIME_IN_TEXT.findall(system + user))
    assert found == {"0:00", "1:01", "2:05", "1:00:00", "1:02:05"}  # each segment's own start only
    assert "segment 0" in system and "3 to 12" in system and "50 characters" in system
    assert client.calls[0]["response_format"] == RESPONSE_FORMAT and client.calls[0]["tools"] is None


def test_segment_number_of_an_empty_segment_is_not_offered(tmp_path: Path) -> None:
    starts = [0, 45, 90, 150, 210, 300]
    texts = ["Intro.", "", "Soil.", "Pots.", "Care.", "Bye."]
    with Store(tmp_path / "q.sqlite") as s:
        s.upsert_video(video(), s.begin_pull("UC_fake", NOW), NOW)
        s.add_transcript(make_transcript(starts, texts=texts), NOW)
        result, _ = run(s, [answer([(0, "Intro"), (1, "Empty"), (2, "Soil"), (3, "Pots")])])
    assert [c.start_s for c in result.chapters] == [0, 90, 150]


def test_long_transcript_is_merged_under_the_first_number_and_stays_near_the_cap(
        tmp_path: Path) -> None:
    count = 600
    starts = [i * 15 for i in range(count)]
    texts = [f"{'word ' * 59}end{i}" for i in range(count)]  # about 300 characters each
    with Store(tmp_path / "m.sqlite") as s:
        s.upsert_video(video(duration="PT3H"), s.begin_pull("UC_fake", NOW), NOW)
        s.add_transcript(make_transcript(starts, texts=texts, duration_ms=9_000_000), NOW)
        # 1 is merged away (offered: 0, 3, 6 ...): dropped. 3 and 300 are offered.
        result, client = run(s, [answer([(0, "Intro"), (1, "Merged away"), (3, "Soil"), (300, "Mid")])])
    _, user = prompt_of(client.calls[0])
    assert len(user) < chapters_module.PROMPT_CHAR_LIMIT + 1000
    numbers = [int(n) for n in re.findall(r"^\[(\d+)\]", user, re.MULTILINE)]
    assert numbers == list(range(0, count, 3))
    assert [c.start_s for c in result.chapters] == [0, 45, 4500]
    assert found_times_match(user, starts)


def found_times_match(user: str, starts: list[int]) -> bool:
    for number, stamp in re.findall(r"^\[(\d+)\] (\d+(?::\d{2}){1,2}) ", user, re.MULTILINE):
        seconds = starts[int(number)]
        h, rest = divmod(seconds, 3600)
        want = f"{h}:{rest // 60:02d}:{rest % 60:02d}" if h else f"{rest // 60}:{rest % 60:02d}"
        if stamp != want:
            return False
    return True


def test_chapters_module_only_imports_local_modules() -> None:
    source = Path(chapters_module.__file__).read_text(encoding="utf-8")
    imports = re.findall(r"^(?:from|import) (\S+)", source, re.MULTILINE)
    assert not [m for m in imports if "google" in m or "httpx" in m or "requests" in m]
    assert not [m for m in imports if m.lstrip(".") in ("youtube", "push", "auth")]
