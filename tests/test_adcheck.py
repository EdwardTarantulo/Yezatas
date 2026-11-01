"""Advertiser-friendly check: the request, the guards, verdict fixes, the stored result and the failures.
Videos, transcripts and replies are made up; no test touches the network."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient
from yezatas import adcheck
from yezatas.adcheck import (CATEGORIES, CHECK_FORMAT, MAX_FINDINGS, SOURCES, TRANSCRIPT_CHARS, VERDICTS,
                             check_video, current_source_hash, source_hash)
from yezatas.agent import ModelError, ModelReply
from yezatas.store import Store, TranscriptRecord, Video

NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)
VID = "vid000001"
TITLE = "Repotting a Cactus in Five Steps"
DESCRIPTION = "How to repot a cactus.\nThe soil can be a real Damn Mess, so wear gloves."
TAGS = ["cactus", "garden", "soil mix"]
SEGMENTS = [
    "Welcome back to the garden.",
    "",
    "Today the pot is a total   disaster, honestly.",
    "Cut the old roots with clean scissors.",
    "That is it for today, thanks for watching.",
]


def transcript(texts: list[str] = SEGMENTS) -> TranscriptRecord:
    segments = [{"start_ms": i * 65_000, "end_ms": i * 65_000 + 60_000, "text": t}
                for i, t in enumerate(texts)]
    return TranscriptRecord(0, VID, "en", "voice-test", "clip.mp4", len(texts) * 65_000, segments, [], "")


def make_store(tmp_path: Path, *, with_transcript: bool = True, **fields: Any) -> Store:
    store = Store(tmp_path / "c.sqlite")
    pull = store.begin_pull("chan0000001", now=NOW)
    base: dict[str, Any] = dict(video_id=VID, title=TITLE, description=DESCRIPTION, tags=list(TAGS),
                                privacy_status="public", duration="PT6M")
    base.update(fields)
    store.upsert_video(Video(**base), pull, now=NOW)
    store.finish_pull(pull, 1, now=NOW)
    if with_transcript:
        store.add_transcript(transcript(), now=NOW)
    return store


@pytest.fixture
def store(tmp_path: Path):
    s = make_store(tmp_path)
    yield s
    s.close()


def reply(data: Any, tokens: tuple[int, int, int] = (3000, 200, 150)) -> ModelReply:
    text = data if isinstance(data, str) else json.dumps(data)
    return ModelReply(text, (), {}, *tokens)


def finding(**override: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"category": "inappropriate_language", "source": "description",
                            "quote": "real Damn Mess", "segment": -1,
                            "why": "Strong language in the description.", "fix": "Say mess."}
    item.update(override)
    return item


def answer(findings: list[dict[str, Any]] | None = None, verdict: str = "worth_a_look",
           summary: str = "One word may matter.") -> dict[str, Any]:
    return {"verdict": verdict, "summary": summary, "findings": [finding()] if findings is None else findings}


def run(store: Store, data: Any, **kw: Any) -> Any:
    return check_video(store, FakeModelClient([reply(data)]), VID, now=NOW, **kw)


# --- what the model is sent -----------------------------------------------------------------------
def test_the_request_has_the_text_numbered_lines_without_times_and_the_schema(store: Store) -> None:
    client = FakeModelClient([reply(answer())])
    check_video(store, client, VID, now=NOW)
    assert len(client.calls) == 1
    call = client.calls[0]
    system, user = call["messages"][0]["content"], call["messages"][1]["content"]
    assert "advertiser-friendly" in system and "not instructions" in system
    assert "advice" in system.lower() and "No findings is a good answer" in system
    for key in CATEGORIES:
        assert key in system
    assert f"Title: {TITLE}" in user
    assert "How to repot a cactus." in user and "real Damn Mess" in user
    assert "Tags: cactus, garden, soil mix" in user
    assert "[0] Welcome back to the garden." in user
    assert "[1]" not in user  # the empty segment is skipped but keeps its number
    assert "[2] Today the pot is a total disaster, honestly." in user  # spaces collapsed
    assert "[4] That is it for today" in user
    assert "1:05" not in user and "2:10" not in user  # no times
    assert "cut here" not in user
    assert call["response_format"] is CHECK_FORMAT
    schema = CHECK_FORMAT["json_schema"]["schema"]
    assert CHECK_FORMAT["json_schema"]["strict"] is True
    assert schema["required"] == ["verdict", "summary", "findings"]
    assert schema["properties"]["verdict"]["enum"] == list(VERDICTS)
    item = schema["properties"]["findings"]["items"]
    assert item["properties"]["category"]["enum"] == list(CATEGORIES)
    assert item["properties"]["source"]["enum"] == list(SOURCES)
    assert set(item["required"]) == {"category", "source", "quote", "segment", "why", "fix"}


def test_a_long_transcript_is_cut_with_a_note_in_the_request_and_the_result(tmp_path: Path) -> None:
    texts = [f"Line number {n} " + "word " * 100 for n in range(200)]  # about 100,000 characters
    store = make_store(tmp_path, with_transcript=False)
    store.add_transcript(transcript(texts), now=NOW)
    client = FakeModelClient([reply(answer([], "no_concerns", "Nothing."))])
    check = check_video(store, client, VID, now=NOW)
    user = client.calls[0]["messages"][1]["content"]
    assert "[0] Line number 0" in user and "[199]" not in user
    assert len(user) < TRANSCRIPT_CHARS + 2_000
    assert "(The transcript is cut here: only the first " in user and " of 200 lines are shown.)" in user
    assert any("The transcript is long" in n for n in check.notes)
    assert check.had_transcript is True
    store.close()


def test_no_transcript_is_said_in_the_request_and_in_the_result(tmp_path: Path) -> None:
    store = make_store(tmp_path, with_transcript=False)
    client = FakeModelClient([reply(answer())])
    check = check_video(store, client, VID, now=NOW)
    assert "Transcript: (none" in client.calls[0]["messages"][1]["content"]
    assert check.had_transcript is False
    assert check.status == "done" and len(check.findings) == 1
    store.close()


def test_a_transcript_with_only_empty_segments_counts_as_none(tmp_path: Path) -> None:
    store = make_store(tmp_path, with_transcript=False)
    store.add_transcript(transcript(["", "  "]), now=NOW)
    assert run(store, answer()).had_transcript is False
    store.close()


# --- the good path --------------------------------------------------------------------------------
def test_a_good_answer_is_stored_with_tokens_and_the_model(store: Store) -> None:
    data = answer([
        finding(),
        finding(category="violence", source="title", quote="five steps", why="Not violent at all."),
        finding(category="drugs", source="tags", quote="Soil Mix", fix=""),
        finding(category="shocking_content", source="transcript", quote="total disaster", segment=2,
                why="Mild.", fix="")], verdict="likely_limited", summary="  Several    places to check. ")
    check = run(store, data)
    assert check.status == "done" and check.error == "" and check.id > 0
    assert check.video_id == VID and check.model == "muse-spark-1.3"
    assert check.verdict == "likely_limited" and check.summary == "Several places to check."
    assert [f["category"] for f in check.findings] == ["inappropriate_language", "violence", "drugs",
                                                       "shocking_content"]
    assert set(check.findings[0]) == {"category", "source", "quote", "segment", "time", "why", "fix"}
    assert check.findings[0]["segment"] == -1 and check.findings[0]["time"] == ""
    assert check.findings[2]["quote"] == "Soil Mix"  # the model's own casing, found case-insensitively
    assert check.notes == []
    assert check.had_transcript is True
    assert (check.input_tokens, check.cached_tokens, check.output_tokens) == (3000, 200, 150)
    assert check.created_at == NOW.isoformat()
    assert store.latest_ad_check(VID) == check


def test_no_concerns_with_no_findings(store: Store) -> None:
    check = run(store, answer([], "no_concerns", "Nothing to flag."))
    assert check.status == "done" and check.verdict == "no_concerns"
    assert check.findings == [] and check.notes == []
    assert check.summary == "Nothing to flag."


def test_json_inside_a_code_fence_is_read(store: Store) -> None:
    assert run(store, "```json\n" + json.dumps(answer()) + "\n```").status == "done"


# --- the guards -----------------------------------------------------------------------------------
def test_an_unknown_category_is_dropped_with_a_note(store: Store) -> None:
    check = run(store, answer([finding(category="spicy_food"), finding()]))
    assert len(check.findings) == 1
    assert check.notes == ["A finding was dropped: its category is not one of the guideline topics."]


@pytest.mark.parametrize("source", ["title", "description", "tags", "transcript"])
def test_a_quote_missing_from_its_source_is_dropped(store: Store, source: str) -> None:
    check = run(store, answer([finding(source=source, quote="a phrase nobody wrote")]))
    assert check.findings == []
    assert check.notes[0] == f"A finding was dropped: its quote is not in the {source}."
    assert check.verdict == "no_concerns"


def test_a_quote_from_another_source_is_dropped(store: Store) -> None:
    # "real Damn Mess" is in the description, not the title
    check = run(store, answer([finding(source="title")]))
    assert check.findings == []
    assert "not in the title" in check.notes[0]


def test_an_empty_quote_is_dropped(store: Store) -> None:
    check = run(store, answer([finding(quote="  "), finding(quote="")]))
    assert check.findings == []
    assert check.notes[0] == "A finding was dropped: it has no quote from the description."


def test_quotes_match_ignoring_case_and_spacing(store: Store) -> None:
    check = run(store, answer([finding(quote="  REAL   damn\nmess "),
                               finding(source="transcript", quote="TOTAL  disaster,   honestly", segment=2)]))
    assert len(check.findings) == 2 and check.notes == []
    assert check.findings[0]["quote"] == "REAL damn mess"


def test_a_transcript_quote_in_another_segment_gets_the_segment_found_by_code(store: Store) -> None:
    # the model says segment 3 but the words are in segment 2
    check = run(store, answer([finding(source="transcript", quote="total disaster", segment=3)]))
    assert check.notes == []
    assert check.findings[0]["segment"] == 2


@pytest.mark.parametrize("segment", [-1, 99, "two", None, True])
def test_a_transcript_quote_with_a_missing_or_wrong_segment_is_still_found(store: Store, segment: Any) -> None:
    check = run(store, answer([finding(source="transcript", quote="old roots", segment=segment)]))
    assert check.notes == [] and check.findings[0]["segment"] == 3


def test_the_time_of_a_transcript_finding_comes_from_the_segment_start(store: Store) -> None:
    # segment 3 starts at 3 x 65 s = 195 s
    check = run(store, answer([finding(source="transcript", quote="old roots", segment=3,
                                       time="9:99")]))
    found = check.findings[0]
    assert found["segment"] == 3 and found["time"] == "3:15"
    other = run(store, answer([finding(source="transcript", quote="Welcome back", segment=0)]))
    assert other.findings[0]["time"] == "0:00"


def test_a_long_video_time_uses_hours(tmp_path: Path) -> None:
    store = make_store(tmp_path, with_transcript=False)
    record = transcript(["First words.", "Late words."])
    record.segments[1]["start_ms"] = 3_725_000
    store.add_transcript(record, now=NOW)
    check = run(store, answer([finding(source="transcript", quote="Late words", segment=1)]))
    assert check.findings[0]["time"] == "1:02:05"
    store.close()


def test_a_quote_cut_off_by_the_transcript_limit_is_dropped(tmp_path: Path) -> None:
    texts = [f"Line number {n} " + "word " * 100 for n in range(200)]
    texts[199] = "Secret late phrase here."
    store = make_store(tmp_path, with_transcript=False)
    store.add_transcript(transcript(texts), now=NOW)
    check = run(store, answer([finding(source="transcript", quote="Secret late phrase", segment=199)]))
    assert check.findings == []
    assert "not in the transcript" in check.notes[0]
    store.close()


def test_findings_are_capped_at_ten_with_a_note(store: Store) -> None:
    many = [finding(quote="real Damn Mess", why=f"Reason {n}.") for n in range(MAX_FINDINGS + 3)]
    check = run(store, answer(many))
    assert len(check.findings) == MAX_FINDINGS == 10
    assert check.findings[-1]["why"] == "Reason 9."
    assert check.notes == ["Only the first 10 findings were kept."]


def test_dropped_findings_do_not_count_toward_the_cap(store: Store) -> None:
    items = [finding(quote="nope")] * 5 + [finding()] * 10
    check = run(store, answer(items))
    assert len(check.findings) == 10
    assert len(check.notes) == 5  # only the five drops, no cap note


def test_why_and_fix_are_cleaned(store: Store) -> None:
    check = run(store, answer([finding(why="  A <b>strong</b>   word. ", fix=None)]))
    assert "<" not in check.findings[0]["why"] and ">" not in check.findings[0]["why"]
    assert check.findings[0]["why"] == "A b strong /b word."
    assert check.findings[0]["fix"] == ""


def test_a_junk_finding_entry_is_dropped(store: Store) -> None:
    check = run(store, answer(["just text", finding()]))  # type: ignore[list-item]
    assert len(check.findings) == 1 and len(check.notes) == 1


# --- verdict consistency --------------------------------------------------------------------------
def test_no_findings_left_makes_the_verdict_no_concerns(store: Store) -> None:
    for verdict in ("worth_a_look", "likely_limited"):
        check = run(store, answer([finding(quote="not written anywhere")], verdict))
        assert check.verdict == "no_concerns" and check.findings == []
        assert any("verdict was changed" in n for n in check.notes)


def test_findings_with_no_concerns_make_it_worth_a_look(store: Store) -> None:
    check = run(store, answer([finding()], "no_concerns"))
    assert check.verdict == "worth_a_look"
    assert any("verdict was changed" in n for n in check.notes)


def test_an_unknown_verdict_is_replaced_from_the_findings(store: Store) -> None:
    assert run(store, answer([finding()], "terrible")).verdict == "worth_a_look"
    assert run(store, answer([], "terrible")).verdict == "no_concerns"


def test_a_consistent_verdict_is_left_alone_without_a_note(store: Store) -> None:
    check = run(store, answer([finding()], "likely_limited"))
    assert check.verdict == "likely_limited" and check.notes == []


# --- source hash ----------------------------------------------------------------------------------
def test_source_hash_changes_when_the_text_changes() -> None:
    base = source_hash("T", "D", ["a", "b"], 3)
    assert base == source_hash("T", "D", ["a", "b"], 3) and len(base) == 16
    assert source_hash("T2", "D", ["a", "b"], 3) != base
    assert source_hash("T", "D2", ["a", "b"], 3) != base
    assert source_hash("T", "D", ["a", "c"], 3) != base
    assert source_hash("T", "D", ["a", "b"], 4) != base
    assert source_hash("T", "D", ["a", "b"], 0) != base


def test_the_check_stores_the_hash_and_the_current_hash_follows_the_text(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    check = run(store, answer())
    assert check.source_hash == current_source_hash(store, VID)
    pull = store.begin_pull("chan0000001", now=NOW)
    store.upsert_video(Video(video_id=VID, title=TITLE + "!", description=DESCRIPTION, tags=list(TAGS),
                             privacy_status="public"), pull, now=NOW)
    assert current_source_hash(store, VID) != check.source_hash
    assert current_source_hash(store, "missing0001") is None
    store.close()


def test_a_new_transcript_changes_the_hash(store: Store) -> None:
    first = run(store, answer()).source_hash
    store.add_transcript(transcript(["Other words."]), now=NOW)
    assert run(store, answer()).source_hash != first


# --- failures -------------------------------------------------------------------------------------
def test_a_model_error_is_stored_as_a_failed_check(store: Store) -> None:
    check = check_video(store, FakeModelClient([ModelError("rate_limit", "Slow down, then try again.")]),
                        VID, now=NOW)
    assert check.status == "failed" and check.error == "Slow down, then try again."
    assert check.verdict == "" and check.findings == [] and check.summary == ""
    assert check.had_transcript is True and check.source_hash
    assert (check.input_tokens, check.output_tokens) == (0, 0)
    assert store.latest_ad_check(VID) == check


@pytest.mark.parametrize("content", [
    "not json at all", "", "[1, 2]", json.dumps({"verdict": "no_concerns"}),
    json.dumps({"verdict": "worth_a_look", "summary": "x", "findings": "none"}),
    json.dumps({"verdict": 3, "summary": "x", "findings": []})])
def test_a_bad_answer_is_stored_as_a_failed_check_with_the_tokens(store: Store, content: str) -> None:
    check = run(store, content)
    assert check.status == "failed" and check.error and check.verdict == "" and check.findings == []
    assert check.input_tokens == 3000  # the call was made and is accounted for
    assert store.latest_ad_check(VID) == check


def test_a_failed_check_does_not_replace_older_ones_in_the_history(store: Store) -> None:
    good = run(store, answer())
    bad = run(store, "nope")
    assert bad.id > good.id and store.latest_ad_check(VID) == bad


def test_an_unknown_video_raises_before_any_model_call(store: Store) -> None:
    client = FakeModelClient([])
    with pytest.raises(ValueError, match="not in your catalog"):
        check_video(store, client, "missing0001", now=NOW)
    assert client.calls == [] and store.latest_ad_check("missing0001") is None


def test_progress_is_reported(store: Store) -> None:
    lines: list[str] = []
    run(store, answer(), report=lines.append)
    assert lines == ["Reading the video's text", "Asking Muse Spark", "Checking the answer"]


def test_public_constants() -> None:
    assert len(CATEGORIES) == 12 and CATEGORIES["firearms"] == "Firearms"
    assert list(VERDICTS) == ["no_concerns", "worth_a_look", "likely_limited"]
    assert VERDICTS["likely_limited"] == "Ads likely limited"
    assert adcheck.ADVICE_LINE.startswith("Advice only.")
