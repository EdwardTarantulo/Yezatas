import json
import re
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fakes import FakeOpener
from yezatas import subtitle_translate as st
from yezatas.store import Store, TranscriptRecord, Video
from yezatas.subtitle_translate import (
    back_segment_messages, check_segments, compare_segment_messages, language_name, make_batches,
    parse_segments, segment_messages, srt_for, translate_subtitles,
)
from yezatas.translate import ChatTranslator, Completion, OllamaVerifier, TranslateError

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
V1 = "vidA00001"
LINK = "https://shop.invalid/cactus"
SEGMENTS = [
    {"start_ms": 0, "end_ms": 3000, "text": "Hello and welcome."},
    {"start_ms": 3000, "end_ms": 4000, "text": ""},  # silence
    {"start_ms": 4000, "end_ms": 9000, "text": f"Today the plant costs 610 dollars, see {LINK}"},
    {"start_ms": 9000, "end_ms": 12000, "text": "Thanks for watching, 1,000 thanks!"},
]


def seg_tags(messages: list[dict[str, str]]) -> list[tuple[int, str]]:
    return [(int(n), t) for n, t in re.findall(r'<s n="(\d+)">(.*?)</s>', messages[-1]["content"],
                                               re.DOTALL)]


def tag_text(items: list[tuple[int, str]]) -> str:
    return "\n".join(f'<s n="{n}">{t}</s>' for n, t in items)


def prefixed(messages: list[dict[str, str]]) -> str:
    """A perfect translation: every segment keeps its text behind a '~' (links and numbers survive)."""
    return tag_text([(n, "~" + t) for n, t in seg_tags(messages)])


class FakeModel:
    """A TextModel. `script` items are used in order: a str (the reply), a Completion, an exception
    to raise or a callable(messages); when it runs out, `default` answers."""

    def __init__(self, script: list[Any] | None = None, default: Any = prefixed,
                 model: str = "spark-test") -> None:
        self.model = model
        self.script = list(script or [])
        self.default = default
        self.calls: list[list[dict[str, str]]] = []
        self._lock = threading.Lock()

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        with self._lock:
            self.calls.append(messages)
            item = self.script.pop(0) if self.script else self.default
        if callable(item):
            item = item(messages)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, Completion) else Completion(item, 10, 20)


def undo_prefix(messages: list[dict[str, str]]) -> str:
    return tag_text([(n, t.removeprefix("~")) for n, t in seg_tags(messages)])


class FakeChecker:
    """A Checker. A call with a schema is the comparison (answered with JSON), a call without is the
    back-translation. `script` items work like FakeModel's, shared by both kinds of call."""

    def __init__(self, script: list[Any] | None = None, up: bool = True,
                 back: Any = undo_prefix, verdict: Any = None, model: str = "glimmer-test") -> None:
        self.model = model
        self.script = list(script or [])
        self.up = up
        self.back = back
        self.verdict = verdict if verdict is not None else {"score": 5, "issues": []}
        self.calls: list[tuple[list[dict[str, str]], Any]] = []

    def available(self) -> bool:
        return self.up

    def chat(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> Any:
        self.calls.append((messages, schema))
        item = self.script.pop(0) if self.script else (
            json.dumps(self.verdict) if schema is not None else self.back)
        if callable(item):
            item = item(messages)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as s:
        pull = s.begin_pull("UC_fake", NOW)
        s.upsert_video(Video(video_id=V1, channel_id="UC_fake", title="Repotting", description="x",
                             default_language="en"), pull, NOW)
        s.add_transcript(TranscriptRecord(0, V1, "en", "asr-test", "clip.mp4", 12000, SEGMENTS, [], ""),
                         NOW)
        yield s


def run(store: Store, model: Any = None, checker: Any = None, targets: Any = ("es",), **kw: Any) -> Any:
    kw.setdefault("workers", 1)
    kw.setdefault("now", NOW)
    return translate_subtitles(store, model or FakeModel(), checker, V1, list(targets), **kw)


# --- prompts ------------------------------------------------------------------------------------------
def test_prompts_carry_numbers_and_text_but_never_a_time(store: Store) -> None:
    model, checker = FakeModel(), FakeChecker()
    run(store, model, checker)
    assert len(model.calls) == 1 and len(checker.calls) == 2
    messages = [model.calls[0]] + [m for m, _ in checker.calls]
    for prompt in messages:
        text = json.dumps(prompt)
        for time in ("3000", "4000", "9000", "12000", "start_ms", "end_ms", "00:00", "-->"):
            assert time not in text
    system, user = model.calls[0][0]["content"], model.calls[0][1]["content"]
    assert "English" in system and "Spanish" in system and "never merge, split, move" in system
    assert "@handles" in system and "hashtags" in system and "URLs" in system
    assert [n for n, _ in seg_tags(model.calls[0])] == [1, 3, 4]  # the silent segment 2 is not sent
    assert user.splitlines()[0] == '<s n="1">Hello and welcome.</s>'
    assert "exactly" not in user


def test_insist_adds_the_count_rule_to_the_user_turn() -> None:
    messages = segment_messages("en", "es", [(3, "a"), (7, "b")], insist=True)
    assert messages[1]["content"].startswith(
        "Your answer must have exactly 2 <s> tags, numbered 3 to 7, every one translated into Spanish.")
    assert "exactly" not in segment_messages("en", "es", [(3, "a")])[1]["content"]


def test_language_names_cover_the_transcript_codes() -> None:
    assert language_name("de") == "German" and language_name("zh") == "Chinese"
    assert language_name("th") == "Thai" and language_name("zh-Hans") == "Simplified Chinese"
    assert language_name("xx") == "xx" and language_name("th-TH") == "Thai"


# --- parsing the answer -------------------------------------------------------------------------------
def test_parse_segments_reads_tags_and_cleans_text() -> None:
    content = 'Sure!\n<s n="1"> Hola   mundo\n  </s>\n<s n="3">Adiós<|eot|></s>'
    assert parse_segments(content, [1, 3]) == {1: "Hola mundo", 3: "Adiós"}


def test_parse_segments_last_one_wins() -> None:
    content = '<s n="1">draft</s> reasoning <s n="2">x</s>\n<s n="1">final</s><s n="2">y</s>'
    assert parse_segments(content, [1, 2]) == {1: "final", 2: "y"}


@pytest.mark.parametrize("content, expected_text", [
    (None, "sent no text"), ("", "sent no text"), ("no tags at all", "missing segments 1, 2"),
    ('<s n="1">a</s>', "missing segment 2"),
    ('<s n="1">a</s><s n="2">b</s><s n="9">c</s>', "segment 9 that was not asked for"),
    ('<s n="1">a</s><s n="2">  </s>', "no text for segment 2"),
    ('<s n="1">a</s><s n="2">b', "missing segment 2"),
])
def test_parse_segments_bad_answers(content: str | None, expected_text: str) -> None:
    with pytest.raises(TranslateError, match=expected_text) as raised:
        parse_segments(content, [1, 2])
    assert raised.value.kind == "answer"


# --- batching -----------------------------------------------------------------------------------------
def test_batches_close_by_count_or_by_characters() -> None:
    items = [(n, "x") for n in range(1, 131)]
    assert [len(b) for b in make_batches(items)] == [60, 60, 10]
    items = [(n, "y" * 2000) for n in range(1, 8)]  # 3 per batch: 6000 chars is the limit
    assert [len(b) for b in make_batches(items)] == [3, 3, 1]
    assert [len(b) for b in make_batches([(1, "a"), (2, "z" * 9000), (3, "b")])] == [1, 1, 1]
    assert make_batches([]) == []


def test_a_long_transcript_is_sent_in_several_calls_in_order(store: Store, monkeypatch) -> None:
    monkeypatch.setattr(st, "BATCH_SEGMENTS", 2)
    model = FakeModel()
    result = run(store, model)
    assert [[n for n, _ in seg_tags(m)] for m in model.calls] == [[1, 3], [4]]
    assert [s["text"] for s in result.records[0].segments] == [
        "~Hello and welcome.", "", f"~Today the plant costs 610 dollars, see {LINK}",
        "~Thanks for watching, 1,000 thanks!"]
    assert result.records[0].input_tokens == 20 and result.records[0].output_tokens == 40


def test_several_languages_run_in_parallel_and_keep_target_order(store: Store) -> None:
    result = run(store, FakeModel(), targets=["de", "fr", "pt"], workers=3)
    assert [r.language for r in result.records] == ["de", "fr", "pt"]
    assert [r.status for r in result.records] == ["unverified"] * 3


# --- retries, failures, the key -----------------------------------------------------------------------
def test_a_bad_answer_is_retried_once_with_the_count_rule(store: Store) -> None:
    model = FakeModel(["I cannot do that.", prefixed])
    record = run(store, model).records[0]
    assert len(model.calls) == 2 and record.status == "unverified"
    assert "exactly 3 <s> tags" not in model.calls[0][1]["content"]
    assert "exactly 3 <s> tags, numbered 1 to 4" in model.calls[1][1]["content"]


def test_network_and_rate_limit_errors_are_retried_without_the_rule(store: Store) -> None:
    for kind in ("network", "rate_limit"):
        model = FakeModel([TranslateError(kind, "Try later."), prefixed])
        assert run(store, model).records[0].status == "unverified"
        assert len(model.calls) == 2 and "exactly" not in model.calls[1][1]["content"]


def test_two_bad_answers_fail_the_language_with_no_partial_subtitles(store: Store) -> None:
    model = FakeModel(["nope", '<s n="1">only one</s>'])
    result = run(store, model, FakeChecker())
    record = result.records[0]
    assert len(model.calls) == 2
    assert record.status == "failed" and record.segments == [] and record.score is None
    assert record.problems == ["The translator's answer is missing segments 3, 4."]
    assert result.message == "1 failed"
    assert (record.input_tokens, record.output_tokens) == (20, 40)  # the failed calls still cost
    assert store.get_subtitle_translation(record.id) == record


def test_other_errors_are_not_retried_and_one_bad_batch_fails_the_whole_language(
        store: Store, monkeypatch) -> None:
    monkeypatch.setattr(st, "BATCH_SEGMENTS", 1)
    # batches: [1], [3], [4]; the second one hits an api error
    model = FakeModel([prefixed, TranslateError("api", "The service broke."), prefixed])
    result = run(store, model, targets=["es"])
    assert len(model.calls) == 3
    assert result.records[0].status == "failed" and result.records[0].problems == ["The service broke."]


def test_an_unexpected_error_in_the_translator_fails_that_language_only(store: Store) -> None:
    def boom(messages: list[dict[str, str]]) -> str:
        if "German" in messages[0]["content"]:
            raise KeyError("x")
        return prefixed(messages)

    result = run(store, FakeModel(default=boom), targets=["de", "es"])
    assert [r.status for r in result.records] == ["failed", "unverified"]
    assert result.records[0].problems == ["The translator failed unexpectedly (KeyError)."]


def test_a_refused_key_aborts_and_stores_nothing(store: Store) -> None:
    def refuse_french(messages: list[dict[str, str]]) -> str:
        if "French" in messages[0]["content"]:
            raise TranslateError("auth", "refused")
        return prefixed(messages)

    model = FakeModel(default=refuse_french)
    with pytest.raises(TranslateError) as raised:
        run(store, model, targets=["es", "fr", "de"])
    assert raised.value.kind == "auth" and "key was refused" in str(raised.value)
    assert store.list_subtitle_translations(V1) == []
    model = FakeModel([TranslateError("auth", "refused")])
    with pytest.raises(TranslateError):
        run(store, model)
    assert len(model.calls) == 1  # never retried


# --- times and empty segments -------------------------------------------------------------------------
def test_times_are_copied_from_the_transcript_and_empty_segments_stay_empty(store: Store) -> None:
    record = run(store).records[0]
    assert len(record.segments) == len(SEGMENTS)
    for stored, original in zip(record.segments, SEGMENTS):
        assert (stored["start_ms"], stored["end_ms"]) == (original["start_ms"], original["end_ms"])
        assert set(stored) == {"start_ms", "end_ms", "text"}
    assert record.segments[1]["text"] == ""
    assert record.transcript_id == store.latest_transcript(V1).id
    assert (record.translator, record.source_language, record.language) == ("spark-test", "en", "es")


def test_a_model_that_writes_times_cannot_change_them(store: Store) -> None:
    def sneaky(messages: list[dict[str, str]]) -> str:
        return tag_text([(n, f"~{t} 00:00:01,000 --> 00:00:02,000") for n, t in seg_tags(messages)])

    record = run(store, FakeModel(default=sneaky)).records[0]
    assert [(s["start_ms"], s["end_ms"]) for s in record.segments] == [
        (s["start_ms"], s["end_ms"]) for s in SEGMENTS]


# --- code checks --------------------------------------------------------------------------------------
def test_check_segments_is_quiet_for_a_good_translation() -> None:
    assert check_segments([1, 2], ["Costs 1,000 dollars.", "Hi @sample_garden"],
                          ["Cuesta 1.000 dólares.", "Hola @sample_garden"]) == []


def test_a_missing_number_is_found_but_separators_do_not_matter() -> None:
    assert check_segments([7], ["It costs 610 dollars."], ["Cuesta mucho."]) == [
        'The number "610" in segment 7 is missing from the translation.']
    assert check_segments([1], ["That is 1,000."], ["Das sind 1.000."]) == []
    assert check_segments([1], ["That is 1.000 euros."], ["That is 1,000 euros."]) == []
    assert check_segments([1], ["About 3.5 meters."], ["Etwa 3,5 Meter."]) == []
    problems = check_segments([1], ["Buy 2 and 2 and 5."], ["Kaufe 2 und 5."])
    assert problems == ['The number "2" in segment 1 is missing from the translation.']  # a multiset
    assert check_segments([1], ["Sector 610"], ["القطاع ٦١٠"]) == []  # other digits are digits
    # a space (also a no-break or narrow no-break space) between groups of three
    assert check_segments([1], ["It costs 1,000 euros."], ["Ça coûte 1 000 euros."]) == []
    assert check_segments([1], ["About 25,000.5 km."], ["Около 25 000,5 км."]) == []
    assert check_segments([1], ["Over 1,500 people."], ["Více než 1 500 lidí."]) == []
    assert check_segments([1], ["Pages 5 and 100."], ["Pages 5 et 100."]) == []


def test_an_added_number_is_fine() -> None:
    assert check_segments([1], ["Take two pots."], ["Toma 2 macetas."]) == []


def test_angle_brackets_are_a_problem() -> None:
    assert check_segments([4], ["Press the button."], ["Press <b>the button</b>."]) == [
        "Segment 4 contains < or >, which YouTube does not allow."]
    assert check_segments([4], ["a"], ["a > b"]) == ["Segment 4 contains < or >, which YouTube does not allow."]


def test_a_translation_much_longer_than_the_source_is_a_problem() -> None:
    assert check_segments([2], ["Hi."], ["x" * 39]) == []
    assert check_segments([2], ["Hi."], ["x" * 40]) == [
        "Segment 2 is much longer than the original; something may have been added."]


def test_links_and_the_like_are_checked_over_the_whole_text() -> None:
    source = [f"Shop: {LINK}", "Mail hi@example.invalid", "Follow @sample_garden #cactus", "Chapter 0:00"]
    assert check_segments([1, 2, 3, 4], source, [f"Tienda: {LINK}", "Correo hi@example.invalid",
                                                 "Sigue @sample_garden #cactus", "Capítulo 0:00"]) == []
    # a link moved to another segment is not a problem for the whole text
    assert check_segments([1, 2], [f"See {LINK}", "ok"], ["Mira", f"vale {LINK}"]) == []
    problems = check_segments([1, 2, 3], [f"See {LINK}", "Mail hi@example.invalid", "plain"],
                              ["Mira", "Correo", "simple https://other.invalid/x @newperson"])
    assert f'The link "{LINK}" is missing from the translation.' in problems
    assert 'The email address "hi@example.invalid" is missing from the translation.' in problems
    assert 'The translation adds the link "https://other.invalid/x".' in problems
    assert 'The translation adds the @handle "@newperson".' in problems


def test_identical_subtitles_are_not_a_translation() -> None:
    sentence = "The subtitles are identical to the source; they were not translated."
    assert check_segments([1, 2], ["Hello.", "Welcome."], ["Hello.", "Welcome."]) == [sentence]
    assert check_segments([1, 2], ["Hello.", "Welcome."], ["Hello.", "Bienvenido."]) == []
    assert check_segments([1], ["123"], ["123"]) == []  # nothing to translate
    assert check_segments([], [], []) == []


def test_code_problems_flag_the_language_even_with_a_perfect_check(store: Store) -> None:
    def drops_the_price(messages: list[dict[str, str]]) -> str:
        return tag_text([(n, "~" + t.replace(" 610", "")) for n, t in seg_tags(messages)])

    record = run(store, FakeModel(default=drops_the_price), FakeChecker()).records[0]
    assert record.status == "flagged" and record.score == 5
    assert record.problems == ['The number "610" in segment 3 is missing from the translation.']


# --- verifying ----------------------------------------------------------------------------------------
def test_the_back_translation_is_blind(store: Store) -> None:
    checker = FakeChecker()
    run(store, FakeModel(), checker)
    (back, schema1), (compare, schema2) = checker.calls
    assert schema1 is None and schema2 is not None and schema2["required"] == ["score", "issues"]
    sent = json.dumps(back)
    assert "~Hello and welcome." in sent and "Spanish" in sent and "English" in sent
    assert "Hello and welcome." not in sent.replace("~Hello and welcome.", "")  # only the translation
    version_a, version_b = compare[1]["content"].split("Version B")
    assert "~" not in version_a and "~" not in version_b  # the comparison sees originals and backs only


def test_back_prompt_never_contains_the_source_text() -> None:
    messages = back_segment_messages("en", "es", [(1, "Hola"), (3, "Adiós")])
    assert "Hello" not in json.dumps(messages) and "Hola" in json.dumps(messages)
    assert "exactly" not in messages[1]["content"]
    insisted = back_segment_messages("en", "es", [(1, "Hola"), (3, "Adiós")], insist=True)
    assert "must be in English, not in Spanish" in insisted[1]["content"]
    compare = compare_segment_messages("en", [(1, "Hello")], [(1, "Hello there")])
    assert 'Version A\n<s n="1">Hello</s>\n\nVersion B\n<s n="1">Hello there</s>' in compare[1]["content"]
    assert "segment number" in compare[0]["content"]


def test_a_clean_check_is_verified_with_the_back_translation_stored(store: Store) -> None:
    messages: list[str] = []
    result = run(store, FakeModel(), FakeChecker(), report=messages.append)
    record = result.records[0]
    assert record.status == "verified" and record.score == 5 and record.issues == []
    assert record.verifier == "glimmer-test" and record.problems == []
    assert record.back_translation == {"segments": [
        {"n": 1, "text": "Hello and welcome."}, {"n": 3, "text": f"Today the plant costs 610 dollars, see {LINK}"},
        {"n": 4, "text": "Thanks for watching, 1,000 thanks!"}]}
    assert result.verifier_available and result.message == "1 verified"
    assert messages == ["Translating subtitles into 1 language", "Checking es (1 of 1)"]


def test_checker_issues_or_a_low_score_flag_the_language(store: Store) -> None:
    low = run(store, FakeModel(), FakeChecker(verdict={"score": 3, "issues": ["Segment 3: the price changed."]}))
    assert low.records[0].status == "flagged" and low.records[0].score == 3
    assert low.records[0].issues == ["Segment 3: the price changed."]
    doubt = run(store, FakeModel(), FakeChecker(verdict={"score": 5, "issues": ["Segment 1: tone differs."]}))
    assert doubt.records[0].status == "flagged" and doubt.records[0].score == 5


def test_a_copied_back_translation_is_retried_then_left_unverified(store: Store) -> None:
    def copy_back(messages: list[dict[str, str]]) -> str:
        return tag_text(seg_tags(messages))

    checker = FakeChecker(back=copy_back)
    record = run(store, FakeModel(), checker).records[0]
    assert len(checker.calls) == 2  # the blind call, and the one retry; no comparison
    assert "must be in English" in checker.calls[1][0][1]["content"]
    assert record.status == "unverified" and record.score is None and record.verifier == ""
    assert record.back_translation == {"note": "The checker copied the translation instead of "
                                               "translating it back, so it could not check it."}
    assert record.segments[0]["text"] == "~Hello and welcome."  # the subtitles are still kept


def test_a_copy_that_is_fixed_by_the_retry_is_checked(store: Store) -> None:
    checker = FakeChecker([lambda m: tag_text(seg_tags(m)), undo_prefix])
    record = run(store, FakeModel(), checker).records[0]
    assert len(checker.calls) == 3 and record.status == "verified"


def test_a_translation_that_kept_most_text_is_not_a_copy(store: Store) -> None:
    # only the brand-name segment is unchanged: copying it back proves nothing
    def keep_names(messages: list[dict[str, str]]) -> str:
        return tag_text([(n, t if n == 1 else "~" + t) for n, t in seg_tags(messages)])

    checker = FakeChecker(back=lambda m: tag_text([(n, t.removeprefix("~")) for n, t in seg_tags(m)]))
    record = run(store, FakeModel(default=keep_names), checker).records[0]
    assert record.status == "verified"


def test_a_long_transcript_is_checked_in_chunks_and_the_lowest_score_counts(
        store: Store, monkeypatch) -> None:
    monkeypatch.setattr(st, "CHECK_SEGMENTS", 2)
    verdicts = iter([{"score": 5, "issues": []}, {"score": 2, "issues": ["Segment 4: dropped."]}])
    checker = FakeChecker()

    def compare(messages: list[dict[str, str]]) -> str:
        return json.dumps(next(verdicts))

    checker.script = [undo_prefix, compare, undo_prefix, compare]
    record = run(store, FakeModel(), checker).records[0]
    assert len(checker.calls) == 4
    assert [n for n, _ in seg_tags(checker.calls[0][0])] == [1, 3]
    assert [n for n, _ in seg_tags(checker.calls[2][0])] == [4]
    assert record.score == 2 and record.issues == ["Segment 4: dropped."] and record.status == "flagged"
    assert [s["n"] for s in record.back_translation["segments"]] == [1, 3, 4]


def test_a_checker_that_cannot_be_reached_is_switched_off_for_the_rest(store: Store) -> None:
    checker = FakeChecker([TranslateError("network", "Could not reach the checker (Ollama). Is it running?")])
    messages: list[str] = []
    result = run(store, FakeModel(), checker, targets=["es", "de"], report=messages.append)
    first, second = result.records
    assert first.status == second.status == "unverified"
    assert first.back_translation == {"note": "Could not reach the checker (Ollama). Is it running?"}
    assert second.back_translation == {}
    assert len(checker.calls) == 1  # German was not even tried
    assert "Checking de (2 of 2)" not in messages and not result.verifier_available
    assert result.message == "2 unverified (the checker was not available)"


def test_other_checker_errors_only_affect_that_language(store: Store) -> None:
    checker = FakeChecker([TranslateError("answer", "The checker sent no back-translation."),
                           undo_prefix, "not json"])
    result = run(store, FakeModel(), checker, targets=["es", "de"])
    first, second = result.records
    assert first.status == "unverified"
    assert first.back_translation == {"note": "The checker sent no back-translation."}
    assert second.status == "unverified"
    assert second.back_translation == {"note": "The checker's answer was not JSON."}
    checker = FakeChecker([KeyError("x")])
    record = run(store, FakeModel(), checker).records[0]
    assert record.back_translation == {"note": "The checker failed unexpectedly (KeyError)."}
    assert record.status == "unverified"


@pytest.mark.parametrize("verdict", [{"score": 9, "issues": []}, {"score": 4}, {"score": True, "issues": []},
                                     {"score": 4, "issues": [1]}, [1, 2]])
def test_a_malformed_comparison_leaves_the_language_unverified(store: Store, verdict: Any) -> None:
    record = run(store, FakeModel(), FakeChecker(verdict=verdict)).records[0]
    assert record.status == "unverified" and "note" in record.back_translation


def test_the_checker_may_end_its_json_with_a_marker(store: Store) -> None:
    checker = FakeChecker()
    checker.script = [lambda m: undo_prefix(m) + "<|eot|>", '{"score": 5.0, "issues": [" "]}<|eot|>']
    record = run(store, FakeModel(), checker).records[0]
    assert record.status == "verified" and record.issues == []


# --- status and summary -------------------------------------------------------------------------------
def test_without_a_checker_the_languages_are_unverified(store: Store) -> None:
    for checker in (None, FakeChecker(up=False)):
        result = run(store, FakeModel(), checker)
        assert result.records[0].status == "unverified" and not result.verifier_available
        assert result.message == "1 unverified (the checker was not available)"
        assert result.records[0].back_translation == {}


def test_the_summary_counts_each_status(store: Store) -> None:
    def per_language(messages: list[dict[str, str]]) -> str:
        if "French" in messages[0]["content"]:
            raise TranslateError("api", "broken")
        if "German" in messages[0]["content"]:  # the number is dropped: flagged by code
            return tag_text([(n, "~" + t.replace(" 610", "")) for n, t in seg_tags(messages)])
        return prefixed(messages)

    result = run(store, FakeModel(default=per_language), FakeChecker(), targets=["es", "pt", "de", "fr"])
    assert [r.status for r in result.records] == ["verified", "verified", "flagged", "failed"]
    assert result.message == "2 verified, 1 flagged, 1 failed"


# --- SRT ----------------------------------------------------------------------------------------------
def test_the_srt_uses_the_transcripts_times(store: Store) -> None:
    record = run(store).records[0]
    assert srt_for(record) == (
        "1\n00:00:00,000 --> 00:00:03,000\n~Hello and welcome.\n\n"
        "2\n00:00:04,000 --> 00:00:09,000\n~Today the plant costs 610 dollars, see\n" + LINK + "\n\n"
        "3\n00:00:09,000 --> 00:00:12,000\n~Thanks for watching, 1,000 thanks!\n")
    failed = replace(record, segments=[])
    assert srt_for(failed) == ""


# --- what is stored and what is refused ----------------------------------------------------------------
def test_records_are_stored_in_target_order(store: Store) -> None:
    result = run(store, FakeModel(), FakeChecker(), targets=["de", "es"])
    stored = store.list_subtitle_translations(V1)
    assert [r.language for r in stored] == ["es", "de"]  # newest first
    assert list(result.records) == [stored[1], stored[0]]
    assert store.latest_subtitle_translations(V1, result.records[0].transcript_id).keys() == {"de", "es"}


def test_an_older_transcript_can_be_chosen(store: Store) -> None:
    first = store.latest_transcript(V1)
    assert first is not None
    store.add_transcript(TranscriptRecord(0, V1, "en", "asr-test", "b.mp4", 1000,
                                          [{"start_ms": 0, "end_ms": 1000, "text": "Only this."}], [], ""), NOW)
    assert run(store).records[0].segments == [{"start_ms": 0, "end_ms": 1000, "text": "~Only this."}]
    assert len(run(store, transcript_id=first.id).records[0].segments) == 4


def test_a_thai_transcript_has_a_name_and_a_chinese_one_blocks_chinese(store: Store) -> None:
    store.add_transcript(TranscriptRecord(0, V1, "th", "asr-test", "b.mp4", 1000,
                                          [{"start_ms": 0, "end_ms": 1000, "text": "สวัสดี"}], [], ""), NOW)
    model = FakeModel()
    run(store, model)
    assert "from Thai into Spanish" in model.calls[0][0]["content"]
    store.add_transcript(TranscriptRecord(0, V1, "zh", "asr-test", "c.mp4", 1000,
                                          [{"start_ms": 0, "end_ms": 1000, "text": "你好"}], [], ""), NOW)
    with pytest.raises(ValueError, match="own language"):
        run(store, targets=["zh-Hans"])


def test_bad_requests_are_refused_before_any_call(store: Store) -> None:
    model = FakeModel()
    with pytest.raises(ValueError, match="no transcript yet"):
        translate_subtitles(store, model, None, "nope", ["es"])
    for targets, message in [([], "at least one"), (["xx"], "Unknown language"), (["en"], "own language"),
                             (["en-GB"], "Unknown language"), (["es", "es"], "only be picked once")]:
        with pytest.raises(ValueError, match=message):
            run(store, model, targets=targets)
    with pytest.raises(ValueError, match="does not belong"):
        run(store, model, transcript_id=999)
    other = store.add_transcript(TranscriptRecord(0, "otherVid1", "en", "m", "f", 1, SEGMENTS, [], ""), NOW)
    with pytest.raises(ValueError, match="does not belong"):
        run(store, model, transcript_id=other)
    assert model.calls == []


def test_a_transcript_without_language_or_text_is_refused(store: Store) -> None:
    store.add_transcript(TranscriptRecord(0, V1, "", "m", "f", 1,
                                          [{"start_ms": 0, "end_ms": 1, "text": "Hi"}], [], ""), NOW)
    with pytest.raises(ValueError, match="no language"):
        run(store)
    store.add_transcript(TranscriptRecord(0, V1, "en", "m", "f", 1,
                                          [{"start_ms": 0, "end_ms": 1, "text": "  "}], [], ""), NOW)
    with pytest.raises(ValueError, match="no text to translate"):
        run(store)
    assert store.list_subtitle_translations(V1) == []


# --- translate.py: complete() and the public chat() ----------------------------------------------------
class StubCompletions:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def chat_translator(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> tuple[ChatTranslator, StubCompletions]:
    import openai

    completions = StubCompletions(outcome)
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=completions)))
    return ChatTranslator("sk-test-key", "spark-open-test", "https://baseten.invalid/v1"), completions


def reply(content: str | None, usage: Any = None) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


def test_complete_returns_the_text_and_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    client, completions = chat_translator(
        monkeypatch, reply("hello", SimpleNamespace(prompt_tokens=7, completion_tokens=3)))
    messages = [{"role": "user", "content": "hi"}]
    assert client.complete(messages) == Completion("hello", 7, 3)
    assert completions.kwargs == {"model": "spark-open-test", "messages": messages}
    client, _ = chat_translator(monkeypatch, reply(None))
    assert client.complete(messages) == Completion(None, 0, 0)


def test_complete_maps_errors_and_translate_still_goes_through_it(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    def status_error(cls: type, status: int, message: str = "It failed.") -> Exception:
        return cls(message, response=SimpleNamespace(request=None, status_code=status, headers={}), body=None)

    cases = [
        (status_error(openai.AuthenticationError, 401), "auth"),
        (status_error(openai.RateLimitError, 429), "rate_limit"),
        (openai.APIConnectionError(request=None), "network"),
        (status_error(openai.BadRequestError, 400, "Bad sk-test-key request."), "api"),
        (SimpleNamespace(choices=[], usage=None), "api"),
    ]
    for outcome, kind in cases:
        client, _ = chat_translator(monkeypatch, outcome)
        for call in (lambda: client.complete([]), lambda: client.translate("en", "es", "a", "b")):
            with pytest.raises(TranslateError) as raised:
                call()
            assert raised.value.kind == kind and "sk-test-key" not in str(raised.value)
    client, _ = chat_translator(monkeypatch, reply("no tags"))
    assert client.complete([]).content == "no tags"  # complete() does not judge the content
    with pytest.raises(TranslateError) as raised:
        client.translate("en", "es", "a", "b")
    assert raised.value.kind == "answer"


def test_the_verifier_chat_is_public() -> None:
    opener = FakeOpener([{"message": {"role": "assistant", "content": "hi"}}])
    checker = OllamaVerifier("http://127.0.0.1:11434", "glimmer-test:30b", opener=opener)
    assert checker.chat([{"role": "user", "content": "x"}]) == "hi"
    assert not hasattr(checker, "_chat")
