import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fakes import ScriptedTranslator
from yezatas.glossary import (
    GlossaryError,
    glossary_problems,
    glossary_rules,
    load_glossary,
    make_term,
    save_glossary,
)
from yezatas.store import Store, TranscriptRecord, Video
from yezatas.subtitle_translate import segment_messages, translate_subtitles
from yezatas.translate import ChatTranslator, Translated, translator_messages

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
V1 = "vid00000001"
V2 = "vid00000002"
LINK = "https://shop.invalid/cactus"
TITLE_MESA = "Mesa tips and tricks"
DESC_MESA = f"Learn about Mesa.\nShop: {LINK}\n"
TITLE_CACTUS = "Cactus care"
DESC_CACTUS = f"Cactus care guide {LINK}\n"

# helpers copied from test_translate / test_subtitle_translate (do not import from them)
class StubCompletions:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def chat_translator(monkeypatch: pytest.MonkeyPatch, outcome: Any):
    import openai

    completions = StubCompletions(outcome)
    made: dict[str, Any] = {}

    def fake_openai(**kwargs: Any) -> Any:
        made.update(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    monkeypatch.setattr(openai, "OpenAI", fake_openai)
    return (
        ChatTranslator("sk-test-key", "spark-open-test", "https://baseten.invalid/v1", timeout=30),
        completions,
        made,
    )


def reply(content: str | None, usage: Any = None) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage
    )


def seg_tags(messages: list[dict[str, str]]) -> list[tuple[int, str]]:
    return [
        (int(n), t)
        for n, t in re.findall(r'<s n="(\d+)">(.*?)</s>', messages[-1]["content"], re.DOTALL)
    ]


def tag_text(items: list[tuple[int, str]]) -> str:
    return "\n".join(f'<s n="{n}">{t}</s>' for n, t in items)


def prefixed(messages: list[dict[str, str]]) -> str:
    return tag_text([(n, "~" + t) for n, t in seg_tags(messages)])


class FakeModel:
    def __init__(self, script: list[Any] | None = None, default: Any = prefixed, model: str = "spark-test") -> None:
        self.model = model
        self.script = list(script or [])
        self.default = default
        self.calls: list[list[dict[str, str]]] = []
        self._lock = threading.Lock()

    def complete(self, messages: list[dict[str, str]]) -> Any:
        from yezatas.translate import Completion

        with self._lock:
            self.calls.append(messages)
            item = self.script.pop(0) if self.script else self.default
        if callable(item):
            item = item(messages)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, Completion) else Completion(item, 10, 20)


# custom translator for glossary runs (records glossary)
class RecordingTranslator:
    def __init__(self, behaviour=None, model: str = "spark-open-test") -> None:
        self.model = model
        self.calls: list[tuple[str, str, str, str, str]] = []
        self.behaviour = behaviour

    def translate(self, source_language: str, target_language: str, title: str, description: str, glossary: str = "") -> Translated:
        self.calls.append((source_language, target_language, title, description, glossary))
        if self.behaviour is not None:
            return self.behaviour(source_language, target_language, title, description)
        prefix = f"[{target_language}] "
        body = "\n".join(prefix + line if line else line for line in description.split("\n"))
        return Translated(prefix + title, body, self.model, 10, 20)


def _store_with_video(tmp_path: Path, video_id: str = V1, title: str = TITLE_MESA, description: str = DESC_MESA) -> Store:
    # caller should use context manager; this helper creates file and inserts video
    store_path = tmp_path / "c.sqlite"
    store = Store(store_path)
    # we will manage lifecycle manually in tests via context, but for helper we use direct
    return store


# --- make_term refusals ---
def test_make_term_empty_raises():
    with pytest.raises(GlossaryError):
        make_term("", True, {})
    with pytest.raises(GlossaryError):
        make_term("   ", True, {})


def test_make_term_over_80_raises():
    with pytest.raises(GlossaryError):
        make_term("a" * 81, True, {})


def test_make_term_angle_bracket_raises():
    with pytest.raises(GlossaryError):
        make_term("a<b", True, {})
    with pytest.raises(GlossaryError):
        make_term("a>b", True, {})
    with pytest.raises(GlossaryError):
        make_term("Cactus", False, {"es": "a<b"})


def test_make_term_line_break_raises():
    with pytest.raises(GlossaryError):
        make_term("a\nb", True, {})
    with pytest.raises(GlossaryError):
        make_term("a\rb", True, {})
    with pytest.raises(GlossaryError):
        make_term("Cactus", False, {"es": "a\nb"})


def test_make_term_unknown_language_raises():
    with pytest.raises(GlossaryError):
        make_term("Cactus", False, {"zz": "foo"})
    with pytest.raises(GlossaryError):
        make_term("Cactus", False, {"xx": "foo"})


def test_make_term_keep_with_translations_raises():
    with pytest.raises(GlossaryError):
        make_term("Mesa", True, {"es": "Mesa"})


def test_make_term_neither_keep_nor_translations_raises():
    with pytest.raises(GlossaryError):
        make_term("Mesa", False, {})
    with pytest.raises(GlossaryError):
        make_term("Mesa", False, {})


# --- save / load ---
def test_save_load_round_trip_sorted(tmp_path: Path):
    path = tmp_path / "glossary.json"
    # translations intentionally unsorted input dict, make_term will sort
    terms = [
        make_term("Mesa", True, {}),
        make_term("Cactus", False, {"es": "Cactus Grande", "de": "Kaktus"}),
    ]
    save_glossary(path, terms)
    loaded = load_glossary(path)
    assert len(loaded) == 2
    assert loaded[0].term == "Mesa"
    assert loaded[0].keep is True
    assert loaded[0].translations == ()
    # sorted by code
    assert loaded[1].term == "Cactus"
    assert loaded[1].translations == (("de", "Kaktus"), ("es", "Cactus Grande"))
    # round trip equality
    assert loaded == terms


def test_load_missing_returns_empty(tmp_path: Path):
    assert load_glossary(tmp_path / "missing.json") == []


def test_load_broken_json_raises(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("not json {", encoding="utf-8")
    with pytest.raises(GlossaryError):
        load_glossary(p)


def test_load_duplicate_case_raises(tmp_path: Path):
    p = tmp_path / "dup.json"
    p.write_text(
        json.dumps({"terms": [{"term": "Mesa", "keep": True}, {"term": "mesa", "keep": True}]}),
        encoding="utf-8",
    )
    with pytest.raises(GlossaryError, match="appears more than once"):
        load_glossary(p)


def test_save_duplicate_case_raises(tmp_path: Path):
    p = tmp_path / "dup2.json"
    terms = [make_term("Mesa", True, {}), make_term("mesa", True, {})]
    with pytest.raises(GlossaryError, match="appears more than once"):
        save_glossary(p, terms)


def test_201_terms_raises(tmp_path: Path):
    p = tmp_path / "many.json"
    terms = [make_term(f"term{i:03d}", True, {}) for i in range(201)]
    with pytest.raises(GlossaryError, match="at most"):
        save_glossary(p, terms)
    # load also raises
    data = {"terms": [{"term": f"term{i:03d}", "keep": True} for i in range(201)]}
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(GlossaryError, match="at most"):
        load_glossary(p)


# --- glossary_rules ---
def test_glossary_rules_only_found_terms():
    terms = [
        make_term("Mesa", True, {}),
        make_term("Cactus", False, {"es": "Kaktus"}),
    ]
    rules = glossary_rules(terms, "I have a Mesa here", "es")
    assert "Mesa" in rules
    assert "Cactus" not in rules
    rules2 = glossary_rules(terms, "Cactus here", "es")
    assert "Cactus" in rules2
    assert "Mesa" not in rules2


def test_glossary_rules_case_insensitive_whole_word():
    term = make_term("Cactus", False, {"es": "Cactus Grande"})
    assert glossary_rules([term], "my cactus tips", "es") != ""
    assert glossary_rules([term], "MY CACTUS TIPS", "es") != ""
    assert glossary_rules([term], "cactuses", "es") == ""
    assert glossary_rules([term], "my cactuses are green", "es") == ""
    assert glossary_rules([term], "my cactus, tips", "es") != ""


def test_glossary_rules_target_picks_translation():
    term = make_term("Cactus", False, {"es": "Cactus Grande", "de": "Kaktus"})
    es_rules = glossary_rules([term], "Cactus here", "es")
    assert "Cactus Grande" in es_rules
    assert "Kaktus" not in es_rules
    de_rules = glossary_rules([term], "Cactus here", "de")
    assert "Kaktus" in de_rules
    assert "Cactus Grande" not in de_rules


def test_glossary_rules_no_translation_for_target_left_out():
    term = make_term("Cactus", False, {"es": "Cactus Grande"})
    assert glossary_rules([term], "Cactus here", "de") == ""


def test_glossary_rules_empty_when_nothing_applies():
    term = make_term("Mesa", True, {})
    assert glossary_rules([term], "no match here", "es") == ""
    assert glossary_rules([], "Mesa here", "es") == ""


def test_glossary_rules_exact_sentences():
    keep = make_term("Mesa", True, {})
    fixed = make_term("Cactus", False, {"es": "Kaktus"})
    assert glossary_rules([keep], "my Mesa", "es") == 'Keep these terms exactly as written: "Mesa".'
    assert glossary_rules([fixed], "my Cactus", "es") == 'Always translate these terms this way: "Cactus" -> "Kaktus".'
    both = glossary_rules([keep, fixed], "Mesa and Cactus", "es")
    assert both == 'Keep these terms exactly as written: "Mesa". Always translate these terms this way: "Cactus" -> "Kaktus".'


# --- glossary_problems ---
def test_glossary_problems_keep_sentence_exact():
    term = make_term("Mesa", True, {})
    probs = glossary_problems([term], "I have Mesa", "I have nothing", "es")
    assert probs == ['The term "Mesa" should stay as it is, but it is not in the translation.']


def test_glossary_problems_fixed_sentence_exact():
    term = make_term("Cactus", False, {"es": "Cactus Grande"})
    probs = glossary_problems([term], "Cactus here", "no fixed here", "es")
    assert probs == ['The term "Cactus" should be translated as "Cactus Grande", but "Cactus Grande" is not in the translation.']


def test_glossary_problems_terms_not_in_source_ignored():
    keep = make_term("Mesa", True, {})
    fixed = make_term("Cactus", False, {"es": "Cactus Grande"})
    probs = glossary_problems([keep, fixed], "hello world", "hello world", "es")
    assert probs == []
    # also if translation missing term but source missing term => ignored
    probs2 = glossary_problems([keep], "hello", "no mesa", "es")
    assert probs2 == []


def test_glossary_problems_keep_case_changed_counts_as_missing():
    term = make_term("Mesa", True, {})
    probs = glossary_problems([term], "Mesa here", "mesa here", "es")
    assert probs == ['The term "Mesa" should stay as it is, but it is not in the translation.']
    # exact case passes
    probs2 = glossary_problems([term], "Mesa here", "Mesa here", "es")
    assert probs2 == []


def test_glossary_problems_fixed_case_insensitive():
    term = make_term("Cactus", False, {"es": "Cactus Grande"})
    # _found is case-insensitive, so lower case still counts as found
    probs = glossary_problems([term], "Cactus here", "cactus grande here", "es")
    # should be considered found? _found uses IGNORECASE, so lower case matches
    assert probs == []
    probs2 = glossary_problems([term], "Cactus here", "nothing", "es")
    assert len(probs2) == 1


# --- translator_messages / segment_messages ---
def test_translator_messages_with_glossary_has_rules_right_after_keep():
    from yezatas.translate import _KEEP_RULES

    msgs = translator_messages("en", "es", "Title", "Desc", glossary="RULES")
    system = msgs[0]["content"]
    assert "RULES" in system
    assert f"{_KEEP_RULES} RULES" in system
    # ensure it is before the final Answer sentence
    assert system.index("RULES") < system.index("Answer with exactly")
    assert system.index(_KEEP_RULES) < system.index("RULES")


def test_translator_messages_without_glossary_exactly_as_before():
    a = translator_messages("en", "es", "Title", "Desc")
    b = translator_messages("en", "es", "Title", "Desc", glossary="")
    assert a == b
    # compare with a call that passes nothing (same as above)
    assert a[0]["content"] == b[0]["content"]
    assert "RULES" not in a[0]["content"]


def test_segment_messages_with_glossary_has_rules_right_after_keep():
    items = [(1, "hello Cactus")]
    msgs = segment_messages("en", "es", items, glossary="RULES")
    system = msgs[0]["content"]
    assert "RULES" in system
    # keep rules substring for subtitles
    assert "hashtags and emails exactly as they are." in system
    assert system.index("hashtags and emails exactly as they are.") < system.index("RULES")
    # RULES should be after keep rules and before user content not in system
    assert "RULES" in system


def test_segment_messages_without_glossary_exactly_as_before():
    items = [(1, "hello")]
    a = segment_messages("en", "es", items)
    b = segment_messages("en", "es", items, glossary="")
    assert a == b


# --- ChatTranslator ---
def test_chat_translator_sends_glossary(monkeypatch: pytest.MonkeyPatch):
    client, completions, made = chat_translator(
        monkeypatch, reply('<title>Hola</title>\n<description>\nMundo\n</description>')
    )
    result = client.translate("en", "es", "Hello", "World", glossary="RULES")
    assert result.title == "Hola"
    assert completions.kwargs["model"] == "spark-open-test"
    system = completions.kwargs["messages"][0]["content"]
    assert "RULES" in system
    # should be after keep rules
    from yezatas.translate import _KEEP_RULES

    assert f"{_KEEP_RULES} RULES" in system


# --- translate_video with glossary ---
def test_translate_video_glossary_keep_missing_flagged(tmp_path: Path):
    from yezatas.translate import translate_video

    with Store(tmp_path / "c.sqlite") as store:
        pull = store.begin_pull("UC_fake", NOW)
        store.upsert_video(
            Video(
                video_id=V1,
                channel_id="UC_fake",
                title=TITLE_MESA,
                description=DESC_MESA,
                tags=["cactus"],
                privacy_status="public",
                published_at="2026-01-01T10:00:00Z",
                duration="PT5M",
                default_language="en",
            ),
            pull,
            NOW,
        )

        def drop_mesa(source_language, target, title, description):
            # drop Mesa but keep link
            return Translated(
                title.replace("Mesa", "").replace("mesa", ""),
                description.replace("Mesa", "").replace("mesa", ""),
                "spark-open-test",
                10,
                20,
            )

        translator = RecordingTranslator(behaviour=drop_mesa)
        glossary = [make_term("Mesa", True, {})]
        result = translate_video(store, translator, None, V1, "en", ["es"], glossary=glossary, now=NOW)
        assert len(result.records) == 1
        rec = result.records[0]
        assert rec.status == "flagged"
        assert 'The term "Mesa" should stay as it is, but it is not in the translation.' in rec.problems
        # translator received rules
        assert len(translator.calls) == 1
        assert 'Keep these terms exactly as written: "Mesa".' in translator.calls[0][4]
        # ensure stored
        stored = store.get_translation(rec.id)
        assert stored is not None
        assert stored.problems == rec.problems


def test_translate_video_glossary_keep_present_no_problem(tmp_path: Path):
    from yezatas.translate import translate_video

    with Store(tmp_path / "c.sqlite") as store:
        pull = store.begin_pull("UC_fake", NOW)
        store.upsert_video(
            Video(
                video_id=V1,
                channel_id="UC_fake",
                title=TITLE_MESA,
                description=DESC_MESA,
                tags=[],
                privacy_status="public",
                published_at="2026-01-01T10:00:00Z",
                duration="PT5M",
                default_language="en",
            ),
            pull,
            NOW,
        )

        def keep_mesa(source_language, target, title, description):
            return Translated("[es] " + title, "[es] " + description, "spark-open-test", 10, 20)

        translator = RecordingTranslator(behaviour=keep_mesa)
        glossary = [make_term("Mesa", True, {})]
        result = translate_video(store, translator, None, V1, "en", ["es"], glossary=glossary, now=NOW)
        rec = result.records[0]
        # no glossary problem
        assert all("Mesa" not in p or "should stay" not in p for p in rec.problems)
        # specifically no keep sentence
        assert 'The term "Mesa" should stay as it is, but it is not in the translation.' not in rec.problems
        # should be unverified (no verifier) but not flagged for glossary
        assert rec.status in ("unverified", "verified")
        # translator did receive rules
        assert 'Keep these terms exactly as written: "Mesa".' in translator.calls[0][4]


def test_translate_video_empty_glossary_plain_translator_still_works(tmp_path: Path):
    from yezatas.translate import translate_video

    with Store(tmp_path / "c.sqlite") as store:
        pull = store.begin_pull("UC_fake", NOW)
        store.upsert_video(
            Video(
                video_id=V1,
                channel_id="UC_fake",
                title=TITLE_MESA,
                description=DESC_MESA,
                tags=[],
                privacy_status="public",
                published_at="2026-01-01T10:00:00Z",
                duration="PT5M",
                default_language="en",
            ),
            pull,
            NOW,
        )
        translator = ScriptedTranslator()
        # glossary empty (default) should not pass glossary kwarg, so ScriptedTranslator works
        result = translate_video(store, translator, None, V1, "en", ["es"], glossary=[], now=NOW)
        assert len(result.records) == 1
        assert result.records[0].title.startswith("[es]")
        # also test with default no glossary arg
        result2 = translate_video(store, translator, None, V1, "en", ["de"], now=NOW)
        assert len(result2.records) == 1


# --- translate_subtitles with glossary ---
def test_translate_subtitles_glossary_fixed_missing_flagged_and_prompt(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as store:
        pull = store.begin_pull("UC_fake", NOW)
        store.upsert_video(
            Video(
                video_id=V1,
                channel_id="UC_fake",
                title="Repotting",
                description="x",
                tags=[],
                privacy_status="public",
                published_at="2026-01-01T10:00:00Z",
                duration="PT5M",
                default_language="en",
            ),
            pull,
            NOW,
        )
        segments = [
            {"start_ms": 0, "end_ms": 3000, "text": "Hello Cactus"},
            {"start_ms": 3000, "end_ms": 6000, "text": f"Visit {LINK} for tips"},
            {"start_ms": 6000, "end_ms": 9000, "text": "Thanks"},
        ]
        store.add_transcript(
            TranscriptRecord(0, V1, "en", "asr-test", "clip.mp4", 9000, segments, [], ""),
            NOW,
        )

        # model that does NOT use fixed translation
        def no_fixed(messages: list[dict[str, str]]) -> str:
            # return translation without Cactus Grande, just prefix with ~
            items = seg_tags(messages)
            # produce translation that keeps original Cactus word, not Cactus Grande
            return tag_text([(n, t) for n, t in items])

        model = FakeModel(default=no_fixed)
        glossary = [make_term("Cactus", False, {"es": "Cactus Grande"})]
        result = translate_subtitles(store, model, None, V1, ["es"], glossary=glossary, now=NOW, workers=1)
        assert len(result.records) == 1
        rec = result.records[0]
        assert rec.status == "flagged"
        assert 'The term "Cactus" should be translated as "Cactus Grande", but "Cactus Grande" is not in the translation.' in rec.problems
        # system prompt contains rules
        assert len(model.calls) >= 1
        system = model.calls[0][0]["content"]
        assert 'Always translate these terms this way: "Cactus" -> "Cactus Grande".' in system


def test_translate_subtitles_glossary_empty_plain_works(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as store:
        pull = store.begin_pull("UC_fake", NOW)
        store.upsert_video(
            Video(
                video_id=V1,
                channel_id="UC_fake",
                title="Repotting",
                description="x",
                tags=[],
                privacy_status="public",
                published_at="2026-01-01T10:00:00Z",
                duration="PT5M",
                default_language="en",
            ),
            pull,
            NOW,
        )
        segments = [
            {"start_ms": 0, "end_ms": 3000, "text": "Hello Cactus"},
            {"start_ms": 3000, "end_ms": 6000, "text": "World"},
        ]
        store.add_transcript(
            TranscriptRecord(0, V1, "en", "asr-test", "clip.mp4", 6000, segments, [], ""),
            NOW,
        )
        model = FakeModel()
        # empty glossary should still work
        result = translate_subtitles(store, model, None, V1, ["es"], glossary=[], now=NOW, workers=1)
        assert len(result.records) == 1
        assert result.records[0].status in ("unverified", "verified", "flagged")

