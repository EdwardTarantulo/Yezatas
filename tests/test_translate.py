import json
import urllib.error
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fakes import FakeOpener, ScriptedTranslator, ScriptedVerifier
from yezatas import agent, translate
from yezatas.config import load_config
from yezatas.edits import save_draft
from yezatas.store import Store, Video
from yezatas.translate import (
    ChatTranslator, OllamaVerifier, TranslateError, Translated, Verdict, check_translation,
    copied_back, parse_translation, same_language, source_hash, translate_video, youtube_key,
)

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
V1 = "vidA00001"
LINK = "https://shop.invalid/cactus"
TITLE = "Repotting a cactus #cactus"
DESCRIPTION = f"How to repot a cactus.\n\n0:00 Intro\n1:30 Soil\nShop: {LINK}\nMail: hi@example.invalid\n@sample_garden\n"


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as s:
        pull = s.begin_pull("UC_fake", NOW)
        s.upsert_video(Video(video_id=V1, channel_id="UC_fake", title=TITLE, description=DESCRIPTION,
                             tags=["cactus"], privacy_status="public",
                             published_at="2026-01-01T10:00:00Z", duration="PT5M",
                             default_language="en"), pull, NOW)
        yield s


def config(tmp_path: Path, **env: str):
    return load_config({"YEZATAS_HOME": str(tmp_path / "home"), **env})


# --- languages and hash ------------------------------------------------------------------------------
def test_languages_and_hash() -> None:
    for code in ("en", "es", "pt", "de", "fr", "it", "ru", "uk", "pl", "lt", "lv", "et", "tr", "ja",
                 "ko", "zh-Hans", "zh-Hant", "hi", "id", "vi", "ar", "nl", "sv", "cs"):
        assert code in translate.LANGUAGES
    assert translate.DEFAULT_TARGETS == ("es", "pt", "de", "fr", "ru")
    digest = source_hash("a", "b")
    assert len(digest) == 16 and int(digest, 16) >= 0
    assert digest == source_hash("a", "b")
    assert digest != source_hash("ab", "") and digest != source_hash("a", "c")


# --- parsing the translator's answer -------------------------------------------------------------------
def test_parse_translation_multiline_description() -> None:
    content = "<title> Trasplantar un cactus </title>\n<description>\nLinea uno.\n\nLinea dos.\n</description>"
    assert parse_translation(content) == ("Trasplantar un cactus", "Linea uno.\n\nLinea dos.")


def test_parse_translation_strips_one_newline_only() -> None:
    assert parse_translation("<title>T</title><description>\n\nA\n\n</description>") == ("T", "\nA\n")
    assert parse_translation("<title>T</title><description>A</description>") == ("T", "A")
    assert parse_translation("<title>T</title><description>\r\nA\r\n</description>") == ("T", "A")


def test_parse_translation_ignores_reasoning_before_the_answer() -> None:
    content = ("I should put the <title> first and then the <description>.\n"
               "<title>Hola</title>\n<description>\nMundo\n</description>")
    assert parse_translation(content) == ("Hola", "Mundo")


@pytest.mark.parametrize("content", [None, "", "no tags here", "<title>Only a title</title>",
                                     "<description>d</description>", "<title>T</title><description>open"])
def test_parse_translation_needs_both_tags(content: str | None) -> None:
    with pytest.raises(TranslateError) as raised:
        parse_translation(content)
    assert raised.value.kind == "answer"


# --- ChatTranslator with a stubbed openai client ----------------------------------------------------------
class StubCompletions:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def chat_translator(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> tuple[ChatTranslator, StubCompletions, dict]:
    import openai

    completions = StubCompletions(outcome)
    made: dict[str, Any] = {}

    def fake_openai(**kwargs: Any) -> Any:
        made.update(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    monkeypatch.setattr(openai, "OpenAI", fake_openai)
    return ChatTranslator("sk-test-key", "spark-open-test", "https://baseten.invalid/v1", timeout=30), completions, made


def reply(content: str | None, usage: Any = None) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=usage)


def test_chat_translator_prompt_and_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=45)
    client, completions, made = chat_translator(
        monkeypatch, reply("<title>Hola</title>\n<description>\nMundo\n</description>", usage))
    result = client.translate("en", "es", "Hello", "World\n0:00 Start")
    assert result == Translated("Hola", "Mundo", "spark-open-test", 120, 45)
    assert made == {"base_url": "https://baseten.invalid/v1", "api_key": "sk-test-key", "timeout": 30}
    assert completions.kwargs["model"] == "spark-open-test"
    system, user = completions.kwargs["messages"]
    assert system["role"] == "system" and user["role"] == "user"
    text = system["content"]
    assert "English" in text and "Spanish" in text
    for rule in ("every URL", "time stamp", "hashtag", "@handle", "email", "emoji", "line breaks",
                 "brand names", "crypto addresses", "<title>", "<description>"):
        assert rule in text
    assert user["content"] == "<title>Hello</title>\n<description>\nWorld\n0:00 Start\n</description>"


def test_chat_translator_without_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = chat_translator(monkeypatch, reply("<title>A</title><description>B</description>"))
    assert client.translate("en", "de", "x", "y") == Translated("A", "B", "spark-open-test", 0, 0)


def test_chat_translator_bad_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = chat_translator(monkeypatch, reply("Sorry, I cannot do that."))
    with pytest.raises(TranslateError) as raised:
        client.translate("en", "es", "a", "b")
    assert raised.value.kind == "answer"
    client, _, _ = chat_translator(monkeypatch, SimpleNamespace(choices=[], usage=None))
    with pytest.raises(TranslateError) as raised:
        client.translate("en", "es", "a", "b")
    assert raised.value.kind == "api"


def status_error(cls: type, status: int, message: str = "It failed.") -> Exception:
    response = SimpleNamespace(request=None, status_code=status, headers={})
    return cls(message, response=response, body=None)


def test_chat_translator_maps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    cases = [
        (status_error(openai.AuthenticationError, 401), "auth"),
        (status_error(openai.PermissionDeniedError, 403), "auth"),
        (status_error(openai.RateLimitError, 429), "rate_limit"),
        (openai.APIConnectionError(request=None), "network"),
        (openai.APITimeoutError(request=None), "network"),
        (status_error(openai.BadRequestError, 400, "Bad sk-test-key request."), "api"),
    ]
    for error, kind in cases:
        client, _, _ = chat_translator(monkeypatch, error)
        with pytest.raises(TranslateError) as raised:
            client.translate("en", "es", "a", "b")
        assert raised.value.kind == kind
        assert "sk-test-key" not in str(raised.value)
    assert str(raised.value) == "The translation service returned an error (HTTP 400): Bad *** request."
    client, _, _ = chat_translator(monkeypatch, status_error(openai.AuthenticationError, 401))
    with pytest.raises(TranslateError, match="translation API key was refused"):
        client.translate("en", "es", "a", "b")


# --- OllamaVerifier with a fake opener ------------------------------------------------------------------------
def chat_reply(payload: Any) -> dict[str, Any]:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return {"message": {"role": "assistant", "content": content}}


def verifier(script: list[Any]) -> tuple[OllamaVerifier, FakeOpener]:
    opener = FakeOpener(script)
    return OllamaVerifier("http://127.0.0.1:11434/", "glimmer-test:30b", opener=opener), opener


def test_verify_makes_a_blind_call_then_a_comparison() -> None:
    v, opener = verifier([
        chat_reply("<title>Repotting a cactus</title>\n<description>\nHow to repot.\n</description>"),
        chat_reply({"score": 5, "issues": []}),
    ])
    verdict = v.verify("en", "es", "SECRET original title", "SECRET original text",
                       "Trasplantar un cactus", "Como trasplantar.")
    assert verdict == Verdict(5, (), "Repotting a cactus", "How to repot.", "glimmer-test:30b")
    blind, compare = opener.requests
    assert blind[:2] == ("POST", "http://127.0.0.1:11434/api/chat")
    body = blind[2]
    assert body["stream"] is False and body["think"] is False and body["options"] == {"temperature": 0}
    assert body["model"] == "glimmer-test:30b"
    assert "format" not in body  # plain text in tags: a JSON format made Glimmer copy the text
    sent = json.dumps(body["messages"])
    assert "Trasplantar un cactus" in sent and "Spanish" in sent and "English" in sent
    assert "SECRET" not in sent  # the first call never sees the original
    body = compare[2]
    assert body["stream"] is False and body["think"] is False and body["options"] == {"temperature": 0}
    assert body["format"]["required"] == ["score", "issues"]
    sent = json.dumps(body["messages"])
    assert "SECRET original title" in sent and "Repotting a cactus" in sent
    assert "Trasplantar" not in sent and "Como trasplantar" not in sent  # not the translation


def test_verify_reads_json_before_trailing_markers() -> None:
    v, _ = verifier([
        chat_reply("<title>T</title><description>D</description><|eot|>"),
        chat_reply('\n{"score": 3, "issues": ["The price was dropped.", " "]}<|eot|>\n'),
    ])
    verdict = v.verify("en", "de", "a", "b", "c", "d")
    assert (verdict.score, verdict.issues) == (3, ("The price was dropped.",))


def test_youtube_key_reuses_a_regional_studio_key() -> None:
    assert youtube_key("de", ["en", "de-DE", "es-US"]) == "de-DE"
    assert youtube_key("de", ["de", "de-DE"]) == "de"  # an exact key wins
    assert youtube_key("pt", ["pt-BR", "pt-PT"]) == "pt"  # two regions: do not guess
    assert youtube_key("zh-Hans", ["zh-Hant"]) == "zh-Hans"  # scripts only match exactly
    assert youtube_key("fr", []) == "fr"


def test_same_language() -> None:
    assert same_language("en", "en-GB") and same_language("en-GB", "en")
    assert same_language("lt", "lt") and same_language(None, None)
    assert not same_language("en-GB", "en-US") and not same_language("zh-Hans", "zh-Hant")
    assert not same_language("en", None)


def test_copied_back_spots_a_back_translation_that_did_not_translate() -> None:
    # Real case: Glimmer "back-translated" German into German and then scored it 5.
    args = ("How to get it", "Short video.", "So bekommst du es", "Kurzes Video.")
    assert copied_back(*args, "So bekommst du es", "Short video.") is True
    assert copied_back(*args, "How to get it", "Kurzes  Video.\n") is True
    assert copied_back(*args, "How you get it", "A short video.") is False
    # a part the translation kept as it was (a brand name) proves nothing either way
    assert copied_back("TF2", "Short video.", "TF2", "Kurzes Video.", "TF2", "Short video.") is False


def test_verify_retries_once_when_the_text_was_copied_back() -> None:
    copied = chat_reply("<title>Trasplantar</title><description>Como hacerlo.</description>")
    fixed = chat_reply("<title>Repotting</title><description>How to do it.</description>")
    v, opener = verifier([copied, fixed, chat_reply({"score": 5, "issues": []})])
    verdict = v.verify("en", "es", "Repot", "How to.", "Trasplantar", "Como hacerlo.")
    assert verdict.back_title == "Repotting" and verdict.score == 5
    retry = json.dumps(opener.requests[1][2]["messages"])
    assert "must be in English, not in Spanish" in retry
    v, opener = verifier([copied, copied])
    with pytest.raises(TranslateError, match="copied the translation") as raised:
        v.verify("en", "es", "Repot", "How to.", "Trasplantar", "Como hacerlo.")
    assert raised.value.kind == "answer" and len(opener.requests) == 2  # no comparison


@pytest.mark.parametrize("second", [
    "not json", "[1, 2]", {"score": 9, "issues": []}, {"score": 0, "issues": []},
    {"score": "5", "issues": []}, {"score": True, "issues": []}, {"score": 4},
    {"score": 4, "issues": "none"}, {"score": 4, "issues": [1]},
])
def test_verify_bad_comparison_answers(second: Any) -> None:
    v, _ = verifier([chat_reply("<title>T</title><description>D</description>"), chat_reply(second)])
    with pytest.raises(TranslateError) as raised:
        v.verify("en", "de", "a", "b", "c", "d")
    assert raised.value.kind == "answer"


@pytest.mark.parametrize("first", [
    "", "no json at all", {"back_title": "T"}, {"back_title": 1, "back_description": "D"},
    {"message": {}},
])
def test_verify_bad_back_translation(first: Any) -> None:
    v, _ = verifier([first if isinstance(first, dict) and "message" in first else chat_reply(first)])
    with pytest.raises(TranslateError) as raised:
        v.verify("en", "de", "a", "b", "c", "d")
    assert raised.value.kind == "answer"


def test_verify_non_json_http_body() -> None:
    v, _ = verifier(["<html>oops</html>"])
    with pytest.raises(TranslateError) as raised:
        v.verify("en", "de", "a", "b", "c", "d")
    assert raised.value.kind == "answer"


def test_verify_maps_connection_errors() -> None:
    for error in (urllib.error.URLError("refused"), ConnectionRefusedError(), TimeoutError()):
        v, _ = verifier([error])
        with pytest.raises(TranslateError) as raised:
            v.verify("en", "de", "a", "b", "c", "d")
        assert raised.value.kind == "network"
    http = urllib.error.HTTPError("http://x.invalid", 404, "Not Found", {}, None)  # type: ignore[arg-type]
    v, _ = verifier([http])
    with pytest.raises(TranslateError) as raised:
        v.verify("en", "de", "a", "b", "c", "d")
    assert raised.value.kind == "api" and "404" in str(raised.value)


def test_available_checks_the_model_list() -> None:
    v, opener = verifier([{"models": [{"name": "other:7b"}, {"name": "glimmer-test:30b"}]}])
    assert v.available() is True
    assert opener.requests[0][:2] == ("GET", "http://127.0.0.1:11434/api/tags")
    assert opener.requests[0][3] is not None and opener.requests[0][3] <= 10  # short timeout
    v, _ = verifier([{"models": [{"name": "other:7b"}]}])
    assert v.available() is False
    v, _ = verifier([{"models": []}])
    assert v.available() is False
    v, _ = verifier([urllib.error.URLError("refused")])
    assert v.available() is False
    v, _ = verifier(["not json"])
    assert v.available() is False
    v, _ = verifier([{"nothing": 1}])
    assert v.available() is False
    opener = FakeOpener([{"models": [{"name": "glimmer-test:latest"}]}])
    assert OllamaVerifier("http://127.0.0.1:11434", "glimmer-test", opener=opener).available() is True


# --- check_translation -----------------------------------------------------------------------------------------
def good() -> tuple[str, str]:
    return "[es] " + TITLE, DESCRIPTION.replace("How to repot a cactus.", "Como trasplantar.")


def test_clean_translation_has_no_problems() -> None:
    assert check_translation(TITLE, DESCRIPTION, *good()) == []


def test_empty_translation_is_a_problem() -> None:
    assert check_translation("T", "D", "", "x") == ["The translated title is empty."]
    assert "The translated description is empty." in check_translation("T", "D", "x", "  ")
    # an empty source is fine
    assert check_translation("T", "", "U", "") == []


def test_length_limits() -> None:
    long_title = "x" * 101
    assert any("101 characters" in p for p in check_translation("T", "D", long_title, "d"))
    assert not any("characters" in p for p in check_translation("T", "D", "x" * 100, "d"))
    # 2,000 two-byte characters = 4,000 bytes (fine); 2,600 = 5,200 bytes (too long)
    assert check_translation("T", "D", "t", "é" * 2000) == []
    assert any("bytes" in p for p in check_translation("T", "D", "t", "é" * 2600))
    assert check_translation("T", "D", "t", "a" * 5000) == []
    assert any("bytes" in p for p in check_translation("T", "D", "t", "a" * 5001))


def test_angle_brackets() -> None:
    assert any("< or >" in p for p in check_translation("T", "D", "a < b", "d"))
    assert any("< or >" in p for p in check_translation("T", "D", "t", "<b>d</b>"))


def test_links_must_match() -> None:
    title, description = good()
    assert check_translation(TITLE, DESCRIPTION, title, description.replace(LINK, "")) == [
        f'The link "{LINK}" is missing from the translation.']
    changed = description.replace(LINK, LINK + "?x=1")
    assert check_translation(TITLE, DESCRIPTION, title, changed) == [
        f'The link "{LINK}" is missing from the translation.',
        f'The translation adds the link "{LINK}?x=1".']
    # trailing punctuation does not matter
    assert check_translation(TITLE, DESCRIPTION, title, description.replace(LINK, LINK + ".")) == []
    assert check_translation(TITLE, DESCRIPTION, title, description + "\nhttps://new.invalid/a\n") == [
        'The translation adds the link "https://new.invalid/a".']


def test_timestamps_are_compared_as_a_multiset() -> None:
    title, description = good()
    assert check_translation(TITLE, DESCRIPTION, title, description.replace("1:30", "1:31")) == [
        'The time "1:30" is missing from the translation.', 'The translation adds the time "1:31".']
    assert check_translation(TITLE, DESCRIPTION, title, description.replace("1:30 Soil", "Soil")) == [
        'The time "1:30" is missing from the translation.']
    assert check_translation(TITLE, DESCRIPTION, title, description + "0:00 again\n") == [
        'The translation adds the time "0:00".']
    twice = "0:00 A\n0:00 B\n"
    assert check_translation("T", twice, "t", "0:00 X\n0:00 Y\n") == []
    # times inside links are part of the link, not a chapter
    assert check_translation("T", f"{LINK}?t=1:30", "t", f"{LINK}?t=1:30 x") == []


def test_hashtags_handles_and_emails() -> None:
    title, description = good()
    assert check_translation(TITLE, DESCRIPTION, "[es] Repotting a cactus #cactus2", description) == [
        'The hashtag "#cactus" is missing from the translation.',
        'The translation adds the hashtag "#cactus2".']
    assert check_translation(TITLE, DESCRIPTION, "[es] Repotting a cactus #Cactus", description) == [
        'The hashtag "#cactus" is missing from the translation.',
        'The translation adds the hashtag "#Cactus".']
    assert check_translation(TITLE, DESCRIPTION, title, description.replace("@sample_garden", "@jardin")) == [
        'The @handle "@sample_garden" is missing from the translation.',
        'The translation adds the @handle "@jardin".']
    assert check_translation(TITLE, DESCRIPTION, title, description.replace("hi@", "hello@")) == [
        'The email address "hi@example.invalid" is missing from the translation.',
        'The translation adds the email address "hello@example.invalid".']
    # an email is not read as a handle
    assert check_translation("T", "Mail: hi@example.invalid", "t", "Correo: hi@example.invalid") == []


def test_identical_to_source_is_a_problem() -> None:
    assert check_translation(TITLE, DESCRIPTION, TITLE, DESCRIPTION) == [
        "The translation is identical to the source; it was not translated."]
    # only the title is the same: fine
    assert check_translation("Cactus", "Some text.", "Cactus", "Algun texto.") == []


def test_check_translation_is_deterministic() -> None:
    args = (TITLE, DESCRIPTION, "x #b #a", "y https://b.invalid https://a.invalid @z @y 2:00 1:00")
    assert check_translation(*args) == check_translation(*args)


# --- translate_video -----------------------------------------------------------------------------------------------
def test_all_verified(store: Store) -> None:
    translator, checker = ScriptedTranslator(), ScriptedVerifier()
    progress: list[str] = []
    result = translate_video(store, translator, checker, V1, "en", ["es", "de"], now=NOW,
                             report=progress.append)
    assert [r.language for r in result.records] == ["es", "de"]
    assert [r.status for r in result.records] == ["verified", "verified"]
    assert result.message == "2 verified" and result.verifier_available is True
    es = result.records[0]
    assert es.id > 0 and store.get_translation(es.id) == es
    assert (es.video_id, es.source_language, es.translator, es.verifier) == (V1, "en", "spark-open-test", "glimmer-test")
    assert es.source_hash == source_hash(TITLE, DESCRIPTION)
    assert (es.title, es.score, es.issues, es.problems) == ("[es] " + TITLE, 5, [], [])
    assert es.back_translation == {"title": TITLE, "description": DESCRIPTION}
    assert (es.input_tokens, es.output_tokens) == (10, 20)
    assert progress == ["Translating 2 languages", "Checking es (1 of 2)", "Checking de (2 of 2)"]
    assert len(checker.calls) == 2
    assert checker.calls[0][:4] == ("en", "es", TITLE, DESCRIPTION)
    assert set(store.latest_translations(V1)) == {"es", "de"}


def test_status_rules(store: Store) -> None:
    def verdict(score: int, *issues: str) -> Verdict:
        return Verdict(score, issues, "t", "d", "glimmer-test")

    def drop_link(source_language, target, title, description):  # noqa: ANN001
        return Translated("[pt] " + title, description.replace(LINK, ""), "spark-open-test")

    translator = ScriptedTranslator({"pt": drop_link})
    checker = ScriptedVerifier({"es": verdict(4), "de": verdict(3), "fr": verdict(5, "A price changed."),
                                "ru": verdict(5)})
    result = translate_video(store, translator, checker, V1, "en",
                             ["es", "de", "fr", "pt", "ru"], now=NOW)
    by = {r.language: r for r in result.records}
    assert by["es"].status == "verified"            # 4 passes
    assert by["de"].status == "flagged" and by["de"].score == 3   # 3 or less is flagged
    assert by["fr"].status == "flagged" and by["fr"].issues == ["A price changed."]
    assert by["pt"].status == "flagged" and by["pt"].score == 5   # code problem beats a good score
    assert any("missing" in p for p in by["pt"].problems)
    assert by["ru"].status == "verified"
    assert result.message == "2 verified, 3 flagged"


def test_unverified_when_the_checker_is_down(store: Store) -> None:
    checker = ScriptedVerifier(up=False)
    progress: list[str] = []
    result = translate_video(store, ScriptedTranslator(), checker, V1, "en", ["es", "de"], now=NOW,
                             report=progress.append)
    assert [r.status for r in result.records] == ["unverified", "unverified"]
    assert result.verifier_available is False and checker.calls == []
    assert all(r.verifier == "" and r.score is None and r.back_translation == {} for r in result.records)
    assert result.message == "2 unverified (the checker was not available)"
    assert progress == ["Translating 2 languages"]
    # no checker at all behaves the same
    result = translate_video(store, ScriptedTranslator(), None, V1, "en", ["fr"], now=NOW)
    assert result.records[0].status == "unverified" and result.verifier_available is False


def test_code_problems_flag_even_when_unverified(store: Store) -> None:
    translator = ScriptedTranslator({"es": Translated("", "x", "spark-open-test")})
    result = translate_video(store, translator, ScriptedVerifier(up=False), V1, "en", ["es"], now=NOW)
    assert result.records[0].status == "flagged"


def test_one_failure_does_not_stop_the_others(store: Store) -> None:
    translator = ScriptedTranslator({
        "de": TranslateError("rate_limit", "The translation service is limiting requests."),
        "fr": RuntimeError("boom with sk-secret-key"),
    })
    checker = ScriptedVerifier()
    progress: list[str] = []
    result = translate_video(store, translator, checker, V1, "en", ["es", "de", "fr", "pt"], now=NOW,
                             report=progress.append)
    assert [(r.language, r.status) for r in result.records] == [
        ("es", "verified"), ("de", "failed"), ("fr", "failed"), ("pt", "verified")]
    failed = result.records[1]
    assert (failed.title, failed.description, failed.score, failed.verifier) == ("", "", None, "")
    assert failed.problems == ["The translation service is limiting requests."]
    assert failed.translator == "spark-open-test"
    assert "sk-secret-key" not in result.records[2].problems[0]
    assert result.message == "2 verified, 2 failed"
    assert len(checker.calls) == 2 and "Checking de (2 of 4)" not in progress
    assert "Checking pt (4 of 4)" in progress


def test_checker_failing_midway(store: Store) -> None:
    checker = ScriptedVerifier({"es": TranslateError("network", "Could not reach the checker (Ollama).")})
    result = translate_video(store, ScriptedTranslator(), checker, V1, "en", ["es", "de"], now=NOW)
    assert [r.status for r in result.records] == ["unverified", "unverified"]
    assert result.verifier_available is False
    assert len(checker.calls) == 1  # it is not asked again once it is gone
    checker = ScriptedVerifier({"es": TranslateError("answer", "bad")})
    result = translate_video(store, ScriptedTranslator(), checker, V1, "en", ["es", "de"], now=NOW)
    assert [r.status for r in result.records] == ["unverified", "verified"]
    assert result.verifier_available is True
    assert result.records[0].back_translation == {"note": "bad"}  # why it was not checked
    checker = ScriptedVerifier({"es": KeyError("bug")})  # a checker bug keeps the translations
    result = translate_video(store, ScriptedTranslator(), checker, V1, "en", ["es", "de"], now=NOW)
    assert [r.status for r in result.records] == ["unverified", "verified"]
    assert result.records[0].back_translation == {
        "note": "The checker failed unexpectedly (KeyError)."}


def test_translates_in_parallel_and_keeps_target_order(store: Store) -> None:
    import threading

    seen: set[int] = set()

    def note(source_language, target, title, description):  # noqa: ANN001
        seen.add(threading.get_ident())
        return Translated(f"[{target}] {title}", description, "spark-open-test")

    translator = ScriptedTranslator({t: note for t in ("es", "pt", "de", "fr", "ru")})
    result = translate_video(store, translator, ScriptedVerifier(), V1, "en",
                             list(reversed(["es", "pt", "de", "fr", "ru"])), now=NOW, workers=4)
    assert [r.language for r in result.records] == ["ru", "fr", "de", "pt", "es"]
    assert len(translator.calls) == 5


def test_source_text_comes_from_the_draft_when_there_is_one(store: Store) -> None:
    translator = ScriptedTranslator()
    translate_video(store, translator, ScriptedVerifier(), V1, "en", ["es"], now=NOW)
    assert translator.calls[0][2:] == (TITLE, DESCRIPTION)  # the catalog
    save_draft(store, V1, "Better title #cactus", DESCRIPTION, ["cactus"], now=NOW)
    translator = ScriptedTranslator()
    result = translate_video(store, translator, ScriptedVerifier(), V1, "en", ["es"], now=NOW)
    assert translator.calls[0][2:] == ("Better title #cactus", DESCRIPTION)
    assert result.records[0].source_hash == source_hash("Better title #cactus", DESCRIPTION)
    # explicit text wins over the draft; the other part still comes from the draft
    translator = ScriptedTranslator()
    translate_video(store, translator, ScriptedVerifier(), V1, "en", ["es"], title="Given #cactus", now=NOW)
    assert translator.calls[0][2:] == ("Given #cactus", DESCRIPTION)
    translator = ScriptedTranslator()
    translate_video(store, translator, ScriptedVerifier(), V1, "en", ["es"], description="Given text", now=NOW)
    assert translator.calls[0][2:] == ("Better title #cactus", "Given text")


@pytest.mark.parametrize("source, targets", [
    ("xx", ["es"]),
    ("en", []),
    ("en", ["xx"]),
    ("en", ["en"]),
    ("en", ["es", "es"]),
    ("en", ["es", "klingon"]),
])
def test_invalid_languages_are_refused_before_any_call(store: Store, source: str, targets: list[str]) -> None:
    translator, checker = ScriptedTranslator(), ScriptedVerifier()
    with pytest.raises(ValueError):
        translate_video(store, translator, checker, V1, source, targets, now=NOW)
    assert translator.calls == [] and checker.calls == []
    assert store.list_translations(V1) == []


def test_unknown_video_or_empty_text_is_refused(store: Store) -> None:
    translator = ScriptedTranslator()
    with pytest.raises(ValueError):
        translate_video(store, translator, None, "nope", "en", ["es"], now=NOW)
    with pytest.raises(ValueError):
        translate_video(store, translator, None, V1, "en", ["es"], title=" ", description="", now=NOW)
    assert translator.calls == []


# --- keys -----------------------------------------------------------------------------------------------------------------
def test_key_source_and_loading(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    assert translate.translator_key_source(cfg, {}) == ""
    assert translate.load_translator_key(cfg, {}) is None
    # the Meta key is the fallback
    assert translate.translator_key_source(cfg, {"MODEL_API_KEY": "meta-key"}) == "meta"
    assert translate.load_translator_key(cfg, {"MODEL_API_KEY": "meta-key"}) is None
    # a saved key
    translate.save_translator_key(cfg, "  saved-key \n")
    assert cfg.translator_key_path.read_text(encoding="utf-8") == "saved-key"
    assert translate.translator_key_source(cfg, {"MODEL_API_KEY": "meta-key"}) == "saved"
    assert translate.load_translator_key(cfg, {}) == "saved-key"
    # the environment wins
    env = {"BASETEN_API_KEY": " env-key "}
    assert translate.translator_key_source(cfg, env) == "environment"
    assert translate.load_translator_key(cfg, env) == "env-key"
    assert translate.delete_translator_key(cfg) is True
    assert translate.delete_translator_key(cfg) is False
    assert translate.translator_key_source(cfg, {}) == ""
    assert not list(cfg.credentials_dir.glob("*.tmp"))


def test_save_translator_key_rejects_bad_keys(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    for bad in ("", "  ", "two words", "line\nbreak"):
        with pytest.raises(ValueError) as raised:
            translate.save_translator_key(cfg, bad)
        assert "two words" not in str(raised.value)
    assert not cfg.translator_key_path.exists()
    with pytest.raises(ValueError):
        translate.save_translator_key(replace(cfg, translator_key_path=None), "k")
    assert translate.delete_translator_key(replace(cfg, translator_key_path=None)) is False


def test_translator_from_config_prefers_baseten_then_meta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    made: list[dict[str, Any]] = []
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: made.append(kw) or SimpleNamespace())
    cfg = config(tmp_path)
    assert translate.translator_from_config(cfg, {}) is None
    assert made == []

    meta = translate.translator_from_config(cfg, {"MODEL_API_KEY": "meta-key"})
    assert meta.model == cfg.model
    assert (made[-1]["base_url"], made[-1]["api_key"]) == (agent.MODEL_BASE_URL, "meta-key")

    both = translate.translator_from_config(cfg, {"MODEL_API_KEY": "meta-key", "BASETEN_API_KEY": "bt-key"})
    assert both.model == cfg.translator_model == "meta-models/muse-spark-open-1026"
    assert (made[-1]["base_url"], made[-1]["api_key"]) == ("https://inference.baseten.co/v1", "bt-key")

    translate.save_translator_key(cfg, "saved-bt")
    saved = translate.translator_from_config(cfg, {"MODEL_API_KEY": "meta-key"})
    assert saved.model == cfg.translator_model and made[-1]["api_key"] == "saved-bt"


def test_verifier_from_config(tmp_path: Path) -> None:
    cfg = config(tmp_path, YEZATAS_VERIFIER_URL="http://127.0.0.1:9999/", YEZATAS_VERIFIER_MODEL="m:1b")
    v = translate.verifier_from_config(cfg)
    assert isinstance(v, OllamaVerifier)
    assert (v.base_url, v.model) == ("http://127.0.0.1:9999", "m:1b")


def test_module_has_no_import_time_dependency_on_openai() -> None:
    import ast

    tree = ast.parse(Path(translate.__file__).read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any(getattr(n, "module", None) == "openai" or any(a.name == "openai" for a in getattr(n, "names", []))
                   for n in top)
