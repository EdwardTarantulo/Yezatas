"""Custom checks by asking: words into one fixed rule, the guards, evaluate and run_rules.
Model replies are scripted with FakeModelClient; no test touches the network or a real model."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient
from yezatas import customchecks
from yezatas.agent import ModelError, ModelReply
from yezatas.customchecks import (FIELDS, KINDS, MAX_TAG_COUNT, MAX_TITLE_CHARS, REQUEST_MAX,
                                  RULE_FORMAT, SYSTEM_PROMPT, VISIBILITIES, RuleDraft, RuleError,
                                  RuleResult, evaluate, explain, rule_from_request, run_rules,
                                  validate_rule)
from yezatas.store import CustomRule, Store, Video

NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)
CHAPTERS = "About this video.\n0:00 Intro\n0:30 First step\n1:30 Second step"


def reply(data: Any, tokens: tuple[int, int, int] = (300, 20, 15)) -> ModelReply:
    text = data if isinstance(data, str) else json.dumps(data)
    return ModelReply(text, (), {}, *tokens)


def model_answer(**override: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"kind": "must_link", "domain": "example.com", "text": "",
                            "field": "", "chars": 0, "count": 0, "visibility": "public"}
    data.update(override)
    return data


def draft(data: Any, request: str | None = None) -> RuleDraft:
    if request is None:  # words that hold every value the answer uses, as a creator would write them
        values = data if isinstance(data, dict) else {}
        request = "check my videos " + " ".join(
            str(values.get(key) or "") for key in ("domain", "text", "chars", "count"))
    return rule_from_request(FakeModelClient([reply(data)]), request)


def make_video(**override: Any) -> Video:
    base: dict[str, Any] = dict(
        video_id="vid00000001", title="Repotting a Cactus in Five Steps",
        description="How to repot a cactus. See https://example.com/guide for more.",
        tags=["cactus", "garden", "soil mix"], privacy_status="public", duration="PT10M",
        view_count=100)
    base.update(override)
    return Video(**base)


def make_store(tmp_path: Path, videos: list[Video]) -> Store:
    store = Store(tmp_path / "c.sqlite")
    pull = store.begin_pull("chan0000001", now=NOW)
    for video in videos:
        store.upsert_video(video, pull, now=NOW)
    store.finish_pull(pull, len(videos), now=NOW)
    return store


# --- constants, schema and the one call -----------------------------------------------------------
def test_public_constants() -> None:
    assert KINDS == {"must_link": ("domain",), "must_contain": ("text", "field"),
                     "must_not_contain": ("text", "field"), "title_max": ("chars",),
                     "tags_min": ("count",), "chapters_required": ()}
    assert FIELDS == ("title", "description", "tags")
    assert VISIBILITIES == ("public", "all")
    assert REQUEST_MAX == 300
    assert MAX_TITLE_CHARS == 1000 and MAX_TAG_COUNT == 100
    assert issubclass(RuleError, ValueError)
    assert "only turns" in customchecks.__doc__  # type: ignore[operator]
    assert "No model-written code is ever run" in customchecks.__doc__  # type: ignore[operator]


def test_the_request_schema_and_prompt() -> None:
    client = FakeModelClient([reply(model_answer())])
    rule_from_request(client, "  every video links example.com  ")
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["response_format"] is RULE_FORMAT
    assert RULE_FORMAT["json_schema"]["strict"] is True
    assert RULE_FORMAT["json_schema"]["name"] == "custom_rule"
    schema = RULE_FORMAT["json_schema"]["schema"]
    assert schema["required"] == ["kind", "domain", "text", "field", "chars", "count", "visibility"]
    assert schema["properties"]["kind"]["enum"] == list(KINDS)
    assert schema["properties"]["field"]["enum"] == ["title", "description", "tags", ""]
    assert schema["properties"]["visibility"]["enum"] == ["public", "all"]
    assert schema["properties"]["domain"] == {"type": "string"}
    assert schema["properties"]["chars"] == {"type": "integer"}
    assert schema["properties"]["count"] == {"type": "integer"}
    system, user = call["messages"][0]["content"], call["messages"][1]["content"]
    assert system is SYSTEM_PROMPT or system == SYSTEM_PROMPT
    assert "only translates the creator's words into one fixed rule" in system
    assert "never writes code" in system
    assert user == "every video links example.com"


def test_the_draft_carries_the_model_cost_record() -> None:
    client = FakeModelClient([reply(model_answer())])
    made = rule_from_request(client, "every video links example.com")
    assert made.model == "muse-spark-1.3"
    assert (made.input_tokens, made.cached_tokens, made.output_tokens) == (300, 20, 15)


def test_a_direct_draft_defaults_to_no_cost_record() -> None:
    made = RuleDraft(rule={"kind": "must_link", "domain": "example.com", "visibility": "public"},
                     explain="Every public video's description links example.com.")
    assert made.model == ""
    assert (made.input_tokens, made.cached_tokens, made.output_tokens) == (0, 0, 0)


# --- each kind end to end: normalized rule + exact explain -----------------------------------------
CASES: list[tuple[dict[str, Any], dict[str, Any], str, str]] = [
    ({"kind": "must_link", "domain": "  Example.COM  "},
     {"kind": "must_link", "domain": "example.com", "visibility": "public"},
     "Every public video's description links example.com.",
     "Every video's description links example.com."),
    ({"kind": "must_contain", "text": "  newsletter  ", "field": "title"},
     {"kind": "must_contain", "text": "newsletter", "field": "title", "visibility": "public"},
     'Every public video\'s title contains "newsletter".',
     'Every video\'s title contains "newsletter".'),
    ({"kind": "must_not_contain", "text": "spoiler", "field": "description"},
     {"kind": "must_not_contain", "text": "spoiler", "field": "description", "visibility": "public"},
     'Every public video\'s description does not contain "spoiler".',
     'Every video\'s description does not contain "spoiler".'),
    ({"kind": "title_max", "chars": 60},
     {"kind": "title_max", "chars": 60, "visibility": "public"},
     "Every public video's title is at most 60 characters.",
     "Every video's title is at most 60 characters."),
    ({"kind": "tags_min", "count": 5},
     {"kind": "tags_min", "count": 5, "visibility": "public"},
     "Every public video has at least 5 tags.",
     "Every video has at least 5 tags."),
    ({"kind": "chapters_required"},
     {"kind": "chapters_required", "visibility": "public"},
     "Every public video longer than a minute has chapters.",
     "Every video longer than a minute has chapters."),
]


@pytest.mark.parametrize("override,rule,explain_public,explain_all", CASES,
                         ids=[c[1]["kind"] for c in CASES])
def test_each_kind_end_to_end(override: dict[str, Any], rule: dict[str, Any],
                              explain_public: str, explain_all: str) -> None:
    public = draft(model_answer(**override))
    assert public.rule == rule and public.explain == explain_public
    every = draft(model_answer(visibility="all", **override))
    assert every.rule == {**rule, "visibility": "all"} and every.explain == explain_all


def test_only_the_kinds_own_keys_are_kept() -> None:
    noisy = model_answer(kind="chapters_required", domain="junk.example", text="junk",
                         field="title", chars=99, count=99)
    assert draft(noisy).rule == {"kind": "chapters_required", "visibility": "public"}
    link = draft(model_answer(kind="must_link", text="junk", field="title", chars=99, count=99))
    assert link.rule == {"kind": "must_link", "domain": "example.com", "visibility": "public"}


def test_singular_explain() -> None:
    assert draft(model_answer(kind="title_max", chars=1)).explain == \
        "Every public video's title is at most 1 character."
    assert draft(model_answer(kind="tags_min", count=1)).explain == \
        "Every public video has at least 1 tag."
    assert draft(model_answer(kind="title_max", chars=1, visibility="all")).explain == \
        "Every video's title is at most 1 character."
    assert draft(model_answer(kind="tags_min", count=1, visibility="all")).explain == \
        "Every video has at least 1 tag."


def test_text_keeps_its_case_and_domain_is_lowercased() -> None:
    made = draft(model_answer(kind="must_contain", text="  NewsLetter ", field="tags"))
    assert made.rule["text"] == "NewsLetter"
    assert made.explain == 'Every public video\'s tags contains "NewsLetter".'


# --- refused before, or instead of, a usable answer -------------------------------------------------
@pytest.mark.parametrize("words", ["", "   ", "\n\t "])
def test_an_empty_request_is_refused_with_no_call(words: str) -> None:
    client = FakeModelClient([])
    with pytest.raises(RuleError, match="Describe the check"):
        rule_from_request(client, words)
    assert client.calls == []


def test_a_long_request_is_refused_with_no_call() -> None:
    client = FakeModelClient([])
    with pytest.raises(RuleError, match="at most 300 characters"):
        rule_from_request(client, "x" * (REQUEST_MAX + 1))
    assert client.calls == []


def test_the_length_counts_the_request_as_given() -> None:
    client = FakeModelClient([])
    with pytest.raises(RuleError):
        rule_from_request(client, " " + "x" * REQUEST_MAX)  # 301 as given, 300 stripped
    assert client.calls == []


def test_a_request_of_exactly_the_limit_is_asked() -> None:
    client = FakeModelClient([reply(model_answer())])
    words = "link example.com "
    made = rule_from_request(client, words + "x" * (REQUEST_MAX - len(words)))
    assert made.rule["kind"] == "must_link" and len(client.calls) == 1


def test_a_model_error_becomes_a_rule_error() -> None:
    client = FakeModelClient([ModelError("rate_limit", "Slow down, then try again.")])
    with pytest.raises(RuleError, match="Slow down, then try again") as caught:
        rule_from_request(client, "titles at most 60 characters")
    assert type(caught.value) is RuleError and len(client.calls) == 1


@pytest.mark.parametrize("content", ["not json at all", "", "[1, 2]",
                                     json.dumps({"kind": "must_link"})])
def test_an_unreadable_answer_becomes_a_rule_error(content: str) -> None:
    # the last one parses but has no usable fields: a missing visibility is a RuleError, never a crash
    with pytest.raises(RuleError):
        rule_from_request(FakeModelClient([reply(content)]), "check my videos")


@pytest.mark.parametrize("override", [
    {"kind": "link_everything"},
    {"kind": ""},
    {"visibility": "private"},
    {"visibility": ""},
    {"kind": "title_max", "chars": 0},
    {"kind": "title_max", "chars": -5},
    {"kind": "title_max", "chars": MAX_TITLE_CHARS + 1},
    {"kind": "title_max", "chars": "60"},
    {"kind": "title_max", "chars": 60.5},
    {"kind": "title_max", "chars": True},
    {"kind": "title_max", "chars": None},
    {"kind": "tags_min", "count": 0},
    {"kind": "tags_min", "count": -1},
    {"kind": "tags_min", "count": MAX_TAG_COUNT + 1},
    {"kind": "tags_min", "count": "5"},
    {"kind": "tags_min", "count": True},
    {"kind": "tags_min", "count": False},
    {"kind": "must_contain", "text": "   ", "field": "title"},
    {"kind": "must_contain", "text": "", "field": "title"},
    {"kind": "must_not_contain", "text": "  ", "field": "tags"},
    {"kind": "must_contain", "text": "news", "field": ""},
    {"kind": "must_contain", "text": "news", "field": "transcript"},
    {"kind": "must_not_contain", "text": "news", "field": "Title"},
])
def test_bad_model_answers_raise_rule_error(override: dict[str, Any]) -> None:
    with pytest.raises(RuleError):
        draft(model_answer(**override))


@pytest.mark.parametrize("domain", [
    "not a domain", "https://example.com/x", "https://example.com", "http://example.com",
    "example", "example.com/path", "example.com:8080", "exam ple.com", "exam_ple.com",
    "-example.com", "example-.com", "example..com", ".example.com", "example.com.", "", "   ",
])
def test_bad_domains_raise_rule_error(domain: str) -> None:
    with pytest.raises(RuleError):
        draft(model_answer(kind="must_link", domain=domain))


@pytest.mark.parametrize("domain,wanted", [("example.com", "example.com"),
                                           ("EXAMPLE.COM", "example.com"),
                                           ("sub.example.co.uk", "sub.example.co.uk")])
def test_good_domains_are_lowercased(domain: str, wanted: str) -> None:
    assert draft(model_answer(kind="must_link", domain=domain)).rule["domain"] == wanted


# --- evaluate: one video ---------------------------------------------------------------------------
def test_evaluate_must_link() -> None:
    rule = {"kind": "must_link", "domain": "example.com", "visibility": "public"}
    assert evaluate(rule, make_video()) is True
    assert evaluate(rule, make_video(description="No links at all.")) is False
    assert evaluate(rule, make_video(description="See https://blog.example.com/post")) is True
    assert evaluate(rule, make_video(description="See https://www.example.com/x")) is True
    assert evaluate(rule, make_video(description="See https://a.b.example.com/x")) is True
    assert evaluate(rule, make_video(description="See https://notexample.com/x")) is False
    assert evaluate(rule, make_video(description="See https://example.com.evil.example/x")) is False
    www_rule = {"kind": "must_link", "domain": "www.example.com", "visibility": "all"}
    assert evaluate(www_rule, make_video(description="See https://example.com/x")) is True


def test_evaluate_must_contain_per_field() -> None:
    title_rule = {"kind": "must_contain", "text": "CACTUS", "field": "title", "visibility": "all"}
    assert evaluate(title_rule, make_video()) is True
    assert evaluate(title_rule, make_video(title="Repotting in Five Steps")) is False
    desc_rule = {"kind": "must_contain", "text": "repot a cactus", "field": "description",
                 "visibility": "all"}
    assert evaluate(desc_rule, make_video()) is True
    tags_rule = {"kind": "must_contain", "text": "soil mix", "field": "tags", "visibility": "all"}
    assert evaluate(tags_rule, make_video()) is True
    assert evaluate(tags_rule, make_video(tags=["cactus", "garden"])) is False
    # tags are joined with newlines: a text spanning two tags does not match
    spanning = {"kind": "must_contain", "text": "cactus garden", "field": "tags", "visibility": "all"}
    assert evaluate(spanning, make_video()) is False


def test_evaluate_must_not_contain() -> None:
    rule = {"kind": "must_not_contain", "text": "spoiler", "field": "description", "visibility": "all"}
    assert evaluate(rule, make_video()) is True
    assert evaluate(rule, make_video(description="Warning: SPOILER ahead.")) is False
    title_rule = {"kind": "must_not_contain", "text": "cactus", "field": "title", "visibility": "all"}
    assert evaluate(title_rule, make_video()) is False


def test_evaluate_title_max_and_tags_min() -> None:
    title_rule = {"kind": "title_max", "chars": 60, "visibility": "all"}
    assert evaluate(title_rule, make_video(title="x" * 60)) is True
    assert evaluate(title_rule, make_video(title="x" * 61)) is False
    tags_rule = {"kind": "tags_min", "count": 3, "visibility": "all"}
    assert evaluate(tags_rule, make_video(tags=["a", "b", "c"])) is True
    assert evaluate(tags_rule, make_video(tags=["a", " ", "", "b"])) is False  # blanks do not count
    assert evaluate(tags_rule, make_video(tags=["a", "b", "c", ""])) is True


def test_evaluate_chapters_required() -> None:
    rule = {"kind": "chapters_required", "visibility": "all"}
    assert evaluate(rule, make_video(description=CHAPTERS, duration="PT10M")) is True
    assert evaluate(rule, make_video(description="No chapters here.", duration="PT10M")) is False
    late = "About.\n1:00 Start\n2:00 Middle\n3:00 End"
    assert evaluate(rule, make_video(description=late, duration="PT10M")) is False  # not from 0:00
    assert evaluate(rule, make_video(description=CHAPTERS, duration="PT1M1S")) is True  # just over a minute


@pytest.mark.parametrize("duration", ["PT1M", "PT59S", "PT45S", "", "nonsense", "PT0S"])
def test_chapters_do_not_apply_to_short_or_unreadable_durations(duration: str) -> None:
    rule = {"kind": "chapters_required", "visibility": "all"}
    assert evaluate(rule, make_video(description=CHAPTERS, duration=duration)) is None


@pytest.mark.parametrize("privacy", ["private", "unlisted", ""])
def test_a_public_rule_does_not_apply_to_non_public_videos(privacy: str) -> None:
    rule = {"kind": "must_contain", "text": "cactus", "field": "title", "visibility": "public"}
    assert evaluate(rule, make_video(privacy_status=privacy)) is None
    chapters = {"kind": "chapters_required", "visibility": "public"}
    assert evaluate(chapters, make_video(privacy_status=privacy, description=CHAPTERS)) is None
    every = {**rule, "visibility": "all"}
    assert evaluate(every, make_video(privacy_status=privacy)) is True


def test_evaluate_accepts_a_custom_rule() -> None:
    stored = CustomRule(id=7, name="Links", request="link example.com",
                        rule={"kind": "must_link", "domain": "example.com", "visibility": "public"},
                        enabled=True, created_at=NOW.isoformat())
    assert evaluate(stored, make_video()) is True
    assert evaluate(stored, make_video(description="Nothing here.")) is False
    assert evaluate(stored, make_video(privacy_status="private")) is None


def test_evaluate_rejects_an_unknown_kind() -> None:
    with pytest.raises(RuleError):
        evaluate({"kind": "nope", "visibility": "all"}, make_video())


# --- run_rules over a real store ---------------------------------------------------------------------
def _catalog(tmp_path: Path) -> Store:
    videos = [
        make_video(video_id="vid00000001", title="Alpha video title", view_count=300,
                   description="See https://example.com/a\n" + CHAPTERS,
                   tags=["one", "two", "three", "four", "five"]),
        make_video(video_id="vid00000002", title="Beta", view_count=200,
                   description="No links here.", tags=["one"]),
        make_video(video_id="vid00000003", title="Gamma private video", view_count=100,
                   privacy_status="private", description="See https://example.com/c",
                   tags=["one", "two", "three", "four", "five", "six"]),
    ]
    return make_store(tmp_path, videos)


LINK_PUBLIC = {"kind": "must_link", "domain": "example.com", "visibility": "public"}
TAGS_ALL = {"kind": "tags_min", "count": 5, "visibility": "all"}
CHAPTERS_PUBLIC = {"kind": "chapters_required", "visibility": "public"}


def test_run_rules_counts_and_catalog_order(tmp_path: Path) -> None:
    store = _catalog(tmp_path)
    results = run_rules(store, [LINK_PUBLIC, TAGS_ALL, CHAPTERS_PUBLIC])
    assert [r.rule for r in results] == [LINK_PUBLIC, TAGS_ALL, CHAPTERS_PUBLIC]
    # catalog order is vid00000001, vid00000002, vid00000003 (most views first)
    assert results[0] == RuleResult(LINK_PUBLIC, 1, ("vid00000002",), 1)
    assert results[1] == RuleResult(TAGS_ALL, 2, ("vid00000002",), 0)
    assert results[2] == RuleResult(CHAPTERS_PUBLIC, 1, ("vid00000002",), 1)
    store.close()


def test_run_rules_keeps_the_given_order_and_empty_is_empty(tmp_path: Path) -> None:
    store = _catalog(tmp_path)
    results = run_rules(store, [TAGS_ALL, LINK_PUBLIC])
    assert results[0].rule is TAGS_ALL and results[1].rule is LINK_PUBLIC
    assert run_rules(store, []) == []
    store.close()


def test_run_rules_with_custom_rules_and_caller_side_disabling(tmp_path: Path) -> None:
    store = _catalog(tmp_path)
    first = store.add_custom_rule("Links", "link example.com", LINK_PUBLIC, now=NOW)
    second = store.add_custom_rule("Tags", "at least 5 tags", TAGS_ALL, now=NOW)
    store.set_custom_rule_enabled(second, False)
    everything = run_rules(store, store.list_custom_rules())
    assert [r.rule.id for r in everything] == [first, second]  # run_rules runs what it is given
    assert everything[0].passing == 1 and everything[1].failing == ("vid00000002",)
    enabled = run_rules(store, store.list_custom_rules(enabled_only=True))
    assert [r.rule.id for r in enabled] == [first]  # filtering disabled rules is the caller's job
    assert isinstance(enabled[0].rule, CustomRule)
    store.close()


def test_run_rules_on_an_empty_catalog(tmp_path: Path) -> None:
    store = make_store(tmp_path, [])
    assert run_rules(store, [LINK_PUBLIC]) == [RuleResult(LINK_PUBLIC, 0, (), 0)]
    store.close()


# --- the rule's values must come from the creator's words ---------------------------------------
@pytest.mark.parametrize("answer, words, message", [
    (model_answer(kind="must_link", domain="other.example"), "link my newsletter", "other.example"),
    (model_answer(kind="must_contain", text="Subscribe", field="description"),
     "every description asks people to follow", "Subscribe"),
    (model_answer(kind="title_max", chars=70), "keep titles short", "70"),
    (model_answer(kind="tags_min", count=8), "use 5 tags or more", "8"),
])
def test_a_value_not_in_the_words_is_refused(answer: dict[str, Any], words: str, message: str) -> None:
    with pytest.raises(RuleError, match="not in your words") as caught:
        draft(answer, words)
    assert message in str(caught.value)


def test_values_match_the_words_ignoring_case() -> None:
    made = draft(model_answer(kind="must_contain", text="Newsletter", field="description"),
                 "every description mentions my NEWSLETTER")
    assert made.rule["text"] == "Newsletter"
    assert draft(model_answer(kind="tags_min", count=5), "at least 5 tags").rule["count"] == 5


def test_validate_rule_refuses_a_value_not_in_the_words() -> None:
    data = model_answer(kind="must_link", domain="other.example")
    with pytest.raises(RuleError, match="not in your words") as caught:
        validate_rule(data, "link my newsletter")
    assert "other.example" in str(caught.value)


def test_validate_rule_returns_the_rule_and_its_explanation() -> None:
    rule, text = validate_rule(model_answer(), "every video links example.com")
    assert rule == {"kind": "must_link", "domain": "example.com", "visibility": "public"}
    assert text == explain(rule) == "Every public video's description links example.com."
