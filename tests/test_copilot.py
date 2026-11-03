import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient
from yezatas import copilot, edits, outliers
from yezatas.agent import ModelError, ModelReply, ToolCall
from yezatas.copilot import (
    MAX_QUESTION_CHARS, TOOLS, CopilotError, ask, catalog_refs, dismiss_proposal, make_draft,
)
from yezatas.store import Store, TranscriptRecord, Video

NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)
AS_OF = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)
CHAPTERS = "A short guide.\n\n0:00 Intro\n0:30 Soil\n1:10 Pots\n"
SHOP = "https://shop.invalid/pot"


def vid(n: int) -> str:
    return f"vid{n:08d}"


def make_video(n: int, **fields: Any) -> Video:
    base = Video(video_id=vid(n), channel_id="UC_fake", title=f"Cactus tip number {n}",
                 description=f"A short guide to cactus care, part {n}.", tags=["cactus", "garden"],
                 privacy_status="public", published_at=f"2026-01-{n:02d}T10:00:00Z",
                 duration="PT5M", view_count=100 * n)
    return replace(base, **fields)


def catalog_videos() -> list[Video]:
    """13 made-up videos. V12 is an outlier, V2 is far below the norm, V13 is too new, V3 is
    private, V4 is a Short, V1 has chapters, V6 mentions Photoshop and a shop link."""
    return [
        make_video(1, description=CHAPTERS),
        make_video(2, view_count=1),
        make_video(3, privacy_status="private"),
        make_video(4, duration="PT45S"),
        make_video(5),
        make_video(6, description=f"Edit the label in Photoshop.\nShop: {SHOP}\n"),
        make_video(7), make_video(8), make_video(9), make_video(10), make_video(11),
        make_video(12, view_count=500_000),
        make_video(13, published_at="2026-05-25T10:00:00Z", view_count=150),
    ]


def fill(store: Store, videos: list[Video]) -> None:
    pull = store.begin_pull("UC_fake", NOW)
    for v in videos:
        store.upsert_video(v, pull, NOW)
    store.finish_pull(pull, len(videos), NOW)


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as s:
        fill(s, catalog_videos())
        segments = [{"start_ms": 0, "end_ms": 5000, "text": "Hello everyone"},
                    {"start_ms": 65_000, "end_ms": 70_000, "text": "Today we open Photoshop for the label"}]
        s.add_transcript(TranscriptRecord(0, vid(8), "en", "voice-test", "clip.mp4", 70_000,
                                          segments, [], ""), NOW)
        yield s


def tool_reply(*calls: tuple[str, dict[str, Any] | str], tokens: tuple[int, int, int] = (0, 0, 0)
               ) -> ModelReply:
    tool_calls = tuple(
        ToolCall(f"call_{i}_{name}", name, args if isinstance(args, str) else json.dumps(args))
        for i, (name, args) in enumerate(calls))
    message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
        for c in tool_calls]}
    return ModelReply(None, tool_calls, message, *tokens)


def answer(text: str, tokens: tuple[int, int, int] = (0, 0, 0)) -> ModelReply:
    return ModelReply(text, (), {"role": "assistant", "content": text}, *tokens)


def run(store: Store, script: list[Any], question: str = "What should I fix?", chat_id: int = 1,
        **kwargs: Any):
    """Ask once; returns (turn, client)."""
    client = FakeModelClient(script)
    turn_id = ask(store, client, question, chat_id=chat_id, as_of=AS_OF, now=NOW, **kwargs)
    turn = store.get_copilot_turn(turn_id)
    assert turn is not None
    return turn, client


@pytest.fixture
def box(store: Store):
    """The toolbox of a running turn, for calling tools one by one."""
    catalog = copilot._Catalog(store.list_videos("video_id"), AS_OF)
    turn_id = store.begin_copilot_turn(1, "q", "muse-spark-1.3", NOW)
    return copilot._Toolbox(store, catalog, turn_id, NOW)


def call(box: copilot._Toolbox, name: str, args: dict[str, Any] | str = "") -> dict[str, Any]:
    text = args if isinstance(args, str) else json.dumps(args)
    return json.loads(box.run(name, text)[1])


# --- before a turn -------------------------------------------------------------------------------
def test_catalog_refs_number_the_oldest_upload_first() -> None:
    refs = catalog_refs(list(reversed(catalog_videos())))
    assert refs["V1"] == vid(1) and refs["V12"] == vid(12) and refs["V13"] == vid(13)


@pytest.mark.parametrize("question", ["", "   \n ", "x" * (MAX_QUESTION_CHARS + 1)])
def test_bad_questions_raise_before_a_turn(store: Store, question: str) -> None:
    client = FakeModelClient([])
    with pytest.raises(CopilotError):
        ask(store, client, question, chat_id=1, as_of=AS_OF)
    assert store.list_copilot_turns(1) == [] and client.calls == []


def test_question_of_the_maximum_length_is_accepted(store: Store) -> None:
    turn, _ = run(store, [answer("Fine.")], question="x" * MAX_QUESTION_CHARS)
    assert turn.status == "answered"


def test_no_catalog_raises_before_a_turn(tmp_path: Path) -> None:
    with Store(tmp_path / "empty.sqlite") as empty:
        client = FakeModelClient([])
        with pytest.raises(CopilotError, match="no pulled catalog yet. Pull it in Configuration first."):
            ask(empty, client, "Hello?", chat_id=1, as_of=AS_OF)
        assert empty.list_copilot_turns(1) == [] and client.calls == []


# --- what the model gets -------------------------------------------------------------------------
def test_first_call_has_system_prompt_with_the_table_and_all_tools(store: Store) -> None:
    turn, client = run(store, [answer("Nothing to fix.")], question="  What should I fix?  ")
    first = client.calls[0]
    assert first["tools"] == TOOLS and first["response_format"] is None
    assert [t["function"]["name"] for t in TOOLS] == [
        "find_videos", "search_videos", "get_video", "get_transcript", "channel_stats",
        "propose_edit"]
    system, last = first["messages"][0], first["messages"][-1]
    assert system["role"] == "system" and system["content"].startswith(
        "You are the channel copilot")
    assert last == {"role": "user", "content": "What should I fix?"}
    assert len(first["messages"]) == 2
    assert turn.question == "What should I fix?" and turn.model == "muse-spark-1.3"
    assert turn.status == "answered" and turn.answer == "Nothing to fix."
    assert turn.finished_at is not None and turn.chat_id == 1


def test_table_has_every_video_newest_first_with_numbers_from_code(store: Store) -> None:
    _, client = run(store, [answer("Ok.")])
    table = client.calls[0]["messages"][0]["content"].split("last pull:\n", 1)[1]
    lines = table.split("\n")
    assert lines[0] == ("ref | published | visibility | views | vs norm | SEO score | chapters "
                        "| short | title")
    assert [line.split(" | ")[0] for line in lines[1:]] == [f"V{n}" for n in range(13, 0, -1)]

    scores = {s.video_id: s for s in outliers.score_videos(store.list_videos("video_id"), AS_OF)}
    cards = copilot._Catalog(store.list_videos("video_id"), AS_OF)
    cells = {line.split(" | ")[0]: line.split(" | ") for line in lines[1:]}
    v12 = cells["V12"]
    assert v12[1:5] == ["2026-01-12", "public", "500,000",
                        outliers.ratio_label(scores[vid(12)].ratio)]
    assert v12[4].endswith("×") and v12[5] == str(cards.by_ref["V12"].card.score)
    assert v12[6:] == ["no", "no", "Cactus tip number 12"]
    assert cells["V13"][4] == "too new"
    assert cells["V3"][2] == "private" and cells["V3"][4] == "not public"
    assert cells["V1"][6] == "yes" and cells["V4"][7] == "yes"
    assert cells["V4"][6] == "n/a"  # a Short: chapters don't apply
    assert cells["V2"][4] == outliers.ratio_label(scores[vid(2)].ratio)


def test_catalog_flags_match_the_outlier_module(store: Store) -> None:
    catalog = copilot._Catalog(store.list_videos("video_id"), AS_OF)
    assert catalog.outlier_ids == {vid(12)} and catalog.below_ids == {vid(2)}


def test_history_is_the_last_six_answered_turns_of_this_chat(store: Store) -> None:
    for n in range(1, 9):
        turn_id = store.begin_copilot_turn(1, f"question {n}", "m", NOW)
        store.finish_copilot_turn(turn_id, "answered", f"answer {n}", {}, [], [], now=NOW)
    failed = store.begin_copilot_turn(1, "failed question", "m", NOW)
    store.finish_copilot_turn(failed, "failed", "model down", {}, [], [], now=NOW)
    other = store.begin_copilot_turn(2, "other chat", "m", NOW)
    store.finish_copilot_turn(other, "answered", "other answer", {}, [], [], now=NOW)

    _, client = run(store, [answer("Ok.")], question="the new one")
    pairs = [(m["role"], m["content"]) for m in client.calls[0]["messages"][1:]]
    expected = [p for n in range(3, 9)
                for p in (("user", f"question {n}"), ("assistant", f"answer {n}"))]
    assert pairs == expected + [("user", "the new one")]


def test_history_after_a_real_turn_is_the_answer_text_only(store: Store) -> None:
    run(store, [answer("V12 is the best.")], question="Which is best?")
    _, client = run(store, [answer("Ok.")], question="And then?")
    assert client.calls[0]["messages"][1:] == [
        {"role": "user", "content": "Which is best?"},
        {"role": "assistant", "content": "V12 is the best."},
        {"role": "user", "content": "And then?"}]


# --- the conversation ----------------------------------------------------------------------------
def test_tool_results_go_back_to_the_model_and_progress_is_reported(store: Store) -> None:
    lines: list[str] = []
    turn, client = run(store, [
        tool_reply(("find_videos", {"visibility": "private"}), ("channel_stats", "")),
        tool_reply(("search_videos", {"query": "photoshop"})),
        tool_reply(("get_video", {"ref": "V6"}), ("get_transcript", {"ref": "V8"})),
        answer("V3 is private."),
    ], report=lines.append)
    assert lines == ["Reading the catalog", "Counting videos", "Counting videos", "Searching",
                     "Reading a video", "Reading a video", "Checking the numbers"]
    second = client.calls[1]["messages"]
    assert [m["role"] for m in second[-3:]] == ["assistant", "tool", "tool"]
    assert second[-2]["tool_call_id"] == "call_0_find_videos"
    assert json.loads(second[-2]["content"])["count"] == 1
    assert [(e["type"], e["name"]) for e in turn.log] == [
        ("tool", "find_videos"), ("tool", "channel_stats"), ("tool", "search_videos"),
        ("tool", "get_video"), ("tool", "get_transcript")]
    assert turn.log[0]["arguments"] == {"visibility": "private"} and turn.log[1]["arguments"] == {}
    assert all(e["result_chars"] > 0 for e in turn.log)


def test_tokens_are_summed_over_every_call(store: Store) -> None:
    turn, _ = run(store, [
        tool_reply(("channel_stats", ""), tokens=(1000, 200, 50)),
        answer("V12 has 7391 views.", tokens=(1500, 800, 30)),  # repair round
        answer("V12 has 500,000 views.", tokens=(1600, 900, 10)),
    ])
    assert (turn.input_tokens, turn.cached_tokens, turn.output_tokens) == (4100, 1900, 90)


def test_max_tool_rounds_then_one_call_without_tools(store: Store) -> None:
    script = [tool_reply(("channel_stats", "")), tool_reply(("channel_stats", "")),
              answer("Here is what I found.")]
    turn, client = run(store, script, max_tool_rounds=2)
    assert len(client.calls) == 3
    assert client.calls[0]["tools"] == TOOLS and client.calls[1]["tools"] == TOOLS
    assert client.calls[2]["tools"] is None
    assert client.calls[2]["messages"][-1]["role"] == "user"
    assert "Do not call more tools" in client.calls[2]["messages"][-1]["content"]
    assert turn.status == "answered" and turn.answer == "Here is what I found."


def test_model_error_fails_the_turn_with_its_message(store: Store) -> None:
    turn, _ = run(store, [tool_reply(("channel_stats", ""), tokens=(10, 0, 5)),
                          ModelError("rate_limit", "The model is busy. Try again soon.")])
    assert turn.status == "failed" and turn.answer == "The model is busy. Try again soon."
    assert turn.log[-1] == {"type": "error", "kind": "rate_limit",
                            "message": "The model is busy. Try again soon."}
    assert (turn.input_tokens, turn.output_tokens) == (10, 5)
    assert turn.refs == {} and turn.finished_at is not None


def test_unexpected_error_closes_the_turn_and_reraises(store: Store) -> None:
    client = FakeModelClient([RuntimeError("boom")])
    with pytest.raises(RuntimeError, match="boom"):
        ask(store, client, "Hello?", chat_id=1, as_of=AS_OF, now=NOW)
    (turn,) = store.list_copilot_turns(1)
    assert turn.status == "failed" and turn.answer == "The run stopped with an unexpected error."


def test_an_empty_answer_fails_the_turn(store: Store) -> None:
    turn, _ = run(store, [ModelReply(None, (), {"role": "assistant", "content": None})])
    assert turn.status == "failed" and "no answer" in turn.answer
    assert turn.log[-1]["kind"] == "answer"


# --- refs ----------------------------------------------------------------------------------------
def test_refs_cover_the_answer_and_proposals_and_unknown_refs_are_noted(store: Store) -> None:
    turn, _ = run(store, [
        tool_reply(("propose_edit", {"ref": "V1", "title": "Cactus tip number one", "reason": "Clearer.",
                                     "description": CHAPTERS, "tags": ["cactus", "garden"]})),
        answer("Look at V12 and V12 again, then V7. V87 is not mine. I offered a title for V1."),
    ])
    (proposal,) = store.list_copilot_proposals([turn.id])
    assert turn.refs == {"V1": vid(1), "V7": vid(7), "V12": vid(12)}
    assert turn.notes == ["The answer mentions V87, which is not in the catalog."]
    assert "V87 is not mine" in turn.answer  # left as text
    assert proposal.video_id == vid(1)


def test_a_proposal_alone_is_a_ref(store: Store) -> None:
    turn, _ = run(store, [
        tool_reply(("propose_edit", {"ref": "V5", "title": "Cactus tip five", "reason": "Shorter.",
                                     "description": "A short guide to cactus care, part 5.",
                                     "tags": ["cactus", "garden"]})),
        answer("I made one proposal."),
    ])
    assert turn.refs == {"V5": vid(5)}


# --- the number guard ----------------------------------------------------------------------------
def test_unsupported_ignores_refs_list_numbers_and_separators() -> None:
    assert copilot._unsupported("V12 and V3 beat V105.", set()) == []
    assert copilot._unsupported("1. First\n2) Second\n10. Tenth", set()) == []
    assert copilot._unsupported("I saw 3 of them, or 1,200.", {"3"}) == ["1,200"]
    assert copilot._unsupported("1 200 views and 1.200 views", {"1200"}) == []
    assert copilot._unsupported("7 and 7 and 8", set()) == ["7", "8"]


def test_numbers_from_the_table_pass_without_a_repair(store: Store) -> None:
    turn, client = run(store, [answer("V12 has 500,000 views, from 2026-01-12.")])
    assert len(client.calls) == 1 and turn.notes == [] and turn.log == []


def test_numbers_from_the_question_pass(store: Store) -> None:
    turn, client = run(store, [answer("No, 7391 is not special.")], question="Is 7391 good?")
    assert len(client.calls) == 1 and turn.notes == []


def test_numbers_from_earlier_turns_pass(store: Store) -> None:
    first = store.begin_copilot_turn(1, "Earlier question", "m", NOW)
    store.finish_copilot_turn(first, "answered", "It was 4242.", {}, [], [], now=NOW)
    turn, client = run(store, [answer("Still 4242.")], question="Again?")
    assert len(client.calls) == 1 and turn.notes == []


def test_numbers_from_a_tool_result_pass_and_others_do_not(store: Store) -> None:
    v12 = store.get_video(vid(12))
    pull = store.begin_pull("UC_fake", NOW)
    store.upsert_video(replace(v12, description="It sold 4321 pots."), pull, NOW)
    table_numbers = copilot._numbers(copilot._Catalog(store.list_videos("video_id"), AS_OF).table())
    assert "4321" not in table_numbers  # only in the get_video result

    turn, client = run(store, [tool_reply(("get_video", {"ref": "V12"})),
                               answer("It sold 4321 pots.")])
    assert len(client.calls) == 2 and turn.notes == []

    turn, client = run(store, chat_id=2, script=[answer("It sold 4321 pots."),
                               tool_reply(("get_video", {"ref": "V12"})),
                               answer("It sold 4321 pots.")])
    assert len(client.calls) == 3 and turn.notes == []  # the repair round may call a tool
    assert turn.answer == "It sold 4321 pots."


def test_repair_round_asks_again_and_keeps_a_clean_answer(store: Store) -> None:
    turn, client = run(store, [answer("V12 has 7391 views."), answer("V12 has 500,000 views.")])
    assert len(client.calls) == 2
    repair = client.calls[1]["messages"]
    assert repair[-2] == {"role": "assistant", "content": "V12 has 7391 views."}
    assert repair[-1]["role"] == "user" and "These numbers came from no tool: 7391." in repair[-1]["content"]
    assert client.calls[1]["tools"] == TOOLS
    assert turn.answer == "V12 has 500,000 views." and turn.notes == []
    assert turn.log == [{"type": "guard", "round": 1, "numbers": ["7391"]}]


def test_unsupported_sentences_are_dropped_with_a_note_after_the_repair(store: Store) -> None:
    bad = "First sentence is fine. V12 has 7391 views and V2 has 8802. Last sentence is fine."
    turn, client = run(store, [answer(bad), answer(bad)])
    assert len(client.calls) == 2
    assert turn.status == "answered"
    assert turn.answer == "First sentence is fine. Last sentence is fine."
    assert turn.notes == ["A sentence was removed: it quoted 7391, 8802, which no tool returned."]
    assert [e["round"] for e in turn.log] == [1, 2]


def test_a_dropped_bullet_leaves_the_other_bullets(store: Store) -> None:
    bad = "Two videos stand out:\n- V12 has 7391 views.\n- V2 is the weakest."
    turn, _ = run(store, [answer(bad), answer(bad)])
    assert turn.answer == "Two videos stand out:\n- V2 is the weakest."
    assert len(turn.notes) == 1 and "7391" in turn.notes[0]


def test_an_answer_with_nothing_left_fails_the_turn(store: Store) -> None:
    turn, _ = run(store, [answer("V12 has 7391 views."), answer("V12 has 7391 views.")])
    assert turn.status == "failed" and "Try asking again" in turn.answer
    assert turn.refs == {}


# --- find_videos ---------------------------------------------------------------------------------
def refs_of(result: dict[str, Any]) -> list[str]:
    return [v["ref"] for v in result["videos"]]


@pytest.mark.parametrize("args, expected", [
    ({"visibility": "private"}, {"V3"}),
    ({"visibility": "unlisted"}, set()),
    ({"min_views": 1000}, {"V10", "V11", "V12"}),
    ({"max_views": 100}, {"V1", "V2"}),
    ({"min_views": 400, "max_views": 600}, {"V4", "V5", "V6"}),
    ({"published_from": "2026-01-11"}, {"V11", "V12", "V13"}),
    ({"published_to": "2026-01-02"}, {"V1", "V2"}),
    ({"published_from": "2026-01-11", "published_to": "2026-01-12"}, {"V11", "V12"}),
    ({"has_chapters": True}, {"V1"}),
    ({"has_chapters": False, "max_views": 600}, {"V2", "V3", "V5", "V6", "V13"}),  # not Short V4
    ({"is_short": True}, {"V4"}),
    ({"outliers_only": True}, {"V12"}),
    ({"below_norm_only": True}, {"V2"}),
    ({"outliers_only": True, "below_norm_only": True}, set()),
    ({"failed_check": "description_chapters", "visibility": "public", "max_views": 600},
     {"V2", "V5", "V6", "V13"}),
])
def test_find_videos_filters(box: copilot._Toolbox, args: dict[str, Any], expected: set[str]) -> None:
    result = call(box, "find_videos", {"limit": 50, **args})
    assert set(refs_of(result)) == expected and result["count"] == len(expected)


def test_find_videos_filters_on_the_seo_score(box: copilot._Toolbox) -> None:
    scores = {r.ref: r.card.score for r in box._catalog.rows}
    mid = sorted(scores.values())[6]
    low = call(box, "find_videos", {"max_score": mid, "limit": 50})
    high = call(box, "find_videos", {"min_score": mid, "limit": 50})
    assert set(refs_of(low)) == {r for r, s in scores.items() if s <= mid}
    assert set(refs_of(high)) == {r for r, s in scores.items() if s >= mid}


def test_find_videos_count_is_all_matches_and_rows_are_limited(box: copilot._Toolbox) -> None:
    result = call(box, "find_videos", {"limit": 3})
    assert result["count"] == 13 and result["shown"] == 3 and refs_of(result) == ["V13", "V12", "V11"]
    default = call(box, "find_videos", {})
    assert default["count"] == 13 and default["shown"] == 13 and refs_of(default)[0] == "V13"


def test_find_videos_rows_carry_the_table_columns(box: copilot._Toolbox) -> None:
    row = call(box, "find_videos", {"outliers_only": True})["videos"][0]
    scores = {s.video_id: s for s in outliers.score_videos(box._store.list_videos(), AS_OF)}
    assert row == {
        "ref": "V12", "title": "Cactus tip number 12", "published": "2026-01-12",
        "visibility": "public", "views": 500_000,
        "vs_norm": outliers.ratio_label(scores[vid(12)].ratio),
        "seo_score": box._catalog.by_ref["V12"].card.score, "chapters": "no", "short": "no"}


def test_find_videos_sorts(box: copilot._Toolbox) -> None:
    def order(sort: str) -> list[str]:
        return refs_of(call(box, "find_videos", {"sort": sort, "limit": 50}))

    assert order("newest")[:2] == ["V13", "V12"] and order("oldest")[:2] == ["V1", "V2"]
    assert order("views")[:3] == ["V12", "V11", "V10"] and order("views")[-1] == "V2"
    ratio = order("ratio")
    assert ratio[0] == "V12" and ratio[-1] in {"V3", "V4", "V13"}  # unscored videos come last
    assert ratio.index("V2") == len([r for r in ratio if r not in {"V3", "V4", "V13"}]) - 1
    scores = [box._catalog.by_ref[r].card.score for r in order("score")]
    assert scores == sorted(scores)


@pytest.mark.parametrize("args", [
    {"visibility": "secret"}, {"min_views": "many"}, {"min_views": -1}, {"min_views": True},
    {"limit": 0}, {"limit": 51}, {"sort": "random"}, {"published_from": "last week"},
    {"published_to": 2026}, {"has_chapters": "yes"}, {"outliers_only": 1},
    {"failed_check": "no_such_check"}, {"min_score": 101}, {"colour": "red"},
])
def test_find_videos_bad_arguments_return_an_error(box: copilot._Toolbox, args: dict[str, Any]) -> None:
    result = call(box, "find_videos", args)
    assert list(result) == ["error"] and result["error"]


def test_bad_calls_return_errors_and_never_raise(box: copilot._Toolbox) -> None:
    assert "not valid JSON" in call(box, "find_videos", "{oops")["error"]
    assert "JSON object" in call(box, "find_videos", "[1]")["error"]
    unknown = call(box, "delete_video", {})["error"]
    assert "Unknown tool" in unknown and "propose_edit" in unknown
    assert "Unknown arguments" in call(box, "channel_stats", {"x": 1})["error"]
    assert "No video V99" in call(box, "get_video", {"ref": "V99"})["error"]
    assert "required" in call(box, "get_video", {})["error"]
    assert "No video 12" in call(box, "get_transcript", {"ref": "12"})["error"]
    assert call(box, "get_video", {"ref": " v12 "})["ref"] == "V12"  # forgiving about case


# --- search_videos -------------------------------------------------------------------------------
def test_search_finds_description_and_transcript_matches(box: copilot._Toolbox) -> None:
    result = call(box, "search_videos", {"query": "Photoshop"})
    assert result["count"] == 2
    by_ref = {m["ref"]: m for m in result["matches"]}
    assert set(by_ref) == {"V6", "V8"}
    assert by_ref["V6"]["where"] == ["description"] and "Photoshop" in by_ref["V6"]["snippet"]
    assert by_ref["V8"]["where"] == ["transcript"]
    assert by_ref["V8"]["snippet"].startswith("1:05 Today we open Photoshop")
    assert by_ref["V6"]["title"] == "Cactus tip number 6"


def test_search_needs_all_words_and_ranks_by_places_matched(box: copilot._Toolbox) -> None:
    both = call(box, "search_videos", {"query": "cactus photoshop"})
    assert [m["ref"] for m in both["matches"]] == ["V8", "V6"]  # the transcript is a fourth place
    assert both["matches"][0]["where"] == ["title", "tags", "description", "transcript"]
    assert both["matches"][1]["where"] == ["title", "tags", "description"]
    assert call(box, "search_videos", {"query": "photoshop zebra"})["count"] == 0
    assert call(box, "search_videos", {"query": "garden"})["matches"][0]["where"] == ["tags"]


def test_search_is_limited_to_twenty_but_counts_everything(store: Store) -> None:
    fill(store, [make_video(n, video_id=f"vidx{n:07d}", title=f"Moss tip {n}",
                            published_at=f"2026-02-{n % 28 + 1:02d}T10:00:00Z") for n in range(30)])
    catalog = copilot._Catalog(store.list_videos("video_id"), AS_OF)
    box = copilot._Toolbox(store, catalog, store.begin_copilot_turn(1, "q", "m", NOW), NOW)
    result = call(box, "search_videos", {"query": "moss"})
    assert result["count"] == 30 and result["shown"] == 20 and len(result["matches"]) == 20


def test_search_bad_queries_return_errors(box: copilot._Toolbox) -> None:
    assert "required" in call(box, "search_videos", {})["error"]
    assert "3 or more letters" in call(box, "search_videos", {"query": "a 12"})["error"]
    assert "Unknown arguments" in call(box, "search_videos", {"query": "cactus", "x": 1})["error"]


# --- get_video and get_transcript ----------------------------------------------------------------
def test_get_video_returns_text_scorecard_and_outlier_fields(box: copilot._Toolbox) -> None:
    data = call(box, "get_video", {"ref": "V12"})
    assert data["title"] == "Cactus tip number 12" and data["tags"] == ["cactus", "garden"]
    assert data["description"] == "A short guide to cactus care, part 12."
    assert data["published"] == "2026-01-12" and data["duration"] == "5:00"
    assert data["views"] == 500_000 and data["likes"] == 0 and data["comments"] == 0
    assert data["seo_score"] == box._catalog.by_ref["V12"].card.score
    assert data["url"] == f"https://www.youtube.com/watch?v={vid(12)}"
    ids = [c["id"] for c in data["failed_checks"]]
    assert "description_chapters" in ids
    chapters = next(c for c in data["failed_checks"] if c["id"] == "description_chapters")
    assert chapters["detail"] and chapters["fix"] and chapters["title"]
    assert data["vs_norm"].endswith("×") and data["outlier_status"] == "scored"
    assert data["peer_median_views"] == 700
    assert data["has_transcript"] is False and data["has_draft"] is False


def test_get_video_knows_transcripts_drafts_and_private_videos(store: Store, box: copilot._Toolbox) -> None:
    assert call(box, "get_video", {"ref": "V8"})["has_transcript"] is True
    edits.save_draft(store, vid(5), "A new title for five", "A short guide to cactus care, part 5.",
                     ["cactus"], now=NOW)
    assert call(box, "get_video", {"ref": "V5"})["has_draft"] is True
    private = call(box, "get_video", {"ref": "V3"})
    assert private["url"] is None and private["vs_norm"] == "not public"
    assert call(box, "get_video", {"ref": "V13"})["vs_norm"] == "too new"
    assert call(box, "get_video", {"ref": "V1"})["failed_checks"] != []  # not empty, but no chapters one
    assert "description_chapters" not in [c["id"] for c in call(box, "get_video", {"ref": "V1"})["failed_checks"]]


def test_get_transcript_returns_timed_lines(box: copilot._Toolbox) -> None:
    data = call(box, "get_transcript", {"ref": "V8"})
    assert data["available"] is True
    assert data["segments"] == ["0:00 Hello everyone", "1:05 Today we open Photoshop for the label"]
    assert call(box, "get_transcript", {"ref": "V1"})["available"] is False


# --- channel_stats -------------------------------------------------------------------------------
def test_channel_stats_are_counted_by_code(box: copilot._Toolbox) -> None:
    stats = call(box, "channel_stats")
    cards = [r.card.score for r in box._catalog.rows if r.video.privacy_status == "public"]
    assert stats["videos_all"] == 13 and stats["videos_public"] == 12
    assert stats["median_views_public"] == 650  # the 12 public videos; the middle pair is 600 and 700
    assert stats["public_videos_of_4_minutes_or_more"] == 11  # all but the Short
    assert stats["of_those_with_chapters"] == 1 and stats["share_with_chapters"] == "9%"
    assert stats["average_seo_score_public"] == round(sum(cards) / len(cards), 1)
    assert stats["outliers"] == 1 and stats["below_norm"] == 1
    failed = stats["most_common_failed_checks"]
    assert failed and all({"id", "title", "videos"} == set(f) for f in failed)
    assert [f["videos"] for f in failed] == sorted((f["videos"] for f in failed), reverse=True)
    assert len(failed) == 8 and failed[0] == {
        "id": "audio_language", "title": "Spoken (audio) language is set", "videos": 12}
    assert call(box, "find_videos", {"failed_check": "description_chapters", "visibility": "public",
                                     "limit": 1})["count"] == 10  # 11 long public videos, one has chapters
    assert stats["top_tags"] == [{"tag": "cactus", "videos": 12}, {"tag": "garden", "videos": 12}]


# --- propose_edit --------------------------------------------------------------------------------
def edit(ref: str = "V1", **fields: Any) -> dict[str, Any]:
    base = {"ref": ref, "title": "Cactus tip number one, repotting", "description": CHAPTERS,
            "tags": ["cactus", "garden"], "reason": "A fuller title."}
    return {**base, **fields}


def test_propose_edit_accepted_is_stored_as_an_open_proposal(store: Store, box: copilot._Toolbox) -> None:
    result = call(box, "propose_edit", edit(title="  Cactus tip number one, repotting  ",
                                            tags=[" cactus ", "garden", "Cactus", ""]))
    assert result["accepted"] is True
    (proposal,) = store.list_copilot_proposals([box._turn_id])
    assert (proposal.video_id, proposal.status, proposal.reason) == (vid(1), "open", "A fuller title.")
    assert proposal.title == "Cactus tip number one, repotting"
    assert proposal.tags == ("cactus", "garden") and proposal.description == CHAPTERS
    assert box.proposed == {"V1": proposal.id}
    assert store.get_draft(vid(1)) is None  # nothing became a draft


def test_propose_edit_lists_problems_so_the_model_can_fix_them(store: Store, box: copilot._Toolbox) -> None:
    bad = call(box, "propose_edit", edit(title="Tip <b>one</b>",
                                          description=CHAPTERS + "Buy: https://evil.invalid/x\n"))
    assert bad["accepted"] is False
    text = " ".join(bad["problems"])
    assert "title" in text and "evil.invalid" in text
    assert store.list_copilot_proposals([box._turn_id]) == [] and box.proposed == {}
    assert call(box, "propose_edit", edit())["accepted"] is True  # the fixed call works


def test_propose_edit_allows_links_that_tools_returned(store: Store, box: copilot._Toolbox) -> None:
    with_link = edit(description=CHAPTERS + f"Pots: {SHOP}\n")
    assert call(box, "propose_edit", with_link)["accepted"] is False
    call(box, "get_video", {"ref": "V6"})  # shows the shop link
    assert call(box, "propose_edit", with_link)["accepted"] is True


def test_propose_edit_with_the_current_text_is_not_a_proposal(store: Store, box: copilot._Toolbox) -> None:
    same = edit(title="Cactus tip number 1", description=CHAPTERS, tags=["cactus", "garden"])
    result = call(box, "propose_edit", same)
    assert result == {"accepted": False, "problems": ["This is the same as the current text."]}
    assert store.list_copilot_proposals([box._turn_id]) == []


def test_propose_edit_refuses_a_video_with_a_draft(store: Store, box: copilot._Toolbox) -> None:
    edits.save_draft(store, vid(1), "Another new title", CHAPTERS, ["cactus"], now=NOW)
    result = call(box, "propose_edit", edit())
    assert "already has a draft" in result["error"]
    assert store.list_copilot_proposals([box._turn_id]) == []


def test_propose_edit_allows_three_per_answer_and_one_per_video(store: Store, box: copilot._Toolbox) -> None:
    for n in (5, 7, 8):
        result = call(box, "propose_edit", edit(f"V{n}", title=f"Cactus tip {n} explained",
                                                description=f"A short guide to cactus care, part {n}."))
        assert result["accepted"] is True
    assert "already has a proposal" in call(box, "propose_edit", edit("V5"))["error"]
    fourth = call(box, "propose_edit", edit("V9", title="Cactus tip nine explained",
                                            description="A short guide to cactus care, part 9."))
    assert "At most 3 proposals" in fourth["error"]
    assert len(store.list_copilot_proposals([box._turn_id])) == 3


@pytest.mark.parametrize("fields", [
    {"ref": "V99"}, {"title": 5}, {"description": None}, {"tags": "cactus"}, {"tags": [1]},
    {"reason": ""}, {"reason": "x" * 301}, {"extra": 1}])
def test_propose_edit_bad_arguments_return_an_error(store: Store, box: copilot._Toolbox,
                                                    fields: dict[str, Any]) -> None:
    result = call(box, "propose_edit", {**edit(), **fields})
    assert list(result) == ["error"]
    assert store.list_copilot_proposals([box._turn_id]) == []
    assert "reason" in call(box, "propose_edit", {k: v for k, v in edit().items() if k != "reason"})["error"]


# --- make_draft and dismiss_proposal -------------------------------------------------------------
def proposal(store: Store, ref_n: int = 1, **fields: Any) -> int:
    turn_id = store.begin_copilot_turn(1, "q", "m", NOW)
    values = {"title": "Cactus tip number one, repotting", "description": CHAPTERS,
              "tags": ["cactus", "garden"], "reason": "A fuller title.", **fields}
    return store.add_copilot_proposal(turn_id, vid(ref_n), values["title"], values["description"],
                                      values["tags"], values["reason"], NOW)


def test_make_draft_saves_a_copilot_draft(store: Store) -> None:
    pid = proposal(store)
    assert make_draft(store, pid, NOW) == vid(1)
    row = store.get_draft(vid(1))
    assert row is not None and row.source == "copilot" and row.approved_at is None
    assert row.title == "Cactus tip number one, repotting" and row.note == "Copilot: A fuller title."
    assert (row.base_title, row.base_tags) == ("Cactus tip number 1", ("cactus", "garden"))
    stored = store.get_copilot_proposal(pid)
    assert stored is not None and stored.status == "drafted"


def test_make_draft_refuses_a_video_that_has_a_draft(store: Store) -> None:
    edits.save_draft(store, vid(1), "Another new title", CHAPTERS, ["cactus"], now=NOW)
    pid = proposal(store)
    with pytest.raises(CopilotError, match="already has a draft"):
        make_draft(store, pid, NOW)
    stored = store.get_copilot_proposal(pid)
    assert stored is not None and stored.status == "open"


def test_make_draft_refuses_a_proposal_that_is_not_open(store: Store) -> None:
    drafted, dismissed = proposal(store, 1), proposal(store, 5, title="Cactus tip five, again")
    make_draft(store, drafted, NOW)
    dismiss_proposal(store, dismissed)
    for pid in (drafted, dismissed):
        with pytest.raises(CopilotError, match="already used or dismissed"):
            make_draft(store, pid, NOW)
    assert store.get_draft(vid(5)) is None


def test_make_draft_of_the_current_text_dismisses_the_proposal(store: Store) -> None:
    pid = proposal(store, 5, title="Cactus tip number 5", description="A short guide to cactus care, part 5.")
    with pytest.raises(CopilotError, match="^That text is already on the video, so no draft was needed.$"):
        make_draft(store, pid, NOW)
    stored = store.get_copilot_proposal(pid)
    assert stored is not None and stored.status == "dismissed"
    assert store.get_draft(vid(5)) is None


def test_make_draft_refuses_unknown_proposals_and_invalid_text(store: Store) -> None:
    with pytest.raises(CopilotError, match="does not exist"):
        make_draft(store, 4242, NOW)
    pid = proposal(store, 5, title="")
    with pytest.raises(CopilotError, match="can't be drafted"):
        make_draft(store, pid, NOW)
    stored = store.get_copilot_proposal(pid)
    assert stored is not None and stored.status == "open" and store.get_draft(vid(5)) is None


def test_dismiss_proposal(store: Store) -> None:
    pid = proposal(store)
    dismiss_proposal(store, pid)
    assert store.get_copilot_proposal(pid).status == "dismissed"  # type: ignore[union-attr]
    kept = proposal(store, 5, title="Cactus tip five, again")
    make_draft(store, kept, NOW)
    dismiss_proposal(store, kept)  # a drafted proposal stays drafted
    assert store.get_copilot_proposal(kept).status == "drafted"  # type: ignore[union-attr]
    with pytest.raises(CopilotError, match="does not exist"):
        dismiss_proposal(store, 4242)


def test_a_proposal_from_ask_can_be_drafted_end_to_end(store: Store) -> None:
    turn, _ = run(store, [
        tool_reply(("propose_edit", edit())),
        answer("I offered a fuller title for V1."),
    ])
    (found,) = store.list_copilot_proposals([turn.id])
    assert make_draft(store, found.id, NOW) == vid(1)
    row = store.get_draft(vid(1))
    assert row is not None and row.source == "copilot" and row.title == found.title
