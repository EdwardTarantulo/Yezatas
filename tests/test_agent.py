import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fakes import FakeModelClient
from yezatas import agent
from yezatas.agent import (
    AgentError, ModelError, ModelReply, RESPONSE_FORMAT, SUGGESTION_SCHEMA, SYSTEM_PROMPT, TOOLS,
    ToolCall, suggest_for_video,
)
from yezatas.edits import save_draft
from yezatas.store import Store, Video

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
V1, V2, V3, V4 = "vidA00001", "vidB00002", "vidC00003", "vidD00004"
RELATED_URL = f"https://www.youtube.com/watch?v={V2}"
SHOP = "https://shop.invalid/cactus"
DESCRIPTION = f"How to repot a cactus.\nShop: {SHOP}\n"


def video(video_id: str, title: str, **fields: Any) -> Video:
    base = Video(video_id=video_id, channel_id="UC_fake", title=title, description="Some text.",
                 tags=["cactus"], privacy_status="public", published_at="2026-01-01T10:00:00Z",
                 duration="PT5M", view_count=100)
    return replace(base, **fields)


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as s:
        pull = s.begin_pull("UC_fake", NOW)
        for v in (
            video(V1, "Cactus repotting guide", description=DESCRIPTION),
            video(V2, "Best cactus soil mix", tags=["soil", "cactus"], view_count=500),
            video(V3, "Cactus secret video", privacy_status="private", view_count=900),
            video(V4, "Orchid care basics", tags=["orchid"], view_count=50),
        ):
            s.upsert_video(v, pull, NOW)
        yield s


def tool_reply(*calls: tuple[str, str], tokens: tuple[int, int, int] = (0, 0, 0)) -> ModelReply:
    tool_calls = tuple(ToolCall(f"call_{i}_{name}", name, args) for i, (name, args) in enumerate(calls))
    message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
        for c in tool_calls]}
    return ModelReply(None, tool_calls, message, *tokens)


def answer(payload: dict[str, Any] | str, tokens: tuple[int, int, int] = (0, 0, 0)) -> ModelReply:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return ModelReply(content, (), {"role": "assistant", "content": content}, *tokens)


def suggestion(**fields: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "title": "How to repot a cactus: a simple guide",
        "description": f"How to repot a cactus, step by step.\nShop: {SHOP}\n",
        "tags": ["cactus", "repotting", "succulents"],
        "summary": "Longer title and a fuller opening line.",
        "changes": [{"field": "title", "check_id": "title_length", "reason": "Closer to 30-60."}],
        "not_fixed": [{"check_id": "description_chapters", "reason": "Needs a transcript."}],
    }
    data.update(fields)
    return data


def basic_script(final: ModelReply, *extra: Any) -> list[Any]:
    """Tool round, a plain reply, then the final answer (and any repair replies)."""
    return [tool_reply(("get_scorecard", "{}"), ("get_current_metadata", "")), answer("Ready."),
            final, *extra]


def test_happy_path_saves_an_agent_draft(store: Store) -> None:
    client = FakeModelClient([
        tool_reply(("get_scorecard", "{}"), ("get_current_metadata", ""), tokens=(1000, 200, 50)),
        tool_reply(("find_related_videos", '{"query": "cactus soil"}'), tokens=(1500, 800, 30)),
        answer("Ready to answer.", tokens=(1600, 900, 10)),
        answer(suggestion(), tokens=(1700, 1000, 300)),
    ])
    lines: list[str] = []
    result = suggest_for_video(store, client, V1, now=NOW, report=lines.append)

    assert result.status == "drafted" and result.video_id == V1
    assert result.suggestion is not None and result.suggestion.title.startswith("How to repot")
    row = store.get_draft(V1)
    assert row is not None and row.source == "agent" and row.approved_at is None
    assert row.title == "How to repot a cactus: a simple guide"
    assert row.note == "Suggested by muse-spark-1.3: Longer title and a fuller opening line."
    assert (row.base_title, row.base_tags) == ("Cactus repotting guide", ("cactus",))

    run = store.get_agent_run(result.run_id)
    assert run is not None and run.status == "drafted" and run.video_id == V1
    assert run.model == "muse-spark-1.3" and run.finished_at is not None
    assert (run.input_tokens, run.cached_tokens, run.output_tokens) == (5800, 2900, 390)
    assert run.summary == "Longer title and a fuller opening line."
    assert run.result["title"] == row.title and run.result["tags"] == list(row.tags)
    assert run.result["changes"] == [{"field": "title", "check_id": "title_length",
                                      "reason": "Closer to 30-60."}]
    assert run.result["not_fixed"][0]["check_id"] == "description_chapters"
    assert [(e["type"], e["name"]) for e in run.log] == [
        ("tool", "get_scorecard"), ("tool", "get_current_metadata"), ("tool", "find_related_videos")]
    assert run.log[0]["arguments"] == {} and run.log[1]["arguments"] == {}
    assert run.log[2]["arguments"] == {"query": "cactus soil"}
    assert all(isinstance(e["result_chars"], int) and e["result_chars"] > 0 for e in run.log)
    assert lines[0] == "Reading the scorecard" and lines[-1] == "Writing the suggestion"
    assert "Looking for related videos" in lines


def test_conversation_shape(store: Store) -> None:
    client = FakeModelClient(basic_script(answer(suggestion())))
    suggest_for_video(store, client, V1, now=NOW)

    first, second, final = client.calls
    assert first["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = first["messages"][1]["content"]
    assert V1 in user and "Cactus repotting guide" in user
    assert first["tools"] == TOOLS and first["response_format"] is None
    # the assistant message with the tool calls comes first, then one tool message per call
    sent = second["messages"]
    assistant = sent[2]
    assert assistant["role"] == "assistant" and len(assistant["tool_calls"]) == 2
    tool_messages = sent[3:5]
    assert [m["role"] for m in tool_messages] == ["tool", "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == [c["id"] for c in assistant["tool_calls"]]
    scorecard = json.loads(tool_messages[0]["content"])
    assert 0 <= scorecard["score"] <= 100
    passed = [c["passed"] for c in scorecard["checks"]]
    assert passed == sorted(passed)  # failed checks first
    assert json.loads(tool_messages[1]["content"])["title"] == "Cactus repotting guide"
    # the final call has the schema and no tools
    assert final["tools"] is None and final["response_format"] == RESPONSE_FORMAT
    assert final["messages"][-1]["role"] == "user" and "JSON" in final["messages"][-1]["content"]
    assert final["messages"][-2] == {"role": "assistant", "content": "Ready."}


def test_system_prompt_states_the_rules() -> None:
    for phrase in ("get_scorecard", "get_current_metadata", "Never invent facts", "not_fixed",
                   "100 characters", "5,000 bytes", "500 characters"):
        assert phrase in SYSTEM_PROMPT


def test_schema_is_strict_everywhere() -> None:
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert sorted(node["required"]) == sorted(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(SUGGESTION_SCHEMA)
    assert sorted(SUGGESTION_SCHEMA["properties"]) == [
        "changes", "description", "not_fixed", "summary", "tags", "title"]
    assert RESPONSE_FORMAT["json_schema"]["strict"] is True
    assert [t["function"]["name"] for t in TOOLS] == [
        "get_current_metadata", "get_scorecard", "find_related_videos", "get_channel_profile",
        "get_transcript"]


def test_missing_video_and_existing_draft_are_refused_before_a_run(store: Store) -> None:
    client = FakeModelClient([])
    with pytest.raises(AgentError, match="No such video."):
        suggest_for_video(store, client, "nope", now=NOW)
    save_draft(store, V1, "Mine", "Mine", ["x"], now=NOW)
    with pytest.raises(AgentError, match="already has a draft"):
        suggest_for_video(store, client, V1, now=NOW)
    assert client.calls == [] and store.list_agent_runs() == []


def test_no_change(store: Store) -> None:
    v = store.get_video(V1)
    same = suggestion(title=v.title, description=v.description, tags=list(v.tags), changes=[],
                      summary="Already good.")
    result = suggest_for_video(store, FakeModelClient(basic_script(answer(same))), V1, now=NOW)
    assert result.status == "no_change"
    assert store.get_draft(V1) is None
    run = store.get_agent_run(result.run_id)
    assert run.status == "no_change" and run.result["title"] == v.title


def test_normalises_the_answer(store: Store) -> None:
    messy = suggestion(title="  A tidy cactus repotting title  ",
                       description=f"Line one.\r\nShop: {SHOP}\r\n",
                       tags=[" cactus ", "", "  ", "soil"])
    result = suggest_for_video(store, FakeModelClient(basic_script(answer(messy))), V1, now=NOW)
    assert result.status == "drafted"
    row = store.get_draft(V1)
    assert row.title == "A tidy cactus repotting title"
    assert row.description == f"Line one.\nShop: {SHOP}\n"
    assert row.tags == ("cactus", "soil")


def test_invented_link_is_rejected_after_the_repair_round(store: Store) -> None:
    bad = suggestion(description=f"Intro.\nShop: {SHOP}\nBuy at https://invented.invalid/x\n")
    client = FakeModelClient(basic_script(answer(bad, tokens=(10, 0, 5)),
                                          answer(bad, tokens=(20, 0, 5))))
    result = suggest_for_video(store, client, V1, now=NOW)

    assert result.status == "rejected" and result.suggestion is None
    assert "https://invented.invalid/x" in result.message
    assert store.get_draft(V1) is None
    assert len(client.calls) == 4  # tool round, plain reply, final, one repair round
    repair = client.calls[3]["messages"]
    assert repair[-2] == {"role": "assistant", "content": json.dumps(bad)}
    assert repair[-1]["role"] == "user" and "https://invented.invalid/x" in repair[-1]["content"]
    assert client.calls[3]["response_format"] == RESPONSE_FORMAT and client.calls[3]["tools"] is None
    run = store.get_agent_run(result.run_id)
    assert run.status == "rejected" and run.result == {}
    assert (run.input_tokens, run.output_tokens) == (30, 10)
    problems = [e for e in run.log if e["type"] == "problems"]
    assert [e["round"] for e in problems] == [1, 2]
    assert "invented.invalid" in problems[0]["problems"][0]


def test_repair_round_can_succeed(store: Store) -> None:
    bad = suggestion(title="Cactus <b>guide</b> for everyone and more")
    good = suggestion()
    client = FakeModelClient(basic_script(answer(bad), answer(good)))
    result = suggest_for_video(store, client, V1, now=NOW)
    assert result.status == "drafted"
    assert store.get_draft(V1).title == good["title"]
    run = store.get_agent_run(result.run_id)
    assert [e["round"] for e in run.log if e["type"] == "problems"] == [1]
    assert "< or >" in run.log[-1]["problems"][0]


def test_invented_timestamp_handle_and_email_are_caught(store: Store) -> None:
    bad = suggestion(description=f"Intro 0:00\nPots 2:30\nAsk @newperson or me@mail.invalid\n"
                                 f"Shop: {SHOP}\n")
    client = FakeModelClient(basic_script(answer(bad), answer(bad)))
    result = suggest_for_video(store, client, V1, now=NOW)
    assert result.status == "rejected"
    for value in ("0:00", "2:30", "@newperson", "me@mail.invalid"):
        assert value in result.message


def test_related_video_url_is_allowed_once_a_tool_returned_it(store: Store) -> None:
    linked = suggestion(description=f"Intro.\nShop: {SHOP}\nWatch next: {RELATED_URL}\n")
    client = FakeModelClient([
        tool_reply(("find_related_videos", '{"query": "cactus soil"}')),
        answer("Done."), answer(linked)])
    result = suggest_for_video(store, client, V1, now=NOW)
    assert result.status == "drafted"
    assert RELATED_URL in store.get_draft(V1).description

    # without asking the tool, the same link is an invention
    store.delete_draft(V1)
    result = suggest_for_video(store, FakeModelClient(basic_script(answer(linked), answer(linked))),
                               V1, now=NOW)
    assert result.status == "rejected"


def test_common_closing_line_urls_are_allowed(store: Store) -> None:
    pull = store.begin_pull("UC_fake", NOW)
    closing = "Join the club: https://club.invalid/join"
    for vid in (V2, V4):
        store.upsert_video(replace(store.get_video(vid), description=f"About.\n{closing}\n"),
                           pull, NOW)
    text = suggestion(description=f"Intro.\nShop: {SHOP}\n{closing}\n")
    client = FakeModelClient([tool_reply(("get_channel_profile", "{}")), answer("Done."),
                              answer(text)])
    result = suggest_for_video(store, client, V1, now=NOW)
    assert result.status == "drafted"
    profile = json.loads(client.calls[1]["messages"][-1]["content"])
    assert profile["video_count"] == 3  # the private video is not counted
    assert {"line": closing, "count": 2} in profile["common_closing_lines"]


def test_model_error_fails_the_run(store: Store) -> None:
    refused = "The Meta API key was refused. Check it in Configuration."
    client = FakeModelClient([
        tool_reply(("get_scorecard", "{}"), tokens=(100, 0, 5)), ModelError("auth", refused)])
    result = suggest_for_video(store, client, V1, now=NOW)
    assert result.status == "failed" and result.message == refused
    run = store.get_agent_run(result.run_id)
    assert run.status == "failed" and run.finished_at is not None
    assert run.log[-1] == {"type": "error", "kind": "auth", "message": refused}
    assert run.input_tokens == 100 and run.result == {}
    assert store.get_draft(V1) is None


def test_invalid_json_twice_fails_the_run(store: Store) -> None:
    script = basic_script(answer("not json {"), answer("still not json"))
    result = suggest_for_video(store, FakeModelClient(script), V1, now=NOW)
    assert result.status == "failed"
    assert result.message == "The model's answer was not valid JSON."
    assert store.get_agent_run(result.run_id).status == "failed"
    assert store.get_draft(V1) is None


def test_invalid_json_gets_a_repair_round(store: Store) -> None:
    client = FakeModelClient(basic_script(answer("not json {"), answer(suggestion())))
    result = suggest_for_video(store, client, V1, now=NOW)
    assert result.status == "drafted"
    run = store.get_agent_run(result.run_id)
    problems = [e for e in run.log if e["type"] == "problems"]
    assert len(problems) == 1 and "not valid JSON" in problems[0]["problems"][0]
    repair = client.calls[-1]["messages"][-1]
    assert repair["role"] == "user" and "only the JSON object" in repair["content"]


def test_json_wrapped_in_a_sentence_is_used(store: Store) -> None:
    wrapped = "Here is my answer: " + json.dumps(suggestion()) + " Hope it helps."
    result = suggest_for_video(store, FakeModelClient(basic_script(answer(wrapped))), V1, now=NOW)
    assert result.status == "drafted"


def test_wrong_shape_fails_the_run(store: Store) -> None:
    script = basic_script(answer({"title": "x"}), answer({"title": "x"}))
    result = suggest_for_video(store, FakeModelClient(script), V1, now=NOW)
    assert result.status == "failed" and "expected format" in result.message


def test_unexpected_error_closes_the_run_and_is_raised(store: Store) -> None:
    client = FakeModelClient([RuntimeError("boom")])
    with pytest.raises(RuntimeError, match="boom"):
        suggest_for_video(store, client, V1, now=NOW)
    run = store.list_agent_runs(V1)[0]
    assert run.status == "failed" and run.finished_at is not None


def test_tool_rounds_are_capped(store: Store) -> None:
    def more_tools(messages: Any, tools: Any, response_format: Any) -> ModelReply:
        return tool_reply(("get_transcript", "{}"))

    client = FakeModelClient([more_tools, more_tools, answer(suggestion())])
    result = suggest_for_video(store, client, V1, now=NOW, max_tool_rounds=2)
    assert result.status == "drafted"
    assert [c["tools"] is not None for c in client.calls] == [True, True, False]
    # the last tool calls were still answered before the final request
    last = client.calls[2]["messages"]
    assert [m["role"] for m in last[2:]] == ["assistant", "tool", "assistant", "tool", "user"]


def test_unknown_tool_and_bad_arguments_return_error_results(store: Store) -> None:
    client = FakeModelClient([
        tool_reply(("delete_everything", "{}"), ("find_related_videos", "{not json"),
                   ("find_related_videos", "[1]"), ("find_related_videos", "{}"),
                   ("get_current_metadata", "")),
        answer("Done."), answer(suggestion())])
    result = suggest_for_video(store, client, V1, now=NOW)
    assert result.status == "drafted"
    sent = client.calls[1]["messages"]
    results = [json.loads(m["content"]) for m in sent if m["role"] == "tool"]
    assert "Unknown tool" in results[0]["error"] and "get_transcript" in results[0]["error"]
    assert "not valid JSON" in results[1]["error"]
    assert "JSON object" in results[2]["error"]
    assert "query" in results[3]["error"]
    assert results[4]["title"] == "Cactus repotting guide"
    log = store.get_agent_run(result.run_id).log
    assert log[0]["arguments"] == {} and log[0]["name"] == "delete_everything"
    assert log[1]["arguments"] == "{not json" and log[2]["arguments"] == "[1]"


def toolbox(store: Store, video_id: str = V1) -> agent._Toolbox:
    return agent._Toolbox(store, store.get_video(video_id))


def test_find_related_videos_ranking_and_url_list(store: Store) -> None:
    box = toolbox(store)
    _, text = box.run("find_related_videos", '{"query": "Cactus SOIL mix"}')
    found = json.loads(text)["videos"]
    # this video and the private one are excluded; videos sharing no word are not listed
    assert found == [{"video_id": V2, "title": "Best cactus soil mix", "url": RELATED_URL,
                      "views": 500}]
    assert box.allowed_urls == {RELATED_URL}
    orchid = json.loads(box.run("find_related_videos", '{"query": "orchid"}')[1])["videos"]
    assert [v["video_id"] for v in orchid] == [V4]


def test_find_related_videos_limit_is_capped_at_five(store: Store) -> None:
    pull = store.begin_pull("UC_fake", NOW)
    for n in range(8):
        store.upsert_video(video(f"vidX{n:05d}", f"Cactus extra {n}", view_count=n), pull, NOW)
    box = toolbox(store)
    found = json.loads(box.run("find_related_videos", '{"query": "cactus", "limit": 50}')[1])["videos"]
    assert len(found) == 5
    views = [v["views"] for v in found]
    assert views == sorted(views, reverse=True)
    two = json.loads(box.run("find_related_videos", '{"query": "cactus", "limit": 2}')[1])["videos"]
    assert len(two) == 2
    assert "error" in json.loads(box.run("find_related_videos", '{"query": "x", "limit": "y"}')[1])


def test_transcript_tool_says_there_is_none(store: Store) -> None:
    data = json.loads(toolbox(store).run("get_transcript", "")[1])
    assert data["available"] is False and "transcript" in data["reason"]


def add_transcript(store: Store, texts: list[str], video_id: str = V1, language: str = "en") -> None:
    from yezatas.store import TranscriptRecord

    segments = [{"start_ms": i * 65_000, "end_ms": (i + 1) * 65_000, "text": t}
                for i, t in enumerate(texts)]
    store.add_transcript(TranscriptRecord(0, video_id, language, "voice-test", "clip.mp4",
                                          len(texts) * 65_000, segments, [], ""), NOW)


def test_transcript_tool_returns_timed_lines_and_a_note(store: Store) -> None:
    add_transcript(store, ["Welcome to the channel.", "", "Today we repot a cactus."])
    data = json.loads(toolbox(store).run("get_transcript", "")[1])
    assert data["available"] is True and data["language"] == "en"
    assert data["segments"] == ["0:00 Welcome to the channel.", "2:10 Today we repot a cactus."]
    assert "truncated" not in data
    assert "must not" in data["note"] or "Do not add chapters" in data["note"]


def test_transcript_tool_uses_the_latest_transcript_of_this_video(store: Store) -> None:
    add_transcript(store, ["Old words."])
    add_transcript(store, ["New words."])
    add_transcript(store, ["Other video."], video_id=V2)
    data = json.loads(toolbox(store).run("get_transcript", "")[1])
    assert data["segments"] == ["0:00 New words."]
    assert json.loads(toolbox(store, V3).run("get_transcript", "")[1])["available"] is False


def test_transcript_tool_cuts_a_long_transcript_and_says_so(store: Store) -> None:
    add_transcript(store, [f"Segment {i} " + "word " * 60 for i in range(200)])
    data = json.loads(toolbox(store).run("get_transcript", "")[1])
    assert data["truncated"] is True
    assert 0 < len(data["segments"]) < 200
    assert sum(len(line) for line in data["segments"]) <= agent.TRANSCRIPT_CHAR_LIMIT
    assert data["segments"][0].startswith("0:00 Segment 0")


def test_the_prompt_still_forbids_adding_chapters_and_points_to_the_button() -> None:
    assert "Never write timestamps or chapters yourself" in SYSTEM_PROMPT
    assert "Make chapters" in SYSTEM_PROMPT
    assert "there is no transcript" not in SYSTEM_PROMPT


def test_scorecard_tool_lists_failed_checks_by_loss(store: Store) -> None:
    checks = json.loads(toolbox(store).run("get_scorecard", "{}")[1])["checks"]
    assert set(checks[0]) == {"id", "title", "points", "max_points", "passed", "detail", "fix"}
    failed = [c for c in checks if not c["passed"]]
    assert failed and checks[:len(failed)] == failed
    losses = [c["max_points"] - c["points"] for c in failed]
    assert losses == sorted(losses, reverse=True)


def test_profile_tool_counts_tags_and_chapters(store: Store) -> None:
    chaptered = "Intro\n0:00 Start\n1:00 Middle\n2:00 End\n"
    pull = store.begin_pull("UC_fake", NOW)
    store.upsert_video(replace(store.get_video(V2), description=chaptered), pull, NOW)
    data = json.loads(toolbox(store).run("get_channel_profile", "{}")[1])
    assert data["video_count"] == 3
    assert data["top_tags"][0] == {"tag": "cactus", "count": 2}
    assert data["share_with_chapters"] == 0.33
    assert data["common_closing_lines"] == []


# --- MetaModelClient with a stubbed openai client ---------------------------------------------
class StubCompletions:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def meta_client(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> tuple[Any, StubCompletions, dict]:
    import openai
    from yezatas.modelclient import MetaModelClient

    completions = StubCompletions(outcome)
    made: dict[str, Any] = {}

    def fake_openai(**kwargs: Any) -> Any:
        made.update(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    monkeypatch.setattr(openai, "OpenAI", fake_openai)
    return MetaModelClient("sk-test-key", "muse-spark-1.3", timeout=30), completions, made


def status_error(cls: type, status: int) -> Exception:
    response = SimpleNamespace(request=None, status_code=status, headers={})
    return cls("It failed.", response=response, body=None)


def test_meta_client_maps_a_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="get_scorecard", arguments="{}"))
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=15,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=100)))
    client, completions, made = meta_client(monkeypatch, response)
    reply = client.complete([{"role": "user", "content": "hi"}], tools=TOOLS)
    assert client.model == "muse-spark-1.3"
    assert made == {"base_url": agent.MODEL_BASE_URL, "api_key": "sk-test-key", "timeout": 30}
    assert completions.kwargs == {"model": "muse-spark-1.3", "tools": TOOLS,
                                  "messages": [{"role": "user", "content": "hi"}]}
    assert reply.tool_calls == (ToolCall("c1", "get_scorecard", "{}"),)
    assert reply.message == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_scorecard", "arguments": "{}"}}]}
    assert (reply.input_tokens, reply.cached_tokens, reply.output_tokens) == (120, 100, 15)


def test_meta_client_text_reply_without_details(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, prompt_tokens_details=None))
    client, completions, _ = meta_client(monkeypatch, response)
    reply = client.complete([], response_format=RESPONSE_FORMAT)
    assert completions.kwargs["response_format"] == RESPONSE_FORMAT and "tools" not in completions.kwargs
    assert reply.message == {"role": "assistant", "content": "{}"}
    assert (reply.input_tokens, reply.cached_tokens, reply.output_tokens) == (5, 0, 2)


def test_meta_client_maps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    cases = [
        (status_error(openai.AuthenticationError, 401), "auth"),
        (status_error(openai.PermissionDeniedError, 403), "auth"),
        (status_error(openai.RateLimitError, 429), "rate_limit"),
        (openai.APIConnectionError(request=None), "network"),
        (openai.APITimeoutError(request=None), "network"),
        (status_error(openai.BadRequestError, 400), "api"),
    ]
    for error, kind in cases:
        client, _, _ = meta_client(monkeypatch, error)
        with pytest.raises(ModelError) as raised:
            client.complete([])
        assert raised.value.kind == kind
        assert "sk-test-key" not in str(raised.value)
    assert str(raised.value) == "The Meta API returned an error (HTTP 400): It failed."
    client, _, _ = meta_client(monkeypatch, status_error(openai.AuthenticationError, 401))
    with pytest.raises(ModelError, match="The Meta API key was refused. Check it in Configuration."):
        client.complete([])
