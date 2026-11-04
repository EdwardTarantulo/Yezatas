"""The Bulk edit page's agent: starting, the live log, stop, replies, review, Make drafts and discard.

The model is a scripted fake and the agent function runs the real `bulkagent.run`, so the page shows what
the real tools did to a made-up catalog. No network, and nothing here can reach YouTube."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient
from webhelpers import make_ui, make_video, seed_catalog
from yezatas import agent, bulk, bulkagent, edits
from yezatas.agent import ModelError, ModelReply, ToolCall
from yezatas.store import Store
from yezatas.web import app as web_app
from yezatas.web.app import NO_PACKAGING_KEY
from yezatas.web.views import BULK_EXAMPLES

A, B, C, D, E = (f"vid{n:08d}" for n in range(1, 6))
OLD = "https://discord.gg/old"
NEW = "https://discord.gg/new"
REQUEST = f"Swap {OLD} for {NEW} on videos from 2024"
SWAP = {"steps": [{"kind": "replace", "find": OLD, "replace_with": NEW, "fields": ["description"]}],
        "all_matching": {"text": OLD, "published_from": "2024-01-01", "published_to": "2024-12-31"}}
EVIL = "<script>alert(1)</script>"


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


def tool_reply(*calls: tuple[str, Any]) -> ModelReply:
    tool_calls = tuple(
        ToolCall(f"call_{i}_{name}", name, args if isinstance(args, str) else json.dumps(args))
        for i, (name, args) in enumerate(calls))
    message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
        for c in tool_calls]}
    return ModelReply(None, tool_calls, message, 3000, 200, 150)


def answer(text: str) -> ModelReply:
    return ModelReply(text, (), {"role": "assistant", "content": text}, 1000, 0, 50)


def standard_script() -> list[Any]:
    return [tool_reply(("find_text", {"text": OLD})), tool_reply(("apply_steps", SWAP)),
            answer("I swapped the old invite in 3 videos. One of them already has a draft.")]


def real_agent(client: FakeModelClient) -> Any:
    calls: list[str] = []

    def run(store: Store, session: Any, text: str, as_of: Any, report: Any) -> None:
        calls.append(text)
        bulkagent.run(store, client, session, text, as_of=as_of, report=report)

    run.calls = calls  # type: ignore[attr-defined]
    return run


def catalog(ui: Any) -> None:
    seed_catalog(ui.config, [
        make_video(1, video_id=A, privacy_status="public", title="Cactus one",
                   description=f"Intro line\nJoin: {OLD}\nBye", view_count=900,
                   published_at="2024-02-01T10:00:00Z"),
        make_video(2, video_id=B, privacy_status="public", title="Cactus two",
                   description=f"Join: {OLD} and again {OLD}", view_count=500,
                   published_at="2024-03-01T10:00:00Z"),
        make_video(3, video_id=C, privacy_status="public", title="Cactus three",
                   description="Nothing to see here", view_count=300,
                   published_at="2024-04-01T10:00:00Z"),
        make_video(4, video_id=D, privacy_status="private", title="Cactus four",
                   description=f"Private: {OLD}", view_count=200, published_at="2024-05-01T10:00:00Z"),
        make_video(5, video_id=E, privacy_status="public", title="Cactus five",
                   description=f"Draft already: {OLD}", view_count=100,
                   published_at="2024-06-01T10:00:00Z"),
    ])
    with Store(ui.config.catalog_db) as store:
        edits.save_draft(store, E, "Cactus five (edited)", f"Draft already: {OLD}", ["cactus"])


def make_agent_ui(tmp_path: Path, script: list[Any] | None = None, key: bool = True) -> Any:
    client = FakeModelClient(script if script is not None else standard_script())
    run = real_agent(client)
    ui = make_ui(tmp_path, bulk_agent=run, packaging_key=lambda cfg: key)
    catalog(ui)
    ui.model = client
    ui.agent_calls = run.calls
    return ui


def start(ui: Any, text: str = REQUEST) -> Any:
    resp = ui.post("/bulk/agent", data={"request": text}, follow_redirects=False)
    ui.web.bulk_agent_job.wait()
    return resp


def page(ui: Any) -> str:
    resp = ui.client.get("/bulk")
    assert resp.status_code == 200
    return resp.text


def drafts(ui: Any) -> dict[str, Any]:
    with Store(ui.config.catalog_db) as store:
        return {d.video_id: d for d in store.list_drafts()}


def flash_text(ui: Any, path: str = "/bulk") -> str:
    return page(ui) if path == "/bulk" else ui.client.get(path).text


def session_of(ui: Any) -> bulkagent.BulkSession:
    session = ui.web.bulk_session()
    assert session is not None
    return session


# --- the start card ---------------------------------------------------------------------------------
def test_empty_state_has_the_start_card_the_chips_and_the_honest_hints(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    html = page(ui)
    assert "Tell it what to change" in html and 'action="/bulk/agent"' in html
    assert 'id="ag-request"' in html and 'maxlength="1000"' in html
    assert "Start with Muse Spark" in html
    for example in BULK_EXAMPLES:
        assert f'data-fill-text="{example}"' in html
    assert len(BULK_EXAMPLES) == 3 and 'data-fill-target="ag-request"' in html
    assert "titles, descriptions, tags and transcripts" in html and "may write new text" in html
    assert "fact guard refuses new numbers and links that are not in your words or your videos" in html
    assert "Only drafts are made" in html
    assert "Meta's Model API (muse-spark-1.3)" in html and 'href="/configuration#ai-data"' in html
    assert "Do it yourself" in html and 'id="bulk-form"' in html and 'name="footer"' in html
    assert "agent-no-key" not in html and 'id="agent-request"' not in html


def test_the_intro_stays_true(tmp_path: Path) -> None:
    html = page(make_agent_ui(tmp_path))
    assert "Say what you want in plain words and Muse Spark" in html
    assert "which use no model" in html and "Only drafts are made" in html
    assert "Nothing reaches YouTube until you approve and push on the Changes page" in html
    assert "History can undo that whole push in one step." in html
    assert "No model is used here, and only drafts are made." in html


def test_local_mode_names_glimmer_and_says_nothing_leaves_the_computer(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YEZATAS_PRIMARY_AI", "local")
    html = page(make_agent_ui(tmp_path))
    assert "Start with Muse Glimmer" in html and "on this computer. Nothing leaves it." in html
    assert "Meta's Model API" not in html


def test_the_page_has_no_inline_styles_or_scripts_and_never_marks_text_safe(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    start(ui)
    html = page(ui)
    assert "style=" not in html and "onclick" not in html and "javascript:" not in html
    assert re.findall(r"<script[^>]*>", html) and all('src="' in t for t in re.findall(r"<script[^>]*>", html))
    source = (Path(web_app.__file__).parent / "templates" / "bulk.html").read_text(encoding="utf-8")
    assert "|safe" not in source and "autoescape" not in source


def test_without_a_key_the_form_is_disabled_and_nothing_is_sent(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path, key=False)
    html = page(ui)
    assert 'id="agent-no-key"' in html and "There is no Meta API key. Add one in Configuration." in html
    assert re.search(r'<button type="submit" class="button primary" id="agent-start" disabled>', html)
    assert re.search(r'<textarea id="ag-request"[^>]* disabled>', html)
    start(ui)
    assert NO_PACKAGING_KEY in page(ui)
    assert ui.agent_calls == [] and ui.web.bulk_session() is None


@pytest.mark.parametrize("text, message", [
    ("", "Describe the change in a few words first. Nothing was sent."),
    ("   ", "Describe the change in a few words first."),
    ("x" * 1001, "at most 1,000 are allowed"),
])
def test_bad_words_are_refused_before_the_model(tmp_path: Path, text: str, message: str) -> None:
    ui = make_agent_ui(tmp_path)
    start(ui, text)
    assert message in page(ui) and ui.agent_calls == [] and ui.web.bulk_session() is None


def test_words_of_the_maximum_length_are_accepted(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path, [answer("Which videos?")])
    start(ui, "x" * 1000)
    assert ui.agent_calls and session_of(ui).closing == "Which videos?"


# --- csrf ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", ["/bulk/agent", "/bulk/agent/continue", "/bulk/agent/stop",
                                  "/bulk/agent/apply", "/bulk/agent/discard"])
def test_every_post_needs_the_csrf_token(tmp_path: Path, path: str) -> None:
    ui = make_agent_ui(tmp_path)
    start(ui)
    before = (len(drafts(ui)), len(session_of(ui).staged_list()))
    for data in ({}, {"csrf": "wrong"}, {"request": "x", "reply": "x", "video_id": A}):
        assert ui.client.post(path, data=data, follow_redirects=False).status_code == 403
    assert (len(drafts(ui)), len(session_of(ui).staged_list())) == before
    assert ui.client.get(path).status_code == 405  # nothing changes on a GET


# --- a finished turn ---------------------------------------------------------------------------------
def test_a_turn_shows_the_log_the_answer_the_usage_and_the_review(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    resp = start(ui)
    assert resp.status_code == 303 and resp.headers["location"].startswith("/bulk")
    html = page(ui)
    assert "What it did" in html and 'id="ag-log"' in html
    for line in ("Read the catalog: 5 videos.", f'Searched for &#34;{OLD}&#34;: 3 videos.',
                 "Replaced text in 3 videos."):
        assert f'<li class="ag-line">{line}</li>' in html
    assert "I swapped the old invite in 3 videos." in html and 'id="agent-closing"' in html
    assert "muse-spark-1.3" in html and "tokens in" in html and "about $" in html
    assert "2 tool calls" in html and "Show what it looked at" in html
    assert "Reply to change the plan" in html and 'id="agent-continue"' in html
    assert "2 videos ready, 1 blocked" in html and 'id="agent-summary"' in html
    assert "3 videos staged" in html
    for video_id in (A, B):
        row = re.search(rf'<li class="ag-row" id="agv-{video_id}">.*?</li>', html, re.S)
        assert row and f'value="{video_id}" aria-label' in row.group(0) and " checked" in row.group(0)
        assert "Show the changes" in row.group(0) and "diff-block" in row.group(0)
        assert '<span class="pill">Description</span>' in row.group(0)
        assert "Replaced" in row.group(0)  # the reason, written by code
    blocked = re.search(rf'<li class="ag-row blocked" id="agv-{E}">.*?</li>', html, re.S)
    assert blocked and "disabled" in blocked.group(0) and "checked" not in blocked.group(0)
    assert "already has a draft" in blocked.group(0)
    assert f'id="agv-{C}"' not in html and f'id="agv-{D}"' not in html
    assert re.search(r'id="agent-make-drafts"', html) and 'id="agent-discard"' in html
    assert 'formaction="/bulk/agent/discard"' in html
    assert html.index('id="agent-summary"') < html.index('id="do-it-yourself"')
    assert drafts(ui).keys() == {E}  # nothing but the draft the creator already had


def test_review_rows_come_ready_first_and_the_diff_is_the_changes_page_diff(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    start(ui)
    html = page(ui)
    assert html.index(f'id="agv-{A}"') < html.index(f'id="agv-{E}"')
    row = re.search(rf'id="agv-{A}">.*?</li>', html, re.S).group(0)  # type: ignore[union-attr]
    assert "difflines" in row and NEW in row and OLD in row


def test_model_text_is_escaped_everywhere(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path, [
        tool_reply(("get_video", {"ref": "V3"}),
                   ("edit_videos", {"edits": [{"ref": "V3", "title": "Cactus three, tidied",
                                               "reason": EVIL}]})),
        answer(f"All done {EVIL} and **<img src=x onerror=alert(1)>** for V3."),
    ])
    with Store(ui.config.catalog_db) as store:
        store.upsert_video(make_video(3, video_id=C, title=f"Cactus three {EVIL}", description="Plain.",
                                      published_at="2024-04-01T10:00:00Z", privacy_status="public"),
                           store.latest_pull()["id"])
    start(ui, f"Fix V3 {EVIL}")
    html = page(ui)
    assert EVIL not in html and "<img src=x" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    # in the log (a title from the catalog), the reason, the closing message, the words and the diff
    assert 'Read &#34;Cactus three &lt;script&gt;alert(1)&lt;/script&gt;&#34;.' in html
    assert "<strong>&lt;img src=x onerror=alert(1)&gt;</strong>" in html
    assert html.count("&lt;script&gt;alert(1)&lt;/script&gt;") >= 5


def test_a_model_error_midway_shows_the_message_and_keeps_the_stage(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path, [tool_reply(("apply_steps", SWAP)),
                                  ModelError("rate_limit", "Meta's Model API is busy.")])
    start(ui)
    html = page(ui)
    assert 'id="agent-error"' in html and "Meta&#39;s Model API is busy." in html
    assert "The model could not continue" in html and "3 videos staged" in html
    assert "2 videos ready, 1 blocked" in html


def test_a_failure_before_anything_started_goes_back_to_the_start_card(tmp_path: Path) -> None:
    def broken(store: Store, session: Any, text: str, as_of: Any, report: Any) -> None:
        raise ModelError("auth", NO_PACKAGING_KEY)

    ui = make_ui(tmp_path, bulk_agent=broken, packaging_key=lambda cfg: True)
    catalog(ui)
    start(ui)
    html = page(ui)
    assert 'id="agent-failed"' in html and "There is no Meta API key." in html
    assert 'id="agent-form"' in html and ui.web.bulk_session() is None
    assert 'id="agent-failed"' not in page(ui)  # shown once


def test_the_limits_notes_show_on_the_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bulkagent, "MAX_ROUNDS", 1)
    ui = make_agent_ui(tmp_path, [tool_reply(("apply_steps", SWAP))])
    start(ui)
    html = page(ui)
    assert html.count("The agent reached its limit of 1 model calls") >= 2  # in the log and the notes
    assert "reply to continue" in html and "2 videos ready, 1 blocked" in html


# --- running: the poll, the stop button, one session at a time -------------------------------------------
def gated_ui(tmp_path: Path) -> tuple[Any, threading.Event, threading.Event]:
    entered, gate = threading.Event(), threading.Event()

    def first(messages: Any, tools: Any, fmt: Any) -> ModelReply:
        entered.set()
        assert gate.wait(10), "the test never opened the gate"
        return tool_reply(("apply_steps", SWAP))

    ui = make_agent_ui(tmp_path, [first, answer("Done.")])
    return ui, entered, gate


def test_while_running_the_page_polls_and_shows_the_log_the_spinner_and_stop(tmp_path: Path) -> None:
    ui, entered, gate = gated_ui(tmp_path)
    try:
        ui.post("/bulk/agent", data={"request": REQUEST}, follow_redirects=False)
        assert entered.wait(10)
        html = page(ui)
        assert 'data-poll="1"' in html
        assert 'id="agent-stop"' in html and 'action="/bulk/agent/stop"' in html
        assert '<li class="ag-line">Read the catalog: 5 videos.</li>' in html
        assert 'id="ag-now"' in html and 'class="spinner"' in html and 'id="bulk_agent-msg"' in html
        assert 'class="ag-log live"' in html and 'data-count="1"' in html
        assert REQUEST in html
        assert 'id="agent-review"' not in html and 'id="agent-reply-form"' not in html
        assert 'id="agent-make-drafts"' not in html and 'id="agent-restart"' not in html
        status = ui.client.get("/api/status").json()["bulk_agent"]
        assert status["status"] == "running"
        live = ui.client.get("/api/bulk-agent?after=0").json()
        assert live["status"] == "running" and live["log"] == ["Read the catalog: 5 videos."]
        assert live["total"] == 1 and live["staged"] == 0 and live["stopping"] is False
        jobs = ui.client.get("/api/activity").json()["jobs"]
        assert any(j["job"] == "bulk_agent" and j["status"] == "running" and j["href"] == "/bulk" for j in jobs)
    finally:
        gate.set()
        ui.web.bulk_agent_job.wait()
    done = ui.client.get("/api/bulk-agent?after=1").json()
    assert done["status"] == "done" and done["total"] == 2 and done["staged"] == 3
    assert done["log"] == ["Replaced text in 3 videos."]  # only the lines after `after`
    assert ui.client.get("/api/bulk-agent?after=99").json()["log"] == []
    assert 'data-poll="1"' not in page(ui) and 'id="agent-review"' in page(ui)


def test_a_second_session_is_refused_while_one_runs(tmp_path: Path) -> None:
    ui, entered, gate = gated_ui(tmp_path)
    try:
        ui.post("/bulk/agent", data={"request": REQUEST}, follow_redirects=False)
        assert entered.wait(10)
        ui.post("/bulk/agent", data={"request": "Something else"}, follow_redirects=False)
        ui.post("/bulk/agent/continue", data={"reply": "More"}, follow_redirects=False)
        html = page(ui)
        assert html.count("The bulk edit agent is still working. Wait for it to finish, or press Stop.") >= 1
        assert ui.agent_calls == [REQUEST]
        assert ui.client.post("/bulk/agent/discard", data={"csrf": ui.csrf()},
                              follow_redirects=False).status_code == 303
        assert "The agent is still working. Press Stop first." in page(ui)
        assert ui.web.bulk_session() is not None
        ui.post("/bulk/agent/apply", data={"video_id": A}, follow_redirects=False)
        assert "The bulk edit agent is still working" in page(ui) and drafts(ui).keys() == {E}
    finally:
        gate.set()
        ui.web.bulk_agent_job.wait()
    assert session_of(ui).requests == [REQUEST]


def test_stop_keeps_what_is_staged_and_says_so(tmp_path: Path) -> None:
    ui, entered, gate = gated_ui(tmp_path)
    try:
        ui.post("/bulk/agent", data={"request": REQUEST}, follow_redirects=False)
        assert entered.wait(10)
        ui.post("/bulk/agent/stop", data={}, follow_redirects=False)
        html = page(ui)
        assert "Stopping after the current step. What is staged is kept." in html
        assert "Stopping after this step." in html and re.search(r'id="agent-stop" disabled', html)
        assert ui.client.get("/api/bulk-agent").json()["stopping"] is True
    finally:
        gate.set()
        ui.web.bulk_agent_job.wait()
    html = page(ui)
    assert "Stopped at your request. What is staged is kept" in html
    assert "2 videos ready, 1 blocked" in html and len(ui.model.calls) == 1


def test_stop_with_nothing_running_does_nothing(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    resp = ui.post("/bulk/agent/stop", data={}, follow_redirects=False)
    assert resp.status_code == 303 and "Stopping" not in page(ui)


# --- replies -----------------------------------------------------------------------------------------
def test_a_reply_goes_on_in_the_same_session(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path, standard_script() + [
        tool_reply(("show_staged", {}), ("unstage", {"refs": ["V2"]})),
        answer("I left V2 out."),
    ])
    start(ui)
    resp = ui.post("/bulk/agent/continue", data={"reply": "Skip the second one"}, follow_redirects=False)
    assert resp.status_code == 303
    ui.web.bulk_agent_job.wait()
    html = page(ui)
    assert ui.agent_calls == [REQUEST, "Skip the second one"]
    assert "Your words and replies" in html and "Skip the second one" in html
    assert 'Read your reply: &#34;Skip the second one&#34;.' in html
    assert "I left " in html and "I swapped the old invite" not in html  # the closing is per turn
    assert "1 video ready, 1 blocked" in html and f'id="agv-{B}"' not in html
    assert len(session_of(ui).requests) == 2
    assert any(m.get("content") == "Skip the second one" for m in ui.model.calls[-1]["messages"])


def test_a_reply_needs_words_a_session_and_a_key(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    ui.post("/bulk/agent/continue", data={"reply": "Hello"}, follow_redirects=False)
    assert "There is no session to continue." in page(ui)
    start(ui)
    ui.post("/bulk/agent/continue", data={"reply": "  "}, follow_redirects=False)
    assert "Describe the change in a few words first." in page(ui)
    ui.post("/bulk/agent/continue", data={"reply": "y" * 1001}, follow_redirects=False)
    assert "at most 1,000 are allowed" in page(ui)
    assert ui.agent_calls == [REQUEST]


def test_starting_again_while_a_session_is_open_is_refused(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    start(ui)
    start(ui, "A different request")
    assert "A session is already open. Reply to it, or discard it first." in page(ui)
    assert ui.agent_calls == [REQUEST] and session_of(ui).requests == [REQUEST]


# --- make drafts, discard ------------------------------------------------------------------------------
def test_make_drafts_for_the_ticked_videos_only(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    start(ui)
    resp = ui.post("/bulk/agent/apply", data={"video_id": [A]}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/changes"
    rows = drafts(ui)
    assert rows.keys() == {A, E}  # E was the creator's own draft
    assert rows[A].source == "bulk-agent" and NEW in rows[A].draft.description
    assert rows[A].approved_at is None
    changes = ui.client.get("/changes").text
    assert "Drafts made for 1 video." in changes and "nothing changes on YouTube until you approve" in changes
    assert "Bulk agent: Replaced" in changes
    html = page(ui)  # the other staged video is still there to review
    assert "1 video ready, 1 blocked" in html and f'id="agv-{B}"' in html and f'id="agv-{A}"' not in html
    assert "Made drafts for 1 video." in html


def test_make_drafts_with_nothing_ticked_or_a_blocked_video_makes_none(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    start(ui)
    ui.post("/bulk/agent/apply", data={}, follow_redirects=False)
    assert "Tick at least one video to make drafts." in page(ui)
    resp = ui.post("/bulk/agent/apply", data={"video_id": [E]}, follow_redirects=False)
    assert resp.headers["location"] == "/bulk"
    html = page(ui)
    assert "No drafts were made. 1 video blocked." in html
    assert drafts(ui).keys() == {E}


def test_make_drafts_refuses_more_than_the_maximum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ui = make_agent_ui(tmp_path)
    start(ui)
    monkeypatch.setattr(bulk, "MAX_BULK", 1)
    ui.post("/bulk/agent/apply", data={"video_id": [A, B]}, follow_redirects=False)
    assert "at most 1 can be changed at once" in page(ui)
    assert drafts(ui).keys() == {E}


def test_a_video_changed_since_it_was_read_is_blocked_on_the_page_and_at_apply(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    start(ui)
    with Store(ui.config.catalog_db) as store:
        store.upsert_video(make_video(1, video_id=A, privacy_status="public", title="Cactus one",
                                      description="Edited in Studio.", published_at="2024-02-01T10:00:00Z"),
                           store.latest_pull()["id"])
    html = page(ui)
    assert "1 video ready, 2 blocked" in html and "changed since the agent read it" in html
    ui.post("/bulk/agent/apply", data={"video_id": [A, B]}, follow_redirects=False)
    assert drafts(ui).keys() == {B, E}


def test_stale_ids_that_are_not_staged_are_counted_not_drafted(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    start(ui)
    ui.post("/bulk/agent/apply", data={"video_id": [C, "nope"]}, follow_redirects=False)
    assert "2 videos no longer matched." in page(ui) and drafts(ui).keys() == {E}


def test_discard_forgets_the_session_and_writes_nothing(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    start(ui)
    assert session_of(ui).staged_list()
    resp = ui.post("/bulk/agent/discard", data={}, follow_redirects=False)
    assert resp.status_code == 303
    html = page(ui)
    assert ui.web.bulk_session() is None and 'id="agent-form"' in html and 'id="agent-review"' not in html
    assert drafts(ui).keys() == {E}
    # and a new session can start
    ui.model.script.extend([answer("What would you like?")])
    start(ui, "Hello")
    assert session_of(ui).closing == "What would you like?"


def test_start_over_and_discard_use_the_same_route(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    start(ui)
    html = page(ui)
    assert re.search(r'action="/bulk/agent/discard"><input type="hidden" name="csrf"[^>]*><button type="submit" '
                     r'class="button secondary" id="agent-restart">Start over', html)


def test_nothing_staged_says_so(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path, [answer("Which videos do you mean?")])
    start(ui, "Fix them")
    html = page(ui)
    assert "0 videos ready, 0 blocked" in html and 'id="agent-empty"' in html
    assert "Which videos do you mean?" in html and 'id="agent-make-drafts"' not in html


def test_the_session_is_not_stored_across_a_restart(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    start(ui)
    fresh = make_ui(tmp_path / "again", bulk_agent=real_agent(FakeModelClient([])), packaging_key=lambda c: True)
    assert fresh.web.bulk_session() is None and 'id="agent-form"' in page(fresh)


# --- the rest of the page and the labels -----------------------------------------------------------------
def test_the_manual_forms_still_preview_and_make_drafts(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    url = ("/bulk?kind=replace&find=" + OLD + "&replace_with=" + NEW +
           "&field=description&visibility=public&preview=1")
    html = ui.client.get(url).text
    assert 'id="bulk-preview"' in html and "videos would change" in html and 'action="/bulk/apply"' in html
    resp = ui.post("/bulk/apply", data={"kind": "replace", "find": OLD, "replace_with": NEW,
                                        "field": "description", "visibility": "public", "video_id": [A]},
                   follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/changes"
    assert drafts(ui)[A].source == "bulk"


def test_drafts_from_the_agent_and_the_old_flow_show_a_label_on_changes(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    with Store(ui.config.catalog_db) as store:
        edits.save_draft(store, A, "Cactus one (a)", "x", ["cactus"], source="bulk-agent")
        edits.save_draft(store, B, "Cactus two (b)", "y", ["cactus"], source="bulk-ask")
    html = ui.client.get("/changes").text
    assert "Bulk edit agent" in html and "Bulk edit by asking" in html
    assert "Bulk-agent" not in html and "Bulk-ask" not in html


def test_error_mapping_is_plain() -> None:
    assert web_app.map_bulk_agent_error(ModelError("rate_limit", "Busy.")) == ("rate_limit", "Busy.")
    assert web_app.map_bulk_agent_error(bulkagent.BulkAgentError("No catalog.")) == ("bulk_agent", "No catalog.")
    kind, message = web_app.map_bulk_agent_error(RuntimeError("secret detail"))
    assert kind == "bulk_agent" and "secret detail" not in message and "RuntimeError" in message


def test_the_timeout_is_fifteen_minutes_and_the_job_uses_it(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    assert web_app.BULK_AGENT_TIMEOUT_SECONDS == 15 * 60
    assert ui.web.bulk_agent_job._timeout == 15 * 60  # type: ignore[attr-defined]
    assert "15 minutes" in web_app.BULK_AGENT_TIMEOUT_MESSAGE


def test_a_timed_out_turn_is_told_to_stop(tmp_path: Path) -> None:
    ui, entered, gate = gated_ui(tmp_path)
    try:
        ui.post("/bulk/agent", data={"request": REQUEST}, follow_redirects=False)
        assert entered.wait(10)
        ui.web.bulk_agent_job._clock = lambda: 10 ** 9  # type: ignore[attr-defined]
        snap = ui.web.bulk_agent_state()
        assert snap.error_kind == "timeout" and "15 minutes" in snap.message
        assert session_of(ui).stop.is_set()
    finally:
        gate.set()
        ui.web.bulk_agent_job.wait()
    assert len(session_of(ui).staged_list()) <= 3


def test_the_log_endpoint_without_a_session_is_empty(tmp_path: Path) -> None:
    ui = make_agent_ui(tmp_path)
    data = ui.client.get("/api/bulk-agent").json()
    assert data == {"status": "idle", "message": "", "log": [], "total": 0, "staged": 0, "stopping": False}
