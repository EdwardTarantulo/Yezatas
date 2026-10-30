"""AI suggestions in the web UI: the video page card, fix-next, the agent log, "Why" on Changes
and the key form in Setup. The model is a fake suggest function; nothing touches the network."""

from __future__ import annotations

import re
import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest

from webhelpers import make_ui, make_video, seed_catalog
from yezatas import agent, edits
from yezatas.agent import AgentError, SuggestResult
from yezatas.store import Store
from yezatas.web.app import make_default_suggest

MODEL = "muse-spark-1.3"
FAKE_KEY = "FAKE-KEY-0123456789abcdef"
V1, V2, V3, V4, V5 = "vidaaa001", "vidaaa002", "vidaaa003", "vidaaa004", "vidaaa005"


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


class FakeSuggest:
    """Stands in for the agent: writes a draft and a run like the real one, records its calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.outcomes: dict[str, str] = {}  # video id -> "drafted" (default) | "no_change" | ...
        self.gate: threading.Event | None = None  # when set, a call waits for it
        self.entered = threading.Event()

    def __call__(self, store: Store, video_id: str, report: Any) -> SuggestResult:
        self.calls.append(video_id)
        report("Reading the scorecard")
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10)
        outcome = self.outcomes.get(video_id, "drafted")
        if outcome == "skip":
            raise AgentError("This video already has a draft. Push or discard it first.")
        if outcome == "crash":
            raise RuntimeError("boom")
        run_id = store.begin_agent_run(video_id, MODEL)
        if outcome in ("auth", "rate_limit"):
            store.finish_agent_run(
                run_id, "failed", "The Meta API key was refused. Check it in Setup.",
                [{"type": "error", "kind": outcome, "message": "refused"}], {})
            return SuggestResult(run_id, video_id, "failed",
                                 "The Meta API key was refused. Check it in Setup.")
        if outcome == "no_change":
            store.finish_agent_run(run_id, "no_change", "Nothing to improve.", [], {})
            return SuggestResult(run_id, video_id, "no_change", "The model found nothing to change.")
        video = store.get_video(video_id)
        assert video is not None
        edits.save_draft(store, video_id, video.title + " improved", video.description,
                         list(video.tags), source="agent",
                         note=f"Suggested by {MODEL}: Sharper title.")
        result = {
            "title": video.title + " improved", "description": video.description,
            "tags": list(video.tags), "summary": "Sharper title.",
            "changes": [{"field": "title", "check_id": "title_length",
                         "reason": "Added a clearer ending."}],
            "not_fixed": [{"check_id": "description_chapters",
                           "reason": "Chapters need a transcript."}],
        }
        log = [{"type": "tool", "name": "get_scorecard", "arguments": {}, "result_chars": 812},
               {"type": "tool", "name": "find_related_videos",
                "arguments": {"query": "cactus", "limit": 3}, "result_chars": 240},
               {"type": "problems", "round": 1, "problems": ["Added a link that is not in the text."]}]
        store.finish_agent_run(run_id, "drafted", "Sharper title.", log, result,
                               tokens=(12000, 2000, 800))
        return SuggestResult(run_id, video_id, "drafted", "Draft made for review.")


def make_suggest_ui(tmp_path: Path, *, key: bool = True, count: int = 5):
    fake = FakeSuggest()
    ui = make_ui(tmp_path, suggest=fake)
    videos = [
        make_video(n, video_id=f"vidaaa{n:03d}", title=f"Cactus video {n}", description="",
                   tags=[], privacy_status="public", view_count=views)
        for n, views in zip(range(1, count + 1), (100, 90_000, 5_000, 300_000, 20_000))
    ]
    seed_catalog(ui.config, videos)
    if key:
        agent.save_api_key(ui.config, FAKE_KEY)
    return ui, fake


def nf(ui: Any, path: str, **kwargs: Any) -> Any:
    """POST with the CSRF token and without following the redirect, so the flash waits for us."""
    kwargs.setdefault("follow_redirects", False)
    return ui.post(path, **kwargs)


def draft(ui: Any, video_id: str, **kw: Any) -> None:
    with Store(ui.config.catalog_db) as store:
        video = store.get_video(video_id)
        edits.save_draft(store, video_id, video.title + " (new)", video.description,
                         list(video.tags), **kw)


def finish(ui: Any) -> None:
    ui.web.suggest_job.wait()


# --- the video page -----------------------------------------------------------------------------
def test_video_page_has_the_card_button_and_privacy_sentence(tmp_path: Path) -> None:
    ui, _ = make_suggest_ui(tmp_path)
    html = ui.client.get(f"/videos/{V1}").text
    assert "Suggestions from Muse Spark" in html and "Suggest improvements" in html
    assert (f"This video's title, description, tags, stats and scorecard go to Meta's Model API "
            f"({MODEL}, Standard mode: Meta does not use them for training).") in html
    assert f'action="/videos/{V1}/suggest"' in html
    assert 'id="suggest-no-key"' not in html and "latest-run" not in html
    assert "<script>" not in html and "style=" not in html


def test_without_a_key_the_video_page_points_to_setup(tmp_path: Path) -> None:
    ui, _ = make_suggest_ui(tmp_path, key=False)
    html = ui.client.get(f"/videos/{V1}").text
    assert 'id="suggest-no-key"' in html and 'href="/setup#step-ai"' in html


def test_suggest_without_a_key_flashes_and_starts_nothing(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path, key=False)
    resp = nf(ui, f"/videos/{V1}/suggest", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}"
    html = ui.client.get(f"/videos/{V1}").text
    assert "Add a Meta API key first." in html and 'href="/setup#step-ai"' in html
    assert fake.calls == [] and ui.web.suggest_job.snapshot().status == "idle"


def test_suggest_is_refused_when_the_video_has_a_draft(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    draft(ui, V1)
    html = ui.client.get(f"/videos/{V1}").text
    assert 'id="suggest-has-draft"' in html
    assert re.search(r'<button type="submit" class="button primary" disabled>Suggest improvements', html)
    nf(ui, f"/videos/{V1}/suggest")
    html = ui.client.get(f"/videos/{V1}").text
    assert "This video already has a draft. Push or discard it first." in html
    assert fake.calls == [] and ui.web.suggest_job.snapshot().status == "idle"


def test_suggest_for_an_unknown_video_is_a_404(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    resp = nf(ui, "/videos/nope/suggest")
    assert resp.status_code == 404 and "Video not found" in resp.text
    assert fake.calls == []


def test_a_run_makes_a_draft_and_the_video_page_shows_it(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    resp = nf(ui, f"/videos/{V1}/suggest", follow_redirects=False)
    assert resp.status_code == 303
    finish(ui)
    assert fake.calls == [V1]
    html = ui.client.get(f"/videos/{V1}").text
    assert "Draft made for review." in html  # the job's message, as a flash
    run = html.split('id="latest-run"')[1].split("</section>")[0]
    assert "Draft made" in run and "Sharper title." in run
    assert "Added a clearer ending." in run and "Title is 30 to 60 characters" in run
    assert "Left for you" in run and "Long video has chapters starting at 0:00" in run
    assert "Chapters need a transcript." in run
    assert "12,000 tokens in (2,000 cached), 800 out" in run
    assert "Estimated cost: $0.016" in run
    assert f'href="/changes#d-{V1}"' in run and 'href="/agent#run-1"' in run
    # the flash is shown once
    assert "Draft made for review." not in ui.client.get(f"/videos/{V1}").text
    with Store(ui.config.catalog_db) as store:
        assert store.get_draft(V1).source == "agent"


def test_the_card_shows_progress_while_a_run_is_going(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    fake.gate = threading.Event()
    nf(ui, f"/videos/{V1}/suggest")
    assert fake.entered.wait(10)
    html = ui.client.get(f"/videos/{V1}").text
    assert 'id="suggest-msg"' in html and "Reading the scorecard" in html
    assert 'data-poll="1"' in html and "Suggest improvements" not in html
    assert ui.client.get("/api/status").json()["suggest"]["status"] == "running"
    other = ui.client.get(f"/videos/{V2}").text
    assert 'id="suggest-busy"' in other and 'data-poll="1"' in other
    assert re.search(r'class="button primary" disabled>Suggest improvements', other)
    fake.gate.set()
    finish(ui)
    assert ui.client.get("/api/status").json()["suggest"]["status"] == "done"


def test_only_one_suggestion_runs_at_a_time(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    fake.gate = threading.Event()
    nf(ui, f"/videos/{V1}/suggest")
    assert fake.entered.wait(10)
    nf(ui, f"/videos/{V2}/suggest")
    assert "A suggestion is already running." in ui.client.get(f"/videos/{V2}").text
    fake.gate.set()
    finish(ui)
    assert fake.calls == [V1]


def test_a_run_with_no_change_and_a_failed_run_are_reported(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    fake.outcomes[V1] = "no_change"
    nf(ui, f"/videos/{V1}/suggest")
    finish(ui)
    html = ui.client.get(f"/videos/{V1}").text
    assert "The model found nothing to change." in html and "No change needed" in html
    fake.outcomes[V2] = "auth"
    nf(ui, f"/videos/{V2}/suggest")
    finish(ui)
    html = ui.client.get(f"/videos/{V2}").text
    assert 'class="flash error"' in html and "The Meta API key was refused." in html
    assert 'class="pill problem">Failed' in html


def test_an_unexpected_error_is_reported_without_details(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    fake.outcomes[V1] = "crash"
    nf(ui, f"/videos/{V1}/suggest")
    finish(ui)
    html = ui.client.get(f"/videos/{V1}").text
    assert "Stopped unexpectedly (RuntimeError)." in html and "boom" not in html


# --- fix next on the dashboard --------------------------------------------------------------------
def test_dashboard_form_with_a_key_and_a_link_without(tmp_path: Path) -> None:
    ui, _ = make_suggest_ui(tmp_path)
    html = ui.client.get("/").text
    assert 'action="/suggest/fix-next"' in html and "Suggest fixes for the top" in html
    assert re.search(r'<option value="3" selected>3</option>', html)
    assert all(f'<option value="{n}"' in html for n in (1, 3, 5, 10))
    assert "Meta's Model API" in html and 'id="suggest-no-key"' not in html
    ui2, _ = make_suggest_ui(tmp_path / "other", key=False)
    html = ui2.client.get("/").text
    assert "/suggest/fix-next" not in html
    assert 'id="suggest-no-key"' in html and 'href="/setup#step-ai"' in html


def test_fix_next_takes_the_top_public_videos_without_a_draft_in_order(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    # V4 (300k views) has a draft, V2 (90k) is unlisted: neither is picked
    seed_catalog(ui.config, [make_video(2, video_id=V2, title="Cactus video 2", description="",
                                        tags=[], privacy_status="unlisted", view_count=90_000)])
    draft(ui, V4)
    nf(ui, "/suggest/fix-next", data={"count": "3"})
    finish(ui)
    assert fake.calls == [V5, V3, V1]
    html = ui.client.get("/").text
    result = html.split('id="suggest-result"')[1].split("</section>")[0]
    assert "3 drafts made" in result
    assert re.findall(r'id="sr-(\w+)"', result) == [V5, V3, V1]
    assert "Draft made for review." in result and 'href="/changes"' in result
    # dismissing clears the card
    resp = nf(ui, "/suggest/done", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/"
    assert 'id="suggest-result"' not in ui.client.get("/").text


def test_fix_next_shows_progress_with_the_position(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    fake.gate = threading.Event()
    nf(ui, "/suggest/fix-next", data={"count": "3"})
    assert fake.entered.wait(10)
    html = ui.client.get("/").text
    assert "1 of 3: Cactus video 4 (Reading the scorecard)" in html
    assert 'data-poll="1"' in html and 'id="suggest-progress"' in html
    fake.gate.set()
    finish(ui)


def test_fix_next_stops_after_an_auth_failure(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    fake.outcomes[V2] = "auth"
    nf(ui, "/suggest/fix-next", data={"count": "5"})
    finish(ui)
    assert fake.calls == [V4, V2]
    result = ui.client.get("/").text.split('id="suggest-result"')[1].split("</section>")[0]
    assert "Stopped early: The Meta API key was refused." in result
    assert result.count("Not attempted") >= 3 and "1 failed" in result
    assert "1 draft made" in result


def test_fix_next_does_not_stop_after_other_failures_and_lists_skips(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    fake.outcomes[V4] = "skip"
    fake.outcomes[V2] = "no_change"
    nf(ui, "/suggest/fix-next", data={"count": "3"})
    finish(ui)
    assert fake.calls == [V4, V2, V5]
    result = ui.client.get("/").text.split('id="suggest-result"')[1].split("</section>")[0]
    assert "1 draft made, 1 needed no change, 1 skipped." in result
    assert "This video already has a draft." in result


def test_fix_next_needs_a_valid_count_and_a_key(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    for bad in ("2", "", "abc", "-3", "100"):
        nf(ui, "/suggest/fix-next", data={"count": bad})
        assert "Choose 1, 3, 5 or 10 videos." in ui.client.get("/").text
    assert fake.calls == []
    ui2, fake2 = make_suggest_ui(tmp_path / "nokey", key=False)
    nf(ui2, "/suggest/fix-next", data={"count": "3"})
    html = ui2.client.get("/").text
    assert "Add a Meta API key first." in html and 'href="/setup#step-ai"' in html
    assert fake2.calls == [] and ui2.web.suggest_job.snapshot().status == "idle"


def test_fix_next_with_nothing_to_fix_says_so(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path, count=1)
    draft(ui, V1)
    nf(ui, "/suggest/fix-next", data={"count": "3"})
    assert "No public video needs a fix" in ui.client.get("/").text
    assert fake.calls == []


# --- the agent log --------------------------------------------------------------------------------
def test_the_agent_log_lists_runs_with_their_steps(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    fake.outcomes[V2] = "no_change"
    nf(ui, "/suggest/fix-next", data={"count": "3"})
    finish(ui)
    html = ui.client.get("/agent").text
    assert "Agent log" in html and "audit trail" in html
    assert 'class="nav-item active" href="/agent"' in html
    rows = re.findall(r'id="run-(\d+)"', html)
    assert rows == ["3", "2", "1"]  # newest first
    first = html.split('id="run-1"')[1].split("</details>")[0]  # V4, drafted
    assert "Draft made" in first and "Cactus video 4" in first and MODEL in first
    assert "Sharper title." in first and "Estimated cost: $0.016" in first
    assert "<details" in first
    assert "<code>get_scorecard</code> <code>{}</code>" in first
    assert '<code>{&#34;limit&#34;:3,&#34;query&#34;:&#34;cactus&#34;}</code>' in first
    assert "812 characters returned" in first
    assert "Problems found by the checks" in first and "Added a link that is not in the text." in first
    second = html.split('id="run-2"')[1].split("</li>")[0]  # V2, no change
    assert "No change needed" in second and "<details" not in second


def test_the_agent_log_shows_errors_and_the_empty_state(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    assert "No suggestions yet" in ui.client.get("/agent").text
    fake.outcomes[V1] = "rate_limit"
    nf(ui, f"/videos/{V1}/suggest")
    finish(ui)
    html = ui.client.get("/agent").text
    assert "Failed" in html and "Errors" in html and "<li>refused</li>" in html


def test_the_agent_log_works_before_the_first_pull(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    assert ui.client.get("/agent").status_code == 200


# --- "Why" on the Changes page ----------------------------------------------------------------------
def test_changes_shows_why_for_agent_drafts_only(tmp_path: Path) -> None:
    ui, _ = make_suggest_ui(tmp_path)
    nf(ui, f"/videos/{V1}/suggest")
    finish(ui)
    draft(ui, V2)  # a manual draft
    # an older agent run on V2 must not leak into the manual draft
    with Store(ui.config.catalog_db) as store:
        run_id = store.begin_agent_run(V2, MODEL)
        store.finish_agent_run(run_id, "drafted", "Old run.", [], {
            "changes": [{"field": "tags", "check_id": "", "reason": "Old reason."}],
            "not_fixed": []})
    html = ui.client.get("/changes").text
    agent_card = html.split(f'id="d-{V1}"')[1].split("</article>")[0]
    manual_card = html.split(f'id="d-{V2}"')[1].split("</article>")[0]
    assert 'class="why"' in agent_card and ">Why<" in agent_card
    assert "Sharper title." in agent_card and "Added a clearer ending." in agent_card
    assert "Chapters need a transcript." in agent_card
    assert f"Suggested by {MODEL}: Sharper title." in agent_card  # the origin line
    assert 'class="why"' not in manual_card and "Old reason." not in html


# --- Setup: the key -------------------------------------------------------------------------------
def test_setup_shows_the_ai_section_without_a_key(tmp_path: Path) -> None:
    ui, _ = make_suggest_ui(tmp_path, key=False)
    html = ui.setup_html()
    section = html.split('id="step-ai"')[1].split("</section>")[0]
    assert "AI suggestions (optional)" in section and "No key" in section
    assert 'action="/setup/model-key"' in section and 'type="password"' in section
    assert "Create an API key in Meta's developer dashboard under API keys." in section
    assert "Standard mode: Meta does not use them for training" in section
    assert "/setup/model-key/delete" not in section


def test_saving_a_key_writes_the_file_and_never_shows_it(tmp_path: Path) -> None:
    ui, _ = make_suggest_ui(tmp_path, key=False)
    resp = nf(ui, "/setup/model-key", data={"api_key": f"  {FAKE_KEY}  "}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/setup#step-ai"
    assert ui.config.model_key_path.read_text(encoding="utf-8") == FAKE_KEY
    assert agent.key_source(ui.config) == "saved"
    setup = ui.setup_html()
    assert "Saved the key on this computer." in setup and "Key saved" in setup
    assert "/setup/model-key/delete" in setup and "Replace the saved key" in setup
    for path in ("/setup", "/", "/agent", f"/videos/{V1}", "/changes", "/history", "/videos",
                 "/api/status"):
        body = ui.client.get(path).text
        assert FAKE_KEY not in body and FAKE_KEY[:12] not in body, path
    # the redirect page did not echo it either, and nothing about it is in the database
    assert FAKE_KEY not in resp.text
    assert FAKE_KEY.encode() not in ui.config.catalog_db.read_bytes()


@pytest.mark.parametrize("bad", ["", "   ", "two words", "a\nb", "x" * 501])
def test_bad_keys_are_refused(tmp_path: Path, bad: str) -> None:
    ui, _ = make_suggest_ui(tmp_path, key=False)
    nf(ui, "/setup/model-key", data={"api_key": bad})
    assert agent.key_source(ui.config) == ""
    html = ui.setup_html()
    assert 'class="flash error"' in html and ("Paste your key first." in html or "one line" in html)


def test_removing_the_saved_key(tmp_path: Path) -> None:
    ui, _ = make_suggest_ui(tmp_path)
    assert not ui.config.model_key_path is None and ui.config.model_key_path.is_file()
    resp = nf(ui, "/setup/model-key/delete", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/setup#step-ai"
    assert not ui.config.model_key_path.exists() and agent.key_source(ui.config) == ""
    assert "The saved key was deleted." in ui.setup_html()
    nf(ui, "/setup/model-key/delete")
    assert "There was no saved key." in ui.setup_html()


def test_a_key_from_the_environment_hides_the_form(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ui, _ = make_suggest_ui(tmp_path, key=False)
    monkeypatch.setenv(agent.KEY_ENV, FAKE_KEY)
    html = ui.setup_html()
    section = html.split('id="step-ai"')[1].split("</section>")[0]
    assert "Key from the environment" in section and "environment variable" in section
    assert 'type="password"' not in section and "/setup/model-key" not in section
    assert FAKE_KEY not in html
    nf(ui, "/setup/model-key", data={"api_key": "FAKE-OTHER-KEY"})
    assert not ui.config.model_key_path.exists()
    assert "so nothing was saved" in ui.setup_html()
    assert 'action="/suggest/fix-next"' in ui.client.get("/").text  # the key counts


# --- the real suggest function --------------------------------------------------------------------
def test_the_default_suggest_builds_a_client_from_the_saved_key(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    ui, _ = make_suggest_ui(tmp_path)
    seen: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, api_key: str, model: str) -> None:
            seen["key"], seen["model"] = api_key, model

    def fake_suggest_for_video(store: Any, client: Any, video_id: str, *, report: Any = None,
                               **_: Any) -> SuggestResult:
        seen["client"], seen["video"] = client, video_id
        report("Writing the suggestion")
        return SuggestResult(1, video_id, "no_change", "Nothing to change.")

    module = types.ModuleType("yezatas.modelclient")
    module.MetaModelClient = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yezatas.modelclient", module)
    monkeypatch.setattr(agent, "suggest_for_video", fake_suggest_for_video, raising=False)
    messages: list[str] = []
    with Store(ui.config.catalog_db) as store:
        result = make_default_suggest(ui.config)(store, V1, messages.append)
    assert result.status == "no_change" and messages == ["Writing the suggestion"]
    assert seen["key"] == FAKE_KEY and seen["model"] == ui.config.model and seen["video"] == V1
    assert isinstance(seen["client"], FakeClient)


def test_the_default_suggest_without_a_key_is_a_skipped_video(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)  # no suggest injected: the default one
    seed_catalog(ui.config, [make_video(1, video_id=V1, privacy_status="public", view_count=10,
                                        description="", tags=[])])
    with Store(ui.config.catalog_db) as store:
        with pytest.raises(AgentError, match="No Meta API key"):
            make_default_suggest(ui.config)(store, V1, lambda text: None)
    assert nf(ui, f"/videos/{V1}/suggest").status_code == 303  # refused up front, with a flash
    assert ui.web.suggest_job.snapshot().status == "idle"


# --- security -------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", [f"/videos/{V1}/suggest", "/suggest/fix-next", "/suggest/done",
                                  "/setup/model-key", "/setup/model-key/delete"])
def test_every_new_post_needs_the_csrf_token(tmp_path: Path, path: str) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    assert ui.client.post(path, data={}).status_code == 403
    assert ui.client.post(path, data={"csrf": "wrong", "count": "3", "api_key": "FAKE"}).status_code == 403
    assert fake.calls == [] and ui.web.suggest_job.snapshot().status == "idle"
    assert agent.key_source(ui.config) == "saved"  # the key was not replaced or removed
    assert ui.config.model_key_path.read_text(encoding="utf-8") == FAKE_KEY


def test_the_new_pages_need_the_session(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, signed_in_browser=False)
    assert ui.client.get("/agent").status_code == 403


def test_new_pages_have_no_inline_script_or_style(tmp_path: Path) -> None:
    ui, fake = make_suggest_ui(tmp_path)
    nf(ui, "/suggest/fix-next", data={"count": "3"})
    finish(ui)
    for path in ("/", "/agent", "/changes", "/setup", f"/videos/{V4}"):
        html = ui.client.get(path).text
        assert "style=" not in html and html.count("<script") == html.count("<script src="), path
        assert not re.search(r"\son\w+=", html), path
