"""The Advertiser-friendly check card on the video page: the card with and without a check, each verdict,
the stale and no-transcript lines, a failed check, escaping, the POST that starts the job with a fake
check function, the no-key path, an unknown video, CSRF and /api/status. No model or network call."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from markupsafe import escape

from fakes import FakeModelClient
from webhelpers import make_ui, make_video, seed_catalog, seed_transcript
from yezatas import adcheck, agent
from yezatas.agent import ModelError, ModelReply
from yezatas.store import AdCheck, Store
from yezatas.web import app as web_app
from yezatas.web.app import NO_PACKAGING_KEY
from yezatas.web.views import AD_CHANGED_LINE, AD_NO_TRANSCRIPT_LINE, build_adcheck_view

V1, V2 = "vid00000001", "vid00000002"
NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)
DESCRIPTION = "How to repot a cactus. The soil is a real Damn Mess, so wear gloves."


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


# --- fakes ---------------------------------------------------------------------------------------
def reply(data: Any) -> ModelReply:
    return ModelReply(data if isinstance(data, str) else json.dumps(data), (), {}, 3000, 200, 150)


def good_answer(**override: Any) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "verdict": "worth_a_look", "summary": "One word may matter.",
        "findings": [{"category": "inappropriate_language", "source": "description",
                      "quote": "real Damn Mess", "segment": -1, "why": "Strong language.",
                      "fix": "Say mess."}]}
    answer.update(override)
    return answer


class FakeAdcheck:
    """The injected check function: the real adcheck.check_video with a scripted model."""

    def __init__(self, script: list[Any] | None = None, gate: threading.Event | None = None,
                 error: Exception | None = None) -> None:
        self.client = FakeModelClient(script if script is not None else [reply(good_answer())])
        self.gate = gate
        self.error = error
        self.entered = threading.Event()
        self.calls: list[str] = []

    def __call__(self, store: Store, video_id: str, now: datetime, report: Any) -> AdCheck:
        self.calls.append(video_id)
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10), "the test never opened the gate"
        if self.error is not None:
            raise self.error
        return adcheck.check_video(store, self.client, video_id, now=now, report=report)


def make_ac_ui(tmp_path: Path, fn: Any = "default", key: bool = True, **kwargs: Any) -> Any:
    fn = FakeAdcheck() if fn == "default" else fn
    ui = make_ui(tmp_path, adcheck=fn, packaging_key=lambda cfg: key, **kwargs)
    seed_catalog(ui.config, [
        make_video(1, video_id=V1, title="Cactus repotting guide", description=DESCRIPTION,
                   tags=["cactus"], privacy_status="public"),
        make_video(2, video_id=V2, title="Second video", description="Plain.", tags=[],
                   privacy_status="public")])
    ui.fn = fn
    return ui


def nf(ui: Any, path: str, **kwargs: Any) -> Any:
    kwargs.setdefault("follow_redirects", False)
    return ui.post(path, **kwargs)


def page(ui: Any, video_id: str = V1) -> str:
    resp = ui.client.get(f"/videos/{video_id}")
    assert resp.status_code == 200
    return resp.text


def card(html: str) -> str:
    return html.split('id="adcheck"')[1].split("</section>")[0]


def run_check(ui: Any, video_id: str = V1) -> Any:
    resp = nf(ui, f"/videos/{video_id}/adcheck")
    ui.web.adcheck_job.wait()
    return resp


def seed_check(ui: Any, video_id: str = V1, **fields: Any) -> AdCheck:
    """Store a check made from the video's current text (so it is not stale) unless fields say otherwise."""
    with Store(ui.config.catalog_db) as store:
        base: dict[str, Any] = dict(
            id=0, video_id=video_id, created_at="", model="muse-spark-1.3", status="done",
            verdict="worth_a_look", summary="Something to look at.", findings=[], notes=[],
            source_hash=adcheck.current_source_hash(store, video_id) or "", had_transcript=True,
            error="", input_tokens=3000, cached_tokens=200, output_tokens=150)
        base.update(fields)
        store.save_ad_check(AdCheck(**base), now=NOW)
        saved = store.latest_ad_check(video_id)
    assert saved is not None
    return saved


def finding(**override: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"category": "inappropriate_language", "source": "description",
                            "quote": "real Damn Mess", "segment": -1, "time": "",
                            "why": "Strong language.", "fix": "Say mess."}
    item.update(override)
    return item


# --- the card ------------------------------------------------------------------------------------
def test_the_card_renders_before_any_check(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path)
    html = page(ui)
    assert html.index('id="edit"') < html.index('id="packaging"') < html.index('id="adcheck"')
    section = card(html)
    assert "Advertiser-friendly check" in section
    assert str(escape(adcheck.ADVICE_LINE)) in section
    assert f'action="/videos/{V1}/adcheck"' in section and 'name="csrf"' in section
    assert 'class="button primary">Check this video<' in section
    assert "disabled" not in section and "adcheck-no-key" not in section
    assert "adcheck-result" not in section
    assert ui.fn.calls == []


def test_no_key_disables_the_button_with_a_plain_reason(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path, key=False)
    section = card(page(ui))
    assert 'id="adcheck-no-key"' in section and "no Meta API key" in section
    assert "button primary\" disabled>" in section


@pytest.mark.parametrize("verdict, label, tone", [
    ("no_concerns", "No concerns found", "done"),
    ("worth_a_look", "Worth a look", "warn"),
    ("likely_limited", "Ads likely limited", "problem")])
def test_each_verdict_has_its_label_and_pill_tone(tmp_path: Path, verdict: str, label: str, tone: str) -> None:
    ui = make_ac_ui(tmp_path)
    seed_check(ui, verdict=verdict, summary="A short summary.")
    section = card(page(ui))
    assert f'<span class="pill {tone}" id="adcheck-verdict">{label}</span>' in section
    assert 'id="adcheck-summary">A short summary.<' in section
    assert str(escape(adcheck.ADVICE_LINE)) in section
    assert "Check again" in section


def test_findings_show_label_where_quote_why_and_fix(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path)
    seed_check(ui, findings=[
        finding(),
        finding(category="violence", source="transcript", quote="old roots", segment=3, time="3:15",
                why="Mild.", fix=""),
        finding(category="drugs", source="tags", quote="cactus", why="", fix="")],
        notes=["A finding was dropped: its quote is not in the title."])
    section = card(page(ui))
    assert "Inappropriate language" in section and "Description" in section
    assert "Violence" in section and "Transcript at 3:15" in section
    assert "Recreational drugs" in section
    assert "real Damn Mess" in section and "Strong language." in section
    assert "Suggested change:</span> Say mess." in section
    assert section.count("Suggested change:") == 1  # empty fixes show nothing
    assert "A finding was dropped: its quote is not in the title." in section
    assert section.count('class="ac-finding"') == 3


def test_the_cost_line_shows_model_tokens_and_cost(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path)
    seed_check(ui)
    section = card(page(ui))
    assert "muse-spark-1.3 &middot; 3,150 tokens &middot; cost $" in section
    assert "Last check: 20" in section  # the local time of the check


def test_a_changed_text_shows_the_stale_line(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path)
    seed_check(ui)
    assert AD_CHANGED_LINE not in card(page(ui))
    seed_check(ui, source_hash="0000000000000000")
    section = card(page(ui))
    assert AD_CHANGED_LINE in section and 'id="adcheck-changed"' in section


def test_a_new_transcript_makes_the_check_stale(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path)
    seed_check(ui)
    seed_transcript(ui.config, V1)
    assert AD_CHANGED_LINE in card(page(ui))


def test_no_transcript_line(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path)
    seed_check(ui, had_transcript=False)
    section = card(page(ui))
    assert AD_NO_TRANSCRIPT_LINE in section and 'href="#transcript"' in section
    seed_check(ui, had_transcript=True)
    assert AD_NO_TRANSCRIPT_LINE not in card(page(ui))


def test_a_failed_check_shows_its_error_and_try_again(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path)
    seed_check(ui, status="failed", verdict="", summary="", error="Slow down, then try again.",
               input_tokens=0, cached_tokens=0, output_tokens=0)
    section = card(page(ui))
    assert 'id="adcheck-error"' in section and "Slow down, then try again." in section
    assert 'class="button primary">Try again<' in section
    assert "adcheck-verdict" not in section and "adcheck-findings" not in section
    assert AD_CHANGED_LINE not in section and AD_NO_TRANSCRIPT_LINE not in section
    assert str(escape(adcheck.ADVICE_LINE)) in section


def test_model_text_is_escaped(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path)
    evil = "<script>alert(1)</script>"
    seed_check(ui, summary=evil, notes=[evil], findings=[
        finding(quote=evil, why=evil, fix=evil, category=evil, source="transcript", time=evil)])
    section = card(page(ui))
    assert "<script>alert(1)" not in section
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in section


def test_view_builder_without_a_check() -> None:
    view = build_adcheck_view(None, "abc")
    assert view.check is None and view.advice == adcheck.ADVICE_LINE and view.findings == ()


# --- running it ----------------------------------------------------------------------------------
def test_post_starts_the_job_and_the_card_shows_the_stored_result(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path)
    resp = run_check(ui)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#adcheck"
    assert ui.fn.calls == [V1]
    assert ui.web.adcheck_job.snapshot().status == "done"
    with Store(ui.config.catalog_db) as store:
        stored = store.latest_ad_check(V1)
    assert stored is not None and stored.status == "done" and len(stored.findings) == 1
    section = card(page(ui))
    assert "Worth a look" in section and "real Damn Mess" in section
    assert "Check again" in section and AD_CHANGED_LINE not in section


def test_a_running_check_shows_progress_and_polls(tmp_path: Path) -> None:
    gate = threading.Event()
    ui = make_ac_ui(tmp_path, FakeAdcheck(gate=gate))
    try:
        nf(ui, f"/videos/{V1}/adcheck")
        assert ui.fn.entered.wait(10)
        html = page(ui)
        section = card(html)
        assert 'id="adcheck-msg"' in section and "Check this video" not in section
        assert 'data-poll="1"' in html
        other = card(page(ui, V2))
        assert 'id="adcheck-busy"' in other and "button primary\" disabled>" in other
        assert ui.client.get("/api/status").json()["adcheck"]["status"] == "running"
        again = nf(ui, f"/videos/{V2}/adcheck")
        assert again.status_code == 303
        assert "already running" in ui.client.get(f"/videos/{V2}").text
    finally:
        gate.set()
        ui.web.adcheck_job.wait()
    assert ui.fn.calls == [V1]


def test_a_model_failure_is_stored_and_shown_as_a_failed_check(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path, FakeAdcheck([ModelError("rate_limit", "Slow down, then try again.")]))
    run_check(ui)
    assert ui.web.adcheck_job.snapshot().status == "done"
    section = card(page(ui))
    assert "Slow down, then try again." in section and 'class="button primary">Try again<' in section


def test_an_unexpected_error_gets_a_plain_sentence_and_can_be_dismissed(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path, FakeAdcheck(error=KeyError("x")))
    run_check(ui)
    snap = ui.web.adcheck_job.snapshot()
    assert snap.status == "failed"
    assert snap.message == "The advertiser-friendly check stopped unexpectedly (KeyError)."
    assert 'id="adcheck-failed"' in card(page(ui))
    resp = nf(ui, "/adcheck/done", data={"video_id": V1})
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#adcheck"
    assert 'id="adcheck-failed"' not in card(page(ui))


def test_a_refused_key_fails_the_job_with_the_auth_kind(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path, FakeAdcheck(error=ModelError("auth", "The Meta API key was refused.")))
    run_check(ui)
    snap = ui.web.adcheck_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "auth"
    assert "The Meta API key was refused." in card(page(ui))


def test_the_error_mapper() -> None:
    assert web_app.map_adcheck_error(ModelError("network", "Offline.")) == ("network", "Offline.")
    assert web_app.map_adcheck_error(ValueError("Not here.")) == ("adcheck", "Not here.")
    kind, text = web_app.map_adcheck_error(RuntimeError("boom"))
    assert kind == "adcheck" and "RuntimeError" in text and "boom" not in text


def test_a_job_that_runs_too_long_gets_the_timeout_message(tmp_path: Path) -> None:
    assert web_app.ADCHECK_TIMEOUT_SECONDS == 180.0 and "3 minutes" in web_app.ADCHECK_TIMEOUT_MESSAGE
    now = [0.0]
    gate = threading.Event()
    ui = make_ac_ui(tmp_path, FakeAdcheck(gate=gate), clock=lambda: now[0])
    try:
        nf(ui, f"/videos/{V1}/adcheck")
        assert ui.fn.entered.wait(10)
        now[0] = 181.0
        snap = ui.web.adcheck_job.snapshot()
        assert snap.status == "failed" and snap.message == web_app.ADCHECK_TIMEOUT_MESSAGE
        assert web_app.ADCHECK_TIMEOUT_MESSAGE in card(page(ui))
    finally:
        gate.set()
        ui.web.adcheck_job.wait()


def test_no_key_flashes_and_runs_nothing(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path, key=False)
    resp = nf(ui, f"/videos/{V1}/adcheck")
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#adcheck"
    assert ui.fn.calls == [] and ui.web.adcheck_job.snapshot().status == "idle"
    assert NO_PACKAGING_KEY in ui.client.get(f"/videos/{V1}").text
    with Store(ui.config.catalog_db) as store:
        assert store.latest_ad_check(V1) is None


def test_an_unknown_video_is_a_404_and_runs_nothing(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path)
    resp = nf(ui, "/videos/missing0001/adcheck")
    assert resp.status_code == 404 and "Video not found" in resp.text
    assert ui.fn.calls == []


def test_post_without_csrf_is_refused_and_runs_nothing(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path)
    resp = ui.client.post(f"/videos/{V1}/adcheck", data={}, follow_redirects=False)
    assert resp.status_code == 403
    bad = ui.client.post(f"/videos/{V1}/adcheck", data={"csrf": "wrong"}, follow_redirects=False)
    assert bad.status_code == 403
    assert ui.fn.calls == []
    assert ui.client.post("/adcheck/done", data={}, follow_redirects=False).status_code == 403


def test_api_status_has_adcheck(tmp_path: Path) -> None:
    ui = make_ac_ui(tmp_path)
    status = ui.client.get("/api/status").json()
    assert status["adcheck"] == {"status": "idle", "message": "", "error_kind": ""}


def test_app_js_polls_the_adcheck_job() -> None:
    js = (Path(web_app.__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
    assert '"adcheck"' in js


def test_the_default_check_without_a_key_stores_nothing(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, packaging_key=lambda cfg: True)  # the real default function, no key in the env
    seed_catalog(ui.config, [make_video(1, video_id=V1)])
    run_check(ui)
    snap = ui.web.adcheck_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "auth" and snap.message == NO_PACKAGING_KEY
    with Store(ui.config.catalog_db) as store:
        assert store.latest_ad_check(V1) is None
