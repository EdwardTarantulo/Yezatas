"""Shorts finder card on the video page. No model or network call."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from markupsafe import escape

from webhelpers import make_ui, make_video, seed_catalog, seed_transcript
from yezatas import shorts
from yezatas.store import ShortsFind, Store
from yezatas.translate import Completion, TranslateError
from yezatas.web import app as web_app
from yezatas.web.app import NO_SHORTS_KEY
from yezatas.web.views import SHORTS_STALE_LINE

V1, V2 = "vid00000001", "vid00000002"
NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BAS_API_KEY", raising=False)


# --- fakes -------------------------------------------------------------------

GOOD_SHORT = (
    "<short><start>1</start><end>1</end>"
    "<title>Cactus soil trick</title><why>One clear tip.</why></short>"
)


class FakeText:
    """Scripted text model."""

    model = "spark-open-test"

    def __init__(self, answers: list[Any] | None = None) -> None:
        self.answers: list[Any] = answers or [GOOD_SHORT]
        self.calls = 0

    def complete(self, messages: Any) -> Completion:
        ans = self.answers[self.calls] if self.calls < len(self.answers) else self.answers[-1]
        self.calls += 1
        if isinstance(ans, BaseException):
            raise ans
        return Completion(ans, 900, 120)


class FakeShorts:
    """Injected shorts function using the real shorts.find_shorts."""

    def __init__(
        self,
        script: Any | None = None,
        gate: threading.Event | None = None,
        error: Exception | None = None,
    ) -> None:
        if isinstance(script, list):
            self.client = FakeText(script)
        elif isinstance(script, str):
            self.client = FakeText([script])
        elif script is None:
            self.client = FakeText([GOOD_SHORT])
        else:
            self.client = FakeText([GOOD_SHORT])
        self.gate = gate
        self.error = error
        self.entered = threading.Event()
        self.calls: list[str] = []

    def __call__(self, store: Store, video_id: str, now: datetime, report: Any) -> ShortsFind:
        self.calls.append(video_id)
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10), "the test never opened the gate"
        if self.error is not None:
            raise self.error
        return shorts.find_shorts(store, self.client, video_id, now=now, report=report)


def make_sh_ui(tmp_path: Path, fn: Any = "default", key: Any = True, **kw: Any) -> Any:
    fn = FakeShorts() if fn == "default" else fn
    ui = make_ui(tmp_path, shorts=fn, shorts_key=lambda cfg: key, **kw)
    seed_catalog(
        ui.config,
        [
            make_video(1, video_id=V1, title="Cactus repotting guide", privacy_status="public"),
            make_video(2, video_id=V2, title="Second video", privacy_status="public"),
        ],
    )
    seed_transcript(ui.config, V1, segments=6)
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
    return html.split('id="shorts"')[1].split("</section>")[0]


def run_find(ui: Any, video_id: str = V1) -> Any:
    resp = nf(ui, f"/videos/{video_id}/shorts")
    ui.web.shorts_job.wait()
    return resp


# --- tests -------------------------------------------------------------------

def test_card_renders(tmp_path: Path) -> None:
    ui = make_sh_ui(tmp_path)
    sec = card(page(ui))
    assert "Find Shorts" in sec
    sec2 = card(page(ui, V2))
    assert 'id="shorts-no-transcript"' in sec2
    assert "disabled" in sec2
    ui2 = make_sh_ui(tmp_path, key=False)
    sec3 = card(page(ui2))
    assert 'id="shorts-no-key"' in sec3
    assert "disabled" in sec3


def test_post_redirects_and_shows_result(tmp_path: Path) -> None:
    ui = make_sh_ui(tmp_path)
    resp = run_find(ui)
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/videos/{V1}#shorts"
    assert ui.fn.calls == [V1]
    sec = card(page(ui))
    assert "Cactus soil trick" in sec
    assert "0:50 \u2013 1:40 (50 s)" in sec
    assert "Piece 1 talks about topic 1." in sec
    assert "https://www.youtube.com/watch?v=vid00000001&amp;t=50s" in sec
    assert "Find again" in sec


def test_running_job_shows_poll_and_busy(tmp_path: Path) -> None:
    gate = threading.Event()
    ui = make_sh_ui(tmp_path, FakeShorts(gate=gate))
    try:
        nf(ui, f"/videos/{V1}/shorts")
        assert ui.fn.entered.wait(10)
        html = page(ui)
        assert 'id="shorts-msg"' in html
        assert 'data-poll="1"' in html
        other = card(page(ui, V2))
        assert 'id="shorts-busy"' in other
        again = nf(ui, f"/videos/{V1}/shorts")
        assert again.status_code == 303
        assert "already running" in ui.client.get(f"/videos/{V1}").text.lower()
        assert ui.client.get("/api/status").json()["shorts"]["status"] == "running"
    finally:
        gate.set()
        ui.web.shorts_job.wait()
    assert ui.fn.calls == [V1]


def test_model_failure_shown(tmp_path: Path) -> None:
    ui = make_sh_ui(tmp_path, FakeShorts(script=[TranslateError("rate_limit", "Slow down, then try again.")]))
    run_find(ui)
    sec = card(page(ui))
    assert "Slow down, then try again." in sec
    assert "Try again" in sec


def test_unexpected_error_and_dismiss(tmp_path: Path) -> None:
    ui = make_sh_ui(tmp_path, FakeShorts(error=KeyError("x")))
    run_find(ui)
    snap = ui.web.shorts_job.snapshot()
    assert snap.status == "failed"
    assert snap.message == "The Shorts finder stopped unexpectedly (KeyError)."
    assert 'id="shorts-failed"' in card(page(ui))
    resp = nf(ui, "/shorts/done", data={"video_id": V1})
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/videos/{V1}#shorts"
    assert 'id="shorts-failed"' not in card(page(ui))


def test_stale_line(tmp_path: Path) -> None:
    ui = make_sh_ui(tmp_path)
    run_find(ui)
    seed_transcript(ui.config, V1, segments=6)
    sec = card(page(ui))
    assert SHORTS_STALE_LINE in sec


def test_escaping(tmp_path: Path) -> None:
    evil = (
        "<short><start>1</start><end>1</end>"
        "<title><script>x</script> Mesa</title><why>One clear tip.</why></short>"
    )
    ui = make_sh_ui(tmp_path, FakeShorts(script=evil))
    run_find(ui)
    html = page(ui)
    assert "<script>x" not in html
    assert "Mesa" in html


def test_no_key_flashes_and_runs_nothing(tmp_path: Path) -> None:
    ui = make_sh_ui(tmp_path, key=False)
    resp = nf(ui, f"/videos/{V1}/shorts")
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/videos/{V1}#shorts"
    assert ui.fn.calls == []
    assert ui.web.shorts_job.snapshot().status == "idle"
    assert NO_SHORTS_KEY in ui.client.get(f"/videos/{V1}").text


def test_unknown_video_404(tmp_path: Path) -> None:
    ui = make_sh_ui(tmp_path)
    resp = nf(ui, "/videos/missing0001/shorts")
    assert resp.status_code == 404
    assert "Video not found" in resp.text
    assert ui.fn.calls == []


def test_csrf(tmp_path: Path) -> None:
    ui = make_sh_ui(tmp_path)
    resp = ui.client.post(f"/videos/{V1}/shorts", data={}, follow_redirects=False)
    assert resp.status_code == 403
    bad = ui.client.post(f"/videos/{V1}/shorts", data={"csrf": "wrong"}, follow_redirects=False)
    assert bad.status_code == 403
    assert ui.fn.calls == []
    assert ui.client.post("/shorts/done", data={}, follow_redirects=False).status_code == 403


def test_api_status_has_shorts(tmp_path: Path) -> None:
    ui = make_sh_ui(tmp_path)
    status = ui.client.get("/api/status").json()
    assert status["shorts"] == {"status": "idle", "message": "", "error_kind": ""}


def test_app_js_contains_shorts(tmp_path: Path) -> None:
    js = (Path(web_app.__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
    assert '"shorts"' in js


def test_map_shorts_error() -> None:
    assert web_app.map_shorts_error(TranslateError("auth", "x")) == ("auth", "x")
    assert web_app.map_shorts_error(ValueError("y")) == ("shorts", "y")
    kind, text = web_app.map_shorts_error(KeyError("boom"))
    assert kind == "shorts"
    assert "stopped unexpectedly" in text
    assert "KeyError" in text
