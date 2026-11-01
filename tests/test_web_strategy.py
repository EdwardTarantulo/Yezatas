"""Strategy in the web UI: the page, the run job and the saved report. The strategy function is a
fake; nothing touches the network and no real model is called."""

from __future__ import annotations

import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from webhelpers import make_ui, make_video
from yezatas import agent, strategy
from yezatas.store import Store
from yezatas.strategy import StrategyIdea, StrategyPoint, StrategyReport
from yezatas.web import app as web_app
from yezatas.web.app import NO_PACKAGING_KEY

PULLED = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
START = datetime(2025, 1, 1, tzinfo=timezone.utc)
HIGH, LOW = 5, 20  # indexes of the made-up videos far above and far below the norm
SENT_LINE = ("The video table, the outlier findings and up to 30 viewer questions go to Meta's "
             "Model API (muse-spark-1.3). Meta handles them under your Meta account's terms; "
             "on free credits Meta may use them to improve its products. Nothing is sent to YouTube.")


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


def vid(n: int) -> str:
    return f"vid{n:06d}"


def catalog() -> list[Any]:
    """30 made-up public videos with 100 views each (a day apart), one far above and one far below."""
    videos = []
    for n in range(30):
        views = {HIGH: 5000, LOW: 5}.get(n, 100)
        title = {HIGH: "Big cactus hit", LOW: "Quiet cactus video"}.get(n, f"Cactus tip {n}")
        videos.append(make_video(
            n, video_id=vid(n), title=title, privacy_status="public",
            published_at=(START + timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            view_count=views, description=f"A cactus video about topic {n}.", tags=["cactus"]))
    return videos


def seed(config: Any, when: datetime = PULLED) -> int:
    """Store the catalog through one finished pull made at `when`; returns the pull id."""
    videos = catalog()
    with Store(config.catalog_db) as store:
        pull_id = store.begin_pull("UC_fake_channel", when)
        for video in videos:
            store.upsert_video(video, pull_id, when)
        store.finish_pull(pull_id, len(videos), when)
    return pull_id


def canned(as_of: datetime, works: tuple[StrategyPoint, ...] | None = None) -> StrategyReport:
    return StrategyReport(
        works=works if works is not None else (
            StrategyPoint("Big promise titles do well.", (vid(HIGH),)),),
        gaps=(StrategyPoint("Quiet videos get few views.", (vid(LOW),)),),
        ideas=(StrategyIdea("A bigger cactus promise", "It builds on the big hit.", (vid(HIGH),)),),
        notes=("One note kept by code.",), model=strategy.STRATEGY_MODEL,
        input_tokens=2000, cached_tokens=0, output_tokens=300, created_at=as_of.isoformat())


def fake_strategy(store: Any, home: Path, as_of: datetime, report: Any) -> StrategyReport:
    report("Asking the model")
    made = canned(as_of)
    strategy.save_report(home, made)
    return made


def make_strategy_ui(tmp_path: Path, fake: Any = None, key: bool = True, seeded: bool = True,
                     **kwargs: Any) -> Any:
    ui = make_ui(tmp_path, strategy=fake or fake_strategy,
                 packaging_key=lambda cfg: key, **kwargs)
    if seeded:
        seed(ui.config)
    return ui


def page(ui: Any, path: str = "/strategy") -> str:
    resp = ui.client.get(path)
    assert resp.status_code == 200
    return resp.text


def section(html: str, marker: str, end: str = "</section>") -> str:
    return html.split(marker)[1].split(end)[0]


def run(ui: Any) -> Any:
    resp = ui.post("/strategy/run", follow_redirects=False)
    ui.web.strategy_job.wait()
    return resp


# --- the page and the key ------------------------------------------------------------------------
def test_no_pull_yet_goes_to_setup(tmp_path: Path) -> None:
    ui = make_strategy_ui(tmp_path, seeded=False)
    resp = ui.client.get("/strategy", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/setup"


def test_without_a_key_the_button_is_disabled_and_a_run_starts_no_job(tmp_path: Path) -> None:
    ui = make_strategy_ui(tmp_path, key=False)
    html = page(ui)
    assert 'id="strategy-no-key"' in html and 'href="/setup#step-ai"' in html
    assert re.search(r'class="button primary" disabled>Make the strategy report<', html)
    resp = ui.post("/strategy/run", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/strategy"
    assert ui.web.strategy_job.snapshot().status == "idle"
    assert NO_PACKAGING_KEY in page(ui)


def test_the_page_names_what_is_sent_and_what_a_run_costs(tmp_path: Path) -> None:
    ui = make_strategy_ui(tmp_path)
    html = page(ui)
    assert SENT_LINE in html
    assert "One run costs about $" in html
    assert 'action="/strategy/run"' in html and 'name="csrf"' in html
    assert re.search(r'class="button primary">Make the strategy report<', html)


# --- the run and the report ------------------------------------------------------------------------
def test_a_run_shows_the_saved_report(tmp_path: Path) -> None:
    ui = make_strategy_ui(tmp_path)
    resp = run(ui)
    assert resp.status_code == 303 and resp.headers["location"] == "/strategy"
    assert ui.web.strategy_job.snapshot().status == "done"

    html = page(ui)
    works = section(html, 'id="strategy-works"')
    assert "What works" in works and "Big promise titles do well." in works
    assert f'href="/videos/{vid(HIGH)}"' in works and "Big cactus hit" in works
    assert "<strong>50×</strong>" in works
    gaps = section(html, 'id="strategy-gaps"')
    assert "Gaps" in gaps and "Quiet videos get few views." in gaps
    assert f'href="/videos/{vid(LOW)}"' in gaps and "Quiet cactus video" in gaps
    ideas = section(html, 'id="strategy-ideas"')
    assert "Next video ideas" in ideas
    assert "A bigger cactus promise" in ideas and "It builds on the big hit." in ideas
    assert 'id="strategy-notes"' in html and "One note kept by code." in html
    assert "muse-spark-1.3" in html and "Cost $" in html
    assert "2,000 in, 300 out" in html
    assert f"Made {web_app._local_time(PULLED.isoformat())}" in html
    assert 'id="strategy-stale"' not in html


def test_the_running_state_shows_while_the_fake_works(tmp_path: Path) -> None:
    gate = threading.Event()
    entered = threading.Event()

    def slow(store: Any, home: Path, as_of: datetime, report: Any) -> StrategyReport:
        entered.set()
        assert gate.wait(10), "the test never opened the gate"
        return fake_strategy(store, home, as_of, report)

    ui = make_strategy_ui(tmp_path, fake=slow)
    ui.post("/strategy/run", follow_redirects=False)
    assert entered.wait(10)
    try:
        html = page(ui)
        assert 'id="strategy-msg"' in html and 'data-poll="1"' in html
        assert ui.client.get("/api/status").json()["strategy"]["status"] == "running"
    finally:
        gate.set()
        ui.web.strategy_job.wait()
    assert ui.web.strategy_job.snapshot().status == "done"


def test_a_failed_run_shows_its_sentence_until_dismissed(tmp_path: Path) -> None:
    def broken(store: Any, home: Path, as_of: datetime, report: Any) -> StrategyReport:
        raise strategy.StrategyError("The model gave nothing that could be kept. Try again.")

    ui = make_strategy_ui(tmp_path, fake=broken)
    run(ui)
    assert ui.web.strategy_job.snapshot().status == "failed"
    html = page(ui)
    assert 'id="strategy-failed"' in html
    assert "The model gave nothing that could be kept. Try again." in html
    assert ui.post("/strategy/done", follow_redirects=False).headers["location"] == "/strategy"
    assert 'id="strategy-failed"' not in page(ui)


def test_model_text_is_escaped(tmp_path: Path) -> None:
    def evil(store: Any, home: Path, as_of: datetime, report: Any) -> StrategyReport:
        made = canned(as_of, works=(
            StrategyPoint("<script>alert(1)</script>", (vid(HIGH),)),))
        strategy.save_report(home, made)
        return made

    ui = make_strategy_ui(tmp_path, fake=evil)
    run(ui)
    html = page(ui)
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_refs_in_the_text_show_as_titles(tmp_path: Path) -> None:
    def fake(store: Any, home: Path, as_of: datetime, report: Any) -> StrategyReport:
        made = canned(as_of, works=(StrategyPoint("V6 promises a big result, unlike V21.", (vid(HIGH),)),))
        strategy.save_report(home, made)
        return made

    ui = make_strategy_ui(tmp_path, fake=fake)
    run(ui)
    works = section(page(ui), 'id="strategy-works"')
    assert "&#34;Big cactus hit&#34; promises a big result, unlike V21." in works
    assert "V6 promises" not in works


def test_after_a_failed_run_the_older_report_is_labelled(tmp_path: Path) -> None:
    calls: list[int] = []

    def fake(store: Any, home: Path, as_of: datetime, report: Any) -> StrategyReport:
        calls.append(1)
        if len(calls) == 1:
            made = canned(as_of)
            strategy.save_report(home, made)
            return made
        raise strategy.StrategyError("The model gave nothing that could be kept. Try again.")

    ui = make_strategy_ui(tmp_path, fake=fake)
    run(ui)
    assert 'id="strategy-older"' not in page(ui)
    run(ui)
    html = page(ui)
    assert 'id="strategy-failed"' in html
    assert 'id="strategy-older"' in html
    assert f"older report from {web_app._local_time(PULLED.isoformat())}" in html
    ui.post("/strategy/done", follow_redirects=False)
    html = page(ui)
    assert 'id="strategy-failed"' not in html
    assert 'id="strategy-older"' not in html


def test_a_report_from_before_the_last_pull_is_marked_stale(tmp_path: Path) -> None:
    ui = make_strategy_ui(tmp_path)
    run(ui)
    assert 'id="strategy-stale"' not in page(ui)
    seed(ui.config, when=PULLED + timedelta(days=1))
    html = page(ui)
    assert 'id="strategy-stale"' in html
    assert "made before your last pull" in html


# --- status, nav and CSRF --------------------------------------------------------------------------
def test_status_has_a_strategy_job(tmp_path: Path) -> None:
    ui = make_strategy_ui(tmp_path)
    status = ui.client.get("/api/status").json()
    assert "strategy" in status and status["strategy"]["status"] == "idle"


def test_nav_has_a_strategy_item_right_after_outliers(tmp_path: Path) -> None:
    ui = make_strategy_ui(tmp_path)
    html = page(ui)
    items = re.findall(r'<a class="nav-item[^"]*" href="([^"]+)"', html)
    assert items.index("/strategy") == items.index("/outliers") + 1
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html) == ["/strategy"]


@pytest.mark.parametrize("path", ["/strategy/run", "/strategy/done"])
def test_every_post_needs_the_csrf_token(tmp_path: Path, path: str) -> None:
    calls: list[int] = []

    def counting(store: Any, home: Path, as_of: datetime, report: Any) -> StrategyReport:
        calls.append(1)
        return fake_strategy(store, home, as_of, report)

    ui = make_strategy_ui(tmp_path, fake=counting)
    assert ui.client.post(path, data={}, follow_redirects=False).status_code == 403
    assert ui.client.post(path, data={"csrf": "wrong"}, follow_redirects=False).status_code == 403
    assert calls == [] and ui.web.strategy_job.snapshot().status == "idle"


def test_the_page_script_polls_the_strategy_job() -> None:
    js = (Path(web_app.__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
    assert '"strategy"' in js
