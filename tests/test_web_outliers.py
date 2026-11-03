"""Outliers in the web UI: the page, the explain job, the report with its evidence chips, the video page
line, the dashboard count and the nav item. The model and the thumbnail download are fakes; nothing
touches the network, and nothing here ever calls YouTube."""

from __future__ import annotations

import io
import json
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from fakes import FakeModelClient
from webhelpers import make_ui, make_video
from yezatas import agent, outliers
from yezatas.agent import ModelError, ModelReply
from yezatas.modelclient import AUTH_MESSAGE
from yezatas.store import Store
from yezatas.web import app as web_app
from yezatas.web.app import NO_PACKAGING_KEY

PULLED = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
START = datetime(2025, 1, 1, tzinfo=timezone.utc)
HIGH, MID, LOW = 5, 15, 20  # indexes of the made-up videos far above, above and far below the norm
TITLES = {HIGH: "Big cactus hit", MID: "Second cactus hit", LOW: "Quiet cactus video"}


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


def vid(n: int) -> str:
    return f"vid{n:06d}"


def catalog() -> list[Any]:
    """30 made-up public videos with 100 views each (a day apart), three of them far from the norm, and
    three that cannot be scored: a private one, a live stream and one only 10 days old."""
    videos = []
    for n in range(30):
        views = {HIGH: 5000, MID: 2000, LOW: 5}.get(n, 100)
        videos.append(make_video(
            n, video_id=vid(n), title=TITLES.get(n, f"Cactus tip {chr(65 + n)}"), privacy_status="public",
            published_at=(START + timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ"), view_count=views,
            description=f"A cactus video about topic {chr(65 + n)}.", tags=["cactus"]))
    videos.append(make_video(100, video_id=vid(100), title="Private draft", privacy_status="private",
                             published_at="2025-02-01T10:00:00Z", view_count=7))
    videos.append(make_video(101, video_id=vid(101), title="Live stream", privacy_status="public",
                             published_at="2025-02-02T10:00:00Z", duration="P0D", view_count=0))
    videos.append(make_video(102, video_id=vid(102), title="Brand new", privacy_status="public",
                             published_at=(PULLED - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             view_count=40))
    return videos


def seed(config: Any, videos: list[Any] | None = None, when: datetime = PULLED) -> int:
    """Store the videos through one finished pull made at `when`; returns the pull id."""
    with Store(config.catalog_db) as store:
        pull_id = store.begin_pull("UC_fake_channel", when)
        for video in videos if videos is not None else catalog():
            store.upsert_video(video, pull_id, when)
        store.finish_pull(pull_id, len(videos) if videos is not None else 33, when)
    return pull_id


def picture() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (320, 180), (200, 200, 200)).save(out, "JPEG")
    return out.getvalue()


class Fetch:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        return picture()


def answer(data: dict[str, Any] | None = None) -> ModelReply:
    data = data if data is not None else {
        "patterns": [{"text": "Both titles promise a big result.", "evidence": ["O1", "O2"], "kind": "title"}],
        "per_video": [{"label": "O1", "why": "The title promises a big result."},
                      {"label": "O2", "why": "A short, direct title."}],
        "ideas": ["Try a big promise in the next title."]}
    return ModelReply(json.dumps(data), (), {}, 2000, 0, 300)


def make_ol_ui(tmp_path: Path, script: list[Any] | None = None, key: bool = True, seeded: bool = True,
               **kwargs: Any) -> tuple[Any, FakeModelClient, Fetch]:
    client = FakeModelClient(script if script is not None else [answer()])
    fetch = Fetch()
    tools = SimpleNamespace(vision=client)
    ui = make_ui(tmp_path, packaging_tools=(lambda cfg: tools) if key else (lambda cfg: None),
                 packaging_fetch=fetch, **kwargs)
    if seeded:
        seed(ui.config)
    return ui, client, fetch


def page(ui: Any, path: str = "/outliers") -> str:
    resp = ui.client.get(path)
    assert resp.status_code == 200
    return resp.text


def explain(ui: Any) -> Any:
    resp = ui.post("/outliers/explain", follow_redirects=False)
    ui.web.outliers_job.wait()
    return resp


def body_rows(table: str) -> list[str]:
    return table.split("<tbody>")[1].split("<tr>")[1:]


def section(html: str, marker: str, end: str = "</section>") -> str:
    return html.split(marker)[1].split(end)[0]


# --- the page ------------------------------------------------------------------------------------
def test_page_shows_the_line_the_outliers_and_the_collapsed_below_norm_rows(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path)
    html = page(ui)
    assert "Videos that beat your channel's norm for their age" in html
    about = section(html, 'id="outliers-about"', "</p>")
    assert f"median views of the {outliers.PEERS} public videos published nearest to it" in about
    assert f"Views as of your last pull on {web_app._local_time(PULLED.isoformat())}." in about
    assert f"younger than {outliers.MIN_AGE_DAYS} days" in about
    assert ">3 left out<" in about  # the private one, the live stream and the 10 day old video

    table = section(html, 'id="outliers-list"')
    assert "Outliers (10× or more)" in table
    assert table.index("Big cactus hit") < table.index("Second cactus hit")  # highest ratio first
    assert "Quiet cactus video" not in table
    first = body_rows(table)[0]
    assert f'href="/videos/{vid(HIGH)}"' in first
    assert f"https://i.ytimg.com/vi/{vid(HIGH)}/mqdefault.jpg" in first
    assert "5,000" in first and ">100<" in first  # views and the peers' median
    assert "<strong>50×</strong>" in first
    assert "Jan 6, 2025" in first
    assert "<strong>20×</strong>" in table

    below = section(html, 'id="outliers-below"', "</details>")
    assert html.count("<details") == 1 and '<details class="card ol-below" id="outliers-below">' in html
    assert "Below the norm (0.1× or less)" in below
    assert "Quiet cactus video" in below
    assert "Big cactus hit" not in below


def test_below_norm_shows_the_small_ratio_with_one_decimal(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path)
    below = section(page(ui), 'id="outliers-below"', "</details>")
    assert f"<strong>{outliers.ratio_label(5 / 100)}</strong>" in below


def test_no_outliers_and_no_below_rows_say_so(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path, seeded=False)
    seed(ui.config, [v for v in catalog() if v.video_id not in (vid(HIGH), vid(MID), vid(LOW))])
    html = page(ui)
    assert 'id="no-outliers"' in html and 'id="no-below"' in html
    assert "0 videos" in html


def test_no_pull_yet_goes_to_setup(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path, seeded=False)
    resp = ui.client.get("/outliers", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/configuration"
    assert ui.post("/outliers/explain", follow_redirects=False).headers["location"] == "/configuration"


def test_titles_are_escaped(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path, seeded=False)
    videos = catalog()
    videos[HIGH] = make_video(HIGH, video_id=vid(HIGH), title="<script>alert(1)</script>",
                              privacy_status="public", published_at=videos[HIGH].published_at,
                              view_count=5000)
    seed(ui.config, videos)
    html = page(ui)
    assert "<script>alert" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_the_page_has_no_inline_scripts_or_styles(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path)
    explain(ui)
    html = page(ui)
    assert " style=" not in html and "<style" not in html
    assert not re.search(r"<script(?![^>]*\bsrc=)", html)


def test_nav_has_an_outliers_item_after_affiliate(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path)
    html = page(ui)
    items = re.findall(r'<a class="nav-item[^"]*" href="([^"]+)"', html)
    assert items.index("/outliers") == items.index("/replies") + 1
    assert items.index("/strategy") == items.index("/competitors") + 1
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html) == ["/outliers"]
    assert re.findall(r'class="nav-item active" href="([^"]+)"', page(ui, "/videos")) == ["/videos"]


# --- the explain button and the job --------------------------------------------------------------
def test_the_button_is_disabled_without_a_key_and_points_to_setup(tmp_path: Path) -> None:
    ui, client, _ = make_ol_ui(tmp_path, key=False)
    card = section(page(ui), 'id="why"')
    assert 'id="outliers-no-key"' in card and 'href="/configuration#step-ai"' in card
    assert re.search(r'class="button primary" disabled>Explain with Muse Spark<', card)
    resp = explain(ui)
    assert resp.status_code == 303 and resp.headers["location"] == "/outliers#why"
    assert NO_PACKAGING_KEY in page(ui)  # the flash
    assert ui.web.outliers_job.snapshot().status == "idle" and client.calls == []


def test_the_button_is_enabled_with_a_key(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path)
    card = section(page(ui), 'id="why"')
    assert 'id="outliers-no-key"' not in card
    assert re.search(r'class="button primary">Explain with Muse Spark<', card)
    assert 'action="/outliers/explain"' in card and 'name="csrf"' in card
    assert f"first {outliers.DESCRIPTION_CHARS} characters" in card
    assert "Nothing is sent to YouTube" in card
    assert "Last explanation" not in card


def test_explaining_runs_the_job_and_shows_the_report(tmp_path: Path) -> None:
    ui, client, fetch = make_ol_ui(tmp_path)
    resp = explain(ui)
    assert resp.status_code == 303 and resp.headers["location"] == "/outliers#why"
    assert ui.web.outliers_job.snapshot().status == "done"
    assert len(client.calls) == 1
    sent = json.dumps(client.calls[0]["messages"])
    assert vid(HIGH) not in sent  # the model only sees labels
    assert fetch.urls  # thumbnails go along (fetched with the injected downloader)

    with Store(ui.config.catalog_db) as store:
        saved = store.latest_outlier_report()
        pull = store.latest_pull()
    assert saved is not None and saved.status == "done" and saved.pull_id == int(pull["id"])
    assert saved.as_of == PULLED.isoformat()  # the time of the pull, not now

    html = page(ui)
    report = section(html, 'id="outliers-report"', "</section>")
    assert "Last explanation: " in report
    assert "Both titles promise a big result." in report
    chips = re.findall(r'<a class="chip" href="([^"]+)"><span class="ol-label">(O\d)</span>&nbsp;([^<]+)</a>',
                       report)
    assert chips == [(f"/videos/{vid(HIGH)}", "O1", "Big cactus hit"),
                     (f"/videos/{vid(MID)}", "O2", "Second cactus hit")]
    assert "Try a big promise in the next title." in report
    assert "older pull" not in html
    assert "Cost $" in report and "muse-spark-1.3" in report and "2,000 in, 300 out tokens" in report
    # the per-video lines sit next to their rows
    table = section(html, 'id="outliers-list"')
    first, second = body_rows(table)[:2]
    assert "The title promises a big result." in first and "A short, direct title." in second


def test_an_older_report_is_shown_with_a_note_and_without_the_row_lines(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path)
    explain(ui)
    seed(ui.config, when=PULLED + timedelta(days=1))  # a newer pull: the report is now from an older one
    html = page(ui)
    assert 'id="outliers-old"' in html and "made from an older pull" in html
    assert "Both titles promise a big result." in section(html, 'id="outliers-report"', "</section>")
    assert "The title promises a big result." not in section(html, 'id="outliers-list"')


def test_fewer_than_two_outliers_makes_no_model_call(tmp_path: Path) -> None:
    ui, client, _ = make_ol_ui(tmp_path, seeded=False)
    seed(ui.config, [v for v in catalog() if v.video_id != vid(MID)])  # only one outlier left
    explain(ui)
    assert client.calls == []
    report = section(page(ui), 'id="outliers-report"', "</section>")
    assert 'id="outliers-too-few"' in report and "fewer than 2 outliers" in report


def test_a_model_failure_shows_the_plain_message(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path, script=[ModelError("server", "The model service had a problem.")])
    explain(ui)
    report = section(page(ui), 'id="outliers-report"', "</section>")
    assert 'id="outliers-error"' in report and "The model service had a problem." in report
    assert "What the outliers share" not in report


def test_guard_problems_are_listed_under_the_patterns(tmp_path: Path) -> None:
    data = {"patterns": [{"text": "Both titles promise a big result.", "evidence": ["O1", "O2"],
                          "kind": "title"},
                         {"text": "Lone pattern.", "evidence": ["O1"], "kind": "other"}],
            "per_video": [], "ideas": []}
    ui, _, _ = make_ol_ui(tmp_path, script=[answer(data)])
    explain(ui)
    report = section(page(ui), 'id="outliers-report"', "</section>")
    assert 'id="outliers-problems"' in report and "cites fewer than 2 of the outliers" in report
    assert "Lone pattern." not in report.split('id="outliers-problems"')[0]


def test_a_refused_key_fails_the_job_and_can_be_dismissed(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path, script=[ModelError("auth", AUTH_MESSAGE)])
    explain(ui)
    snap = ui.web.outliers_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "auth"
    html = page(ui)
    assert 'id="outliers-failed"' in html and AUTH_MESSAGE in html
    with Store(ui.config.catalog_db) as store:
        assert store.latest_outlier_report() is None  # nothing stored for a refused key
    assert ui.post("/outliers/done", follow_redirects=False).headers["location"] == "/outliers#why"
    assert 'id="outliers-failed"' not in page(ui)


def test_a_second_explain_while_one_runs_is_refused(tmp_path: Path) -> None:
    gate = threading.Event()
    entered = threading.Event()

    def slow(messages: Any, tools: Any, response_format: Any) -> ModelReply:
        entered.set()
        assert gate.wait(10), "the test never opened the gate"
        return answer()

    ui, client, _ = make_ol_ui(tmp_path, script=[slow])
    ui.post("/outliers/explain", follow_redirects=False)
    assert entered.wait(10)
    try:
        html = page(ui)
        assert 'id="outliers-msg"' in html and 'data-poll="1"' in html
        assert not re.search(r'class="button primary"[^>]*>Explain with Muse Spark<', html)  # form is hidden
        assert ui.client.get("/api/status").json()["outliers"]["status"] == "running"
        ui.post("/outliers/explain", follow_redirects=False)
        assert "already running" in page(ui)
        assert len(client.calls) == 1
    finally:
        gate.set()
        ui.web.outliers_job.wait()
    assert ui.client.get("/api/status").json()["outliers"]["status"] == "done"


@pytest.mark.parametrize("path", ["/outliers/explain", "/outliers/done"])
def test_every_post_needs_the_csrf_token(tmp_path: Path, path: str) -> None:
    ui, client, _ = make_ol_ui(tmp_path)
    assert ui.client.post(path, data={}, follow_redirects=False).status_code == 403
    assert ui.client.post(path, data={"csrf": "wrong"}, follow_redirects=False).status_code == 403
    assert client.calls == []


def test_the_page_script_polls_the_outliers_job() -> None:
    js = (Path(web_app.__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
    assert '"outliers"' in js


# --- the video page and the dashboard ------------------------------------------------------------
def test_the_video_page_says_how_far_above_its_norm_the_video_is(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path)
    html = page(ui, f"/videos/{vid(HIGH)}")
    line = section(html, 'id="video-outlier"', "</p>")
    assert 'href="/outliers"' in line
    assert "50× the channel norm for its age (median 100 views of 20 videos published around it)" in line
    typical = section(page(ui, f"/videos/{vid(0)}"), 'id="video-outlier"', "</p>")
    assert "1.0× the channel norm for its age (median 100 views of 20 videos" in typical


def test_the_video_page_line_for_a_new_video_and_for_one_left_out(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path)
    new = page(ui, f"/videos/{vid(102)}")
    assert f"Too new to compare (under {outliers.MIN_AGE_DAYS} days)" in new
    assert 'href="/outliers"' not in section(new, 'id="video-outlier"', "</p>")
    for left_out in (vid(100), vid(101)):  # private, live stream
        assert 'id="video-outlier"' not in page(ui, f"/videos/{left_out}")


def test_the_dashboard_counts_the_outliers(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path)
    line = section(page(ui, "/seo"), 'id="dash-outliers"', "</p>")
    assert '<a href="/outliers">2 outliers</a>' in line and "10× or more" in line


def test_the_dashboard_says_one_outlier_in_the_singular(tmp_path: Path) -> None:
    ui, _, _ = make_ol_ui(tmp_path, seeded=False)
    seed(ui.config, [v for v in catalog() if v.video_id != vid(MID)])
    assert '<a href="/outliers">1 outlier</a>' in page(ui, "/seo")
