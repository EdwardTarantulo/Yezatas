"""Competitors in the web UI: the page, the read job and the explanation. The reader and the
explanation are fakes; nothing touches the network."""

import dataclasses
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from webhelpers import make_ui

from test_competitors import FakeChannelReader, _chan_item
from yezatas import competitors
from yezatas.competitors import ChannelRef, CompetitorChannel, CompetitorInsight
from yezatas.quota import QuotaGuard
from yezatas.store import Store
from yezatas.web.app import NO_PACKAGING_KEY

CID = "UC" + "a" * 22


def make_reader() -> FakeChannelReader:
    ids = [f"fv{i:06d}" for i in range(40)]
    views = {f"fv{i:06d}": 100 for i in range(40)}
    for i in (5, 6, 7):
        views[f"fv{i:06d}"] = 5000
    item = _chan_item(CID, "Sample Fern Corner", "@fern.corner")
    return FakeChannelReader([item], ids, views)


def insight(**override: Any) -> CompetitorInsight:
    base = CompetitorInsight(
        labels={},
        patterns=(),
        per_video=(),
        ideas=(),
        problems=(),
        status="done",
        error="",
        model="muse-spark-1.3",
        input_tokens=0,
        cached_tokens=0,
        output_tokens=0,
        cost_usd=None,
        created_at="2026-06-01T12:00:00+00:00",
    )
    if override:
        base = dataclasses.replace(base, **override)
    return base


def make_competitors_ui(
    tmp_path: Path,
    *,
    signed_in: bool = True,
    key: bool = True,
    explain: Callable[..., Any] | None = None,
    reader: FakeChannelReader | None = None,
) -> tuple[Any, FakeChannelReader]:
    r = reader or make_reader()
    exp_fn = explain or (lambda ch, report: insight())
    ui = make_ui(
        tmp_path,
        credentials_loader=lambda cfg: object() if signed_in else None,
        competitor_reader=lambda creds: r,
        competitor_explain=exp_fn,
        packaging_key=lambda cfg: key,
    )
    return ui, r


def add(ui: Any, text: str = "@fern.corner") -> Any:
    resp = ui.post("/competitors/add", data={"channel": text})
    ui.web.competitors_job.wait()
    return resp


def page(ui: Any) -> str:
    resp = ui.client.get("/competitors")
    assert resp.status_code == 200
    return resp.text


def saved(channel_id: str, title: str = "Saved") -> CompetitorChannel:
    return CompetitorChannel(
        channel_id=channel_id,
        title=title,
        handle="",
        units=1,
        videos=(),
        fetched_at="2026-06-01T12:00:00+00:00",
    )


def test_nav_link(tmp_path: Path) -> None:
    ui, _ = make_competitors_ui(tmp_path)
    html = page(ui)
    items = re.findall(r'<a class="nav-item[^"]*" href="([^"]+)"', html)
    assert items.index("/competitors") == items.index("/translations") + 1
    assert 'id="competitors-about"' in html
    assert "about 9 of your" in html


def test_not_signed_in(tmp_path: Path) -> None:
    ui, reader = make_competitors_ui(tmp_path, signed_in=False)
    html = ui.post("/competitors/add", data={"channel": "@fern.corner"}).text
    ui.web.competitors_job.wait()
    assert "Reading other channels needs the YouTube sign-in" in html
    assert ui.web.competitors_job.snapshot().status == "idle"
    assert reader.calls == []
    assert 'id="competitors-no-sign-in"' in html


def test_bad_text(tmp_path: Path) -> None:
    ui, reader = make_competitors_ui(tmp_path)
    html = ui.post("/competitors/add", data={"channel": "fern"}).text
    ui.web.competitors_job.wait()
    assert "Give a channel handle" in html
    assert ui.web.competitors_job.snapshot().status == "idle"
    assert reader.calls == []


def test_add_shows_card_and_quota(tmp_path: Path) -> None:
    ui, reader = make_competitors_ui(tmp_path)
    add(ui)
    assert reader.calls[0] == ("channel", ChannelRef("handle", "@fern.corner"))
    chans = competitors.load_channels(ui.config.home)
    assert len(chans) == 1
    html = page(ui)
    assert "Sample Fern Corner" in html
    assert "https://www.youtube.com/watch?v=fv000005" in html
    assert "50×" in html
    assert ">Refresh<" in html
    assert ">Remove<" in html
    with Store(ui.config.catalog_db) as store:
        used = QuotaGuard(store, ui.config.daily_quota_budget).used()
    assert used == 3


def test_full_limit(tmp_path: Path) -> None:
    ui, reader = make_competitors_ui(tmp_path)
    for c in "bcdef":
        competitors.save_channel(ui.config.home, saved("UC" + c * 22))
    html = page(ui)
    assert 'id="competitors-full"' in html
    assert re.search(r'class="button primary" disabled>Add\n\s*channel<', html) or re.search(
        r'class="button primary" disabled>Add channel<', html
    )
    html2 = ui.post("/competitors/add", data={"channel": "@fern.corner"}).text
    ui.web.competitors_job.wait()
    assert "at most 5" in html2
    assert ui.web.competitors_job.snapshot().status == "idle"
    assert reader.calls == []


def test_refresh(tmp_path: Path) -> None:
    ui, reader = make_competitors_ui(tmp_path)
    html = ui.post("/competitors/refresh", data={"channel_id": "UCxxxx"}).text
    ui.web.competitors_job.wait()
    assert "not on your list" in html
    add(ui)
    reader.calls.clear()
    ui.post("/competitors/refresh", data={"channel_id": CID})
    ui.web.competitors_job.wait()
    assert reader.calls[0] == ("channel", ChannelRef("id", CID))


def test_remove(tmp_path: Path) -> None:
    ui, _ = make_competitors_ui(tmp_path)
    add(ui)
    html = ui.post("/competitors/remove", data={"channel_id": CID}).text
    ui.web.competitors_job.wait()
    assert "The channel was removed from your list." in html
    assert competitors.load_channels(ui.config.home) == []
    html2 = page(ui)
    assert 'id="competitor-UCaaaa' not in html2


def test_explain_without_key(tmp_path: Path) -> None:
    ui, _ = make_competitors_ui(tmp_path, key=False)
    add(ui)
    html = ui.post("/competitors/explain", data={}).text
    ui.web.competitors_explain_job.wait()
    assert NO_PACKAGING_KEY in html
    assert ui.web.competitors_explain_job.snapshot().status == "idle"


def test_explain_with_fake(tmp_path: Path) -> None:
    captured: list[list[CompetitorChannel]] = []

    def fake_explain(
        chs: list[CompetitorChannel], report: Any
    ) -> CompetitorInsight:
        captured.append(list(chs))
        # compute outliers for first channel
        outs = competitors.channel_outliers(chs[0])
        o1 = outs[0]
        o2 = outs[1]
        labels = {
            "O1": {
                "title": o1.video.title,
                "video_id": o1.video.video_id,
                "channel_title": o1.channel_title,
            },
            "O2": {
                "title": o2.video.title,
                "video_id": o2.video.video_id,
                "channel_title": o2.channel_title,
            },
        }
        patterns = (
            {"text": "Short how-to titles", "kind": "title", "evidence": ["O1", "O2"]},
        )
        per_video = ({"label": "O1", "why": "A clear promise"},)
        made = CompetitorInsight(
            labels=labels,
            patterns=patterns,
            per_video=per_video,
            ideas=("Try a question title",),
            problems=("One line was dropped.",),
            status="done",
            error="",
            model="muse-spark-1.3",
            input_tokens=2000,
            cached_tokens=0,
            output_tokens=300,
            cost_usd=0.004,
            created_at="2026-06-01T12:00:00+00:00",
        )
        competitors.save_insight(ui_ref["ui"].config.home, made)
        return made

    ui_ref: dict[str, Any] = {}
    ui, _ = make_competitors_ui(tmp_path, explain=fake_explain)
    ui_ref["ui"] = ui
    add(ui)
    ui.post("/competitors/explain", data={})
    ui.web.competitors_explain_job.wait()
    assert captured
    assert captured[0][0].channel_id == CID
    html = page(ui)
    assert "Short how-to titles" in html
    assert "A clear promise" in html
    assert "Try a question title" in html
    assert "One line was dropped." in html
    assert "2,000 in, 300 out" in html
    assert "Cost $0.004" in html
    # O1 title link
    outs2 = competitors.channel_outliers(
        competitors.load_channels(ui.config.home)[0]
    )
    o1_title = outs2[0].video.title
    assert o1_title in html
    assert f"https://www.youtube.com/watch?v={outs2[0].video.video_id}" in html


def test_insight_statuses(tmp_path: Path) -> None:
    ui, _ = make_competitors_ui(tmp_path)
    add(ui)
    competitors.save_insight(
        ui.config.home,
        insight(status="too_few"),
    )
    html = page(ui)
    assert "Fewer than 2 outliers" in html
    competitors.save_insight(
        ui.config.home,
        insight(status="error", error="The model's answer was incomplete."),
    )
    html2 = page(ui)
    assert "answer was incomplete" in html2


def test_api_status(tmp_path: Path) -> None:
    ui, _ = make_competitors_ui(tmp_path)
    resp = ui.client.get("/api/status", headers={"X-CSRF-Token": ui.csrf()})
    assert resp.status_code == 200
    data = resp.json()
    assert "competitors" in data
    assert "competitors_explain" in data


def test_escaping(tmp_path: Path) -> None:
    ui, _ = make_competitors_ui(tmp_path)
    competitors.save_channel(ui.config.home, saved(CID, "<b>x</b>"))
    html = page(ui)
    assert "&lt;b&gt;x&lt;/b&gt;" in html
    assert "<b>x</b>" not in html
