"""Search gaps in the web UI: the page, the sign-in, the read job and the ranking. The sign-in, the
Analytics reader and the model are fakes; nothing touches the network."""

import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest

from test_searchgaps import FakeAnalyticsReader
from webhelpers import http_error, make_ui, make_video, seed_catalog, write_client_file, write_token
from yezatas import auth, searchgaps
from yezatas.web.app import NO_PACKAGING_KEY


def ranking(**override: Any) -> searchgaps.GapRanking:
    base = searchgaps.GapRanking(
        picks=(),
        problems=(),
        status="done",
        error="",
        model="muse-spark",
        input_tokens=0,
        cached_tokens=0,
        output_tokens=0,
        cost_usd=None,
        created_at="2026-06-01T12:00:00+00:00",
        terms_fetched_at="",
    )
    if override:
        base = dataclasses.replace(base, **override)
    return base


def make_sg_ui(
    tmp_path: Path, *, connected: bool = True, key: bool = True, reader: Any = None, rank: Any = None,
    sign_in: Any = None, seed: bool = True
) -> tuple[Any, Any]:
    r = reader or FakeAnalyticsReader(
        {
            None: [("cactus soil recipe", 300)],
            "vid000001": [("repot cactus winter", 120), ("repotting cactus", 80)],
        }
    )

    def default_rank(terms, gaps, report):
        return ranking(terms_fetched_at=terms.fetched_at if terms else "")

    ui = make_ui(
        tmp_path,
        analytics_credentials_loader=lambda cfg: object() if connected else None,
        analytics_reader=lambda creds: r,
        gap_rank=rank or default_rank,
        packaging_key=lambda cfg: key,
        analytics_sign_in=sign_in or (lambda cfg: None),
    )
    if connected:
        write_token(ui.config, [auth.READONLY, auth.ANALYTICS])
    if seed:
        seed_catalog(ui.config, [make_video(n) for n in range(6)])
    return ui, r


def test_nav(tmp_path: Path) -> None:
    ui, _ = make_sg_ui(tmp_path)
    html = ui.client.get("/search-gaps").text
    items = re.findall(r'<a class="nav-item[^"]*" href="([^"]+)"', html)
    assert items.index("/search-gaps") == items.index("/outliers") + 1
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html) == ["/search-gaps"]
    assert 'id="searchgaps-about"' in html


def test_not_connected(tmp_path: Path) -> None:
    ui, reader = make_sg_ui(tmp_path, connected=False)
    html = ui.client.get("/search-gaps").text
    assert 'id="searchgaps-connect-hint"' in html
    assert "Connect search terms" in html
    html2 = ui.post("/search-gaps/read", data={}).text
    assert "needs the search-terms sign-in" in html2
    assert reader.calls == []


def test_connect(tmp_path: Path) -> None:
    calls: list[Any] = []

    def fake_sign_in(cfg):
        calls.append(cfg)
        return None

    ui, _ = make_sg_ui(tmp_path, connected=True, sign_in=fake_sign_in)
    html = ui.post("/search-gaps/connect", data={}).text
    assert "Add the OAuth client file first" in html
    assert calls == []
    write_client_file(ui.config)
    ui.post("/search-gaps/connect", data={})
    ui.web.searchgaps_signin_job.wait()
    assert len(calls) == 1
    assert calls[0] is ui.config


def test_read(tmp_path: Path) -> None:
    ui, reader = make_sg_ui(tmp_path)
    ui.post("/search-gaps/read", data={})
    ui.web.searchgaps_job.wait()
    html = ui.client.get("/search-gaps").text
    section = html.split('id="searchgaps-gaps"')[1].split("</section>")[0]
    assert "2. Gaps (2)" in section
    assert "<td>repot cactus winter</td>" in section
    assert "<td>repot, winter</td>" in section
    assert "<td>cactus soil recipe</td>" in section
    assert "No video has these words" in section
    assert '<td class="col-num">120</td>' in section
    assert "repotting cactus" not in section
    calls = reader.calls
    assert calls[0][2] is None
    assert len(calls) == 1 + 4


def test_read_without_catalog(tmp_path: Path) -> None:
    ui, _ = make_sg_ui(tmp_path, seed=False)
    ui.post("/search-gaps/read", data={})
    ui.web.searchgaps_job.wait()
    html = ui.client.get("/search-gaps").text
    assert "There is no pulled catalog yet" in html


def test_analytics_api_disabled(tmp_path: Path) -> None:
    class BadReader:
        def search_terms(self, start_date, end_date, video_id, max_results):
            raise http_error(403, "accessNotConfigured", "YouTube Analytics API has not been used in project 1 before")

    ui, _ = make_sg_ui(tmp_path, reader=BadReader())
    ui.post("/search-gaps/read", data={})
    ui.web.searchgaps_job.wait()
    html = ui.client.get("/search-gaps").text
    assert "YouTube Analytics API is not enabled" in html


def test_rank_without_key(tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_rank(terms, gaps, report):
        calls.append(1)
        return ranking(terms_fetched_at=terms.fetched_at)

    ui, _ = make_sg_ui(tmp_path, key=False, rank=fake_rank)
    html = ui.post("/search-gaps/rank", data={}).text
    assert NO_PACKAGING_KEY in html
    assert calls == []


def test_rank_with_fake(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_rank(terms, gaps, report):
        captured["terms"] = terms
        captured["gaps"] = gaps
        made = ranking(
            picks=(searchgaps.RankedGap(label="G1", gap=gaps[0], priority="high", reason="Add the word winter."),),
            terms_fetched_at=terms.fetched_at,
        )
        searchgaps.save_ranking(ui.config.home, made)
        return made

    ui, _ = make_sg_ui(tmp_path, rank=fake_rank)
    ui.post("/search-gaps/read", data={})
    ui.web.searchgaps_job.wait()
    ui.post("/search-gaps/rank", data={})
    ui.web.searchgaps_rank_job.wait()
    html = ui.client.get("/search-gaps").text
    assert "high" in html
    assert "Add the word winter." in html
    assert 'id="searchgaps-picks"' in html
    assert captured["terms"] is not None
    assert isinstance(captured["gaps"], list)


def test_ranking_error(tmp_path: Path) -> None:
    def fake_rank(terms, gaps, report):
        made = ranking(status="error", error="The model was busy.", terms_fetched_at=terms.fetched_at)
        searchgaps.save_ranking(ui.config.home, made)
        return made

    ui, _ = make_sg_ui(tmp_path, rank=fake_rank)
    ui.post("/search-gaps/read", data={})
    ui.web.searchgaps_job.wait()
    ui.post("/search-gaps/rank", data={})
    ui.web.searchgaps_rank_job.wait()
    html = ui.client.get("/search-gaps").text
    assert "The model was busy." in html


def test_api_status(tmp_path: Path) -> None:
    ui, _ = make_sg_ui(tmp_path)
    data = ui.client.get("/api/status").json()
    assert "searchgaps_signin" in data
    assert "searchgaps" in data
    assert "searchgaps_rank" in data


@pytest.mark.parametrize("path", ["/search-gaps/connect", "/search-gaps/read", "/search-gaps/rank", "/search-gaps/done"])
def test_every_post_needs_csrf(tmp_path: Path, path: str) -> None:
    ui, _ = make_sg_ui(tmp_path)
    assert ui.client.post(path, data={}, follow_redirects=False).status_code == 403
    assert ui.client.post(path, data={"csrf": "wrong"}, follow_redirects=False).status_code == 403


def test_escaping(tmp_path: Path) -> None:
    reader = FakeAnalyticsReader({None: [("<b>winter</b> soil", 10)]})
    ui, _ = make_sg_ui(tmp_path, reader=reader)
    ui.post("/search-gaps/read", data={})
    ui.web.searchgaps_job.wait()
    html = ui.client.get("/search-gaps").text
    assert "&lt;b&gt;winter&lt;/b&gt; soil" in html
    assert "<b>winter</b>" not in html


def test_dismiss_hides_a_failed_sign_in(tmp_path: Path) -> None:
    def fake_sign_in(cfg):
        raise RuntimeError("boom")

    ui, _ = make_sg_ui(tmp_path, sign_in=fake_sign_in)
    write_client_file(ui.config)
    ui.post("/search-gaps/connect", data={})
    ui.web.searchgaps_signin_job.wait()
    html = ui.client.get("/search-gaps").text
    assert 'id="searchgaps-signin-failed"' in html
    ui.post("/search-gaps/done", data={})
    html2 = ui.client.get("/search-gaps").text
    assert 'id="searchgaps-signin-failed"' not in html2


def test_sign_in_with_write_access_keeps_analytics(tmp_path: Path, monkeypatch) -> None:
    from yezatas.web import app as web_app
    from yezatas import auth as auth_mod

    recorded: dict[str, Any] = {}

    def fake_sign_in(config, scopes=None):
        recorded["scopes"] = scopes
        return object()

    monkeypatch.setattr(auth_mod, "sign_in", fake_sign_in)
    ui, _ = make_sg_ui(tmp_path, connected=True)
    web_app.sign_in_with_write_access(ui.config)
    assert recorded["scopes"] == (auth.WRITE, auth.ANALYTICS)
    write_token(ui.config, [auth.READONLY])
    web_app.sign_in_with_write_access(ui.config)
    assert recorded["scopes"] == (auth.WRITE,)
