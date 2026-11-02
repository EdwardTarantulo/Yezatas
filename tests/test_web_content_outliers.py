"""The Content page's vs norm column, its sort and filter chip; the cheap key check on page loads; the
wording of a thumbnail variant whose subject was not found. Made-up videos and fakes only; no network."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from webhelpers import make_ui, make_video
from yezatas import agent, outliers
from yezatas import packaging as packaging_mod
from yezatas.seo import score_catalog
from yezatas.store import Store, ThumbnailVariant
from yezatas.web import views

PULLED = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
START = datetime(2025, 1, 1, tzinfo=timezone.utc)
HIGH, MID, LOW, THREE = 5, 15, 20, 25  # 50x, 20x, 0.05x and 3x the norm of 100 views
STATIC = Path(__file__).parent.parent / "src" / "yezatas" / "web" / "static"


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


def vid(n: int) -> str:
    return f"vid{n:06d}"


def catalog(count: int = 60) -> list[Any]:
    """`count` public videos a day apart with 100 views each (a few far from the norm), then a private
    one, a live stream and one only 10 days old: those three get no ratio."""
    videos = []
    for n in range(count):
        views_ = {HIGH: 5000, MID: 2000, LOW: 5, THREE: 300}.get(n, 100)
        videos.append(make_video(
            n, video_id=vid(n), title=f"Cactus tip {n}", privacy_status="public", view_count=views_,
            published_at=(START + timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ")))
    videos.append(make_video(100, video_id=vid(100), title="Private draft", privacy_status="private",
                             published_at="2025-02-01T10:00:00Z", view_count=7))
    videos.append(make_video(101, video_id=vid(101), title="Live stream", privacy_status="public",
                             published_at="2025-02-02T10:00:00Z", duration="P0D", view_count=0))
    videos.append(make_video(102, video_id=vid(102), title="Brand new", privacy_status="public",
                             published_at=(PULLED - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             view_count=40))
    return videos


def seed(config: Any, videos: list[Any] | None = None) -> None:
    videos = videos if videos is not None else catalog()
    with Store(config.catalog_db) as store:
        pull_id = store.begin_pull("UC_fake_channel", PULLED)
        for video in videos:
            store.upsert_video(video, pull_id, PULLED)
        store.finish_pull(pull_id, len(videos), PULLED)


def content_ui(tmp_path: Path, **kwargs: Any) -> Any:
    ui = make_ui(tmp_path, **kwargs)
    seed(ui.config)
    return ui


def rows(html: str) -> list[str]:
    return html.split("<tbody>")[1].split("</tbody>")[0].split("<tr>")[1:]


def ids_in(table_rows: list[str]) -> list[str]:
    return [re.search(r'class="vtitle" href="/videos/([^"]+)"', r).group(1) for r in table_rows]


def ratio_cell(row: str) -> str:
    return row.split('<td class="col-ratio">')[1].split("</td>")[0]


# --- the column ----------------------------------------------------------------------------------
def test_the_table_has_a_vs_norm_column_between_views_and_seo_score(tmp_path: Path) -> None:
    html = content_ui(tmp_path).client.get("/videos?sort=newest").text
    head = html.split("<thead>")[1].split("</thead>")[0]
    assert head.index(">Views<") < head.index(">vs norm<") < head.index(">SEO score<")
    row = rows(html)[0]
    assert row.index('class="col-num"') < row.index('class="col-ratio"') < row.index('class="col-score"')


def test_labels_come_from_ratio_label_and_outliers_get_the_pill(tmp_path: Path) -> None:
    ui = content_ui(tmp_path)
    html = ui.client.get("/videos?sort=norm").text
    table = rows(html) + rows(ui.client.get("/videos?sort=norm&page=2").text)
    by_id = dict(zip(ids_in(table), table))
    high = ratio_cell(by_id[vid(HIGH)])
    assert outliers.ratio_label(50.0) in high and 'class="pill done"' in high
    assert outliers.ratio_label(20.0) in ratio_cell(by_id[vid(MID)])
    three = ratio_cell(by_id[vid(THREE)])
    assert outliers.ratio_label(3.0) in three and "pill" not in three  # not an outlier
    low = ratio_cell(by_id[vid(LOW)])
    assert outliers.ratio_label(0.05) in low and "pill" not in low
    assert "style=" not in html


def test_unscored_videos_show_a_dash_and_say_why_in_plain_words(tmp_path: Path) -> None:
    ui = content_ui(tmp_path)
    table = rows(ui.client.get("/videos?sort=norm&page=2").text)
    by_id = dict(zip(ids_in(table), table))
    dash = outliers.ratio_label(None)
    for video_id, words in ((vid(102), "Too new to compare"), (vid(100), "Only public videos"),
                            (vid(101), "Live streams")):
        cell = ratio_cell(by_id[video_id])
        assert f">{dash}<" in cell and f'title="{words}' in cell, video_id
        assert "pill" not in cell


def test_a_video_with_too_few_peers_says_so(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    seed(ui.config, catalog(5))
    cell = ratio_cell(rows(ui.client.get("/videos?dir=asc").text)[0])
    assert "Not enough other public videos" in cell and "—" in cell


def test_the_phone_layout_hides_the_new_column() -> None:
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    narrow = css.split("@media (max-width: 760px) {")[1].split("\n}\n")[0]
    assert ".vtable .col-date, .vtable .col-ratio { display: none; }" in narrow
    assert re.search(r"^\.vtable \.col-ratio \{ width: \d+px", css, re.M)


# --- the sort ------------------------------------------------------------------------------------
def test_the_vs_norm_sort_puts_the_highest_ratio_first_and_unscored_last(tmp_path: Path) -> None:
    ui = content_ui(tmp_path)
    page1 = ids_in(rows(ui.client.get("/videos?sort=norm").text))
    page2 = ids_in(rows(ui.client.get("/videos?sort=norm&page=2").text))
    order = page1 + page2
    assert len(order) == 63
    assert order[:3] == [vid(HIGH), vid(MID), vid(THREE)]
    assert order[-4] == vid(LOW)  # the lowest ratio is the last scored one
    assert set(order[-3:]) == {vid(100), vid(101), vid(102)}  # no ratio: after every scored video
    assert order[-3:] == sorted(order[-3:])  # in a stable order


def test_the_vs_norm_title_sorts_and_unknown_values_still_fall_back(tmp_path: Path) -> None:
    ui = content_ui(tmp_path)
    html = ui.client.get("/videos?sort=norm").text
    head = html.split("<thead>")[1].split("</thead>")[0]
    assert re.search(r'<th class="col-ratio" scope="col" aria-sort="descending"><a class="th-sort active" '
                     r'href="/videos\?sort=norm&amp;dir=asc"[^>]*>vs norm<', head)
    assert head.index(">Date<") < head.index(">vs norm<")
    low_first = ids_in(rows(ui.client.get("/videos?sort=norm&dir=asc").text))
    assert low_first[0] == vid(LOW)
    odd = ui.client.get("/videos?sort=nonsense&outliers=maybe")
    assert odd.status_code == 200
    assert ids_in(rows(odd.text)) == ids_in(rows(ui.client.get("/videos").text))


# --- the filter chip -----------------------------------------------------------------------------
def test_the_outliers_chip_filters_to_ratios_at_or_above_the_threshold(tmp_path: Path) -> None:
    ui = content_ui(tmp_path)
    label = f"Outliers ({outliers.ratio_label(outliers.OUTLIER_THRESHOLD)}+)"
    assert label == "Outliers (10×+)"
    score_row = ui.client.get("/videos").text.split('aria-label="SEO score"')[1].split("</div>")[0]
    assert f">{label}<" in score_row and 'href="/videos?outliers=1"' in score_row
    filtered = ui.client.get("/videos?outliers=1").text
    assert set(ids_in(rows(filtered))) == {vid(HIGH), vid(MID)}  # 3x and the rest are not outliers
    assert "2 videos" in filtered
    active = filtered.split('aria-label="SEO score"')[1].split("</div>")[0]
    assert re.search(r'class="chip active" id="chip-outliers" href="/videos"', active)  # click again: off


def test_outliers_combine_with_sort_and_other_filters_and_keep_paging(tmp_path: Path) -> None:
    ui = content_ui(tmp_path)
    html = ui.client.get("/videos?outliers=1&sort=norm").text
    assert ids_in(rows(html)) == [vid(HIGH), vid(MID)]
    head = html.split("<thead>")[1].split("</thead>")[0]
    assert 'href="/videos?sort=views&amp;outliers=1"' in head  # the filter survives a re-sort
    none = ui.client.get("/videos?outliers=1&visibility=private").text
    assert 'id="no-videos"' in none and "Clear all filters" in none
    page1 = ui.client.get("/videos?sort=norm").text
    assert 'rel="next" href="/videos?sort=norm&amp;page=2"' in page1
    assert "Page 2 of 2" in ui.client.get("/videos?sort=norm&page=2").text
    assert "2 videos" in ui.client.get("/videos?outliers=1&page=9").text  # past the end: the last page


def test_build_listing_without_scores_still_works() -> None:
    videos = catalog(3)
    entries = views.make_entries(videos, score_catalog(videos))
    listing = views.build_listing(entries, {"sort": "norm"})
    assert len(listing.entries) == len(entries) and listing.entries[0].ratio_text == "—"
    assert views.build_listing(entries, {"outliers": "1"}).total == 0


# --- the key check on page loads -----------------------------------------------------------------
class CountingFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, config: Any) -> Any:
        self.calls += 1
        return None


def button_after(html: str, text: str) -> str:
    return html.split(text)[0].rsplit("<button", 1)[1]


@pytest.mark.parametrize("key", [True, False])
def test_page_loads_do_not_build_the_packaging_tools(tmp_path: Path, key: bool) -> None:
    factory = CountingFactory()
    ui = make_ui(tmp_path, packaging_tools=factory, packaging_key=lambda cfg: key)
    seed(ui.config)
    video = ui.client.get(f"/videos/{vid(HIGH)}").text
    outlier_page = ui.client.get("/outliers").text
    assert factory.calls == 0
    packaging_card = video.split('id="packaging"')[1].split("</section>")[0]
    assert ('id="packaging-no-key"' in packaging_card) is (not key)
    assert ('id="outliers-no-key"' in outlier_page) is (not key)
    assert (" disabled" in button_after(packaging_card, "Research thumbnail")) is (not key)
    assert (" disabled" in button_after(outlier_page, "Explain with Muse Spark")) is (not key)


def test_the_default_check_reads_the_key_and_builds_no_client(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(config: Any, env: Any = None) -> Any:
        raise AssertionError("built the packaging clients on a page load")

    monkeypatch.setattr(packaging_mod, "packaging_tools_from_config", boom)
    for key in (False, True):
        if key:
            monkeypatch.setenv(agent.KEY_ENV, "fake-key-for-tests")
        ui = make_ui(tmp_path / str(key))
        seed(ui.config)
        video = ui.client.get(f"/videos/{vid(HIGH)}").text
        outlier_page = ui.client.get("/outliers").text
        assert ('id="packaging-no-key"' in video) is (not key)
        assert ('id="outliers-no-key"' in outlier_page) is (not key)


def test_an_injected_factory_still_decides_when_no_key_check_is_given(tmp_path: Path) -> None:
    factory = CountingFactory()
    ui = make_ui(tmp_path, packaging_tools=factory)
    seed(ui.config)
    assert 'id="packaging-no-key"' in ui.client.get(f"/videos/{vid(HIGH)}").text
    assert factory.calls == 1


# --- the variant wording -------------------------------------------------------------------------
def variant(subject: Any) -> ThumbnailVariant:
    return ThumbnailVariant(
        id=1, review_id=1, video_id="vid00000001", instruction="Brighter", overlay_text="",
        image_file="a.jpg", text_read="", text_ok=None, measurements={"subject": subject},
        status="ok", error="", model="m", cost_usd=None, created_at="2026-06-01T00:00:00Z")


def test_subject_share_wording_with_and_without_the_subject_in_the_variant() -> None:
    both = views._variant_view("vid00000001", variant(
        {"before_share_pct": 44, "after": {"found": True, "share_pct": 29}}))
    assert both.subject_change == "Subject share: 44% → 29%"
    gone = views._variant_view("vid00000001", variant(
        {"before_share_pct": 44, "after": {"found": False}}))
    assert gone.subject_change == "Subject share: 44% before; not found in the variant"
    unknown = views._variant_view("vid00000001", variant(
        {"before_share_pct": None, "after": {"found": False}}))
    assert unknown.subject_change == "Subject share: not measured before; not found in the variant"
