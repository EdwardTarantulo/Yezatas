"""The thumbnail swap pages: replace with a variant, restore from History (F3).
YouTube and the download are fakes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httplib2
from googleapiclient.errors import HttpError

from yezatas import packaging as packaging_mod
from yezatas import thumbswap as thumbswap_mod
from yezatas.packaging import PackagingError, stored_image_path
from yezatas.store import PackagingReview, Store, ThumbnailVariant
from yezatas.thumbswap import ThumbnailSwap
from yezatas.web.app import map_thumb_error
from yezatas.web.thumbswap_view import build_swap_rows, swap_label
from webhelpers import make_push_ui

JPEG_OLD = b"\xff\xd8\xff" + b"old"
JPEG_NEW = b"\xff\xd8\xff" + b"new"
JPEG_LIVE = b"\xff\xd8\xff" + b"live"


def http_error(reason: str) -> HttpError:
    resp = httplib2.Response({"status": 403})
    content = json.dumps({"error": {"message": "x", "errors": [{"reason": reason}]}}).encode()
    return HttpError(resp, content)


def make(tmp_path: Path, **kwargs: Any) -> tuple[Any, Any]:
    """Create a UI helper for thumbnail swap tests."""
    kwargs.setdefault("packaging_fetch", lambda url: JPEG_OLD)
    return make_push_ui(tmp_path, 3, **kwargs)


def confirm(ui: Any, url: str) -> Any:
    """Confirm a thumbnail action."""
    return ui.client.post(url, data={"csrf": ui.csrf(), "confirm": "yes"}, follow_redirects=False)


def seed_variant(ui: Any) -> tuple[str, int]:
    """Create a variant for a fake video and return (video_id, variant_id)."""
    V = "vid000001"
    name = packaging_mod.save_image(ui.web.images_dir, V, "variant", JPEG_NEW)
    with Store(ui.config.catalog_db) as store:
        review = PackagingReview(
            id=0,
            video_id=V,
            title="t",
            thumbnail_url="",
            image_file="",
            measurements={},
            reading={},
            critique={},
            problems=[],
            status="done",
            error="",
            models={},
            input_tokens=0,
            cached_tokens=0,
            output_tokens=0,
            cost_usd=None,
            created_at="",
        )
        rid = store.add_packaging_review(review)
        var = ThumbnailVariant(
            id=0,
            review_id=rid,
            video_id=V,
            instruction="make it pop",
            overlay_text="",
            image_file=name,
            text_read="",
            text_ok=None,
            measurements={},
            status="ok",
            error="",
            model="m",
            cost_usd=None,
            created_at="",
        )
        vid = store.add_thumbnail_variant(var)
    return V, vid


def test_variant_card_link(tmp_path: Path) -> None:
    ui, writer = make(tmp_path)
    V, vid = seed_variant(ui)
    html = ui.client.get(f"/videos/{V}").text
    assert "Replace current thumbnail" in html
    assert f"/videos/{V}/packaging/variants/{vid}/replace" in html


def test_replace_page_shows_both(tmp_path: Path) -> None:
    ui, writer = make(tmp_path)
    V, vid = seed_variant(ui)
    html = ui.client.get(f"/videos/{V}/packaging/variants/{vid}/replace").text
    assert "https://example.invalid/1.jpg" in html
    assert f"/videos/{V}/packaging/variants/{vid}.jpg" in html
    assert "51" in html
    assert 'name="confirm"' in html
    assert writer.calls == []


def test_without_tick_box(tmp_path: Path) -> None:
    ui, writer = make(tmp_path)
    V, vid = seed_variant(ui)
    csrf = ui.csrf()
    r = ui.client.post(
        f"/videos/{V}/packaging/variants/{vid}/replace", data={"csrf": csrf}, follow_redirects=False
    )
    assert r.status_code == 303
    html = ui.client.get(r.headers["location"]).text
    assert "Tick the box" in html
    assert writer.calls == []
    assert writer.thumbnails_set == []


def test_without_write_access(tmp_path: Path) -> None:
    ui, writer = make(tmp_path, write_access=False)
    V, vid = seed_variant(ui)
    html = ui.client.get(f"/videos/{V}/packaging/variants/{vid}/replace").text
    assert 'id="thumb-no-write"' in html
    assert 'name="confirm"' not in html
    r = ui.client.post(
        f"/videos/{V}/packaging/variants/{vid}/replace",
        data={"csrf": ui.csrf(), "confirm": "yes"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    html2 = ui.client.get(r.headers["location"]).text
    assert "does not have permission" in html2
    assert writer.thumbnails_set == []


def test_confirmed_replace_runs(tmp_path: Path) -> None:
    ui, writer = make(tmp_path)
    V, vid = seed_variant(ui)
    confirm(ui, f"/videos/{V}/packaging/variants/{vid}/replace")
    ui.web.thumb_job.wait()
    assert writer.thumbnails_set == [(V, JPEG_NEW)]
    html = ui.client.get(f"/videos/{V}/packaging/variants/{vid}/replace").text
    assert 'id="thumb-result"' in html
    assert "Quota used: 51 units" in html
    swaps = thumbswap_mod.load_swaps(ui.web.images_dir)
    assert len(swaps) == 1
    assert swaps[0].kind == "replace"
    assert swaps[0].variant_id == vid
    assert swaps[0].backup_file
    p = stored_image_path(ui.web.images_dir, V, swaps[0].backup_file)
    assert p.read_bytes() == JPEG_OLD


def test_history_and_restore(tmp_path: Path) -> None:
    fetched: list[bytes] = []
    def _fetch(url: str) -> bytes:
        pic = JPEG_LIVE if fetched else JPEG_OLD
        fetched.append(pic)
        return pic

    ui, writer = make(tmp_path, packaging_fetch=_fetch)
    V, vid = seed_variant(ui)
    confirm(ui, f"/videos/{V}/packaging/variants/{vid}/replace")
    ui.web.thumb_job.wait()
    swaps = thumbswap_mod.load_swaps(ui.web.images_dir)
    sid = swaps[0].swap_id
    html = ui.client.get("/history").text
    assert 'id="thumbnails"' in html
    assert f"/history/thumbnails/{sid}/backup.jpg" in html
    assert "Put this thumbnail back" in html
    r = ui.client.get(f"/history/thumbnails/{sid}/backup.jpg")
    assert r.status_code == 200
    assert r.content == JPEG_OLD
    assert "image/jpeg" in r.headers["content-type"]
    html2 = ui.client.get(f"/history/thumbnails/{sid}/restore").text
    assert "Put the old thumbnail back" in html2
    confirm(ui, f"/history/thumbnails/{sid}/restore")
    ui.web.thumb_job.wait()
    assert writer.thumbnails_set[-1] == (V, JPEG_OLD)
    swaps2 = thumbswap_mod.load_swaps(ui.web.images_dir)
    assert len(swaps2) == 2
    assert swaps2[0].kind == "restore"
    assert swaps2[1].kind == "replace"
    p2 = stored_image_path(ui.web.images_dir, V, swaps2[0].backup_file)
    assert p2.read_bytes() == JPEG_LIVE


def test_forbidden_error(tmp_path: Path) -> None:
    ui, writer = make(tmp_path)
    V, vid = seed_variant(ui)
    writer.thumbnail_errors[V] = http_error("forbidden")
    confirm(ui, f"/videos/{V}/packaging/variants/{vid}/replace")
    ui.web.thumb_job.wait()
    html = ui.client.get(f"/videos/{V}/packaging/variants/{vid}/replace").text
    assert 'id="thumb-failed"' in html
    assert "verified channel" in html
    assert "saved in History" not in html
    assert thumbswap_mod.load_swaps(ui.web.images_dir) == []


def test_failed_message_shown_once(tmp_path: Path) -> None:
    ui, writer = make(tmp_path)
    V, vid = seed_variant(ui)
    writer.thumbnail_errors[V] = http_error("forbidden")
    confirm(ui, f"/videos/{V}/packaging/variants/{vid}/replace")
    ui.web.thumb_job.wait()
    url = f"/videos/{V}/packaging/variants/{vid}/replace"
    html1 = ui.client.get(url).text
    assert 'id="thumb-failed"' in html1
    html2 = ui.client.get(url).text
    assert 'id="thumb-failed"' not in html2
    assert 'name="confirm"' in html2


def test_build_swap_rows_and_label() -> None:
    replace = ThumbnailSwap(
        swap_id=1,
        video_id="vid000001",
        kind="replace",
        new_file="variant-a.jpg",
        backup_file="backup-a.jpg",
        variant_id=7,
        units=51,
        at="2026-06-01T12:00:00+00:00",
    )
    restore = ThumbnailSwap(
        swap_id=2,
        video_id="vid000002",
        kind="restore",
        new_file="backup-a.jpg",
        backup_file="backup-b.jpg",
        variant_id=None,
        units=51,
        at="2026-06-02T12:00:00+00:00",
    )
    assert swap_label(replace) == "Replaced with a variant idea"
    assert swap_label(restore) == "Put an old thumbnail back"
    rows = build_swap_rows(
        [restore, replace], {"vid000002": "Winter care"}, lambda iso: "when " + iso
    )
    assert rows[0].title == "Winter care"
    assert rows[1].title == "vid000001"
    assert rows[0].when == "when 2026-06-02T12:00:00+00:00"
    assert rows[1].when == "when 2026-06-01T12:00:00+00:00"
    assert rows[0].backup_url == "/history/thumbnails/2/backup.jpg"
    assert rows[0].restore_url == "/history/thumbnails/2/restore"
    assert rows[0].label == "Put an old thumbnail back"
    assert rows[1].label == "Replaced with a variant idea"


def test_unknown_ids(tmp_path: Path) -> None:
    ui, writer = make(tmp_path)
    V, vid = seed_variant(ui)
    r = ui.client.get(f"/videos/{V}/packaging/variants/99999/replace", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].endswith(f"/videos/{V}#thumbnail")
    r2 = ui.client.get("/history/thumbnails/99999/backup.jpg")
    assert r2.status_code == 404
    r3 = ui.client.get("/history/thumbnails/99999/restore", follow_redirects=False)
    assert r3.status_code == 303
    assert r3.headers["location"].endswith("/history#thumbnails")


def test_api_status_and_js(tmp_path: Path) -> None:
    ui, writer = make(tmp_path)
    data = ui.client.get("/api/status").json()
    assert "thumb" in data
    js = Path(__file__).parent.parent / "src" / "yezatas" / "web" / "static" / "app.js"
    assert '"thumb"' in js.read_text(encoding="utf-8")


def test_map_thumb_error() -> None:
    e1 = thumbswap_mod.ThumbSwapError("bad thumb")
    assert map_thumb_error(e1) == ("thumb", "bad thumb")
    e2 = PackagingError("gone")
    k, m = map_thumb_error(e2)
    assert k == "thumb"
    assert "could not save the current thumbnail" in m
    e3 = http_error("forbidden")
    k3, m3 = map_thumb_error(e3)
    assert k3 == "thumb"
    assert "verified channel" in m3
    e4 = http_error("uploadRateLimitExceeded")
    k4, m4 = map_thumb_error(e4)
    assert k4 == "thumb"
    assert "too many thumbnails" in m4


def test_dismiss_only_goes_back_to_local_pages(tmp_path: Path) -> None:
    ui, writer = make(tmp_path)
    for bad in ("//example.com/x", "/\\example.com/x", "https://example.com/x", ""):
        form = {"csrf": ui.csrf(), "back": bad}
        r = ui.client.post("/thumbnails/done", data=form, follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"].endswith("/history#thumbnails")
    r = ui.client.post("/thumbnails/done", data={"csrf": ui.csrf(), "back": "/videos"}, follow_redirects=False)
    assert r.headers["location"].endswith("/videos")


def test_confirm_page_quota_has_separators(tmp_path: Path) -> None:
    ui, writer = make(tmp_path)
    V, vid = seed_variant(ui)
    html = ui.client.get(f"/videos/{V}/packaging/variants/{vid}/replace").text
    assert "units left today (budget 10,000)" in html
