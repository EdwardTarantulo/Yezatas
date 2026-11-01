from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fakes import FakeYouTubeWriter, make_item
from yezatas.quota import QuotaExceeded, QuotaGuard
from yezatas.store import Store
from yezatas.thumbswap import (
    MAX_BYTES,
    ThumbSwapError,
    backup_image,
    check_jpeg,
    live_thumbnail_url,
    load_swaps,
    save_swap,
    swap_thumbnail,
    swap_units,
)

JPEG_OLD = b"\xff\xd8\xff" + b"old"
JPEG_NEW = b"\xff\xd8\xff" + b"new"
NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _fetch_old(url: str) -> bytes:
    return JPEG_OLD


def test_replace_charges_and_saves(tmp_path):
    writer = FakeYouTubeWriter(video_count=5)
    images_dir = tmp_path / "thumbs"
    fetches: list[str] = []

    def fake_fetch(url: str) -> bytes:
        fetches.append(url)
        return JPEG_OLD

    with Store(tmp_path / "c.sqlite") as store:
        guard = QuotaGuard(store, 10_000)
        reports: list[str] = []
        swap = swap_thumbnail(
            writer,
            guard,
            images_dir,
            "vid000001",
            JPEG_NEW,
            kind="replace",
            new_file="new.jpg",
            variant_id=7,
            fetch=fake_fetch,
            now=NOW,
            report=reports.append,
        )
        assert guard.used() == 51
        assert writer.calls[0][0] == "videos"
        assert writer.calls[1][0] == "thumbnails.set"
        assert fetches == ["https://example.invalid/1.jpg"]
        assert swap.swap_id == 1
        assert swap.kind == "replace"
        assert swap.units == 51
        assert swap.at == NOW.isoformat()
        assert swap.video_id == "vid000001"
        assert swap.new_file == "new.jpg"
        assert swap.variant_id == 7
        backup_path = images_dir / "vid000001" / swap.backup_file
        assert backup_path.read_bytes() == JPEG_OLD
        assert writer.thumbnails_set == [("vid000001", JPEG_NEW)]
        assert reports == [
            "Reading the live thumbnail",
            "Saving the current thumbnail",
            "Uploading the new thumbnail",
        ]


def test_load_swaps_order_and_second_id(tmp_path):
    writer = FakeYouTubeWriter(video_count=5)
    images_dir = tmp_path / "thumbs"
    with Store(tmp_path / "c.sqlite") as store:
        guard = QuotaGuard(store, 10_000)
        s1 = swap_thumbnail(
            writer,
            guard,
            images_dir,
            "vid000001",
            JPEG_NEW,
            kind="replace",
            new_file="a.jpg",
            fetch=_fetch_old,
            now=NOW,
        )
        s2 = swap_thumbnail(
            writer,
            guard,
            images_dir,
            "vid000002",
            JPEG_NEW,
            kind="restore",
            new_file="b.jpg",
            fetch=_fetch_old,
            now=NOW,
        )
        loaded = load_swaps(images_dir)
        assert len(loaded) == 2
        assert loaded[0].swap_id == 2
        assert loaded[1].swap_id == 1
        assert s2.swap_id == 2
        assert loaded[0].video_id == "vid000002"


def test_non_jpeg_and_too_large_raise_before_charge(tmp_path):
    writer = FakeYouTubeWriter(video_count=5)
    images_dir = tmp_path / "thumbs"
    with Store(tmp_path / "c.sqlite") as store:
        guard = QuotaGuard(store, 10_000)
        with pytest.raises(ThumbSwapError, match="not a JPEG"):
            swap_thumbnail(
                writer,
                guard,
                images_dir,
                "vid000001",
                b"not jpeg",
                kind="replace",
                new_file="x.jpg",
            )
        assert writer.calls == []
        assert guard.used() == 0
        big = b"\xff\xd8\xff" + b"x" * (MAX_BYTES + 1 - 3)
        assert len(big) > MAX_BYTES
        with pytest.raises(ThumbSwapError, match="larger than YouTube"):
            swap_thumbnail(
                writer,
                guard,
                images_dir,
                "vid000001",
                big,
                kind="replace",
                new_file="x.jpg",
            )
        assert writer.calls == []
        assert guard.used() == 0


def test_quota_exceeded_no_call(tmp_path):
    writer = FakeYouTubeWriter(video_count=5)
    images_dir = tmp_path / "thumbs"
    with Store(tmp_path / "c.sqlite") as store:
        guard = QuotaGuard(store, 50)
        with pytest.raises(QuotaExceeded):
            swap_thumbnail(
                writer,
                guard,
                images_dir,
                "vid000001",
                JPEG_NEW,
                kind="replace",
                new_file="x.jpg",
            )
        assert writer.calls == []


def test_unknown_video_and_no_thumbnail(tmp_path):
    writer = FakeYouTubeWriter(video_count=5)
    images_dir = tmp_path / "thumbs"
    with Store(tmp_path / "c.sqlite") as store:
        guard = QuotaGuard(store, 10_000)
        with pytest.raises(ThumbSwapError, match="did not find"):
            swap_thumbnail(
                writer,
                guard,
                images_dir,
                "vid999999",
                JPEG_NEW,
                kind="replace",
                new_file="x.jpg",
                fetch=_fetch_old,
                now=NOW,
            )
        assert writer.thumbnails_set == []
    writer2 = FakeYouTubeWriter(video_count=5)
    writer2.items[0]["snippet"]["thumbnails"] = {
        "default": {"url": "http://example.invalid/x.jpg"}
    }
    with Store(tmp_path / "c2.sqlite") as store:
        guard = QuotaGuard(store, 10_000)
        with pytest.raises(ThumbSwapError, match="nothing to keep"):
            swap_thumbnail(
                writer2,
                guard,
                images_dir,
                "vid000001",
                JPEG_NEW,
                kind="replace",
                new_file="x.jpg",
                fetch=_fetch_old,
                now=NOW,
            )
        assert writer2.thumbnails_set == []


def test_set_thumbnail_error_propagates_no_swap(tmp_path):
    writer = FakeYouTubeWriter(video_count=5)
    writer.thumbnail_errors["vid000001"] = RuntimeError("boom")
    images_dir = tmp_path / "thumbs"
    with Store(tmp_path / "c.sqlite") as store:
        guard = QuotaGuard(store, 10_000)
        with pytest.raises(RuntimeError, match="boom"):
            swap_thumbnail(
                writer,
                guard,
                images_dir,
                "vid000001",
                JPEG_NEW,
                kind="replace",
                new_file="x.jpg",
                fetch=_fetch_old,
                now=NOW,
            )
        assert load_swaps(images_dir) == []
        assert guard.used() == 51


def test_backup_image_roundtrip_and_gone(tmp_path):
    writer = FakeYouTubeWriter(video_count=5)
    images_dir = tmp_path / "thumbs"
    with Store(tmp_path / "c.sqlite") as store:
        guard = QuotaGuard(store, 10_000)
        swap = swap_thumbnail(
            writer,
            guard,
            images_dir,
            "vid000001",
            JPEG_NEW,
            kind="replace",
            new_file="x.jpg",
            fetch=_fetch_old,
            now=NOW,
        )
        assert backup_image(images_dir, swap) == JPEG_OLD
        p = images_dir / "vid000001" / swap.backup_file
        p.unlink()
        with pytest.raises(ThumbSwapError, match="gone"):
            backup_image(images_dir, swap)


def test_broken_swaps_json(tmp_path):
    images_dir = tmp_path / "thumbs"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "swaps.json").write_text("not json", encoding="utf-8")
    assert load_swaps(images_dir) == []
    (images_dir / "swaps.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert load_swaps(images_dir) == []
    good = {
        "swap_id": 1,
        "video_id": "vid000001",
        "kind": "replace",
        "new_file": "n.jpg",
        "backup_file": "b.jpg",
        "variant_id": None,
        "units": 51,
        "at": NOW.isoformat(),
    }
    bad = {"swap_id": "oops", "video_id": "vid000001"}
    (images_dir / "swaps.json").write_text(json.dumps([bad, good]), encoding="utf-8")
    loaded = load_swaps(images_dir)
    assert len(loaded) == 1
    assert loaded[0].swap_id == 1


def test_live_thumbnail_url_prefers_maxres(tmp_path):
    item = make_item(1)
    item["snippet"]["thumbnails"]["maxres"] = {"url": "https://example.invalid/max.jpg"}
    item["snippet"]["thumbnails"]["default"] = {
        "url": "https://example.invalid/default.jpg"
    }
    assert live_thumbnail_url(item) == "https://example.invalid/max.jpg"
    item2 = {
        "snippet": {
            "thumbnails": {
                "maxres": {"url": "http://example.invalid/max.jpg"},
                "default": {"url": "http://example.invalid/d.jpg"},
            }
        }
    }
    assert live_thumbnail_url(item2) == ""
    assert live_thumbnail_url({}) == ""
    assert live_thumbnail_url({"snippet": None}) == ""


def test_swap_units():
    assert swap_units() == 51


def test_load_swaps_skips_bool_units_and_str_variant(tmp_path):
    images_dir = tmp_path / "thumbs"
    images_dir.mkdir(parents=True, exist_ok=True)
    good = {
        "swap_id": 1,
        "video_id": "vid000001",
        "kind": "replace",
        "new_file": "n.jpg",
        "backup_file": "b.jpg",
        "variant_id": None,
        "units": 51,
        "at": NOW.isoformat(),
    }
    bad_units = {
        "swap_id": 2,
        "video_id": "vid000002",
        "kind": "replace",
        "new_file": "n.jpg",
        "backup_file": "b.jpg",
        "variant_id": None,
        "units": True,
        "at": NOW.isoformat(),
    }
    bad_variant = {
        "swap_id": 3,
        "video_id": "vid000003",
        "kind": "replace",
        "new_file": "n.jpg",
        "backup_file": "b.jpg",
        "variant_id": "7",
        "units": 51,
        "at": NOW.isoformat(),
    }
    (images_dir / "swaps.json").write_text(
        json.dumps([bad_units, bad_variant, good]), encoding="utf-8"
    )
    loaded = load_swaps(images_dir)
    assert len(loaded) == 1
    assert loaded[0].swap_id == 1
