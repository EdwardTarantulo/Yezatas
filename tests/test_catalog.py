from datetime import datetime, timezone
from pathlib import Path

import pytest

from fakes import CHANNEL_ID, FakeYouTubeReader, make_item
from yezatas.catalog import NoChannelError, VIDEO_PARTS, pull_catalog, video_from_api
from yezatas.quota import QuotaExceeded, QuotaGuard
from yezatas.store import Store

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
DAY = "2026-01-15"


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as s:
        yield s


def make_guard(store: Store, budget: int = 1000) -> QuotaGuard:
    return QuotaGuard(store, budget, lambda: NOW)


def test_full_pull_stores_all_videos_with_one_snapshot(store: Store) -> None:
    result = pull_catalog(FakeYouTubeReader(120), store, make_guard(store), NOW)
    assert result.channel_id == CHANNEL_ID
    assert result.video_count == 120
    assert len(store.list_videos()) == 120
    assert len(store.snapshots_for("vid000001")) == 1
    pull = store.get_pull(result.pull_id)
    assert pull["finished_at"] is not None and pull["video_count"] == 120
    assert store.get_video("vid000005").localizations["de"]["title"] == "Kaktus umtopfen #5"


def test_second_pull_adds_snapshots_and_updates_stats(store: Store) -> None:
    guard = make_guard(store)
    reader = FakeYouTubeReader(120)
    pull_catalog(reader, store, guard, NOW)
    reader.items[0] = make_item(1, views=99999)
    second = pull_catalog(reader, store, guard, NOW)
    assert len(store.list_videos()) == 120
    assert store.get_video("vid000001").view_count == 99999
    snaps = store.snapshots_for("vid000001")
    assert [s[2].view_count for s in snaps] == [10, 99999]
    assert snaps[1][0] == second.pull_id


def test_pagination_and_quota_ledger(store: Store) -> None:
    reader = FakeYouTubeReader(120, page_size=50)
    guard = make_guard(store)
    result = pull_catalog(reader, store, guard, NOW)
    names = [c[0] for c in reader.calls]
    assert names.count("channels_mine") == 1
    assert names.count("playlist_items") == 3
    assert names.count("videos") == 3
    assert result.units_used == 1 + 3 + 3
    assert store.quota_used(DAY) == 7
    ops = [(r["operation"], r["units"]) for r in store.quota_entries(DAY)]
    assert ops == [("channels.list", 1)] + [("playlistItems.list", 1)] * 3 + [("videos.list", 1)] * 3
    video_calls = [c[1] for c in reader.calls if c[0] == "videos"]
    assert [len(ids) for ids, _ in video_calls] == [50, 50, 20]
    assert all(part == VIDEO_PARTS for _, part in video_calls)


def test_budget_too_small_raises_before_fetching_videos(store: Store) -> None:
    reader = FakeYouTubeReader(120)
    guard = make_guard(store, budget=6)  # 1 + 3 pages fit, 3 video batches do not
    with pytest.raises(QuotaExceeded):
        pull_catalog(reader, store, guard, NOW)
    assert store.list_videos() == []
    assert "videos" not in [c[0] for c in reader.calls]
    assert store.quota_used(DAY) == 4  # quota spent on the calls that were made


def test_no_channel(store: Store) -> None:
    with pytest.raises(NoChannelError):
        pull_catalog(FakeYouTubeReader(has_channel=False), store, make_guard(store), NOW)
    assert store.list_videos() == []
    assert store.quota_used(DAY) == 1  # charged before the call


def test_empty_channel(store: Store) -> None:
    result = pull_catalog(FakeYouTubeReader(0), store, make_guard(store), NOW)
    assert result.video_count == 0 and result.units_used == 2


def test_quota_charged_even_if_call_fails(store: Store) -> None:
    class Boom(FakeYouTubeReader):
        def channels_mine(self, part: str):
            raise RuntimeError("network down")

    with pytest.raises(RuntimeError):
        pull_catalog(Boom(), store, make_guard(store), NOW)
    assert store.quota_used(DAY) == 1


def test_video_from_api_full() -> None:
    item = make_item(10)
    v = video_from_api(item)
    assert (v.video_id, v.channel_id, v.title) == ("vid000010", CHANNEL_ID, "Repotting cactus #10")
    assert v.tags == ["cactus", "garden"]
    assert v.privacy_status == "public" and v.default_language == "en"
    assert v.default_audio_language is None
    assert v.duration == "PT5M30S" and v.published_at == "2026-01-01T10:00:00Z"
    assert (v.view_count, v.like_count, v.comment_count) == (100, 10, 2)
    assert v.localizations["de"]["description"] == "Anleitung 10."
    assert "default" in v.thumbnails
    assert v.raw == item


def test_video_from_api_missing_stats_and_localizations() -> None:
    v = video_from_api(make_item(7))  # likes hidden, no localizations
    assert v.like_count == 0 and v.view_count == 70
    assert v.localizations == {}
    bare = video_from_api({"id": "vidbare", "snippet": {"title": "t"}})
    assert (bare.view_count, bare.like_count, bare.comment_count) == (0, 0, 0)
    assert bare.tags == [] and bare.localizations == {} and bare.thumbnails == {}
