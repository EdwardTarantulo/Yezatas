import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from yezatas.store import MIGRATIONS, Store, Video

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def make_video(vid: str = "vid00000001", **kw: object) -> Video:
    base: dict[str, object] = dict(
        video_id=vid,
        channel_id="chan0000001",
        title="How to repot a cactus",
        description="Step by step.\nSecond line.",
        tags=["cactus", "plants"],
        privacy_status="public",
        default_language="en",
        published_at="2026-01-01T00:00:00Z",
        duration="PT5M3S",
        view_count=100,
        like_count=10,
        comment_count=2,
        localizations={"lt": {"title": "Kaip persodinti kaktusą", "description": "x"}},
        thumbnails={"default": {"url": "https://example.invalid/t.jpg"}},
        raw={"kind": "youtube#video"},
    )
    base.update(kw)
    return Video(**base)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "sub" / "c.sqlite") as s:
        yield s


def test_creates_parent_dirs_and_version(tmp_path: Path) -> None:
    with Store(tmp_path / "a" / "b" / "c.sqlite") as s:
        assert s.schema_version == len(MIGRATIONS) == 1
        assert s._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_migrations_idempotent_on_reopen(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    with Store(path) as s:
        pid = s.begin_pull("chan0000001", now=T0)
        s.upsert_video(make_video(), pid, now=T0)
    with Store(path) as s:
        assert s.schema_version == 1
        assert s.get_video("vid00000001") is not None
        assert len(s.snapshots_for("vid00000001")) == 1


def test_upsert_writes_snapshot_per_pull(store: Store) -> None:
    p1 = store.begin_pull("chan0000001", now=T0)
    store.upsert_video(make_video(view_count=100), p1, now=T0)
    store.finish_pull(p1, 1, now=T0 + timedelta(seconds=5))
    later = T0 + timedelta(days=1)
    p2 = store.begin_pull("chan0000001", now=later)
    store.upsert_video(make_video(view_count=250, title="Repotting a cactus"), p2, now=later)
    cur = store.get_video("vid00000001")
    assert cur is not None and cur.view_count == 250 and cur.title == "Repotting a cactus"
    assert len(store.list_videos()) == 1
    snaps = store.snapshots_for("vid00000001")
    assert [(s[0], s[2].view_count) for s in snaps] == [(p1, 100), (p2, 250)]
    assert snaps[0][2].title == "How to repot a cactus"
    pull = store.get_pull(p1)
    assert pull is not None and pull["video_count"] == 1 and pull["finished_at"] is not None
    assert store._conn.execute("SELECT last_pull_id FROM videos").fetchone()[0] == p2


def test_upsert_is_atomic(store: Store) -> None:
    # Unknown pull id violates the FK: neither the video nor a snapshot may be written.
    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_video(make_video(), 999, now=T0)
    assert store.get_video("vid00000001") is None
    assert store.snapshots_for("vid00000001") == []


def test_json_round_trip_unicode_and_emoji(store: Store) -> None:
    v = make_video(
        title="Kaktusų persodinimas 🌵 – 仙人掌",
        description="Ačiū! 😀\nLine two",
        tags=["kaktusas", "🌵", "日本語"],
        localizations={"ja": {"title": "サボテン", "description": "🌵"}},
    )
    pid = store.begin_pull("chan0000001", now=T0)
    store.upsert_video(v, pid, now=T0)
    assert store.get_video(v.video_id) == v
    assert store.snapshots_for(v.video_id)[0][2] == v
    assert Video.from_dict(v.to_dict()) == v


def test_get_missing_video(store: Store) -> None:
    assert store.get_video("nope") is None


def test_list_videos_ordering(store: Store) -> None:
    pid = store.begin_pull("chan0000001", now=T0)
    for i, views in enumerate([5, 50, 20]):
        store.upsert_video(make_video(f"vid0000000{i}", view_count=views), pid, now=T0)
    assert [v.view_count for v in store.list_videos()] == [50, 20, 5]
    assert [v.view_count for v in store.list_videos("view_count asc")] == [5, 20, 50]
    assert [v.video_id for v in store.list_videos("video_id")] == [
        "vid00000000", "vid00000001", "vid00000002"]


@pytest.mark.parametrize("bad", [
    "view_count; DROP TABLE videos", "nope", "view_count sideways", "",
    "view_count desc, title", "1", "view_count desc; --",
])
def test_order_by_whitelist(store: Store, bad: str) -> None:
    with pytest.raises(ValueError):
        store.list_videos(bad)
    store._conn.execute("SELECT 1 FROM videos")  # table still exists


def test_quota_ledger(store: Store) -> None:
    store.add_quota_entry("videos.list", 1, 3, "pull", quota_day="2026-10-01", now=T0)
    store.add_quota_entries([("videos.update", 50, 1, ""), ("thumbnails.set", 50, 1, "")],
                            quota_day="2026-10-01", now=T0)
    store.add_quota_entry("videos.list", 1, quota_day="2026-10-02", now=T0)
    assert store.quota_used("2026-10-01") == 101
    assert store.quota_used("2026-10-02") == 1
    assert store.quota_used("2026-10-03") == 0
    rows = store.quota_entries("2026-10-01")
    assert [r["operation"] for r in rows] == ["videos.list", "videos.update", "thumbnails.set"]
    assert rows[0]["count"] == 3 and rows[0]["note"] == "pull"
    assert rows[0]["ts_utc"] == "2026-10-01T12:00:00+00:00"


def test_naive_timestamp_rejected(store: Store) -> None:
    with pytest.raises(ValueError):
        store.begin_pull("chan0000001", now=datetime(2026, 10, 1))
