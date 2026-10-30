import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from yezatas.store import MIGRATIONS, DraftRow, PushRecord, Store, TranslationRecord, Video

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
        assert s.schema_version == len(MIGRATIONS) == 4
        assert s._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_migrations_idempotent_on_reopen(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    with Store(path) as s:
        pid = s.begin_pull("chan0000001", now=T0)
        s.upsert_video(make_video(), pid, now=T0)
    with Store(path) as s:
        assert s.schema_version == 4
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


def test_v1_database_migrates_to_the_latest_and_keeps_its_data(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    with Store(path) as s:
        pid = s.begin_pull("chan0000001", now=T0)
        s.upsert_video(make_video(), pid, now=T0)
        s.add_quota_entry("videos.list", 1, quota_day="2026-10-01", now=T0)
    conn = sqlite3.connect(path)  # turn it back into a version 1 database
    conn.executescript("DROP TABLE drafts; DROP TABLE pushes; DROP TABLE agent_runs; "
                       "DROP TABLE translations; PRAGMA user_version = 1;")
    conn.close()
    with Store(path) as s:
        assert s.schema_version == 4
        video = s.get_video("vid00000001")
        assert video is not None and video.title == "How to repot a cactus"
        assert s.quota_used("2026-10-01") == 1
        assert s.list_drafts() == [] and s.list_pushes() == [] and s.list_agent_runs() == []
        s.put_draft(make_draft())
        assert s.get_draft("vid00000001") is not None


def test_v3_drafts_get_the_catalog_language_as_their_base(tmp_path: Path) -> None:
    path = tmp_path / "v3.sqlite"
    loc = {"de": {"title": "Kaktus umtopfen", "description": "Schritt für Schritt."}}
    with Store(path) as s:
        pid = s.begin_pull("chan0000001", now=T0)
        s.upsert_video(make_video(default_language="en", localizations=loc), pid, now=T0)
        s.upsert_video(make_video("vid00000002", default_language=None, localizations={}), pid,
                       now=T0)
        s.put_draft(make_draft())
        s.put_draft(make_draft("vid00000002"))
    conn = sqlite3.connect(path)  # turn it back into a version 3 database
    conn.executescript(
        "DROP TABLE translations; ALTER TABLE drafts DROP COLUMN default_language; "
        "ALTER TABLE drafts DROP COLUMN localizations_json; "
        "ALTER TABLE drafts DROP COLUMN base_default_language; "
        "ALTER TABLE drafts DROP COLUMN base_localizations_json; PRAGMA user_version = 3;")
    conn.close()
    with Store(path) as s:
        row = s.get_draft("vid00000001")
        assert row is not None and row.base_default_language == "en"
        assert row.base_localizations == loc
        assert row.default_language is None and row.localizations is None
        other = s.get_draft("vid00000002")
        assert other is not None and other.base_localizations == {}
        assert other.base_default_language is None


def make_draft(vid: str = "vid00000001", **kw: object) -> DraftRow:
    base: dict[str, object] = dict(
        video_id=vid, title="New title", description="New text\nline two", tags=("a", "b"),
        base_title="How to repot a cactus", base_description="Step by step.\nSecond line.",
        base_tags=("cactus", "plants"), source="manual", note="", created_at="t1",
        updated_at="t2", approved_at=None,
    )
    base.update(kw)
    return DraftRow(**base)  # type: ignore[arg-type]


def test_draft_round_trip_and_upsert(store: Store) -> None:
    pid = store.begin_pull("chan0000001", now=T0)
    store.upsert_video(make_video(), pid, now=T0)
    row = make_draft(title="Kaktusų 🌵", tags=("🌵", "kaktusas"))
    store.put_draft(row)
    assert store.get_draft("vid00000001") == row
    assert row.draft.title == "Kaktusų 🌵" and row.draft.tags == ("🌵", "kaktusas")
    assert row.base.description == "Step by step.\nSecond line."
    changed = make_draft(title="Other", note="n", approved_at="t3")
    store.put_draft(changed)
    assert store.list_drafts() == [changed]
    store.set_draft_approval("vid00000001", None)
    assert store.get_draft("vid00000001").approved_at is None  # type: ignore[union-attr]
    store.set_draft_approval("vid00000001", "t9")
    assert store.get_draft("vid00000001").approved_at == "t9"  # type: ignore[union-attr]
    assert store.delete_draft("vid00000001") is True
    assert store.delete_draft("vid00000001") is False
    assert store.get_draft("vid00000001") is None


def test_draft_needs_a_catalog_video(store: Store) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.put_draft(make_draft("unknown00001"))


def test_update_video_text_changes_columns_and_raw_snippet_without_snapshot(store: Store) -> None:
    pid = store.begin_pull("chan0000001", now=T0)
    store.upsert_video(make_video(raw={"snippet": {"title": "old", "categoryId": "22"}}), pid, now=T0)
    later = T0 + timedelta(hours=1)
    store.update_video_text("vid00000001", "T", "D", ["x", "y"], now=later)
    video = store.get_video("vid00000001")
    assert video is not None
    assert (video.title, video.description, video.tags) == ("T", "D", ["x", "y"])
    assert video.raw["snippet"] == {"title": "T", "description": "D", "tags": ["x", "y"],
                                    "categoryId": "22"}
    assert video.view_count == 100
    assert len(store.snapshots_for("vid00000001")) == 1
    row = store._conn.execute("SELECT updated_at FROM videos").fetchone()
    assert row[0] == later.isoformat()
    with pytest.raises(KeyError):
        store.update_video_text("nope", "T", "D", [])


def test_push_records_round_trip_newest_first(store: Store) -> None:
    first = store.begin_push("vid00000001", "manual", {"title": "old", "tags": ["a"]}, now=T0)
    record = store.get_push(first)
    assert isinstance(record, PushRecord)
    assert (record.status, record.finished_at, record.after, record.message) == (
        "pending", None, {}, "")
    assert record.before == {"title": "old", "tags": ["a"]} and record.started_at == T0.isoformat()
    store.finish_push(first, "pushed", {"title": "new"}, "ok", now=T0 + timedelta(seconds=2))
    second = store.begin_push("vid00000002", "restore", {"title": "x"}, now=T0)
    third = store.begin_push("vid00000001", "manual", {"title": "y"}, now=T0)
    done = store.get_push(first)
    assert done is not None and done.status == "pushed" and done.after == {"title": "new"}
    assert done.finished_at == (T0 + timedelta(seconds=2)).isoformat() and done.message == "ok"
    assert [r.id for r in store.list_pushes()] == [third, second, first]
    assert [r.id for r in store.list_pushes("vid00000001")] == [third, first]
    assert [r.id for r in store.list_pushes(limit=2)] == [third, second]
    assert store.get_push(999) is None


def test_agent_runs_round_trip_newest_first(store: Store) -> None:
    first = store.begin_agent_run("vid00000001", "muse-spark-1.3", now=T0)
    second = store.begin_agent_run("vid00000002", "muse-spark-1.3", now=T0)
    run = store.get_agent_run(first)
    assert run is not None and run.status == "running" and run.finished_at is None
    assert run.log == [] and run.result == {} and run.input_tokens == 0
    store.finish_agent_run(first, "drafted", "Longer title", [{"tool": "get_scorecard"}],
                           {"title": "New"}, tokens=(1200, 300, 450), now=T0)
    run = store.get_agent_run(first)
    assert run is not None and run.status == "drafted" and run.summary == "Longer title"
    assert run.log == [{"tool": "get_scorecard"}] and run.result == {"title": "New"}
    assert (run.input_tokens, run.cached_tokens, run.output_tokens) == (1200, 300, 450)
    assert [r.id for r in store.list_agent_runs()] == [second, first]
    assert [r.id for r in store.list_agent_runs("vid00000001")] == [first]
    assert store.get_agent_run(999) is None


def test_draft_language_and_translations_round_trip(store: Store) -> None:
    store.upsert_video(make_video(), store.begin_pull("chan0000001", now=T0), now=T0)
    store.put_draft(make_draft())
    row = store.get_draft("vid00000001")
    assert row is not None and row.default_language is None and row.localizations is None
    assert row.base_default_language is None and row.base_localizations == {}
    loc = {"es": {"title": "Cactus", "description": "Hola"}}
    store.put_draft(replace(make_draft(), default_language="en", localizations=loc,
                            base_default_language="lt", base_localizations={"de": {"title": "x", "description": "y"}}))
    row = store.get_draft("vid00000001")
    assert row is not None and row.default_language == "en" and row.localizations == loc
    assert row.base_default_language == "lt" and row.base_localizations["de"]["title"] == "x"
    store.put_draft(replace(row, localizations={}))  # an empty map is not "leave alone"
    assert store.get_draft("vid00000001").localizations == {}


def test_update_video_text_sets_language_and_translations_only_when_given(store: Store) -> None:
    pid = store.begin_pull("chan0000001", now=T0)
    store.upsert_video(make_video(default_language="lt", localizations={"de": {"title": "a"}}), pid, now=T0)
    store.update_video_text("vid00000001", "T", "D", ["x"], T0)
    video = store.get_video("vid00000001")
    assert video.default_language == "lt" and video.localizations == {"de": {"title": "a"}}
    loc = {"es": {"title": "T es", "description": "D es"}}
    store.update_video_text("vid00000001", "T", "D", ["x"], T0, default_language="en", localizations=loc)
    video = store.get_video("vid00000001")
    assert video.default_language == "en" and video.localizations == loc
    assert video.raw["localizations"] == loc


def test_translations_round_trip_and_latest_per_language(store: Store) -> None:
    def record(language: str, title: str, status: str = "verified") -> TranslationRecord:
        return TranslationRecord(0, "vid00000001", language, "en", "abc123", title, "desc",
                                 "muse-spark-open", "muse-glimmer:30b", status, 5, [], [],
                                 {"title": "back", "description": "back d"}, 100, 50, "")
    first = store.add_translation(record("es", "Uno"), now=T0)
    store.add_translation(record("de", "Eins"), now=T0)
    newest = store.add_translation(record("es", "Uno bis", "flagged"), now=T0)
    got = store.get_translation(first)
    assert got is not None and got.title == "Uno" and got.back_translation["title"] == "back"
    assert got.created_at and got.id == first
    assert [t.id for t in store.list_translations("vid00000001")][0] == newest
    latest = store.latest_translations("vid00000001")
    assert set(latest) == {"es", "de"} and latest["es"].title == "Uno bis"
    assert latest["es"].status == "flagged" and store.list_translations("other") == []

