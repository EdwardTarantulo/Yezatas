from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fakes import FakeYouTubeReader, make_item
from yezatas.catalog import pull_catalog, video_from_api
from yezatas.edits import (
    DraftError, approve, discard, fingerprint, is_stale, issues_for, pending, rebase,
    restore_draft, save_draft, unapprove,
)
from yezatas.quota import QuotaGuard
from yezatas.store import Store

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
VID = "vid000002"  # "Repotting cactus #2", tags cactus + garden


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as s:
        pull_catalog(FakeYouTubeReader(5), s, QuotaGuard(s, 1000, lambda: NOW), NOW)
        yield s


def test_save_normalises_text(store: Store) -> None:
    result = save_draft(store, VID, "  New title  ", "Line one\r\nLine two\rLine three",
                        [" a ", "B", "", "  ", "b", "A", "c"], now=NOW)
    assert result.saved and not result.discarded and result.draft is not None
    row = store.get_draft(VID)
    assert row == result.draft
    assert row.title == "New title"
    assert row.description == "Line one\nLine two\nLine three"
    assert row.tags == ("a", "B", "c")  # empties dropped, case-insensitive duplicates dropped
    assert (row.base_title, row.base_description, row.base_tags) == (
        "Repotting cactus #2", "How to repot cactus number 2.", ("cactus", "garden"))
    assert row.source == "manual" and row.approved_at is None
    assert row.created_at == row.updated_at == NOW.isoformat()


def test_unknown_video_is_refused(store: Store) -> None:
    with pytest.raises(DraftError, match="catalog"):
        save_draft(store, "nope0000001", "t", "d", [])
    assert store.list_drafts() == []


def test_original_base_is_kept_across_edits(store: Store) -> None:
    save_draft(store, VID, "First edit", "d", ["x"], now=NOW)
    later = NOW + timedelta(minutes=5)
    result = save_draft(store, VID, "Second edit", "d2", ["y"], source="restore", note="n",
                        now=later)
    row = result.draft
    assert row is not None
    assert row.title == "Second edit"
    assert row.base_title == "Repotting cactus #2"  # not "First edit"
    assert row.created_at == NOW.isoformat() and row.updated_at == later.isoformat()
    assert row.source == "restore" and row.note == "n"
    assert len(store.list_drafts()) == 1


def test_saving_the_base_text_discards_the_draft(store: Store) -> None:
    save_draft(store, VID, "Changed", "d", [], now=NOW)
    video = store.get_video(VID)
    result = save_draft(store, VID, video.title + " ", video.description.replace("\n", "\r\n"),
                        list(video.tags), now=NOW)
    assert (result.saved, result.discarded, result.draft) == (False, True, None)
    assert store.get_draft(VID) is None
    again = save_draft(store, VID, video.title, video.description, video.tags)  # no draft at all
    assert again.discarded and not again.saved


def test_editing_clears_approval(store: Store) -> None:
    save_draft(store, VID, "Better title", "Better text", ["x"], now=NOW)
    approve(store, VID, now=NOW)
    assert store.get_draft(VID).approved_at is not None
    save_draft(store, VID, "Better title!", "Better text", ["x"], now=NOW)
    assert store.get_draft(VID).approved_at is None


def test_errors_are_saved_but_block_approval(store: Store) -> None:
    result = save_draft(store, VID, "x" * 101, "d", [], now=NOW)
    assert result.saved
    assert [i.code for i in result.issues if i.severity == "error"] == ["title_too_long"]
    assert issues_for(result.draft) == result.issues
    with pytest.raises(DraftError, match="errors"):
        approve(store, VID, now=NOW)
    assert store.get_draft(VID).approved_at is None


def test_approve_and_unapprove(store: Store) -> None:
    with pytest.raises(DraftError, match="no draft"):
        approve(store, VID)
    save_draft(store, VID, "Better title", "d", [], now=NOW)
    row = approve(store, VID, now=NOW)
    assert row.approved_at == NOW.isoformat()
    assert store.get_draft(VID) == row
    unapprove(store, VID)
    assert store.get_draft(VID).approved_at is None


def test_stale_draft_cannot_be_approved_until_rebased(store: Store) -> None:
    save_draft(store, VID, "My title", "My text", ["mine"], now=NOW)
    row = store.get_draft(VID)
    assert not is_stale(row, store.get_video(VID))
    store.update_video_text(VID, "Edited in Studio", "Studio text", ["studio"])
    assert is_stale(row, store.get_video(VID))
    with pytest.raises(DraftError, match="changed"):
        approve(store, VID)
    rebased = rebase(store, VID, now=NOW + timedelta(hours=1))
    assert rebased is not None
    assert (rebased.base_title, rebased.base_description, rebased.base_tags) == (
        "Edited in Studio", "Studio text", ("studio",))
    assert rebased.title == "My title" and rebased.updated_at == (NOW + timedelta(hours=1)).isoformat()
    assert not is_stale(rebased, store.get_video(VID))
    approve(store, VID, now=NOW)


def test_rebase_clears_approval_and_drops_a_draft_that_became_a_no_op(store: Store) -> None:
    save_draft(store, VID, "My title", "My text", ["mine"], now=NOW)
    approve(store, VID, now=NOW)
    store.update_video_text(VID, "Other", "Other text", [])
    assert rebase(store, VID).approved_at is None
    store.update_video_text(VID, "My title", "My text", ["mine"])  # Studio made the same edit
    assert rebase(store, VID) is None
    assert store.get_draft(VID) is None
    with pytest.raises(DraftError):
        rebase(store, VID)


def test_discard(store: Store) -> None:
    assert discard(store, VID) is False
    save_draft(store, VID, "Better title", "d", [], now=NOW)
    assert discard(store, VID) is True
    assert store.get_draft(VID) is None


def test_restore_draft_from_a_push_backup(store: Store) -> None:
    push_id = store.begin_push(
        VID, "manual",
        {"title": "Old title", "description": "Old text", "tags": ["old"], "categoryId": "26"})
    result = restore_draft(store, push_id, now=NOW)
    row = result.draft
    assert result.saved and row is not None
    assert (row.title, row.description, row.tags) == ("Old title", "Old text", ("old",))
    assert row.source == "restore"
    assert row.note == f"Restore of the text from before push #{push_id}"
    assert row.base_title == "Repotting cactus #2" and row.approved_at is None


def test_restore_draft_refuses_unknown_or_empty_backups(store: Store) -> None:
    with pytest.raises(DraftError):
        restore_draft(store, 999)
    empty = store.begin_push(VID, "manual", {})
    with pytest.raises(DraftError, match="no saved text"):
        restore_draft(store, empty)
    assert store.get_draft(VID) is None


def test_restoring_the_current_text_leaves_no_draft(store: Store) -> None:
    video = store.get_video(VID)
    push_id = store.begin_push(
        VID, "manual",
        {"title": video.title, "description": video.description, "tags": video.tags})
    result = restore_draft(store, push_id)
    assert result.discarded and store.get_draft(VID) is None


def test_fingerprint_changes_with_text_or_base(store: Store) -> None:
    save_draft(store, VID, "Title A", "d", ["x"], now=NOW)
    first = fingerprint(store.get_draft(VID))
    assert len(first) == 16 and int(first, 16) >= 0
    assert fingerprint(store.get_draft(VID)) == first
    save_draft(store, VID, "Title B", "d", ["x"], now=NOW)
    second = fingerprint(store.get_draft(VID))
    save_draft(store, VID, "Title B", "d2", ["x"], now=NOW)
    third = fingerprint(store.get_draft(VID))
    save_draft(store, VID, "Title B", "d2", ["x", "y"], now=NOW)
    fourth = fingerprint(store.get_draft(VID))
    store.update_video_text(VID, "Moved", "d", [])
    rebase(store, VID)
    fifth = fingerprint(store.get_draft(VID))
    assert len({first, second, third, fourth, fifth}) == 5


def test_fingerprint_ignores_approval_and_timestamps(store: Store) -> None:
    save_draft(store, VID, "Title A", "d", [], now=NOW)
    before = fingerprint(store.get_draft(VID))
    approve(store, VID, now=NOW + timedelta(days=1))
    assert fingerprint(store.get_draft(VID)) == before


def test_pending_is_ordered_by_views(store: Store) -> None:
    # the fake gives video n exactly n * 10 views; make #3 the most viewed
    store.upsert_video(video_from_api(make_item(3, views=9999)), store.begin_pull("c", NOW), NOW)
    for vid in ("vid000001", "vid000003", "vid000005"):
        save_draft(store, vid, f"New {vid}", "d", [], now=NOW)
    pairs = pending(store)
    assert [row.video_id for row, _ in pairs] == ["vid000003", "vid000005", "vid000001"]
    assert [video.view_count for _, video in pairs] == [9999, 50, 10]
    assert pending(store)[0][1].video_id == pairs[0][0].video_id
