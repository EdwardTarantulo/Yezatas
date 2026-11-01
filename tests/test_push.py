from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeYouTubeWriter, make_http_error
from yezatas.catalog import pull_catalog
from yezatas.edits import (
    DraftError, add_translations, approve, fingerprint, remove_translation, restore_draft,
    restore_run, save_draft, set_language,
)
from yezatas.push import KEEP_SNIPPET_KEYS, backup_of, push_drafts, push_units
from yezatas.quota import QuotaExceeded, QuotaGuard
from yezatas.store import Store

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
DAY = "2026-01-15"
V1, V2, V3 = "vid000001", "vid000002", "vid000003"
V5 = "vid000005"  # has a German translation on YouTube
DE5 = {"title": "Kaktus umtopfen #5", "description": "Anleitung 5."}
ES = ("Replantar un cactus", "Como replantar un cactus.")
PT = ("Replantar um cacto", "Como replantar um cacto.")


def make_guard(store: Store, budget: int = 10000) -> QuotaGuard:
    return QuotaGuard(store, budget, lambda: NOW)


def setup(tmp_path: Path, count: int = 5) -> tuple[Store, FakeYouTubeWriter]:
    store = Store(tmp_path / "c.sqlite")
    writer = FakeYouTubeWriter(count)
    pull_catalog(writer, store, make_guard(store), NOW)
    writer.calls.clear()
    return store, writer


@pytest.fixture
def env(tmp_path: Path):
    store, writer = setup(tmp_path)
    yield store, writer
    store.close()


def test_catalog_keeps_the_tag_order_sent_not_the_response_order(env) -> None:
    # Real YouTube: the videos.update response lists tags alphabetically, a later read gives the
    # order sent. Storing the response's order made the next push of that video a false conflict.
    store, writer = env
    real_update = writer.update_video

    def sorted_response(body, part):
        response = real_update(body, part)
        response["snippet"]["tags"] = sorted(response["snippet"]["tags"])
        return response

    writer.update_video = sorted_response
    save_draft(store, V1, "A better title", "New text", ["zebra", "apple"], now=NOW)
    approve(store, V1, now=NOW)
    seen = fingerprint(store.get_draft(V1))
    result = push_drafts(writer, store, make_guard(store), {V1: seen}, lambda _: None, now=NOW)
    assert result.items[0].status == "pushed"
    assert store.get_video(V1).tags == ["zebra", "apple"]
    approved_draft(store, V1, "Second change")  # the next push is not a conflict
    seen = fingerprint(store.get_draft(V1))
    result = push_drafts(writer, store, make_guard(store), {V1: seen}, lambda _: None, now=NOW)
    assert result.items[0].status == "pushed"


def approved_draft(store: Store, video_id: str, title: str | None = None) -> str:
    """Save and approve a draft; returns the fingerprint the creator saw."""
    save_draft(store, video_id, title or f"New title for {video_id}", "New text", ["x", "y"],
               now=NOW)
    approve(store, video_id, now=NOW)
    return fingerprint(store.get_draft(video_id))


def used(store: Store) -> list[tuple[str, int]]:
    """Quota ledger entries written by pushes (not by the catalog pull in setup)."""
    return [(r["operation"], r["units"]) for r in store.quota_entries(DAY) if r["note"] == "push"]


def test_push_units_and_backup_of() -> None:
    assert push_units(1) == [("videos.list", 1), ("videos.update", 1)]
    assert push_units(50) == [("videos.list", 1), ("videos.update", 50)]
    assert push_units(51) == [("videos.list", 2), ("videos.update", 51)]
    assert push_units(0) == []
    assert KEEP_SNIPPET_KEYS == ("categoryId", "defaultLanguage", "defaultAudioLanguage")
    live = {"title": "T", "description": "D", "categoryId": "22", "channelId": "UC", "tags": None,
            "thumbnails": {}}
    assert backup_of(live) == {"title": "T", "description": "D", "tags": [], "categoryId": "22"}


def test_happy_path(env) -> None:
    store, writer = env
    seen = approved_draft(store, V1, "A better title")
    before_text = store.get_video(V1)
    messages: list[str] = []
    result = push_drafts(writer, store, make_guard(store), {V1: seen}, messages.append, now=NOW)

    assert [(i.video_id, i.status) for i in result.items] == [(V1, "pushed")]
    assert result.units_used == 51
    assert messages  # progress was reported
    # the body keeps the live category/languages and nothing else
    (body, part), = writer.updates
    assert part == "snippet" and set(body) == {"id", "snippet"} and body["id"] == V1
    assert body["snippet"] == {
        "title": "A better title", "description": "New text", "tags": ["x", "y"],
        "categoryId": "26", "defaultLanguage": "en", "defaultAudioLanguage": "en"}
    # YouTube (the fake) kept the fields; the catalog row and its raw snippet follow
    live = writer.item(V1)["snippet"]
    assert live["categoryId"] == "26" and live["title"] == "A better title"
    video = store.get_video(V1)
    assert (video.title, video.description, video.tags) == ("A better title", "New text", ["x", "y"])
    assert video.raw["snippet"]["title"] == "A better title"
    assert store.get_draft(V1) is None
    # the push record holds the backup and what was sent
    record = store.get_push(result.items[0].push_id)
    assert record.status == "pushed" and record.finished_at is not None and record.source == "manual"
    assert record.before == {
        "title": before_text.title, "description": before_text.description,
        "tags": before_text.tags, "categoryId": "26", "defaultLanguage": "en",
        "defaultAudioLanguage": "en"}
    assert record.after == body["snippet"]
    assert [c[0] for c in writer.calls] == ["videos", "update_video"]
    assert writer.calls[0][1] == ((V1,), "snippet,localizations")  # the live read also asks for translations
    assert used(store) == [("videos.list", 1), ("videos.update", 50)]


def test_backup_is_written_before_videos_update(env) -> None:
    store, writer = env
    original = store.get_video(V1)
    seen_at_update: list[Any] = []

    def check(body: dict[str, Any]) -> None:
        records = store.list_pushes(body["id"])
        seen_at_update.append([(r.status, r.before["title"], r.before["description"])
                               for r in records])
        assert store.quota_used(DAY) >= 51  # the update was charged before it was made

    writer.on_update = check
    push_drafts(writer, store, make_guard(store), {V1: approved_draft(store, V1)}, now=NOW)
    assert seen_at_update == [[("pending", original.title, original.description)]]


def test_missing_snippet_keys_are_not_invented(env) -> None:
    store, writer = env
    del writer.item(V1)["snippet"]["defaultAudioLanguage"]
    del writer.item(V1)["snippet"]["categoryId"]
    push_drafts(writer, store, make_guard(store), {V1: approved_draft(store, V1)}, now=NOW)
    assert set(writer.updates[0][0]["snippet"]) == {
        "title", "description", "tags", "defaultLanguage"}


def test_studio_edit_is_a_conflict_and_nothing_is_pushed(env) -> None:
    store, writer = env
    seen = approved_draft(store, V1)
    old_title = store.get_video(V1).title
    writer.studio_edit(V1, title="Edited in Studio", tags=["studio"])
    result = push_drafts(writer, store, make_guard(store), {V1: seen}, now=NOW)

    item, = result.items
    assert item.status == "conflict" and "Studio" in item.message
    assert writer.updates == []  # nothing reached YouTube
    video = store.get_video(V1)
    assert (video.title, video.tags) == ("Edited in Studio", ["studio"])
    assert video.raw["snippet"]["title"] == "Edited in Studio"
    row = store.get_draft(V1)
    assert row is not None and row.approved_at is None
    assert row.base_title == old_title  # the draft is now stale
    record = store.get_push(item.push_id)
    assert record.status == "conflict" and record.after == {}
    assert record.before["title"] == "Edited in Studio"
    assert used(store) == [("videos.list", 1)]
    assert result.units_used == 1
    with pytest.raises(DraftError):  # approving again is refused until the draft is rebased
        approve(store, V1)


def test_changed_description_alone_is_a_conflict(env) -> None:
    store, writer = env
    seen = approved_draft(store, V1)
    writer.studio_edit(V1, description="Different text")
    assert push_drafts(writer, store, make_guard(store), {V1: seen}).items[0].status == "conflict"
    assert writer.updates == []


def test_video_gone_from_youtube_is_missing(env) -> None:
    store, writer = env
    seen = approved_draft(store, V1)
    writer.remove(V1)
    result = push_drafts(writer, store, make_guard(store), {V1: seen}, now=NOW)
    assert result.items[0].status == "missing" and result.items[0].push_id is None
    assert writer.updates == [] and store.list_pushes() == []
    assert store.get_draft(V1) is not None


def test_failed_update_continues_with_the_next_video(env) -> None:
    store, writer = env
    approved = {v: approved_draft(store, v) for v in (V1, V2, V3)}
    writer.update_errors[V2] = make_http_error(400, "invalidDescription", "Bad description.")
    result = push_drafts(writer, store, make_guard(store), approved, now=NOW)
    assert [(i.video_id, i.status) for i in result.items] == [
        (V1, "pushed"), (V2, "failed"), (V3, "pushed")]
    failed = result.items[1]
    assert "400" in failed.message and "invalidDescription" in failed.message
    assert "Bad description." in failed.message
    record = store.get_push(failed.push_id)
    assert record.status == "failed" and record.after == {} and record.message == failed.message
    assert store.get_draft(V2) is not None  # kept so the creator can fix and retry
    assert store.get_draft(V1) is None and store.get_draft(V3) is None
    assert len([c for c in writer.calls if c[0] == "update_video"]) == 3


@pytest.mark.parametrize("status, reason", [
    (403, "quotaExceeded"), (403, "insufficientPermissions"), (401, "authError"),
    (400, "dailyLimitExceeded"),
])
def test_blocking_error_stops_and_marks_the_rest_not_attempted(env, status: int, reason: str) -> None:
    store, writer = env
    approved = {v: approved_draft(store, v) for v in (V1, V2, V3)}
    writer.update_errors[V2] = make_http_error(status, reason)
    result = push_drafts(writer, store, make_guard(store), approved, now=NOW)
    assert [(i.video_id, i.status) for i in result.items] == [
        (V1, "pushed"), (V2, "failed"), (V3, "not_attempted")]
    assert result.items[2].push_id is None
    assert [c[1][0] for c in writer.calls if c[0] == "update_video"] == [V1, V2]
    assert store.get_draft(V3) is not None and store.get_draft(V3).approved_at is not None
    assert len(store.list_pushes()) == 2


def test_failed_live_read_marks_failed_and_pushes_nothing(env) -> None:
    store, writer = env
    approved = {v: approved_draft(store, v) for v in (V1, V2)}

    def broken(ids, part):
        raise make_http_error(403, "forbidden", "No access.")

    writer.videos = broken  # type: ignore[method-assign]
    result = push_drafts(writer, store, make_guard(store), approved, now=NOW)
    assert [i.status for i in result.items] == ["failed", "failed"]
    assert writer.updates == [] and store.list_pushes() == []


def test_changed_after_review_is_skipped(env) -> None:
    store, writer = env
    seen = approved_draft(store, V1, "Reviewed title")
    save_draft(store, V1, "Sneaky different title", "New text", ["x", "y"], now=NOW)
    approve(store, V1, now=NOW)  # approved again, but the creator saw the old fingerprint
    result = push_drafts(writer, store, make_guard(store), {V1: seen}, now=NOW)
    assert result.items[0].status == "skipped" and "reviewed" in result.items[0].message
    assert writer.calls == [] and result.units_used == 0


def test_unapproved_missing_stale_and_invalid_are_skipped(env) -> None:
    store, writer = env
    save_draft(store, V1, "Not approved", "d", [], now=NOW)
    stale_seen = approved_draft(store, V2)
    store.update_video_text(V2, "Studio change", "x", [])  # catalog moved on after approval
    save_draft(store, V3, "x" * 101, "d", [], now=NOW)  # has an error
    bad = store.get_draft(V3)
    store.put_draft(replace(bad, approved_at=NOW.isoformat()))  # approved by force
    approved = {
        V1: fingerprint(store.get_draft(V1)),
        V2: stale_seen,
        V3: fingerprint(store.get_draft(V3)),
        "vid000004": "0123456789abcdef",  # no draft
    }
    result = push_drafts(writer, store, make_guard(store), approved, now=NOW)
    assert [i.status for i in result.items] == ["skipped"] * 4
    messages = [i.message for i in result.items]
    assert "not approved" in messages[0]
    assert "changed" in messages[1]
    assert "errors" in messages[2]
    assert "no draft" in messages[3]
    assert writer.calls == [] and result.units_used == 0


def test_quota_exceeded_before_any_youtube_call(env) -> None:
    store, writer = env
    approved = {v: approved_draft(store, v) for v in (V1, V2)}
    before = store.quota_used(DAY)
    with pytest.raises(QuotaExceeded):
        push_drafts(writer, store, make_guard(store, budget=before + 100), approved, now=NOW)
    assert writer.calls == [] and store.list_pushes() == []
    assert store.quota_used(DAY) == before
    # exactly enough works
    result = push_drafts(writer, store, make_guard(store, budget=before + 101), approved, now=NOW)
    assert [i.status for i in result.items] == ["pushed", "pushed"]


def test_skipped_drafts_do_not_count_towards_the_quota_plan(env) -> None:
    store, writer = env
    seen = approved_draft(store, V1)
    save_draft(store, V2, "Not approved", "d", [], now=NOW)
    before = store.quota_used(DAY)
    result = push_drafts(writer, store, make_guard(store, budget=before + 51),
                         {V1: seen, V2: fingerprint(store.get_draft(V2))}, now=NOW)
    assert [i.status for i in result.items] == ["pushed", "skipped"]


def test_live_text_is_fetched_in_batches_of_50(tmp_path: Path) -> None:
    store, writer = setup(tmp_path, count=51)
    with store:
        approved = {f"vid{n:06d}": approved_draft(store, f"vid{n:06d}") for n in range(1, 52)}
        result = push_drafts(writer, store, make_guard(store, budget=100000), approved, now=NOW)
        reads = [c[1][0] for c in writer.calls if c[0] == "videos"]
        assert [len(ids) for ids in reads] == [50, 1]
        assert [c[0] for c in writer.calls].count("update_video") == 51
        assert [i.status for i in result.items] == ["pushed"] * 51
        assert result.units_used == 2 + 51 * 50
        assert store.list_drafts() == []


def test_report_order_follows_the_request(env) -> None:
    store, writer = env
    approved = {v: approved_draft(store, v) for v in (V3, V1, V2)}
    result = push_drafts(writer, store, make_guard(store), approved, now=NOW)
    assert [i.video_id for i in result.items] == [V3, V1, V2]
    assert result.count("pushed") == 3 and result.count("failed") == 0


def test_pushed_history_can_be_restored_through_the_same_path(env) -> None:
    store, writer = env
    original = store.get_video(V1)
    pushed = push_drafts(writer, store, make_guard(store), {V1: approved_draft(store, V1)}, now=NOW)
    restored = restore_draft(store, pushed.items[0].push_id, now=NOW).draft
    assert restored is not None and restored.title == original.title
    assert restored.approved_at is None  # still needs the creator's approval
    approve(store, V1, now=NOW)
    second = push_drafts(writer, store, make_guard(store), {V1: fingerprint(store.get_draft(V1))}, now=NOW)
    assert second.items[0].status == "pushed"
    assert writer.item(V1)["snippet"]["title"] == original.title
    assert store.get_push(second.items[0].push_id).source == "restore"


# --- push runs ---------------------------------------------------------------------------------


def test_a_multi_video_push_is_one_closed_run(env) -> None:
    store, writer = env
    approved = {v: approved_draft(store, v) for v in (V1, V2, V3)}
    result = push_drafts(writer, store, make_guard(store), approved, now=NOW)
    pushes = store.list_pushes()
    assert len(pushes) == 3 and len({p.run_id for p in pushes}) == 1
    run = store.get_push_run(pushes[0].run_id)
    assert run is not None and run.video_count == 3 and run.finished_at == NOW.isoformat()
    assert [p.id for p in store.list_pushes_of_run(run.id)] == [i.push_id for i in result.items]
    second = push_drafts(writer, store, make_guard(store),
                          {V1: approved_draft(store, V1, "Another title")}, now=NOW)
    assert store.get_push(second.items[0].push_id).run_id != run.id  # each call is its own run


def test_conflicts_and_failures_are_part_of_the_run(env) -> None:
    store, writer = env
    approved = {v: approved_draft(store, v) for v in (V1, V2, V3)}
    writer.studio_edit(V1, title="Changed by hand")
    writer.update_errors[V2] = make_http_error(400, "invalidDescription", "Bad description.")
    push_drafts(writer, store, make_guard(store), approved, now=NOW)
    pushes = store.list_pushes_of_run(store.list_pushes()[0].run_id)
    assert [p.status for p in pushes] == ["conflict", "failed", "pushed"]
    assert store.get_push_run(pushes[0].run_id).video_count == 3


def test_a_stopped_push_still_closes_its_run(env) -> None:
    store, writer = env
    approved = {v: approved_draft(store, v) for v in (V1, V2, V3)}
    writer.update_errors[V2] = make_http_error(403, "quotaExceeded")
    push_drafts(writer, store, make_guard(store), approved, now=NOW)
    run = store.get_push_run(store.list_pushes()[0].run_id)
    assert run.finished_at is not None and run.video_count == 2  # V3 was never attempted


def test_a_crash_inside_a_push_closes_the_run(env) -> None:
    store, writer = env
    approved = {v: approved_draft(store, v) for v in (V1, V2)}

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    store.update_video_text = broken  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        push_drafts(writer, store, make_guard(store), approved, now=NOW)
    run = store.get_push_run(1)
    assert run is not None and run.finished_at is not None and run.video_count == 1


def test_no_run_is_made_when_nothing_is_pushed(env) -> None:
    store, writer = env
    save_draft(store, V1, "Not approved", "d", [], now=NOW)
    push_drafts(writer, store, make_guard(store), {V1: fingerprint(store.get_draft(V1))}, now=NOW)
    assert store.get_push_run(1) is None and writer.calls == []
    approved = {V2: approved_draft(store, V2)}
    with pytest.raises(QuotaExceeded):
        push_drafts(writer, store, make_guard(store, budget=1), approved, now=NOW)
    assert store.get_push_run(1) is None  # refused before any call: no run


def test_a_pushed_run_can_be_undone_into_drafts_without_touching_youtube(env) -> None:
    store, writer = env
    originals = {v: store.get_video(v).title for v in (V1, V2)}
    approved = {v: approved_draft(store, v) for v in (V1, V2)}
    push_drafts(writer, store, make_guard(store), approved, now=NOW)
    writer.calls.clear()
    updates = len(writer.updates)
    result = restore_run(store, store.list_pushes()[0].run_id, now=NOW)
    assert result.count("drafted") == 2
    assert {v: store.get_draft(v).title for v in (V1, V2)} == originals
    assert writer.calls == [] and len(writer.updates) == updates


# --- language and translations--------------------------------------------------------------


def approve_now(store: Store, video_id: str) -> str:
    approve(store, video_id, now=NOW)
    return fingerprint(store.get_draft(video_id))


def test_backup_of_adds_translations_only_when_given() -> None:
    live = {"title": "T", "description": "D", "defaultLanguage": "en"}
    assert backup_of(live) == {"title": "T", "description": "D", "tags": [], "defaultLanguage": "en"}
    assert backup_of(live, None) == backup_of(live)
    loc = {"de": {"title": "a", "description": "b", "extra": "x"}}
    assert backup_of(live, loc)["localizations"] == {"de": {"title": "a", "description": "b"}}
    assert backup_of(live, {})["localizations"] == {}


def test_a_text_only_draft_is_pushed_exactly_as_before(env) -> None:
    store, writer = env
    seen = approved_draft(store, V5, "Only the text")  # V5 has a translation that must not be touched
    result = push_drafts(writer, store, make_guard(store), {V5: seen}, now=NOW)
    assert result.items[0].status == "pushed"
    (body, part), = writer.updates
    assert part == "snippet" and set(body) == {"id", "snippet"}
    assert body["snippet"] == {
        "title": "Only the text", "description": "New text", "tags": ["x", "y"],
        "categoryId": "26", "defaultLanguage": "en", "defaultAudioLanguage": "en"}
    assert writer.item(V5)["localizations"] == {"de": DE5}
    record = store.get_push(result.items[0].push_id)
    assert "localizations" not in record.before and record.after == body["snippet"]
    assert used(store) == [("videos.list", 1), ("videos.update", 50)]


def test_a_translation_only_draft_sends_the_complete_map_and_keeps_the_text(env) -> None:
    store, writer = env
    live_before = writer.item(V5)["snippet"]
    add_translations(store, V5, "en", {"es": ES, "pt": PT}, now=NOW)
    messages: list[str] = []
    result = push_drafts(writer, store, make_guard(store), {V5: approve_now(store, V5)},
                         messages.append, now=NOW)
    assert [(i.video_id, i.status) for i in result.items] == [(V5, "pushed")]
    (body, part), = writer.updates
    assert part == "snippet,localizations" and set(body) == {"id", "snippet", "localizations"}
    # title, description and tags are the live ones, so the translation-only push changes none of them
    assert body["snippet"] == {
        "title": live_before["title"], "description": live_before["description"],
        "tags": live_before["tags"], "categoryId": "26", "defaultLanguage": "en",
        "defaultAudioLanguage": "en"}
    assert body["localizations"] == {"de": DE5, "es": {"title": ES[0], "description": ES[1]},
                                     "pt": {"title": PT[0], "description": PT[1]}}
    assert writer.item(V5)["localizations"] == body["localizations"]
    assert writer.calls[0][1] == ((V5,), "snippet,localizations")
    assert used(store) == [("videos.list", 1), ("videos.update", 50)]  # quota as always
    # the catalog follows YouTube and the draft is done
    video = store.get_video(V5)
    assert video.localizations == body["localizations"] and video.default_language == "en"
    assert video.raw["localizations"] == body["localizations"]
    assert store.get_draft(V5) is None
    # the push record keeps the translations from before and what was sent
    record = store.get_push(result.items[0].push_id)
    assert record.before["localizations"] == {"de": DE5}
    assert record.after == {**body["snippet"], "localizations": body["localizations"]}
    assert record.status == "pushed"


def test_existing_languages_survive_a_push_that_adds_one(env) -> None:
    store, writer = env
    add_translations(store, V5, "en", {"fr": ("Titre", "Texte")}, now=NOW)
    push_drafts(writer, store, make_guard(store), {V5: approve_now(store, V5)}, now=NOW)
    assert set(writer.item(V5)["localizations"]) == {"de", "fr"}


def test_removing_a_translation_pushes_the_smaller_map(env) -> None:
    store, writer = env
    remove_translation(store, V5, "de", now=NOW)
    result = push_drafts(writer, store, make_guard(store), {V5: approve_now(store, V5)}, now=NOW)
    assert result.items[0].status == "pushed"
    assert writer.updates[0][0]["localizations"] == {}
    assert "localizations" not in writer.item(V5)
    assert store.get_video(V5).localizations == {}


def test_a_language_only_draft_sends_the_live_translations(env) -> None:
    store, writer = env
    set_language(store, V5, "es", now=NOW)
    result = push_drafts(writer, store, make_guard(store), {V5: approve_now(store, V5)}, now=NOW)
    assert result.items[0].status == "pushed"
    (body, part), = writer.updates
    assert part == "snippet,localizations"
    assert body["snippet"]["defaultLanguage"] == "es"
    assert body["localizations"] == {"de": DE5}  # the live map, untouched
    assert writer.item(V5)["snippet"]["defaultLanguage"] == "es"
    assert store.get_video(V5).default_language == "es"
    assert store.get_video(V5).raw["snippet"]["defaultLanguage"] == "es"


def test_a_video_without_a_language_gets_the_source_language(tmp_path: Path) -> None:
    store, writer = setup(tmp_path)
    with store:
        del writer.item(V1)["snippet"]["defaultLanguage"]
        pull_catalog(writer, store, make_guard(store), NOW)
        assert store.get_video(V1).default_language is None
        add_translations(store, V1, "en", {"es": ES}, now=NOW)
        result = push_drafts(writer, store, make_guard(store), {V1: approve_now(store, V1)}, now=NOW)
        assert result.items[0].status == "pushed"
        (body, part), = writer.updates
        assert body["snippet"]["defaultLanguage"] == "en"
        assert writer.item(V1)["snippet"]["defaultLanguage"] == "en"
        assert store.get_video(V1).default_language == "en"
        record = store.get_push(result.items[0].push_id)
        assert "defaultLanguage" not in record.before and record.before["localizations"] == {}


def test_text_and_translations_in_one_push(env) -> None:
    store, writer = env
    save_draft(store, V1, "A better title", "New text", ["x"], now=NOW)
    add_translations(store, V1, "en", {"es": ES}, now=NOW)
    push_drafts(writer, store, make_guard(store), {V1: approve_now(store, V1)}, now=NOW)
    (body, part), = writer.updates
    assert part == "snippet,localizations"
    assert (body["snippet"]["title"], body["snippet"]["description"]) == ("A better title", "New text")
    assert set(body["localizations"]) == {"es"}


def test_studio_change_of_translations_is_a_conflict(env) -> None:
    store, writer = env
    add_translations(store, V5, "en", {"es": ES}, now=NOW)
    seen = approve_now(store, V5)
    writer.studio_localizations(V5, {"de": {"title": "Studio Titel", "description": "x"}})
    result = push_drafts(writer, store, make_guard(store), {V5: seen}, now=NOW)
    item, = result.items
    assert item.status == "conflict" and "Studio" in item.message
    assert writer.updates == []
    # the catalog shows the live translations now, so the draft is stale and unapproved
    video = store.get_video(V5)
    assert video.localizations == {"de": {"title": "Studio Titel", "description": "x"}}
    row = store.get_draft(V5)
    assert row.approved_at is None and row.base_localizations == {"de": DE5}
    record = store.get_push(item.push_id)
    assert record.status == "conflict" and record.after == {}
    assert record.before["localizations"] == {"de": {"title": "Studio Titel", "description": "x"}}
    assert used(store) == [("videos.list", 1)]
    with pytest.raises(DraftError):
        approve(store, V5)


def test_studio_change_of_the_language_is_a_conflict_for_a_draft_that_sets_it(env) -> None:
    store, writer = env
    set_language(store, V1, "es", now=NOW)
    seen = approve_now(store, V1)
    writer.studio_edit(V1, defaultLanguage="fr")
    item, = push_drafts(writer, store, make_guard(store), {V1: seen}, now=NOW).items
    assert item.status == "conflict" and writer.updates == []
    assert store.get_video(V1).default_language == "fr"


def test_a_translation_added_in_studio_is_a_conflict_even_when_text_is_unchanged(env) -> None:
    store, writer = env
    add_translations(store, V1, "en", {"es": ES}, now=NOW)
    seen = approve_now(store, V1)
    writer.studio_localizations(V1, {"fr": {"title": "T", "description": "d"}})
    assert push_drafts(writer, store, make_guard(store), {V1: seen}).items[0].status == "conflict"
    assert writer.updates == []


def test_translation_changes_in_studio_do_not_block_a_text_only_draft(env) -> None:
    store, writer = env
    seen = approved_draft(store, V5)
    writer.studio_localizations(V5, {"de": {"title": "Studio Titel", "description": "x"}})
    item, = push_drafts(writer, store, make_guard(store), {V5: seen}, now=NOW).items
    assert item.status == "pushed"
    assert writer.item(V5)["localizations"]["de"]["title"] == "Studio Titel"  # left alone


def test_a_failed_translation_push_keeps_the_draft_and_the_backup(env) -> None:
    store, writer = env
    add_translations(store, V5, "en", {"es": ES}, now=NOW)
    seen = approve_now(store, V5)
    writer.update_errors[V5] = make_http_error(400, "invalidLocalization", "Bad translation.")
    item, = push_drafts(writer, store, make_guard(store), {V5: seen}, now=NOW).items
    assert item.status == "failed" and "invalidLocalization" in item.message
    assert store.get_draft(V5) is not None
    assert store.get_push(item.push_id).before["localizations"] == {"de": DE5}
    assert "es" not in writer.item(V5).get("localizations", {})


def test_a_draft_with_translation_errors_is_not_pushed(env) -> None:
    store, writer = env
    add_translations(store, V1, "en", {"es": ("x" * 101, "d")}, now=NOW)
    store.put_draft(replace(store.get_draft(V1), approved_at=NOW.isoformat()))  # approved by force
    item, = push_drafts(writer, store, make_guard(store), {V1: fingerprint(store.get_draft(V1))}).items
    assert item.status == "skipped" and "errors" in item.message
    assert writer.calls == []


def test_the_backup_of_a_translation_push_restores_the_translations(env) -> None:
    store, writer = env
    original = store.get_video(V5)
    add_translations(store, V5, "en", {"es": ES}, now=NOW)
    pushed = push_drafts(writer, store, make_guard(store), {V5: approve_now(store, V5)}, now=NOW)
    assert set(store.get_video(V5).localizations) == {"de", "es"}
    restored = restore_draft(store, pushed.items[0].push_id, now=NOW).draft
    assert restored.localizations == {"de": DE5}  # the full map from before the push
    assert restored.title == original.title and restored.approved_at is None
    approve(store, V5, now=NOW)
    second = push_drafts(writer, store, make_guard(store), {V5: fingerprint(store.get_draft(V5))}, now=NOW)
    assert second.items[0].status == "pushed"
    assert writer.item(V5)["localizations"] == {"de": DE5}


def test_restoring_a_text_only_push_leaves_later_translations_alone(env) -> None:
    store, writer = env
    pushed = push_drafts(writer, store, make_guard(store), {V1: approved_draft(store, V1)}, now=NOW)
    add_translations(store, V1, "en", {"es": ES}, now=NOW)  # added after, then pushed
    push_drafts(writer, store, make_guard(store), {V1: approve_now(store, V1)}, now=NOW)
    restored = restore_draft(store, pushed.items[0].push_id, now=NOW).draft
    assert restored.localizations is None
