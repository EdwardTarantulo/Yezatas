from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fakes import FakeYouTubeReader, make_item
from yezatas.catalog import pull_catalog, video_from_api
from yezatas.edits import (
    DraftError, add_translations, approve, discard, fingerprint, is_stale, issues_for, pending,
    rebase, remove_translation, restore_draft, save_draft, set_language, unapprove,
)
from yezatas.quota import QuotaGuard
from yezatas.store import Store

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
VID = "vid000002"  # "Repotting cactus #2", tags cactus + garden, language en, no translations
VID5 = "vid000005"  # the same, but with a German translation
DE5 = {"title": "Kaktus umtopfen #5", "description": "Anleitung 5."}


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


def test_restore_draft_never_replaces_an_existing_draft(store: Store) -> None:
    save_draft(store, VID, "Work in progress", "d", [], now=NOW)
    push_id = store.begin_push(VID, "manual", {"title": "Old title", "description": "", "tags": []})
    with pytest.raises(DraftError, match="already has a draft"):
        restore_draft(store, push_id)
    assert store.get_draft(VID).title == "Work in progress"


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


# --- language and translations --------------------------------------------------------------

ES = ("Replantar un cactus", "Como replantar un cactus.")
PT = ("Replantar um cacto", "Como replantar um cacto.")


def loc(title: str, description: str = "d") -> dict:
    return {"title": title, "description": description}


def without_language(store: Store, video_id: str) -> None:
    """Make the catalog video one that has no language set on YouTube."""
    video = store.get_video(video_id)
    video.default_language = None
    video.raw["snippet"].pop("defaultLanguage", None)
    store.upsert_video(video, store.begin_pull("c", NOW), NOW)


def test_new_draft_records_the_language_and_translations_it_was_made_from(store: Store) -> None:
    row = save_draft(store, VID5, "New", "d", [], now=NOW).draft
    assert row.base_default_language == "en" and row.base_localizations == {"de": DE5}
    assert row.default_language is None and row.localizations is None
    plain = save_draft(store, VID, "New", "d", [], now=NOW).draft
    assert plain.base_default_language == "en" and plain.base_localizations == {}


def test_base_translations_keep_only_title_and_description(store: Store) -> None:
    store.update_video_text(VID, "T", "D", [], localizations={"fr": {**loc("Titre"), "extra": "x"},
                                                              "bad": "not a map"})
    row = save_draft(store, VID, "New", "d", [], now=NOW).draft
    assert row.base_localizations == {"fr": loc("Titre")}


def test_the_base_language_is_kept_across_edits(store: Store) -> None:
    save_draft(store, VID5, "First", "d", [], now=NOW)
    store.update_video_text(VID5, "First", "d", [], default_language="fr", localizations={})
    row = save_draft(store, VID5, "Second", "d", [], now=NOW).draft
    assert row.base_default_language == "en" and row.base_localizations == {"de": DE5}


def test_set_language_makes_a_draft_from_the_current_text(store: Store) -> None:
    result = set_language(store, VID, "es", now=NOW)
    row = result.draft
    assert result.saved and row is not None
    assert row.default_language == "es" and row.localizations is None
    video = store.get_video(VID)
    assert (row.title, row.description, row.tags) == (video.title, video.description, tuple(video.tags))
    assert (row.base_title, row.base_description) == (video.title, video.description)
    assert row.source == "manual" and row.approved_at is None
    assert store.get_draft(VID) == row


def test_set_language_keeps_an_edited_text_and_the_translations_and_clears_approval(store: Store) -> None:
    save_draft(store, VID, "My title", "My text", ["x"], now=NOW)
    add_translations(store, VID, "en", {"es": ES}, now=NOW)
    approve(store, VID, now=NOW)
    row = set_language(store, VID, "pt-BR", source="agent", now=NOW).draft
    assert row.default_language == "pt-BR" and row.approved_at is None and row.source == "agent"
    assert (row.title, row.description, row.tags) == ("My title", "My text", ("x",))
    assert set(row.localizations) == {"es"}


def test_set_language_to_the_videos_own_language_leaves_no_draft(store: Store) -> None:
    result = set_language(store, VID, "en", now=NOW)
    assert (result.saved, result.discarded, result.draft) == (False, True, None)
    assert store.get_draft(VID) is None


@pytest.mark.parametrize("code", ["", "EN", "english", "e", "en_US", "en-", "zh-Hans-CN", "es ", "1a"])
def test_set_language_refuses_bad_codes(store: Store, code: str) -> None:
    with pytest.raises(DraftError, match="language code"):
        set_language(store, VID, code)
    assert store.get_draft(VID) is None


@pytest.mark.parametrize("code", ["en", "es", "fil", "pt-BR", "zh-Hans", "zh-Hant", "es"])
def test_set_language_accepts_youtube_codes(store: Store, code: str) -> None:
    set_language(store, VID, code, now=NOW)


def test_set_language_unknown_video(store: Store) -> None:
    with pytest.raises(DraftError, match="catalog"):
        set_language(store, "nope0000001", "es")


def test_add_translations_to_a_translation_only_draft(store: Store) -> None:
    video = store.get_video(VID)
    result = add_translations(store, VID, "en", {"es": ES, "pt": PT}, now=NOW)
    row = result.draft
    assert result.saved and row is not None and result.issues == ()
    # the text is the video's own, the language is the one it already had, only the map is set
    assert (row.title, row.description, row.tags) == (video.title, video.description, tuple(video.tags))
    assert row.default_language is None
    assert row.localizations == {"es": loc(*ES), "pt": loc(*PT)}
    assert row.source == "translation" and row.approved_at is None
    assert (row.base_default_language, row.base_localizations) == ("en", {})


def test_add_translations_keeps_the_existing_languages(store: Store) -> None:
    row = add_translations(store, VID5, "en", {"es": ES}, now=NOW).draft
    assert row.localizations == {"de": DE5, "es": loc(*ES)}  # the base map is the start
    again = add_translations(store, VID5, "en", {"pt": PT, "es": ("Nuevo", "Texto")}, now=NOW).draft
    assert again.localizations == {"de": DE5, "es": loc("Nuevo", "Texto"), "pt": loc(*PT)}
    assert again.created_at == row.created_at and again.base_localizations == {"de": DE5}


def test_add_translations_next_to_studio_translations(store: Store) -> None:
    # Real channels: Studio lists the video's own language and uses region codes.
    video = store.get_video(VID)
    studio = {"en": loc(video.title, video.description), "de-DE": loc("Titel", "Text")}
    store.update_video_text(VID, video.title, video.description, video.tags, localizations=studio)
    result = add_translations(store, VID, "en", {"es-419": ES}, now=NOW)
    assert result.issues == ()
    assert result.draft.localizations == {**studio, "es-419": loc(*ES)}


def test_add_translations_keeps_a_regional_video_language(store: Store) -> None:
    set_language(store, VID, "en-GB", now=NOW)
    row = add_translations(store, VID, "en", {"es": ES}, now=NOW).draft
    assert row.default_language == "en-GB"


def test_add_translations_keeps_text_edits_and_normalises_the_translations(store: Store) -> None:
    save_draft(store, VID, "My title", "My text", ["x"], note="n", now=NOW)
    row = add_translations(store, VID, "en", {"es": ("  Titulo  ", "Uno\r\nDos")}, now=NOW).draft
    assert (row.title, row.description, row.tags, row.note) == ("My title", "My text", ("x",), "n")
    assert row.localizations == {"es": loc("Titulo", "Uno\nDos")}


def test_add_translations_sets_the_language_when_the_video_has_none(store: Store) -> None:
    without_language(store, VID)
    row = add_translations(store, VID, "en", {"es": ES}, now=NOW).draft
    assert row.default_language == "en" and row.base_default_language is None
    assert issues_for(row) == ()


def test_add_translations_changes_a_different_language(store: Store) -> None:
    row = add_translations(store, VID, "fr", {"es": ES}, now=NOW).draft  # the text is French
    assert row.default_language == "fr" and row.base_default_language == "en"
    # a draft that already has its own language: the source language replaces it only if different
    assert add_translations(store, VID, "fr", {"pt": PT}, now=NOW).draft.default_language == "fr"
    assert add_translations(store, VID, "en", {"pt": PT}, now=NOW).draft.default_language == "en"


def test_add_translations_into_the_source_language_is_refused(store: Store) -> None:
    with pytest.raises(DraftError, match="already in en"):
        add_translations(store, VID, "en", {"es": ES, "en": ES})
    assert store.get_draft(VID) is None


def test_add_translations_refuses_bad_codes_and_unknown_videos(store: Store) -> None:
    with pytest.raises(DraftError, match="language code"):
        add_translations(store, VID, "English", {"es": ES})
    with pytest.raises(DraftError, match="language code"):
        add_translations(store, VID, "en", {"Spanish": ES})
    with pytest.raises(DraftError, match="catalog"):
        add_translations(store, "nope0000001", "en", {"es": ES})
    assert store.get_draft(VID) is None


def test_add_translations_clears_approval(store: Store) -> None:
    add_translations(store, VID, "en", {"es": ES}, now=NOW)
    approve(store, VID, now=NOW)
    add_translations(store, VID, "en", {"pt": PT}, now=NOW)
    assert store.get_draft(VID).approved_at is None


def test_the_same_translations_as_the_video_leave_no_draft(store: Store) -> None:
    result = add_translations(store, VID5, "en", {"de": ("Kaktus umtopfen #5", "Anleitung 5.")}, now=NOW)
    assert result.discarded and store.get_draft(VID5) is None


def test_remove_translation(store: Store) -> None:
    add_translations(store, VID, "en", {"es": ES, "pt": PT}, now=NOW)
    result = remove_translation(store, VID, "es", now=NOW)
    assert result.draft.localizations == {"pt": loc(*PT)}
    assert result.draft.approved_at is None
    # removing the last one added leaves the draft equal to the video, so it goes away
    gone = remove_translation(store, VID, "pt", now=NOW)
    assert gone.discarded and store.get_draft(VID) is None


def test_remove_translation_of_a_video_translation_clears_it_on_push_later(store: Store) -> None:
    result = remove_translation(store, VID5, "de", now=NOW)
    assert result.saved and result.draft.localizations == {}  # an empty map: remove all of them
    assert result.draft.base_localizations == {"de": DE5}


def test_remove_translation_that_is_not_there(store: Store) -> None:
    with pytest.raises(DraftError, match="no fr translation"):
        remove_translation(store, VID5, "fr")
    with pytest.raises(DraftError, match="catalog"):
        remove_translation(store, "nope0000001", "de")
    assert store.get_draft(VID5) is None


def test_save_draft_keeps_the_language_and_translations(store: Store) -> None:
    add_translations(store, VID, "fr", {"es": ES}, now=NOW)
    row = save_draft(store, VID, "Better title", "d", [], now=NOW).draft
    assert row.default_language == "fr" and row.localizations == {"es": loc(*ES)}
    assert row.title == "Better title"


def test_text_back_to_the_base_keeps_a_draft_that_still_has_translations(store: Store) -> None:
    add_translations(store, VID, "en", {"es": ES}, now=NOW)
    video = store.get_video(VID)
    result = save_draft(store, VID, video.title, video.description, video.tags, now=NOW)
    assert result.saved and result.draft.localizations == {"es": loc(*ES)}


def test_text_back_to_the_base_keeps_a_draft_that_still_has_a_language(store: Store) -> None:
    set_language(store, VID, "es", now=NOW)
    video = store.get_video(VID)
    assert save_draft(store, VID, video.title, video.description, video.tags, now=NOW).saved


def test_a_draft_with_only_the_base_language_and_translations_is_dropped(store: Store) -> None:
    save_draft(store, VID, "Changed", "d", [], now=NOW)
    store.put_draft(replace(store.get_draft(VID), default_language="en", localizations={}))
    video = store.get_video(VID)
    result = save_draft(store, VID, video.title, video.description, video.tags, now=NOW)
    assert result.discarded and store.get_draft(VID) is None


def test_stale_when_the_language_or_translations_changed_meanwhile(store: Store) -> None:
    add_translations(store, VID5, "en", {"es": ES}, now=NOW)
    row = store.get_draft(VID5)
    assert not is_stale(row, store.get_video(VID5))
    store.update_video_text(VID5, *_text(store, VID5), localizations={"de": DE5, "fr": loc("T")})
    assert is_stale(row, store.get_video(VID5))
    with pytest.raises(DraftError, match="changed"):
        approve(store, VID5)


def test_stale_when_only_the_language_changed_meanwhile(store: Store) -> None:
    set_language(store, VID, "es", now=NOW)
    row = store.get_draft(VID)
    store.update_video_text(VID, *_text(store, VID), default_language="fr")
    assert is_stale(row, store.get_video(VID))


def _text(store: Store, video_id: str):
    video = store.get_video(video_id)
    return video.title, video.description, video.tags


def test_rebase_keeps_the_drafts_own_language_and_translations(store: Store) -> None:
    add_translations(store, VID5, "fr", {"es": ES}, now=NOW)
    store.update_video_text(VID5, "Studio title", "d", [], default_language="pt",
                            localizations={"it": loc("Titolo")})
    rebased = rebase(store, VID5, now=NOW + timedelta(hours=1))
    assert rebased.default_language == "fr" and rebased.localizations == {"de": DE5, "es": loc(*ES)}
    assert rebased.base_default_language == "pt" and rebased.base_localizations == {"it": loc("Titolo")}
    assert (rebased.base_title, rebased.title) == ("Studio title", "Repotting cactus #5")
    assert not is_stale(rebased, store.get_video(VID5)) and rebased.approved_at is None


def test_rebase_drops_a_draft_that_now_equals_the_video_in_every_part(store: Store) -> None:
    add_translations(store, VID, "en", {"es": ES}, now=NOW)
    store.update_video_text(VID, *_text(store, VID), localizations={"es": loc(*ES)})
    assert rebase(store, VID) is None and store.get_draft(VID) is None
    # a draft whose language still differs stays
    set_language(store, VID, "es", now=NOW)
    assert rebase(store, VID).default_language == "es"


def test_fingerprint_changes_with_the_language_and_translations(store: Store) -> None:
    add_translations(store, VID, "en", {"es": ES}, now=NOW)
    first = fingerprint(store.get_draft(VID))
    assert fingerprint(store.get_draft(VID)) == first
    add_translations(store, VID, "en", {"es": ("Otro titulo", ES[1])}, now=NOW)
    second = fingerprint(store.get_draft(VID))
    add_translations(store, VID, "en", {"pt": PT}, now=NOW)
    third = fingerprint(store.get_draft(VID))
    set_language(store, VID, "fr", now=NOW)
    fourth = fingerprint(store.get_draft(VID))
    store.update_video_text(VID, *_text(store, VID), localizations={"it": loc("T")})
    rebase(store, VID)
    fifth = fingerprint(store.get_draft(VID))
    assert len({first, second, third, fourth, fifth}) == 5


def test_fingerprint_does_not_depend_on_the_order_of_translations(store: Store) -> None:
    add_translations(store, VID, "en", {"es": ES, "pt": PT}, now=NOW)
    a = fingerprint(store.get_draft(VID))
    store.put_draft(replace(store.get_draft(VID), localizations={"pt": loc(*PT), "es": loc(*ES)}))
    assert fingerprint(store.get_draft(VID)) == a


def test_issues_for_includes_the_translation_errors(store: Store) -> None:
    result = add_translations(store, VID, "en", {"es": ("x" * 101, "a < b")}, now=NOW)
    errors = {(i.field, i.code) for i in result.issues if i.severity == "error"}
    assert errors == {("localizations.es", "localization_title_too_long"),
                      ("localizations.es", "localization_description_angle_brackets")}
    assert issues_for(result.draft) == result.issues
    with pytest.raises(DraftError, match="errors"):
        approve(store, VID, now=NOW)


def test_translations_need_a_language_before_approval(store: Store) -> None:
    without_language(store, VID)
    save_draft(store, VID, "My title", "d", [], now=NOW)
    store.put_draft(replace(store.get_draft(VID), localizations={"es": loc(*ES)}))
    assert [i.code for i in issues_for(store.get_draft(VID)) if i.severity == "error"] == [
        "localizations_language_missing"]
    with pytest.raises(DraftError, match="errors"):
        approve(store, VID)
    set_language(store, VID, "en", now=NOW)
    approve(store, VID, now=NOW)


def test_a_translation_only_draft_is_not_reported_as_unchanged(store: Store) -> None:
    result = add_translations(store, VID, "en", {"es": ES}, now=NOW)
    assert "no_change" not in [i.code for i in result.issues]
    # a text-only draft is not affected
    store.delete_draft(VID)
    assert "no_change" not in [i.code for i in save_draft(store, VID, "New", "d", [], now=NOW).issues]


def test_restore_draft_with_translations_and_language_in_the_backup(store: Store) -> None:
    push_id = store.begin_push(
        VID5, "manual",
        {"title": "Old title", "description": "Old text", "tags": ["old"], "defaultLanguage": "fr",
         "localizations": {"es": {**loc(*ES), "extra": "x"}}})
    row = restore_draft(store, push_id, now=NOW).draft
    assert (row.title, row.default_language, row.localizations) == (
        "Old title", "fr", {"es": loc(*ES)})
    assert row.base_localizations == {"de": DE5}  # the base is the video as it is now


def test_restore_draft_can_restore_translations_alone(store: Store) -> None:
    video = store.get_video(VID5)
    push_id = store.begin_push(
        VID5, "manual",
        {"title": video.title, "description": video.description, "tags": video.tags,
         "defaultLanguage": "en", "localizations": {}})
    result = restore_draft(store, push_id, now=NOW)
    assert result.saved and result.draft.localizations == {}  # clear the translations
    assert result.draft.default_language is None  # the video has this language already


def test_restore_draft_from_an_old_backup_leaves_language_and_translations_alone(store: Store) -> None:
    push_id = store.begin_push(VID5, "manual", {"title": "Old title", "description": "", "tags": []})
    row = restore_draft(store, push_id, now=NOW).draft
    assert row.default_language is None and row.localizations is None
    # a backup with the video's current language and translations also leaves them alone
    other = store.begin_push(
        VID, "manual",
        {"title": "Old", "description": "", "tags": [], "defaultLanguage": "en", "localizations": {}})
    row = restore_draft(store, other, now=NOW).draft
    assert row.default_language is None and row.localizations is None
