from __future__ import annotations

import pytest

from yezatas.diffs import (
    DraftDiff,
    Line,
    LocalizationsDiff,
    Segment,
    diff_draft,
    diff_localizations,
    diff_row,
    diff_tags,
    diff_text,
    diff_words,
)
from yezatas.store import DraftRow
from yezatas.validate import Draft


def round_trips(old: str, new: str) -> tuple[Segment, ...]:
    segments = diff_words(old, new)
    assert "".join(s.text for s in segments if s.kind in ("same", "removed")) == old
    assert "".join(s.text for s in segments if s.kind in ("same", "added")) == new
    for first, second in zip(segments, segments[1:]):
        assert first.kind != second.kind
    return segments


def test_diff_words_identical():
    assert diff_words("hello world", "hello world") == (Segment("same", "hello world"),)


def test_diff_words_both_empty():
    assert diff_words("", "") == ()


def test_diff_text_identical():
    assert diff_text("a\nb", "a\nb") == diff_text("a\nb", "a\nb")
    result = diff_text("a\nb", "a\nb")
    assert result.changed is False
    assert result.lines == ()
    assert (result.added, result.removed) == (0, 0)


def test_diff_tags_identical():
    result = diff_tags(("a", "b"), ("a", "b"))
    assert result.changed is False
    assert result.kept == ("a", "b")
    assert result.added == () and result.removed == ()
    assert result.reordered is False


def test_diff_draft_identical():
    draft = Draft(title="T", description="a\nb", tags=("x",))
    result = diff_draft(draft, draft)
    assert result.changed is False
    assert result.title_changed is False
    assert result.description.changed is False
    assert result.tags.changed is False


@pytest.mark.parametrize("old,new", [
    ("hello world", "hello brave world"),
    ("  padded  ", "padded"),
    ("", "something"),
    ("something", ""),
    ("Kaip persodinti kaktusą", "Kaip persodinti kaktusą namuose"),
    ("a  b", "a b"),
])
def test_diff_words_round_trip(old, new):
    round_trips(old, new)


def test_diff_words_replaced_run_order():
    segments = round_trips("cat sat", "dog sat")
    assert segments[0] == Segment("removed", "cat")
    assert segments[1] == Segment("added", "dog")
    assert segments[-1].kind == "same"


def test_description_one_line_change_in_middle():
    old = "\n".join(f"line {i}" for i in range(1, 21))
    new = "\n".join(f"line {i}" if i != 10 else "line 10 edited" for i in range(1, 21))
    result = diff_text(old, new)
    assert result.changed is True
    assert (result.added, result.removed) == (1, 1)
    assert result.lines == (
        Line("gap", "7 unchanged lines", None, None),
        Line("same", "line 8", 8, 8),
        Line("same", "line 9", 9, 9),
        Line("removed", "line 10", 10, None),
        Line("added", "line 10 edited", None, 10),
        Line("same", "line 11", 11, 11),
        Line("same", "line 12", 12, 12),
        Line("gap", "8 unchanged lines", None, None),
    )


def test_description_change_first_line():
    old = "first\nline 2\nline 3\nline 4\nline 5"
    new = "FIRST\nline 2\nline 3\nline 4\nline 5"
    result = diff_text(old, new)
    assert result.lines == (
        Line("removed", "first", 1, None),
        Line("added", "FIRST", None, 1),
        Line("same", "line 2", 2, 2),
        Line("same", "line 3", 3, 3),
        Line("gap", "2 unchanged lines", None, None),
    )


def test_description_change_last_line():
    old = "line 1\nline 2\nline 3\nline 4\nline 5"
    new = "line 1\nline 2\nline 3\nline 4\nline 5 edited"
    result = diff_text(old, new)
    assert result.lines == (
        Line("gap", "2 unchanged lines", None, None),
        Line("same", "line 3", 3, 3),
        Line("same", "line 4", 4, 4),
        Line("removed", "line 5", 5, None),
        Line("added", "line 5 edited", None, 5),
    )


def test_description_context_zero():
    result = diff_text("a\nb\nc", "a\nB\nc", context=0)
    assert result.lines == (
        Line("gap", "1 unchanged line", None, None),
        Line("removed", "b", 2, None),
        Line("added", "B", None, 2),
        Line("gap", "1 unchanged line", None, None),
    )
    assert (result.added, result.removed) == (1, 1)


def test_description_adding_to_empty():
    result = diff_text("", "hello\nworld")
    assert result.changed is True
    assert result.lines == (
        Line("added", "hello", None, 1),
        Line("added", "world", None, 2),
    )
    assert (result.added, result.removed) == (2, 0)


def test_tags_added():
    result = diff_tags(("a",), ("a", "b"))
    assert result.changed is True
    assert result.kept == ("a",)
    assert result.added == ("b",)
    assert result.removed == ()
    assert result.reordered is False


def test_tags_removed():
    result = diff_tags(("a", "b"), ("a",))
    assert result.changed is True
    assert result.kept == ("a",)
    assert result.added == ()
    assert result.removed == ("b",)
    assert result.reordered is False


def test_tags_reordered():
    result = diff_tags(("a", "b"), ("b", "a"))
    assert result.changed is True
    assert result.kept == ("b", "a")
    assert result.added == () and result.removed == ()
    assert result.reordered is True


def test_tags_added_and_removed():
    result = diff_tags(("a", "b"), ("b", "c"))
    assert result.changed is True
    assert result.kept == ("b",)
    assert result.added == ("c",)
    assert result.removed == ("a",)
    assert result.reordered is False


def test_tags_duplicates_as_multiset():
    result = diff_tags(("a", "a", "b"), ("a", "b", "b"))
    assert result.kept == ("a", "b")
    assert result.added == ("b",)
    assert result.removed == ("a",)
    assert result.changed is True
    assert result.reordered is False


def test_tags_case_sensitive():
    result = diff_tags(("Baking",), ("baking",))
    assert result.changed is True
    assert result.kept == ()
    assert result.added == ("baking",)
    assert result.removed == ("Baking",)


def test_diff_draft_tags_only_change():
    original = Draft(title="My title", description="line 1\nline 2", tags=("a", "b"))
    draft = Draft(title="My title", description="line 1\nline 2", tags=("a", "c"))
    result = diff_draft(original, draft)
    assert result.title == (Segment("same", "My title"),)
    assert result.title_changed is False
    assert result.description.changed is False
    assert result.tags.changed is True
    assert result.tags.kept == ("a",)
    assert result.tags.added == ("c",)
    assert result.tags.removed == ("b",)
    assert result.changed is True


def test_diff_words_shows_a_mostly_rewritten_title_as_old_then_new() -> None:
    old = "Repotting cactus #12"
    new = "How to Repot a Cactus Without Getting Hurt"
    assert diff_words(old, new) == (Segment("removed", old), Segment("added", new))


def test_diff_words_keeps_word_level_detail_for_small_edits() -> None:
    segments = diff_words("How to repot a cactus at home", "How to repot a big cactus at home")
    assert Segment("added", "big ") in segments
    assert segments[0].kind == "same"


def loc(title: str, description: str = "d") -> dict:
    return {"title": title, "description": description}


def test_diff_draft_without_language_parts_still_works():
    result = diff_draft(Draft("A", "d"), Draft("B", "d"))
    assert result.language is None
    assert result.localizations == LocalizationsDiff(False, (), (), (), ())
    assert isinstance(result, DraftDiff)


def test_diff_localizations_unchanged_when_not_set():
    base = {"de": loc("T"), "fr": loc("T")}
    assert diff_localizations(base, None) == LocalizationsDiff(False, (), (), (), ("de", "fr"))
    assert diff_localizations({}, None) == LocalizationsDiff(False, (), (), (), ())
    assert diff_localizations(base, dict(base)).changed is False


def test_diff_localizations_added_removed_changed_kept():
    base = {"de": loc("T"), "fr": loc("T"), "it": loc("T", "old")}
    new = {"de": loc("T"), "it": loc("T", "new"), "es": loc("T"), "pt": loc("T")}
    result = diff_localizations(base, new)
    assert result == LocalizationsDiff(
        True, added=("es", "pt"), removed=("fr",), changed_langs=("it",), kept=("de",))


def test_diff_localizations_ignores_other_keys_and_clearing_all_is_a_change():
    assert diff_localizations({"de": loc("T")}, {"de": {**loc("T"), "extra": 1}}).changed is False
    cleared = diff_localizations({"de": loc("T")}, {})
    assert cleared.changed and cleared.removed == ("de",)


def make_row(**overrides) -> DraftRow:
    values = dict(
        video_id="vid1", title="T", description="d", tags=("a",), base_title="T",
        base_description="d", base_tags=("a",), source="manual", note="", created_at="x",
        updated_at="x", approved_at=None, base_default_language="en",
        base_localizations={"de": loc("T")},
    )
    values.update(overrides)
    return DraftRow(**values)


def test_diff_row_text_only_is_like_diff_draft():
    row = make_row(title="New")
    result = diff_row(row)
    assert (result.title, result.description, result.tags) == (
        diff_draft(row.base, row.draft).title, diff_draft(row.base, row.draft).description,
        diff_draft(row.base, row.draft).tags)
    assert result.changed and result.title_changed and result.language is None


def test_diff_row_untouched_draft_is_unchanged():
    result = diff_row(make_row())
    assert result.changed is False and result.language is None
    assert result.localizations.changed is False


def test_diff_row_language_only_counts_as_a_change():
    result = diff_row(make_row(default_language="es"))
    assert result.changed and not result.title_changed
    assert result.language == ("en", "es")
    # a language equal to the base is not a change
    same = diff_row(make_row(default_language="en"))
    assert same.language is None and same.changed is False


def test_diff_row_translations_only_count_as_a_change():
    result = diff_row(make_row(localizations={"de": loc("T"), "es": loc("Titulo")}))
    assert result.changed and not result.title_changed and not result.description.changed
    assert result.localizations.added == ("es",) and result.localizations.kept == ("de",)
    unset = diff_row(make_row(localizations=None))
    assert unset.localizations.kept == ("de",) and unset.changed is False


def test_diff_row_from_no_language():
    result = diff_row(make_row(default_language="en", base_default_language=None))
    assert result.language == (None, "en") and result.changed
