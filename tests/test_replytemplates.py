"""Reply templates: checking, filling and pure-code suggestions (no model, no network)."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from yezatas.replytemplates import BODY_MAX, NAME_MAX, TemplateError, check_template, fill, suggest
from yezatas.store import ReplyTemplate, Store

T0 = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def make_template(id: int, name: str, body: str) -> ReplyTemplate:
    return ReplyTemplate(id, name, body, T0.isoformat(), T0.isoformat())


# --- check_template ------------------------------------------------------------------------------


def test_check_template_trims_and_returns_the_cleaned_pair() -> None:
    assert check_template("  Thanks  ", "  Hi {name}, thanks!  \n") == ("Thanks", "Hi {name}, thanks!")
    assert check_template("Schedule", "New videos every week.") == ("Schedule", "New videos every week.")


def test_check_template_refuses_an_empty_name() -> None:
    for name in ("", "   ", "\n\t"):
        with pytest.raises(TemplateError) as caught:
            check_template(name, "Hi {name}, thanks!")
        assert str(caught.value) == "Enter a template name."


def test_check_template_refuses_an_empty_body() -> None:
    for body in ("", "   ", "\n\t"):
        with pytest.raises(TemplateError) as caught:
            check_template("Thanks", body)
        assert str(caught.value) == "Enter a reply text."


def test_check_template_refuses_non_text_with_a_sentence() -> None:
    with pytest.raises(TemplateError) as caught:
        check_template(None, "Hi {name}, thanks!")  # type: ignore[arg-type]
    assert str(caught.value) == "Enter a template name."
    with pytest.raises(TemplateError) as caught:
        check_template("Thanks", None)  # type: ignore[arg-type]
    assert str(caught.value) == "Enter a reply text."
    with pytest.raises(TemplateError) as caught:
        check_template(123, "Hi {name}, thanks!")  # type: ignore[arg-type]
    assert str(caught.value) == "Enter a template name."


def test_check_template_allows_the_exact_limits() -> None:
    name, body = check_template("n" * NAME_MAX, "b" * BODY_MAX)
    assert (len(name), len(body)) == (NAME_MAX, BODY_MAX)
    # the limits are measured after trimming
    assert check_template("  " + "n" * NAME_MAX + "  ", "x") == ("n" * NAME_MAX, "x")


def test_check_template_refuses_a_name_over_the_limit() -> None:
    with pytest.raises(TemplateError) as caught:
        check_template("n" * (NAME_MAX + 1), "Hi {name}, thanks!")
    assert str(caught.value) == f"The template name is {NAME_MAX + 1} characters; the limit is {NAME_MAX}."


def test_check_template_refuses_a_body_over_the_limit() -> None:
    with pytest.raises(TemplateError) as caught:
        check_template("Thanks", "b" * (BODY_MAX + 1))
    assert str(caught.value) == f"The reply text is {BODY_MAX + 1} characters; the limit is {BODY_MAX}."


def test_check_template_refuses_a_body_that_is_only_the_placeholder() -> None:
    for body in ("{name}", "  {name}  ", "\n{name}\n"):
        with pytest.raises(TemplateError) as caught:
            check_template("Thanks", body)
        assert str(caught.value) == "The reply text must say more than just {name}."
    assert check_template("Thanks", "Hi {name}!") == ("Thanks", "Hi {name}!")


def test_template_error_is_a_value_error() -> None:
    assert issubclass(TemplateError, ValueError)


# --- fill ----------------------------------------------------------------------------------------


def test_fill_replaces_the_name() -> None:
    assert fill("Hi {name}, welcome!", "Alex Rivers") == "Hi Alex Rivers, welcome!"
    assert fill("Hi {name}, welcome!", "  Alex Rivers  ") == "Hi Alex Rivers, welcome!"


def test_fill_uses_there_without_a_name() -> None:
    assert fill("Hi {name},", "") == "Hi there,"
    assert fill("Hi {name},", "   ") == "Hi there,"
    assert fill("Hi {name},", " \n\t ") == "Hi there,"


def test_fill_replaces_every_placeholder() -> None:
    assert fill("{name} asked, hi {name}!", "Sam") == "Sam asked, hi Sam!"
    assert fill("No placeholder here.", "Sam") == "No placeholder here."


def test_fill_leaves_other_braces_as_typed() -> None:
    assert fill("Hi {name}, see {foo} and {} and { alone", "Sam") == "Hi Sam, see {foo} and {} and { alone"
    assert fill("Hi {Name} and {NAME}", "Sam") == "Hi {Name} and {NAME}"


# --- suggest -------------------------------------------------------------------------------------


def test_suggest_ranks_by_distinct_shared_words() -> None:
    templates = [
        make_template(1, "Watching", "Thanks for watching the garden tour."),
        make_template(2, "Uploads", "New garden videos arrive every week."),
        make_template(3, "Thanks", "Thanks, {name}! Glad it helped."),
    ]
    # "garden" twice in the comment still counts once, so Uploads (garden, videos, arrive)
    # beats Watching (garden, watching, tour is not shared: "tour" is in Watching only... kept simple)
    got = suggest("Loved the garden videos, when do new garden videos arrive?", templates)
    assert [t.id for t in got] == [2, 1]


def test_suggest_counts_a_repeated_comment_word_once() -> None:
    templates = [
        make_template(1, "One match", "The garden is lovely."),
        make_template(2, "Two matches", "Garden videos weekly."),
    ]
    # without distinct counting, "garden garden garden" would outscore the two-word match
    got = suggest("garden garden garden videos", templates)
    assert [t.id for t in got] == [2, 1]


def test_suggest_ignores_stop_words_and_short_words() -> None:
    templates = [
        make_template(1, "Only filler", "The and with your this that for you."),
        make_template(2, "Short only", "Go on up by it is to my we."),
    ]
    assert suggest("the and with your this that go on up it is", templates) == []
    real = make_template(3, "Garden help", "Garden watering tips inside.")
    assert [t.id for t in suggest("the garden and the watering", [templates[0], real])] == [3]


def test_suggest_ignores_the_placeholder_token() -> None:
    templates = [make_template(1, "Hello", "Hi {name}, welcome aboard.")]
    assert suggest("name name name", templates) == []
    assert [t.id for t in suggest("welcome aboard, friend", templates)] == [1]


def test_suggest_is_case_insensitive_and_splits_on_punctuation() -> None:
    templates = [make_template(1, "Garden", "Watering, pruning; repotting!")]
    assert [t.id for t in suggest("WATERING... pruning?", templates)] == [1]


def test_suggest_no_match_is_empty() -> None:
    templates = [make_template(1, "Garden", "Watering and pruning tips.")]
    assert suggest("completely unrelated words here", templates) == []
    assert suggest("", templates) == []
    assert suggest("   ", templates) == []
    assert suggest("a an the", templates) == []


def test_suggest_respects_the_limit() -> None:
    templates = [
        make_template(1, "Garden one", "Garden watering"),
        make_template(2, "Garden two", "Garden pruning"),
        make_template(3, "Garden three", "Garden repotting"),
        make_template(4, "Garden four", "Garden planting"),
    ]
    assert [t.id for t in suggest("garden", templates)] == [1, 2, 3]  # default limit is 3
    assert [t.id for t in suggest("garden", templates, limit=2)] == [1, 2]
    assert [t.id for t in suggest("garden", templates, limit=10)] == [1, 2, 3, 4]
    assert suggest("garden", templates, limit=0) == []
    assert suggest("garden", templates, limit=-1) == []


def test_suggest_keeps_the_given_order_on_a_tie_and_does_not_mutate() -> None:
    templates = [
        make_template(3, "Third given", "Garden tips"),
        make_template(1, "First given", "Garden help"),
        make_template(2, "Second given", "Garden notes"),
    ]
    before = list(templates)
    assert [t.id for t in suggest("garden", templates, limit=3)] == [3, 1, 2]
    assert templates == before


def test_suggest_matches_words_from_the_name_too() -> None:
    templates = [
        make_template(1, "Watering schedule", "See the pinned note."),
        make_template(2, "Other", "Nothing useful here."),
    ]
    assert [t.id for t in suggest("What is the watering plan?", templates)] == [1]


# --- together with the store ---------------------------------------------------------------------


def test_templates_round_trip_through_the_store(tmp_path: Path) -> None:
    with Store(tmp_path / "templates.sqlite") as store:
        name, body = check_template("  Thanks  ", "  Thanks, {name}! Glad the guide helped.  ")
        thanks = store.save_reply_template(name, body, now=T0)
        name, body = check_template("Schedule", "Hi {name}, new videos arrive every week.")
        schedule = store.save_reply_template(name, body, now=T0)

        stored = store.list_reply_templates()
        assert [t.name for t in stored] == ["Schedule", "Thanks"]  # by name
        assert fill(stored[1].body, "Alex Rivers") == "Thanks, Alex Rivers! Glad the guide helped."
        assert fill(stored[1].body, "  ") == "Thanks, there! Glad the guide helped."

        got = suggest("Thanks, when do new videos arrive?", stored)
        assert [t.id for t in got] == [schedule, thanks]

        name, body = check_template("Thanks a lot", "Thank you, {name}!")
        assert store.save_reply_template(name, body, now=T0, template_id=thanks) == thanks
        updated = {t.id: t for t in store.list_reply_templates()}[thanks]
        assert (updated.name, updated.body) == ("Thanks a lot", "Thank you, {name}!")
        assert updated.created_at == T0.isoformat()

        assert store.delete_reply_template(schedule) is True
        assert store.delete_reply_template(schedule) is False
        assert [t.id for t in store.list_reply_templates()] == [thanks]
