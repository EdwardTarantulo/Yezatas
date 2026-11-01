from __future__ import annotations

import pytest

from yezatas.validate import Draft, has_errors, validate_draft, validate_localizations

CHAPTERS = "0:00 Intro\n0:30 Tools\n1:30 Build\n4:00 Wrap up"


def make_draft(**overrides) -> Draft:
    base = Draft(
        title="How to bake sourdough bread at home",
        description=(
            "In this video I show how to bake a crusty sourdough loaf from scratch, "
            "including the starter, the folds and the bake itself.\n\n"
            "Full recipe: https://site.invalid/sourdough-recipe\n"
            "#sourdough #baking"
        ),
        tags=("sourdough", "baking", "bread"),
    )
    return Draft(
        title=overrides.get("title", base.title),
        description=overrides.get("description", base.description),
        tags=overrides.get("tags", base.tags),
    )


def codes(draft: Draft, original: Draft | None = None) -> list[str]:
    return [issue.code for issue in validate_draft(draft, original)]


def one(code: str, draft: Draft, original: Draft | None = None):
    return next(i for i in validate_draft(draft, original) if i.code == code)


def test_clean_draft_returns_no_issues():
    assert validate_draft(make_draft()) == []


def test_draft_tags_default_to_empty():
    assert Draft(title="T", description="D").tags == ()


# --- title ---

def test_title_empty_fails():
    assert "title_empty" in codes(make_draft(title=""))
    assert "title_empty" in codes(make_draft(title="   "))
    issue = one("title_empty", make_draft(title=""))
    assert issue.severity == "error" and issue.field == "title"


def test_title_present_passes():
    assert "title_empty" not in codes(make_draft())


def test_title_boundary_100_passes_101_fails():
    assert "title_too_long" not in codes(make_draft(title="t" * 100))
    issue = one("title_too_long", make_draft(title="t" * 101))
    assert "101" in issue.message and "100" in issue.message
    assert issue.severity == "error"


def test_title_angle_brackets():
    assert "title_angle_brackets" in codes(make_draft(title="A <b> story"))
    assert "title_angle_brackets" in codes(make_draft(title="Bread > cake"))
    assert "title_angle_brackets" not in codes(make_draft())


# --- description ---

def test_description_byte_boundary():
    assert "description_too_long" not in codes(make_draft(description="a" * 5000))
    issue = one("description_too_long", make_draft(description="a" * 5001))
    assert "5001" in issue.message and "5,000" in issue.message


def test_description_multibyte_bytes_not_characters():
    desc = "ą" * 3000  # 3000 characters but 6000 bytes in UTF-8
    assert len(desc) < 5000
    assert len(desc.encode("utf-8")) > 5000
    assert "description_too_long" in codes(make_draft(description=desc))


def test_description_exactly_5000_bytes_with_multibyte_passes():
    desc = "я" * 2500  # 2500 characters and exactly 5000 bytes
    assert len(desc.encode("utf-8")) == 5000
    assert "description_too_long" not in codes(make_draft(description=desc))


def test_description_angle_brackets():
    assert "description_angle_brackets" in codes(make_draft(description="a <b> c"))
    assert "description_angle_brackets" in codes(make_draft(description="a > b"))
    assert "description_angle_brackets" not in codes(make_draft())


@pytest.mark.parametrize("text,quoted", [
    ("TODO add the links", "TODO"),
    ("visit example.com now", "example.com"),
    ("see [link] for details", "[link]"),
    ("LINK HERE for the plans", "LINK HERE"),
    ("see more at https:// ", "https://"),
])
def test_description_placeholder_link_fails(text, quoted):
    issue = one("description_placeholder_link", make_draft(description="Notes. " + text))
    assert f'"{quoted}"' in issue.message
    assert issue.severity == "error"


def test_description_placeholder_link_passes():
    good = "Plans at https://site.invalid/plans and my todos for next week."
    assert "description_placeholder_link" not in codes(make_draft(description=good))


def test_description_hashtag_boundary():
    assert "description_too_many_hashtags" not in codes(
        make_draft(description="Intro. " + "#garden " * 60))
    issue = one("description_too_many_hashtags",
                make_draft(description="Intro. " + "#garden " * 61))
    assert "61" in issue.message and "60" in issue.message


# --- chapters_invalid ---

def test_valid_chapters_pass():
    assert "chapters_invalid" not in codes(make_draft(description=CHAPTERS))


def test_fewer_than_two_timestamps_is_not_chapters():
    assert "chapters_invalid" not in codes(make_draft(description="See 0:00 for the start"))
    assert "chapters_invalid" not in codes(make_draft(description="No timestamps here"))


def test_chapters_need_three():
    issue = one("chapters_invalid", make_draft(description="0:00 A\n1:30 B"))
    assert "3" in issue.message


def test_chapters_must_start_at_zero():
    issue = one("chapters_invalid", make_draft(description="0:05 A\n1:30 B\n4:00 C"))
    assert "0:00" in issue.message


def test_chapters_must_go_up():
    issue = one("chapters_invalid", make_draft(description="0:00 A\n2:00 B\n1:30 C"))
    assert "up" in issue.message


def test_chapters_need_ten_second_gaps():
    issue = one("chapters_invalid", make_draft(description="0:00 A\n0:09 B\n1:30 C"))
    assert "10" in issue.message
    assert "chapters_invalid" not in codes(
        make_draft(description="0:00 A\n0:10 B\n1:30 C"))


def test_chapters_invalid_is_a_description_warning():
    issue = one("chapters_invalid", make_draft(description="0:00 A\n1:30 B"))
    assert issue.field == "description" and issue.severity == "warning"


# --- tags ---

def test_tags_char_boundary():
    assert "tags_too_long" not in codes(make_draft(tags=("a" * 500,)))
    issue = one("tags_too_long", make_draft(tags=("a" * 501,)))
    assert "501" in issue.message and "500" in issue.message


def test_tags_char_count_includes_commas():
    assert "tags_too_long" in codes(make_draft(tags=("ab", "cd") * 200))


def test_tag_invalid_cases():
    assert "tag_invalid" in codes(make_draft(tags=("ok", "")))
    assert "tag_invalid" in codes(make_draft(tags=("   ",)))
    assert "tag_invalid" in codes(make_draft(tags=("a<b",)))
    assert "tag_invalid" in codes(make_draft(tags=("a>b",)))
    assert "tag_invalid" in codes(make_draft(tags=("a,b",)))
    assert "tag_invalid" not in codes(make_draft())


def test_tag_invalid_names_the_tag():
    issue = one("tag_invalid", make_draft(tags=("ok", "a,b")))
    assert "a,b" in issue.message


def test_one_issue_per_bad_tag():
    issues = [i for i in validate_draft(make_draft(tags=("a,b", "", "ok"))) if i.code == "tag_invalid"]
    assert len(issues) == 2


# --- original-only checks ---

def test_link_removed_fails_and_quotes_url():
    original = make_draft()
    draft = make_draft(description="Same video without the recipe link.")
    issues = [i for i in validate_draft(draft, original) if i.code == "link_removed"]
    assert len(issues) == 1
    assert "https://site.invalid/sourdough-recipe" in issues[0].message
    assert issues[0].field == "description" and issues[0].severity == "warning"


def test_link_kept_passes():
    original = make_draft()
    assert "link_removed" not in codes(make_draft(), original)


def test_link_kept_when_only_surrounding_punctuation_changes():
    original = make_draft(description="Recipe (see https://site.invalid/a). More at https://site.invalid/b.")
    draft = make_draft(description="Recipe: https://site.invalid/a\nMore: https://site.invalid/b")
    assert "link_removed" not in codes(draft, original)


def test_link_kept_when_upgraded_to_https():
    original = make_draft(description="More at http://site.invalid/a and https://site.invalid/b")
    upgraded = make_draft(description="More at https://site.invalid/a and https://site.invalid/b")
    downgraded = make_draft(description="More at http://site.invalid/a and http://site.invalid/b")
    assert "link_removed" not in codes(upgraded, original)
    issues = [i for i in validate_draft(downgraded, original) if i.code == "link_removed"]
    assert [i.message for i in issues] == [
        'The link "https://site.invalid/b" was in the original description but is missing now.']


def test_link_removed_needs_original():
    draft = make_draft(description="No links here.")
    assert "link_removed" not in codes(draft)


def test_repeated_removed_url_is_one_issue_but_each_url_is_listed():
    original = make_draft(description="See https://site.invalid/a twice: https://site.invalid/a")
    draft = make_draft(description="No links here.")
    first = make_draft(description="Only https://site.invalid/a here.")
    assert codes(draft, original).count("link_removed") == 1
    assert "link_removed" not in codes(first, make_draft(description=first.description))


def test_chapters_changed():
    original = make_draft(description=CHAPTERS)
    same = make_draft(description=CHAPTERS)
    changed = make_draft(description="0:00 Intro\n0:30 Tools\n2:00 Build\n4:00 Wrap up")
    assert "chapters_changed" not in codes(same, original)
    issue = one("chapters_changed", changed, original)
    assert issue.field == "description" and issue.severity == "warning"


def test_chapters_changed_needs_valid_original():
    original = make_draft(description="0:00 A\n1:30 B")  # only two stamps
    draft = make_draft(description="0:00 A\n1:00 B\n2:00 C")
    assert "chapters_changed" not in codes(draft, original)


def test_chapters_changed_needs_original():
    assert "chapters_changed" not in codes(make_draft(description=CHAPTERS))


def test_no_change():
    original = make_draft()
    issues = validate_draft(make_draft(), original)
    assert [i.code for i in issues] == ["no_change"]
    assert issues[0].field == "title" and issues[0].severity == "warning"


def test_no_change_needs_original_and_exact_match():
    assert "no_change" not in codes(make_draft())
    original = make_draft()
    assert "no_change" not in codes(make_draft(title="New title"), original)
    assert "no_change" not in codes(make_draft(tags=("other",)), original)


# --- order, fields and has_errors ---

def test_stable_order_title_then_description_then_tags():
    title = "T" * 99 + "<>"  # 101 characters with angle brackets
    description = (
        "0:05 Intro\n"
        "0:08 Setup\n"
        "TODO replace this link <draft>\n"
        + "#topic " * 61 + "\n"
        + "x" * 4900
    )
    issues = validate_draft(Draft(title=title, description=description,
                                  tags=("y" * 501, "bad,tag", "")))
    assert [i.code for i in issues] == [
        "title_too_long",
        "title_angle_brackets",
        "description_too_long",
        "description_angle_brackets",
        "description_placeholder_link",
        "description_too_many_hashtags",
        "chapters_invalid",
        "tags_too_long",
        "tag_invalid",
        "tag_invalid",
    ]
    assert [i.field for i in issues] == ["title"] * 2 + ["description"] * 5 + ["tags"] * 3


CASES = {
    "title_empty": lambda: (make_draft(title=""), None),
    "title_too_long": lambda: (make_draft(title="t" * 101), None),
    "title_angle_brackets": lambda: (make_draft(title="A <b>"), None),
    "description_too_long": lambda: (make_draft(description="x" * 5001), None),
    "description_angle_brackets": lambda: (make_draft(description="a > b"), None),
    "description_placeholder_link": lambda: (make_draft(description="TODO x"), None),
    "description_too_many_hashtags": lambda: (make_draft(description="#t " * 61), None),
    "tags_too_long": lambda: (make_draft(tags=("z" * 501,)), None),
    "tag_invalid": lambda: (make_draft(tags=("a,b",)), None),
    "chapters_invalid": lambda: (make_draft(description="0:00 A\n1:30 B"), None),
    "link_removed": lambda: (make_draft(description="No links."),
                             make_draft(description="See https://site.invalid/a")),
    "chapters_changed": lambda: (make_draft(description="0:00 A\n0:30 B\n2:00 C"),
                                 make_draft(description=CHAPTERS)),
    "no_change": lambda: (make_draft(), make_draft()),
}

EXPECTED = {
    "title_empty": ("title", "error"),
    "title_too_long": ("title", "error"),
    "title_angle_brackets": ("title", "error"),
    "description_too_long": ("description", "error"),
    "description_angle_brackets": ("description", "error"),
    "description_placeholder_link": ("description", "error"),
    "description_too_many_hashtags": ("description", "error"),
    "tags_too_long": ("tags", "error"),
    "tag_invalid": ("tags", "error"),
    "chapters_invalid": ("description", "warning"),
    "link_removed": ("description", "warning"),
    "chapters_changed": ("description", "warning"),
    "no_change": ("title", "warning"),
}


@pytest.mark.parametrize("code", sorted(EXPECTED))
def test_field_and_severity(code):
    draft, original = CASES[code]()
    issue = one(code, draft, original)
    assert (issue.field, issue.severity) == EXPECTED[code]
    assert len(issue.message) > 0


def test_has_errors():
    assert not has_errors([])
    assert not has_errors(iter([]))
    assert has_errors(validate_draft(make_draft(title="")))
    warnings_only = validate_draft(make_draft(description="0:00 A\n1:30 B"))
    assert warnings_only and not has_errors(warnings_only)
    assert not has_errors(i for i in warnings_only)
    assert has_errors(i for i in validate_draft(make_draft(title="")))


def loc(title: str = "Titulo", description: str = "Texto") -> dict:
    return {"title": title, "description": description}


def loc_codes(default, localizations, base=None) -> list[tuple[str, str]]:
    return [(i.field, i.code) for i in validate_localizations(default, localizations, base)]


def test_localizations_nothing_to_check():
    assert validate_localizations(None, None, None) == []
    assert validate_localizations(None, {}, None) == []  # no map: no language needed
    assert loc_codes("en", {"es": loc(), "de": loc("Titel", "")}) == []
    assert loc_codes(None, {"es": loc()}, "en") == []  # the draft's base language counts


def test_localizations_as_on_youtube_are_not_judged():
    # Studio lists the video's own language among the translations; sending it back is fine.
    studio = {"en": loc("How to repot", "Step by step."), "de-DE": loc("x" * 120)}
    assert validate_localizations("en", {**studio, "es": loc()}, "en", studio) == []
    changed = {**studio, "en": loc("Changed title")}
    assert [(i.field, i.code) for i in validate_localizations("en", changed, "en", studio)] == [
        ("localizations.en", "localization_is_default")]


def test_localizations_need_a_video_language():
    issue, = validate_localizations(None, {"es": loc()}, None)
    assert (issue.field, issue.code, issue.severity) == (
        "localizations", "localizations_language_missing", "error")
    assert "Set the video language first" in issue.message


def test_localization_in_the_video_language_is_refused():
    assert loc_codes("en", {"en": loc()}) == [("localizations.en", "localization_is_default")]
    # the draft's own language wins over the base one
    assert loc_codes("es", {"en": loc()}, "en") == []
    assert loc_codes("es", {"es": loc()}, "en") == [("localizations.es", "localization_is_default")]


def test_localization_title_rules():
    assert loc_codes("en", {"es": loc(title="  ")}) == [("localizations.es", "localization_title_empty")]
    assert loc_codes("en", {"es": loc(title="x" * 100)}) == []
    assert loc_codes("en", {"es": loc(title="x" * 101)}) == [
        ("localizations.es", "localization_title_too_long")]
    assert loc_codes("en", {"es": loc(title="a < b")}) == [
        ("localizations.es", "localization_title_angle_brackets")]


def test_localization_description_rules():
    assert loc_codes("en", {"es": loc(description="x" * 5000)}) == []
    assert loc_codes("en", {"es": loc(description="x" * 5001)}) == [
        ("localizations.es", "localization_description_too_long")]
    assert loc_codes("en", {"es": loc(description="é" * 2501)}) == [  # 2 bytes each
        ("localizations.es", "localization_description_too_long")]
    assert loc_codes("en", {"es": loc(description="a > b")}) == [
        ("localizations.es", "localization_description_angle_brackets")]
    issue, = validate_localizations("en", {"es": loc(description="x" * 5001)}, None)
    assert issue.severity == "error" and "5,000 bytes" in issue.message and "es" in issue.message


def test_localization_issues_come_in_sorted_language_order():
    result = loc_codes("en", {"pt": loc(title=""), "de": loc(title="")})
    assert result == [("localizations.de", "localization_title_empty"),
                      ("localizations.pt", "localization_title_empty")]
