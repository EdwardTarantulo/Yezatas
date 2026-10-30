from yezatas.facts import check_facts
from yezatas.validate import Draft

ORIGINAL = Draft(
    "Repotting cactus",
    "How to repot a cactus.\nMore: https://example.invalid/shop?ref=abc\n"
    "Intro 0:00\nSoil 1:05\nContact: hello@example.invalid or @sample_garden\n",
)


def kinds(draft: Draft, original: Draft = ORIGINAL, allowed: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    return [(p.kind, p.value) for p in check_facts(draft, original, allowed)]


def with_description(text: str, title: str = "Repotting cactus") -> Draft:
    return Draft(title, text)


def test_unchanged_text_has_no_problems() -> None:
    assert check_facts(ORIGINAL, ORIGINAL) == []
    shuffled = with_description("Soil 1:05\nIntro 0:00\nWrite to hello@example.invalid. "
                                "Shop https://example.invalid/shop?ref=abc\n@sample_garden")
    assert check_facts(shuffled, ORIGINAL) == []


def test_new_link() -> None:
    draft = with_description("See https://other.invalid/page for more.")
    problems = check_facts(draft, ORIGINAL)
    assert [(p.field, p.kind, p.value) for p in problems] == [
        ("description", "new_link", "https://other.invalid/page")]
    assert "other.invalid" in problems[0].message


def test_new_link_in_title() -> None:
    problems = check_facts(Draft("Cactus www.other.invalid", ORIGINAL.description), ORIGINAL)
    assert [(p.field, p.kind) for p in problems] == [("title", "new_link")]


def test_trailing_punctuation_is_stripped() -> None:
    assert kinds(with_description("(see https://example.invalid/shop?ref=abc).")) == []
    assert kinds(with_description("Go to www.other.invalid/a, then rest.")) == [
        ("new_link", "www.other.invalid/a")]


def test_link_host_is_case_insensitive_and_scheme_free() -> None:
    assert kinds(with_description("http://EXAMPLE.invalid/shop?ref=abc")) == []
    assert kinds(with_description("https://www.example.invalid/shop?ref=abc")) == []
    assert kinds(with_description("https://example.invalid/shop?ref=other")) == [
        ("new_link", "https://example.invalid/shop?ref=other")]


def test_allowed_urls_are_not_new() -> None:
    draft = with_description("Next: https://www.youtube.com/watch?v=vid000007")
    assert len(check_facts(draft, ORIGINAL)) == 1
    assert check_facts(draft, ORIGINAL, ["https://www.youtube.com/watch?v=vid000007"]) == []
    assert check_facts(draft, ORIGINAL, {"https://www.youtube.com/watch?v=vid000007."}) == []


def test_youtu_be_and_watch_links_are_the_same() -> None:
    original = with_description("Part one: https://youtu.be/AbC123xyz_0")
    same = with_description("Part one: https://www.youtube.com/watch?v=AbC123xyz_0&t=5")
    other = with_description("Part one: https://youtu.be/ZZZ")
    assert check_facts(same, original) == []
    assert [p.kind for p in check_facts(other, original)] == ["new_link"]
    assert check_facts(with_description("https://youtu.be/AbC123xyz_0"), same) == []


def test_new_timestamp() -> None:
    draft = with_description("Intro 0:00\nSoil 1:05\nPots 3:45\nAll 1:02:03\nAgain 3:45")
    assert kinds(draft) == [("new_timestamp", "3:45"), ("new_timestamp", "1:02:03")]


def test_timestamp_forms_compare_by_value() -> None:
    assert kinds(with_description("00:00 Intro\n01:05 Soil")) == []
    assert kinds(with_description("0:00:00 Intro")) == []


def test_ratio_and_times_inside_words_are_not_timestamps() -> None:
    assert kinds(with_description("Filmed in 16:9 and 4:3, version 2:0")) == []
    assert kinds(with_description("Code a12:30 and 5:61")) == []


def test_timestamp_in_a_url_is_not_a_timestamp() -> None:
    original = with_description("x")
    problems = check_facts(with_description("https://example.invalid/a/12:34"), original)
    assert [p.kind for p in problems] == ["new_link"]


def test_new_handle() -> None:
    draft = with_description("Follow @new_person and @Sample_Garden, @new_person again.")
    assert kinds(draft) == [("new_handle", "@new_person")]


def test_handle_inside_email_or_url_is_not_a_handle() -> None:
    problems = check_facts(with_description("Mail me@other.invalid or https://x.invalid/@name."),
                           ORIGINAL)
    assert sorted(p.kind for p in problems) == ["new_email", "new_link"]


def test_new_email() -> None:
    draft = with_description("Write to Hello@Example.invalid or new.person@mail.invalid.")
    assert kinds(draft) == [("new_email", "new.person@mail.invalid")]


def test_one_problem_per_value_across_fields() -> None:
    draft = Draft("Visit https://other.invalid", "Visit https://other.invalid again")
    assert [(p.field, p.kind) for p in check_facts(draft, ORIGINAL)] == [("title", "new_link")]


def test_original_title_counts_as_known() -> None:
    original = Draft("Cactus @sample_garden 5:00", "text")
    draft = Draft("Other title", "Thanks @sample_garden, see 5:00")
    assert check_facts(draft, original) == []
