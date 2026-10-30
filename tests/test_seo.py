from __future__ import annotations

from dataclasses import replace

import pytest

from yezatas import seo
from yezatas.seo import (
    CATEGORIES,
    CHECKS,
    ChannelContext,
    channel_context,
    fix_priority,
    parse_chapters,
    parse_duration,
    score_catalog,
    score_video,
    tags_char_count,
)
from yezatas.store import Video

CHAPTERS = "0:00 Intro\n1:30 Setting up the garden bed\n4:10 Planting the seeds\n8:45 Wrap up"

DESCRIPTION = (
    "In this garden bed tutorial we build a raised bed from scratch and plant the first seeds "
    "of the season. You will see the tools, the soil mix and the mistakes to avoid.\n\n"
    + CHAPTERS
    + "\n\nWatch next: https://youtu.be/abc123XYZ00 for the compost bin build.\n"
    "Subscribe for more weekly garden projects and leave a comment with your own tips. "
    "Every week we show one practical build with a full materials list, honest costs and "
    "the small mistakes that cost us time, so you can skip them.\n"
    "#gardening #raisedbed"
)


def make_video(**overrides) -> Video:
    base = Video(
        video_id="v1",
        title="How to build a raised garden bed in one day",
        description=DESCRIPTION,
        tags=["garden bed", "raised bed", "gardening", "diy garden", "vegetables", "soil"],
        privacy_status="public",
        default_language="en",
        default_audio_language="en",
        duration="PT10M3S",
        localizations={code: {"title": "Titel", "description": "Text"}
                       for code in ("de", "fr", "es", "it", "pt")},
        thumbnails={"maxres": {"url": "https://example.invalid/t.jpg", "width": 1280, "height": 720}},
    )
    return replace(base, **overrides)


def card(video: Video, context: ChannelContext | None = None):
    return score_video(video, context)


def check(video: Video, check_id: str, context: ChannelContext | None = None):
    return next(c for c in card(video, context).checks if c.check_id == check_id)


def ctx_for(*videos: Video) -> ChannelContext:
    return channel_context(videos)


# --- whole scorecard ---

def test_perfect_video_scores_100():
    video = make_video()
    result = card(video, ctx_for(video))
    assert result.score == 100
    assert result.failed == []
    assert all(c.applicable and c.passed for c in result.checks)


def test_perfect_video_total_is_100_points():
    result = card(make_video(), ctx_for(make_video()))
    assert sum(c.max_points for c in result.checks) == 100
    assert len(result.checks) == len(CHECKS) == 25


def test_max_points_of_all_checks_sum_to_100():
    result = card(make_video(), ctx_for(make_video()))
    assert all(c.applicable for c in result.checks)
    assert sum(c.max_points for c in result.checks) == 100


def test_empty_video_scores_low_and_does_not_crash():
    result = card(Video(video_id="x"), ChannelContext())
    assert isinstance(result.score, int)
    assert 0 <= result.score <= 10
    assert result.failed


def test_empty_video_without_context():
    assert 0 <= card(Video(video_id="x")).score <= 10


def test_score_is_int_between_0_and_100():
    for video in (Video(video_id="a"), make_video(), make_video(title="X" * 300, tags=["a"] * 40)):
        value = card(video).score
        assert isinstance(value, int) and 0 <= value <= 100


def test_nothing_applicable_scores_100(monkeypatch):
    monkeypatch.setattr(seo, "CHECKS", ())
    assert seo.score_video(Video(video_id="x")).score == 100


def test_not_applicable_checks_do_not_change_score():
    video = make_video()
    without = card(video, None)  # title_unique not applicable
    with_ctx = card(video, ctx_for(video))
    assert not check(video, "title_unique", None).applicable
    assert without.score == with_ctx.score == 100
    short = make_video(duration="PT45S", description="x" * 300)
    assert not check(short, "description_chapters").applicable
    applicable = [c for c in card(short).checks if c.applicable]
    earned = sum(c.points for c in applicable)
    total = sum(c.max_points for c in applicable)
    assert card(short).score == round(100 * earned / total)


def test_check_metadata_invariants():
    videos = [Video(video_id="x"), make_video(), make_video(title="<>" * 50, tags=["a", "A"])]
    for video in videos:
        result = card(video, ctx_for(video))
        ids = [c.check_id for c in result.checks]
        assert len(ids) == len(set(ids))
        for c in result.checks:
            assert c.title
            assert c.category in CATEGORIES
            assert 0 <= c.points <= c.max_points
            if c.applicable and not c.passed:
                assert c.fix, c.check_id
            if c.passed or not c.applicable:
                assert c.fix == ""


def test_every_category_has_checks():
    categories = {c.category for c in card(make_video()).checks}
    assert categories == set(CATEGORIES)


# --- catalog and priority ---

def test_score_catalog_detects_duplicate_titles():
    a = make_video(video_id="a")
    b = make_video(video_id="b")
    c = make_video(video_id="c", title="A completely different title for the third video")
    cards = score_catalog([a, b, c])
    assert set(cards) == {"a", "b", "c"}
    for vid in ("a", "b"):
        unique = next(x for x in cards[vid].checks if x.check_id == "title_unique")
        assert unique.points == 0 and "2 videos" in unique.detail
    assert next(x for x in cards["c"].checks if x.check_id == "title_unique").passed


def test_duplicate_titles_ignore_case_and_spacing():
    a = make_video(video_id="a", title="Hello   World again and again and again")
    b = make_video(video_id="b", title="hello world AGAIN and again and again")
    assert not check(a, "title_unique", ctx_for(a, b)).passed


def test_fix_priority_is_views_times_missing():
    video = make_video(view_count=1000, title="")
    result = card(video, ctx_for(video))
    assert fix_priority(video, result) == 1000 * result.missing
    assert fix_priority(make_video(view_count=0), card(make_video())) == 0.0
    perfect = make_video(view_count=500)
    assert fix_priority(perfect, card(perfect, ctx_for(perfect))) == 0.0


def test_channel_context_counts():
    ctx = channel_context([make_video(video_id="a"), make_video(video_id="b")])
    assert ctx.title_counts == {"how to build a raised garden bed in one day": 2}


# --- title ---

@pytest.mark.parametrize("n,points", [
    (0, 0), (19, 0), (20, 4), (29, 4), (30, 8), (60, 8), (61, 4), (70, 4), (71, 0), (90, 0),
])
def test_title_length_limits(n, points):
    result = check(make_video(title="a" * n), "title_length")
    assert result.points == points
    assert str(n) in result.detail
    if points < 8:
        assert result.fix
    if n > 60:
        assert "phones" in result.fix


@pytest.mark.parametrize("n,ok", [(1, True), (70, True), (71, False), (0, False)])
def test_title_not_truncated(n, ok):
    result = check(make_video(title="a" * n), "title_not_truncated")
    assert result.passed is ok
    assert bool(result.fix) is (not ok)


def test_title_caps():
    assert check(make_video(title="How to build a RAISED bed"), "title_caps").passed
    assert not check(make_video(title="HOW TO BUILD A RAISED BED NOW"), "title_caps").passed
    # exactly 50% capitals passes, just over fails
    assert check(make_video(title="ABCdef"), "title_caps").passed
    assert not check(make_video(title="ABCDefg"), "title_caps").passed
    # short words are ignored, numbers only is not applicable
    assert check(make_video(title="Top 10 DIY ideas for a small plot"), "title_caps").passed
    assert not check(make_video(title="12345"), "title_caps").applicable


def test_title_unique_not_applicable_without_context_or_title():
    assert not check(make_video(), "title_unique").applicable
    assert not check(make_video(title=""), "title_unique", ChannelContext()).applicable
    assert check(make_video(), "title_unique", ChannelContext()).passed


def test_title_angle_brackets():
    assert check(make_video(), "title_no_angle_brackets").passed
    assert not check(make_video(title="Build <fast> beds"), "title_no_angle_brackets").passed
    assert not check(make_video(title="Beds > sheds"), "title_no_angle_brackets").passed


def test_title_punctuation():
    ok = "title_punctuation"
    assert check(make_video(title="Is this the best bed design?"), ok).passed
    assert check(make_video(title="What a bed!"), ok).passed
    assert not check(make_video(title="You will not believe this bed!!"), ok).passed
    assert not check(make_video(title="Is this the best bed design??"), ok).passed
    assert check(make_video(title="Great beds \U0001F331\U0001F331"), ok).passed
    assert not check(make_video(title="Great beds \U0001F331\U0001F331\U0001F331"), ok).passed
    assert "3 emojis" in check(make_video(title="\U0001F331\U0001F331\U0001F331"), ok).detail


# --- description ---

@pytest.mark.parametrize("n,points", [
    (0, 0), (249, 0), (250, 2), (499, 2), (500, 5), (900, 5),
])
def test_description_length(n, points):
    result = check(make_video(description="a" * n), "description_length")
    assert result.points == points
    assert str(n) in result.detail


def test_description_hook():
    hook = "description_hook"
    assert check(make_video(), hook).passed
    links_first = "https://site.invalid/a https://site.invalid/b #one #two " + "x" * 40
    assert not check(make_video(description=links_first), hook).passed
    assert check(make_video(description="x" * 50), hook).passed
    assert not check(make_video(description="x" * 49), hook).passed
    # text after the 150 character window does not count
    assert not check(make_video(description="#tag " * 40 + "y" * 80), hook).passed
    assert not check(make_video(description=""), hook).passed


SENTENCE = "This video shows how to build a raised bed and plant the first seeds of spring."


@pytest.mark.parametrize("first_line,problem", [
    ("https://site.invalid/shop " + SENTENCE, "link"),
    ("HTTP://site.invalid/shop", "link"),
    ("www.site.invalid " + SENTENCE, "link"),
    ("______________________________", "separator"),
    ("----------", "separator"),
    ("= = = = =", "separator"),
    ("•••", "separator"),
    ("=== LINKS ===", "separator"),
    ("Subscribe for more garden builds every week, with full plans and costs.", "promotion"),
    ("Use code SAVE10 for a discount on the tools I use in this video.", "promotion"),
    ("Support my channel on Patreon to see new builds early, thank you.", "promotion"),
    ("Follow me for daily garden updates and behind the scenes photos.", "promotion"),
    ("Buy the same raised bed kit from my affiliate store today.", "promotion"),
])
def test_hook_fails_on_link_separator_or_promo_first_line(first_line, problem):
    result = check(make_video(description=first_line + "\n" + SENTENCE + " " + SENTENCE), "description_hook")
    assert not result.passed and result.points == 0
    assert problem in result.detail
    assert "move links and promos lower" in result.fix


def test_hook_skips_leading_blank_lines_and_checks_first_real_line():
    result = check(make_video(description="\n\n  \nhttps://site.invalid\n" + SENTENCE),
                   "description_hook")
    assert not result.passed and "link" in result.detail


def test_hook_passes_with_a_sentence_first_even_if_links_and_promos_follow():
    desc = SENTENCE + " It takes one day.\n\nhttps://site.invalid/plans\nSubscribe and use code SAVE10"
    result = check(make_video(description=desc), "description_hook")
    assert result.passed and result.points == 8
    assert check(make_video(description=SENTENCE + "\n- a bullet\n---"), "description_hook").passed


def test_hook_is_worth_8_points():
    assert check(make_video(), "description_hook").max_points == 8


def test_description_title_words():
    words = "description_title_words"
    assert check(make_video(), words).passed
    assert not check(make_video(description="Nothing relevant here at all. " * 10), words).passed
    late = "filler " * 40 + "garden"
    assert not check(make_video(description=late), words).passed
    assert not check(make_video(title="How to do it", description="x"), words).applicable
    assert check(make_video(title="Raised Beds", description="build RAISED beds"), words).passed


@pytest.mark.parametrize("text", [
    "see http:// for more", "https://", "visit example.com now", "Link: [link]", "LINK HERE",
    "TODO add sources", "http://\nnext line", "buy at [URL]",
])
def test_description_links_broken(text):
    assert not check(make_video(description=DESCRIPTION + "\n" + text), "description_links_ok").passed


def test_description_links_fine():
    good = DESCRIPTION + "\nhttps://site.invalid/page?x=1\nmy-example.org and todos"
    assert check(make_video(description=good), "description_links_ok").passed
    assert not check(make_video(description=""), "description_links_ok").applicable


def test_chapters_pass_and_first_must_be_zero():
    chap = "description_chapters"
    assert check(make_video(), chap).passed
    late_start = "0:05 Intro\n1:30 Middle\n4:10 End"
    result = check(make_video(description=late_start), chap)
    assert not result.passed and "0:00" in result.detail
    assert result.fix


def test_chapters_out_of_order_or_too_close():
    chap = "description_chapters"
    assert not check(make_video(description="0:00 A\n4:10 B\n1:30 C"), chap).passed
    assert not check(make_video(description="0:00 A\n0:09 B\n1:30 C"), chap).passed
    assert check(make_video(description="0:00 A\n0:10 B\n1:30 C"), chap).passed
    assert not check(make_video(description="0:00 A\n1:00 A again\n1:00 B"), chap).passed


def test_chapters_need_three():
    assert not check(make_video(description="0:00 A\n1:30 B"), "description_chapters").passed
    assert not check(make_video(description="no chapters at all"), "description_chapters").passed


def test_chapters_must_start_the_line():
    desc = "Intro at 0:00 and 1:30 and 4:10 are not chapters"
    assert not check(make_video(description=desc), "description_chapters").passed


def test_chapters_hour_format_and_bullets():
    desc = "0:00 Intro\n- 12:30 Middle\n1:02:03 End"
    assert parse_chapters(desc) == [0, 750, 3723]
    assert check(make_video(description=desc, duration="PT1H5M"), "description_chapters").passed


def test_chapters_not_applicable_for_short_or_unknown_duration():
    chap = "description_chapters"
    for duration in ("PT3M59S", "PT45S", "PT1M", "", "P0D", "garbage"):
        assert not check(make_video(description="none", duration=duration), chap).applicable
    assert "4 minutes" in check(make_video(description="none", duration="PT3M59S"), chap).detail
    assert check(make_video(description="none", duration="PT4M"), chap).applicable
    assert not check(make_video(description="none", duration="PT4M"), chap).passed
    assert check(make_video(description="none", duration="PT5M59S"), chap).applicable
    assert check(make_video(description="0:00 A\n0:30 B\n1:30 C", duration="PT4M"), chap).passed


def test_hook_not_applicable_for_shorts():
    assert not check(make_video(duration="PT45S", description="x"), "description_hook").applicable
    assert not check(make_video(duration="PT1M", description="x"), "description_hook").applicable
    assert check(make_video(duration="PT1M1S", description="x"), "description_hook").applicable
    assert check(make_video(duration="", description="x"), "description_hook").applicable


@pytest.mark.parametrize("n,points", [(0, 1), (1, 3), (3, 3), (4, 1), (15, 1), (16, 0)])
def test_hashtags(n, points):
    tags = " ".join(f"#tag{chr(97 + i)}" for i in range(n))
    result = check(make_video(description="Intro text. " + tags), "description_hashtags")
    assert result.points == points
    assert f"{n} hashtags" in result.detail


def test_hashtags_ignore_numbers_and_duplicates():
    desc = "Issue #1 and #2 plus #Garden #garden"
    assert check(make_video(description=desc), "description_hashtags").passed


def test_description_angle_brackets():
    key = "description_no_angle_brackets"
    assert check(make_video(), key).passed
    assert not check(make_video(description=DESCRIPTION + " <b>"), key).passed
    assert not check(make_video(description="a > b"), key).passed
    assert not check(make_video(description=""), key).applicable


def test_description_limit():
    assert check(make_video(description="a" * 5000), "description_limit").passed
    over = check(make_video(description="a" * 5001), "description_limit")
    assert not over.passed and "1 characters" in over.fix
    assert not check(make_video(description=""), "description_limit").applicable


def test_description_not_title_copy():
    key = "description_not_title_copy"
    title = "How to build a raised garden bed in one day"
    assert not check(make_video(title=title, description=title), key).passed
    assert not check(make_video(title=title, description="  " + title.upper() + " "), key).passed
    assert not check(make_video(title=title, description=title + " - video"), key).passed
    assert check(make_video(title=title, description=title + " " + "more words " * 3), key).passed
    assert check(make_video(), key).passed
    assert not check(make_video(title="", description="abc"), key).applicable


def test_engagement_prompt():
    key = "engagement_prompt"
    assert check(make_video(description="Please SUBSCRIBE!"), key).passed
    assert check(make_video(description="leave a comment"), key).passed
    assert check(make_video(description="hit like"), key).passed
    assert not check(make_video(description="A plain description."), key).passed
    assert not check(make_video(description="unlikely and dislikeable"), key).passed


# --- tags ---

@pytest.mark.parametrize("n,points", [(0, 0), (1, 2), (4, 2), (5, 5), (15, 5)])
def test_tags_present(n, points):
    tags = [f"tag{chr(97 + i)}" for i in range(n)]
    assert check(make_video(tags=tags), "tags_present").points == points


def test_tags_char_count_rules():
    assert tags_char_count([]) == 0
    assert tags_char_count(["abc"]) == 3
    assert tags_char_count(["abc", "def"]) == 7  # comma between
    assert tags_char_count(["a b"]) == 5  # quotes
    assert tags_char_count(["a b", "cd"]) == 5 + 2 + 1


def test_tags_length_limit_counts_spaces_and_commas():
    # 100 tags of 4 letters + 99 commas = 499 -> ok; 100 tags of "a b" = 100*5 + 99 > 500
    ok = [f"{i:04d}" for i in range(100)]
    assert tags_char_count(ok) == 499
    assert check(make_video(tags=ok), "tags_length").passed
    assert check(make_video(tags=ok[:99] + ["abcde"]), "tags_length").passed  # exactly 500
    assert not check(make_video(tags=ok[:99] + ["abcdef"]), "tags_length").passed
    spaced = [f"a {i:02d}" for i in range(60)]  # (4+2)*60 + 59 = 419
    assert check(make_video(tags=spaced), "tags_length").passed
    over = [f"a {i:02d}" for i in range(80)]  # 480 + 79 = 559
    result = check(make_video(tags=over), "tags_length")
    assert not result.passed and "559" in result.detail
    assert not check(make_video(tags=[]), "tags_length").applicable


def test_tags_title_match():
    key = "tags_title_match"
    assert check(make_video(), key).passed
    assert not check(make_video(tags=["unrelated", "stuff"]), key).passed
    assert check(make_video(tags=["Raised Beds Plans"]), key).passed
    assert not check(make_video(tags=[]), key).applicable
    assert not check(make_video(title="How to do it"), key).applicable


def test_tags_unique():
    key = "tags_unique"
    assert check(make_video(), key).passed
    result = check(make_video(tags=["Garden", "soil", "garden "]), key)
    assert not result.passed and "garden" in result.detail
    assert not check(make_video(tags=["a b", "A  B"]), key).passed
    assert not check(make_video(tags=[]), key).applicable


# --- details ---

def test_languages():
    assert check(make_video(), "default_language").passed
    assert not check(make_video(default_language=None), "default_language").passed
    assert not check(make_video(default_language=""), "default_language").passed
    assert check(make_video(), "audio_language").passed
    assert not check(make_video(default_audio_language=None), "audio_language").passed


def test_localizations():
    assert check(make_video(), "localizations").passed
    result = check(make_video(localizations={}), "localizations")
    assert not result.passed and result.points == 0 and "Yezatas" in result.fix


@pytest.mark.parametrize("n,points", [(0, 0), (1, 4), (3, 4), (4, 4), (5, 8), (12, 8)])
def test_localizations_need_five_languages(n, points):
    locs = {f"l{i}": {"title": "t", "description": "d"} for i in range(n)}
    result = check(make_video(localizations=locs), "localizations")
    assert result.points == points and result.max_points == 8
    if points < 8:
        assert "Yezatas" in result.fix


def test_localizations_do_not_count_the_video_language():
    # Studio lists the video's own language among the translations.
    loc = {"title": "t", "description": "d"}
    result = check(make_video(default_language="en", localizations={"en": loc}), "localizations")
    assert result.points == 0 and "0 translations" in result.detail
    locs = {"en-GB": loc, "de-DE": loc, "es": loc}
    result = check(make_video(default_language="en", localizations=locs), "localizations")
    assert "2 translations" in result.detail
    locs = {"zh-Hant": loc}
    result = check(make_video(default_language="zh-Hans", localizations=locs), "localizations")
    assert "1 translation." in result.detail


def test_related_links():
    key = "description_related_links"
    assert check(make_video(), key).passed
    for link in ("https://youtu.be/abc123", "https://www.youtube.com/watch?v=abc123",
                 "https://youtube.com/playlist?list=PL123", "https://www.youtube.com/shorts/abc"):
        assert check(make_video(description=SENTENCE + "\n" + link), key).passed, link
    plain = check(make_video(description=SENTENCE + "\nMore at https://site.invalid/plans"), key)
    assert not plain.passed and "Link a related video" in plain.fix
    assert not check(make_video(description=SENTENCE + "\nhttps://www.youtube.com/@name"),
                     key).passed
    assert not check(make_video(description=""), key).applicable
    assert not check(make_video(duration="PT45S", description="https://youtu.be/a"), key).applicable
    assert [c.__name__ for c in CHECKS].index("check_description_related_links") == \
        [c.__name__ for c in CHECKS].index("check_description_links_ok") + 1


def test_thumbnail_hd():
    key = "thumbnail_hd"
    assert check(make_video(thumbnails={"standard": {}}), key).passed
    assert check(make_video(thumbnails={"default": {}, "maxres": {}}), key).passed
    assert not check(make_video(thumbnails={"default": {}, "high": {}}), key).passed
    assert not check(make_video(thumbnails={}), key).applicable


# --- helpers ---

@pytest.mark.parametrize("text,seconds", [
    ("PT1H2M3S", 3723), ("PT45S", 45), ("PT10M3S", 603), ("PT1H", 3600), ("PT5M", 300),
    ("P1DT1H", 90000), ("PT0S", None), ("P0D", None), ("", None), ("junk", None),
    ("PT", None), ("PT1.5S", 1),
])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


def test_parse_duration_none_input():
    assert parse_duration(None) is None  # type: ignore[arg-type]


def _result_of(check_id: str, **fields) -> seo.CheckResult:
    card = seo.score_video(Video(video_id="v1", **fields))
    return next(c for c in card.checks if c.check_id == check_id)


def test_lowercase_todo_is_not_a_placeholder_link() -> None:
    ok = _result_of("description_links_ok", description="Best todo app for planning your week.")
    bad = _result_of("description_links_ok", description="Links: TODO add the store page.")
    assert ok.passed and not bad.passed


def test_call_to_action_needs_a_phrase_not_just_like_or_comment() -> None:
    plain = _result_of("engagement_prompt", description="It looks like a comment on the case.")
    asks = _result_of("engagement_prompt", description="Let me know what you think below.")
    assert not plain.passed and asks.passed


def test_buying_question_is_a_fine_description_opening() -> None:
    hook = _result_of("description_hook", duration="PT8M",
                      description="Which budget laptop should you buy this year? I compare three of them "
                                  "for battery, screen and speed.")
    assert hook.passed
