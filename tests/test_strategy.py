"""Channel strategy: the context made by code, the guarded model report and the saved files.

Videos, comments and replies are made up; the model is the fake. No test touches the network.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient
from yezatas import seo
from yezatas.agent import ModelError, ModelReply
from yezatas.store import Comment, OutlierReport, Store, Video
from yezatas.strategy import (
    MAX_IDEAS, MAX_POINTS, MAX_QUESTIONS, QUESTION_CHARS, STRATEGY_FORMAT, STRATEGY_MODEL,
    TEXT_MAX, StrategyError, StrategyIdea, StrategyPoint, StrategyReport, build_context,
    SYSTEM_PROMPT, latest_report, make_report, save_report,
)

START = datetime(2025, 1, 1, tzinfo=timezone.utc)
AS_OF = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)


def vid(n: int) -> str:
    return f"vid{n:08d}"


def video(n: int, views: int = 100, **kw: Any) -> Video:
    published = START + timedelta(days=n)
    base: dict[str, Any] = dict(
        video_id=vid(n),
        title=f"Cactus {chr(65 + n % 26)}{chr(65 + (n // 26) % 26)}",
        description="How to repot a cactus and keep it healthy in a bright spot.",
        tags=["cactus", "garden"], privacy_status="public",
        published_at=published.strftime("%Y-%m-%dT%H:%M:%SZ"), duration="PT5M30S",
        view_count=views,
        thumbnails={"medium": {"url": f"https://i.ytimg.com/vi/{vid(n)}/mqdefault.jpg"}})
    base.update(kw)
    return Video(**base)


def catalog(count: int = 30, views: dict[int, int] | None = None) -> list[Video]:
    views = views or {}
    return [video(n, views.get(n, 100)) for n in range(1, count + 1)]


def make_store(tmp_path: Path, videos: list[Video]) -> Store:
    store = Store(tmp_path / "c.sqlite")
    pull = store.begin_pull("chan0000001", now=NOW)
    for v in videos:
        store.upsert_video(v, pull, now=NOW)
    store.finish_pull(pull, len(videos), now=NOW)
    return store


def reply(data: Any, tokens: tuple[int, int, int] = (1000, 100, 200)) -> ModelReply:
    text = data if isinstance(data, str) else json.dumps(data)
    return ModelReply(text, (), {}, *tokens)


def good_answer(**override: Any) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "works": [{"text": "Cactus question titles do well.", "refs": ["V15"]}],
        "gaps": [{"text": "Some cactus videos have thin descriptions.", "refs": ["V3"]}],
        "ideas": [{"title": "Try a question title", "why": "It builds on the cactus video V15.",
                   "refs": ["V15"]}],
    }
    answer.update(override)
    return answer


def run_report(store: Store, data: Any, **kwargs: Any) -> StrategyReport:
    client = FakeModelClient([reply(data)], model=STRATEGY_MODEL)
    return make_report(store, client, as_of=AS_OF, **kwargs)


@pytest.fixture
def store(tmp_path: Path):
    with make_store(tmp_path, catalog(30, {15: 5000, 3: 1})) as s:
        yield s


# --- the context -------------------------------------------------------------------------------
def test_context_holds_the_catalog_outliers_and_seo(store: Store) -> None:
    context = build_context(store, AS_OF)
    assert context.refs["V1"] == vid(1) and context.refs["V15"] == vid(15)
    assert "V1" in context.catalog_table and "Cactus" in context.catalog_table
    assert context.catalog_table in context.text
    assert context.outliers == (("V15", "50×"),)
    assert context.below_norm == (("V3", "0.0×"),)
    assert "V15 50×" in context.text and "V3 0.0×" in context.text
    videos = store.list_videos("video_id")
    cards = [seo.score_video(v, seo.channel_context(videos)) for v in videos]
    assert context.average_seo == round(sum(c.score for c in cards) / len(cards), 1)
    assert 0.0 <= context.average_seo <= 100.0
    assert context.failed_checks and len(context.failed_checks) <= 5
    counts = [n for _, n in context.failed_checks]
    assert counts == sorted(counts, reverse=True)
    assert all(isinstance(title, str) and title for title, _ in context.failed_checks)
    assert f"Average SEO score: {context.average_seo:.1f}" in context.text
    assert context.valid_refs == frozenset(context.refs)
    assert "V999" not in context.valid_refs


def test_context_has_no_questions_or_insights_when_there_are_none(store: Store) -> None:
    context = build_context(store, AS_OF)
    assert context.questions == () and context.insight_lines == ()
    assert "Viewer questions" not in context.text


def test_context_holds_insight_lines_of_a_done_outlier_report(store: Store) -> None:
    store.save_outlier_report(OutlierReport(
        id=0, pull_id=None, as_of=AS_OF.isoformat(), params={}, labels={},
        insight={"patterns": [{"text": "Titles ask a question."}],
                 "per_video": [{"label": "O1", "why": "It asks a clear question."}],
                 "ideas": ["Ask a question next time."]},
        problems=[], status="done", error="", model=STRATEGY_MODEL, input_tokens=0,
        cached_tokens=0, output_tokens=0, cost_usd=None, created_at=""), now=NOW)
    context = build_context(store, AS_OF)
    assert context.insight_lines == (
        "Titles ask a question.", "It asks a clear question.", "Ask a question next time.")
    for line in context.insight_lines:
        assert line in context.text


def test_context_ignores_an_unfinished_outlier_report(store: Store) -> None:
    store.save_outlier_report(OutlierReport(
        id=0, pull_id=None, as_of=AS_OF.isoformat(), params={}, labels={},
        insight={"patterns": [{"text": "Titles ask a question."}], "per_video": [],
                 "ideas": []},
        problems=[], status="error", error="It failed.", model=STRATEGY_MODEL,
        input_tokens=0, cached_tokens=0, output_tokens=0, cost_usd=None, created_at=""),
        now=NOW)
    assert build_context(store, AS_OF).insight_lines == ()


def test_context_holds_questions_newest_first_and_cut(store: Store) -> None:
    long_text = "How do I water this plant? " + "please " * 40
    store.upsert_comments([
        Comment(comment_id="cmt00000001", video_id=vid(2), author="Viewer A",
                text="An older question about soil?", published_at="2026-05-10T10:00:00Z"),
        Comment(comment_id="cmt00000002", video_id=vid(5), author="Viewer B",
                text=long_text, published_at="2026-05-20T10:00:00Z"),
        Comment(comment_id="cmt00000003", video_id=vid(2), author="Viewer C",
                text="Great video, thanks.", published_at="2026-05-25T10:00:00Z"),
    ], now=NOW)
    store.set_comment_categories(
        {"cmt00000001": "question", "cmt00000002": "question", "cmt00000003": "praise"},
        now=NOW)
    context = build_context(store, AS_OF)
    assert context.questions == (
        ("V5", long_text[:QUESTION_CHARS]), ("V2", "An older question about soil?"))
    assert len(context.questions[0][1]) == QUESTION_CHARS == 200
    assert f"V5: {long_text[:QUESTION_CHARS]}" in context.text
    assert "Great video" not in context.text
    assert len(context.questions) <= MAX_QUESTIONS


# --- the report ----------------------------------------------------------------------------------
def test_make_report_happy_path(store: Store) -> None:
    assert STRATEGY_MODEL == "muse-wonder-1.0"
    lines: list[str] = []
    client = FakeModelClient([reply(good_answer())], model=STRATEGY_MODEL)
    report = make_report(store, client, as_of=AS_OF, now=NOW, report=lines.append)
    assert len(client.calls) == 1 and lines
    call = client.calls[0]
    assert call["response_format"] is STRATEGY_FORMAT
    fmt = call["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["name"] == "channel_strategy"
    assert fmt["json_schema"]["strict"] is True
    assert set(fmt["json_schema"]["schema"]["properties"]) == {"works", "gaps", "ideas"}
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert call["messages"][1]["content"] == build_context(store, AS_OF).text
    assert report.works == (StrategyPoint("Cactus question titles do well.", (vid(15),)),)
    assert report.gaps == (StrategyPoint("Some cactus videos have thin descriptions.", (vid(3),)),)
    assert report.ideas == (
        StrategyIdea("Try a question title", "It builds on the cactus video V15.", (vid(15),)),)
    assert report.notes == () and report.model == STRATEGY_MODEL
    assert (report.input_tokens, report.cached_tokens, report.output_tokens) == (1000, 100, 200)
    assert report.created_at == NOW.isoformat()


def test_invented_refs_are_filtered_and_a_point_without_one_is_dropped(store: Store) -> None:
    report = run_report(store, good_answer(
        works=[{"text": "Cactus question titles do well.", "refs": ["V15", "V999", "V15"]},
               {"text": "Only an invented video.", "refs": ["V999"]}],
        gaps=[], ideas=[]))
    assert report.works == (StrategyPoint("Cactus question titles do well.", (vid(15),)),)
    assert report.notes == ("Dropped a point that named no video of the channel.",)


def test_an_idea_without_a_ref_is_dropped_with_its_note(store: Store) -> None:
    report = run_report(store, good_answer(
        ideas=[{"title": "A loose idea", "why": "It builds on nothing.", "refs": []}]))
    assert report.ideas == ()
    assert report.notes == ("Dropped an idea that named no video of the channel.",)


def test_invented_numbers_are_dropped_with_a_note(store: Store) -> None:
    assert "999999" not in build_context(store, AS_OF).numbers
    report = run_report(store, good_answer(
        works=[{"text": "It reached 999999 views.", "refs": ["V15"]},
               {"text": "Cactus question titles do well.", "refs": ["V15"]}],
        ideas=[{"title": "Repeat it", "why": "It reached 999999 views.", "refs": ["V15"]},
               {"title": "Try a question title", "why": "It builds on the cactus video V15.",
                "refs": ["V15"]}]))
    assert [p.text for p in report.works] == ["Cactus question titles do well."]
    assert [i.title for i in report.ideas] == ["Try a question title"]
    assert report.notes == (
        "Dropped a point that uses a number that was not in the tables.",
        "Dropped an idea that uses a number that was not in the tables.")


def test_everything_dropped_raises(store: Store) -> None:
    client = FakeModelClient([reply(good_answer(
        works=[{"text": "Only invented.", "refs": ["V999"]}], gaps=[], ideas=[]))],
        model=STRATEGY_MODEL)
    with pytest.raises(StrategyError, match="nothing that could be kept"):
        make_report(store, client, as_of=AS_OF)


def test_no_videos_raises_before_a_model_call(tmp_path: Path) -> None:
    with Store(tmp_path / "empty.sqlite") as empty:
        client = FakeModelClient([], model=STRATEGY_MODEL)
        with pytest.raises(StrategyError, match="no pulled catalog yet"):
            build_context(empty, AS_OF)
        with pytest.raises(StrategyError, match="no pulled catalog yet"):
            make_report(empty, client, as_of=AS_OF)
        assert client.calls == []


def test_a_model_error_and_a_bad_answer_become_strategy_errors(store: Store) -> None:
    failing = FakeModelClient([ModelError("rate_limit", "Slow down.")], model=STRATEGY_MODEL)
    with pytest.raises(StrategyError, match="Slow down"):
        make_report(store, failing, as_of=AS_OF)
    for content in ("not json at all", json.dumps({"works": []})):
        client = FakeModelClient([reply(content)], model=STRATEGY_MODEL)
        with pytest.raises(StrategyError, match="could not be read"):
            make_report(store, client, as_of=AS_OF)


def test_texts_are_cut_and_sections_are_limited(store: Store) -> None:
    long_text = "Helpful cactus opening. " + "word " * 90
    report = run_report(store, {
        "works": [{"text": long_text, "refs": ["V15"]} for _ in range(MAX_POINTS + 2)],
        "gaps": [{"text": long_text, "refs": ["V15"]} for _ in range(MAX_POINTS + 2)],
        "ideas": [{"title": long_text, "why": "It builds on the cactus video V15.", "refs": ["V15"]}
                  for _ in range(MAX_IDEAS + 2)],
    })
    assert len(report.works) == MAX_POINTS and len(report.gaps) == MAX_POINTS
    assert len(report.ideas) == MAX_IDEAS and report.notes == ()
    assert all(len(p.text) <= TEXT_MAX for p in report.works + report.gaps)
    assert all(len(i.title) <= TEXT_MAX and len(i.why) <= TEXT_MAX for i in report.ideas)


@pytest.fixture
def titled(tmp_path: Path):
    with make_store(
        tmp_path,
        [
            video(1, title="How to connect optical cable to a Mesa TV"),
            video(2, title="Hearthstone Arena - Reaching Level 5 Key"),
            video(3, title="Repot a Cactus in Spring"),
        ],
    ) as s:
        yield s


KEPT = {"text": "The optical cable guide V1 has no next step.", "refs": ["V1"]}


def test_a_point_that_does_not_match_its_titles_is_dropped(titled) -> None:
    context = build_context(titled, AS_OF)
    assert context.titles["V3"] == "Repot a Cactus in Spring"
    report = run_report(
        titled,
        good_answer(
            works=[
                {"text": "V2 sits on one side of a boot-manager choice.", "refs": ["V2"]},
                KEPT,
            ],
            gaps=[],
            ideas=[],
        ),
    )
    assert [p.text for p in report.works] == [KEPT["text"]]
    assert report.notes == (
        "Dropped a point that does not match the titles of the videos it names.",
    )


def test_an_idea_that_does_not_match_its_titles_is_dropped(titled) -> None:
    report = run_report(
        titled,
        good_answer(
            works=[KEPT],
            gaps=[],
            ideas=[
                {"title": "GRUB vs rEFInd", "why": "Builds on V2.", "refs": ["V2"]},
                {
                    "title": "Repotting in winter",
                    "why": "Builds on the cactus video V3.",
                    "refs": ["V3"],
                },
            ],
        ),
    )
    assert [i.title for i in report.ideas] == ["Repotting in winter"]
    assert report.notes == (
        "Dropped an idea that does not match the titles of the videos it names.",
    )


def test_a_point_must_match_every_video_it_names(titled) -> None:
    report = run_report(
        titled,
        good_answer(
            works=[
                {"text": "Optical cable tips for V1 and V2.", "refs": ["V1", "V2"]},
                {"text": "Optical cable and arena tips.", "refs": ["V1", "V2"]},
            ],
            gaps=[],
            ideas=[],
        ),
    )
    assert [p.text for p in report.works] == ["Optical cable and arena tips."]


def test_a_plural_matches_a_title_word(titled) -> None:
    report = run_report(
        titled,
        good_answer(works=[{"text": "Better cables for V1.", "refs": ["V1"]}], gaps=[], ideas=[]),
    )
    assert len(report.works) == 1


def test_the_prompt_asks_for_a_title_word() -> None:
    assert "use at least one word of its title" in SYSTEM_PROMPT


# --- saving --------------------------------------------------------------------------------------
def sample_report(created_at: str = "2026-06-01T12:00:00+00:00") -> StrategyReport:
    return StrategyReport(
        works=(StrategyPoint("Cactus question titles do well.", (vid(15),)),),
        gaps=(StrategyPoint("Some cactus videos have thin descriptions.", (vid(3),)),),
        ideas=(StrategyIdea("Try a question title", "It builds on the cactus video V15.",
                (vid(15),)),),
        notes=("A note.",), model=STRATEGY_MODEL, input_tokens=1000, cached_tokens=100,
        output_tokens=200, created_at=created_at)


def test_save_and_latest_round_trip(tmp_path: Path) -> None:
    home = tmp_path / "home"
    report = sample_report()
    path = save_report(home, report)
    assert path == home / "reports" / "strategy-20260601-120000-000000.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "works": [{"text": "Cactus question titles do well.", "video_ids": [vid(15)]}],
        "gaps": [{"text": "Some cactus videos have thin descriptions.", "video_ids": [vid(3)]}],
        "ideas": [{"title": "Try a question title", "why": "It builds on the cactus video V15.",
                   "video_ids": [vid(15)]}],
        "notes": ["A note."], "model": STRATEGY_MODEL, "input_tokens": 1000,
        "cached_tokens": 100, "output_tokens": 200, "created_at": report.created_at}
    assert latest_report(home) == report
    second = save_report(home, report)
    assert second.name == "strategy-20260601-120000-000000-1.json" and path.exists()
    assert latest_report(tmp_path / "nowhere") is None


def test_latest_is_the_second_save_with_the_same_created_at(tmp_path: Path) -> None:
    home = tmp_path / "home"
    first = sample_report()
    second = StrategyReport(
        works=first.works, gaps=first.gaps, ideas=first.ideas,
        notes=("Second save.",), model=first.model, input_tokens=first.input_tokens,
        cached_tokens=first.cached_tokens, output_tokens=first.output_tokens,
        created_at=first.created_at)
    save_report(home, first)
    save_report(home, second)
    assert latest_report(home) == second


def test_latest_skips_broken_files(tmp_path: Path) -> None:
    home = tmp_path / "home"
    good = sample_report()
    save_report(home, good)
    folder = home / "reports"
    (folder / "strategy-20260602-120000.json").write_text("{not json", encoding="utf-8")
    (folder / "strategy-20260603-120000.json").write_text(
        json.dumps({"works": []}), encoding="utf-8")
    (folder / "strategy-20260604-120000.json").write_text(
        json.dumps(["not", "a", "report"]), encoding="utf-8")
    assert latest_report(home) == good


def test_save_keeps_only_ten_files(tmp_path: Path) -> None:
    home = tmp_path / "home"
    paths = []
    for minute in range(12):
        moment = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=minute)
        paths.append(save_report(home, sample_report(moment.isoformat())))
    files = sorted((home / "reports").glob("strategy-*.json"))
    assert len(files) == 10
    assert not paths[0].exists() and not paths[1].exists()
    assert paths[-1].exists() and latest_report(home) is not None


def test_make_and_save_write_nothing_to_the_store(store: Store, tmp_path: Path) -> None:
    before_videos = store.list_videos("video_id")
    before_report = store.latest_outlier_report()
    report = run_report(store, good_answer())
    save_report(tmp_path / "home", report)
    assert store.list_videos("video_id") == before_videos
    assert store.latest_outlier_report() == before_report
    assert (tmp_path / "home" / "reports").is_dir()
    assert not (tmp_path / "reports").exists()
