"""Own-channel outliers: the score (eligibility, peers, median, ratio), the helpers, and the Muse Spark
explanation with its guards. Videos and replies are made up; no test touches the network."""

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from fakes import FakeModelClient
from yezatas.agent import ModelError, ModelReply, estimate_cost
from yezatas.outliers import (BELOW_THRESHOLD, MAX_EXPLAINED, MAX_TYPICAL, MIN_AGE_DAYS,
                              OUTLIER_THRESHOLD, PEERS, OutlierScore, below_norm, duration_seconds,
                              explain_outliers, outliers, ratio_label, score_videos, typical)
from yezatas.packaging import PackagingError
from yezatas.store import Store, Video

START = datetime(2025, 1, 1, tzinfo=timezone.utc)
AS_OF = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)


def video(n: int, views: int = 100, *, day: int | None = None, privacy: str = "public",
          duration: str = "PT5M30S", **kw: Any) -> Video:
    """Video n, published on day `day` (default n) after START. Titles are letters, not numbers."""
    published = START + timedelta(days=n if day is None else day)
    base: dict[str, Any] = dict(
        video_id=f"vid{n:06d}", title=f"Cactus {chr(65 + n % 26)}{chr(65 + n // 26)}",
        description="How to repot a cactus.", tags=["cactus", "garden"], privacy_status=privacy,
        published_at=published.strftime("%Y-%m-%dT%H:%M:%SZ"), duration=duration, view_count=views,
        thumbnails={"medium": {"url": f"https://i.ytimg.com/vi/vid{n:06d}/mqdefault.jpg"}})
    base.update(kw)
    return Video(**base)


def catalog(count: int = 30, boosted: dict[int, int] | None = None) -> list[Video]:
    """`count` videos with 100 views each, except the indexes in `boosted` (index -> views)."""
    boosted = boosted or {}
    return [video(n, boosted.get(n, 100)) for n in range(count)]


def by_id(scores: list[OutlierScore]) -> dict[str, OutlierScore]:
    return {s.video_id: s for s in scores}


# --- eligibility ----------------------------------------------------------------------------------
def test_constants_and_duration_reading() -> None:
    assert (OUTLIER_THRESHOLD, BELOW_THRESHOLD, PEERS, MIN_AGE_DAYS) == (10.0, 0.1, 20, 28)
    assert (MAX_EXPLAINED, MAX_TYPICAL) == (8, 8)
    assert duration_seconds("PT5M30S") == 330
    assert duration_seconds("PT1H2M3S") == 3723
    assert duration_seconds("P1DT1S") == 86401
    assert duration_seconds("P0D") == 0 and duration_seconds("PT0S") == 0
    assert duration_seconds("") == 0 and duration_seconds("garbage") == 0


def test_ineligible_videos_get_the_status_that_left_them_out() -> None:
    videos = catalog(12)
    videos.append(video(100, privacy="private"))
    videos.append(video(101, duration="P0D"))
    videos.append(video(102, duration=""))
    videos.append(video(103, day=(AS_OF - START).days - 10))  # 10 days old
    videos.append(video(104, privacy="unlisted", duration="P0D"))  # not_public is checked first
    videos.append(video(105, duration="P0D", day=(AS_OF - START).days - 1))  # no_duration before too_new
    videos.append(video(106, published_at="not a date"))
    scores = by_id(score_videos(videos, AS_OF))
    assert scores["vid000100"].status == "not_public"
    assert scores["vid000101"].status == "no_duration"
    assert scores["vid000102"].status == "no_duration"
    assert scores["vid000103"].status == "too_new" and scores["vid000103"].age_days == 10
    assert scores["vid000104"].status == "not_public"
    assert scores["vid000105"].status == "no_duration"
    assert scores["vid000106"].status == "too_new"
    for key in ("vid000100", "vid000101", "vid000103", "vid000106"):
        s = scores[key]
        assert s.peer_median is None and s.ratio is None and s.peer_ids == ()
    assert scores["vid000003"].status == "scored"


def test_age_threshold_is_in_whole_days_from_as_of() -> None:
    old = video(0)
    edge = video(1, day=(AS_OF - START).days - MIN_AGE_DAYS)  # published 28 days (and a bit) before
    young = video(2, day=(AS_OF - START).days - MIN_AGE_DAYS + 1)
    scores = by_id(score_videos([old, edge, young, *catalog(12)[3:]], AS_OF))
    assert scores["vid000001"].age_days == MIN_AGE_DAYS and scores["vid000001"].status != "too_new"
    assert scores["vid000002"].age_days == MIN_AGE_DAYS - 1 and scores["vid000002"].status == "too_new"
    # a later as_of ages the same video; the clock is not read
    later = by_id(score_videos([young, *catalog(12)[3:]], AS_OF + timedelta(days=2)))
    assert later["vid000002"].status != "too_new"


def test_naive_as_of_is_read_as_utc() -> None:
    a = score_videos(catalog(12), AS_OF.replace(tzinfo=None))
    b = score_videos(catalog(12), AS_OF)
    assert a == b


# --- peers ----------------------------------------------------------------------------------------
def test_peers_are_the_nearest_by_publish_time_with_ties_by_video_id() -> None:
    scores = by_id(score_videos(catalog(30), AS_OF, peers=3))
    # video 10: 9 and 11 are one day away (tie, by id: 9 first), then 8 and 12 are two days away (8 first)
    assert scores["vid000010"].peer_ids == ("vid000009", "vid000011", "vid000008")
    assert all("vid000010" not in s.peer_ids or s.video_id != "vid000010" for s in scores.values())


def test_peers_at_the_edges_come_from_one_side() -> None:
    scores = by_id(score_videos(catalog(30), AS_OF, peers=4))
    assert scores["vid000000"].peer_ids == ("vid000001", "vid000002", "vid000003", "vid000004")
    assert scores["vid000029"].peer_ids == ("vid000028", "vid000027", "vid000026", "vid000025")


def test_peer_choice_ignores_ineligible_videos_and_input_order() -> None:
    videos = catalog(30)
    videos.append(video(200, 999_999, privacy="private", day=10))  # same day as video 10, not a peer
    forward = by_id(score_videos(videos, AS_OF, peers=5))
    backward = by_id(score_videos(list(reversed(videos)), AS_OF, peers=5))
    assert forward == backward
    assert not any("vid000200" in s.peer_ids for s in forward.values())


def test_equal_publish_times_are_broken_by_video_id() -> None:
    videos = [video(n, day=0) for n in range(12)]
    scores = by_id(score_videos(videos, AS_OF, peers=3))
    assert scores["vid000005"].peer_ids == ("vid000000", "vid000001", "vid000002")
    assert scores["vid000000"].peer_ids == ("vid000001", "vid000002", "vid000003")


def test_default_peer_count_is_twenty() -> None:
    scores = by_id(score_videos(catalog(40), AS_OF))
    assert len(scores["vid000020"].peer_ids) == 20
    assert len(by_id(score_videos(catalog(15), AS_OF))["vid000007"].peer_ids) == 14  # fewer exist


def test_few_peers_below_eight_other_eligible_videos() -> None:
    eight = by_id(score_videos(catalog(9), AS_OF))  # each has 8 others
    assert {s.status for s in eight.values()} == {"scored"}
    seven = by_id(score_videos(catalog(8), AS_OF))  # each has 7 others
    assert {s.status for s in seven.values()} == {"few_peers"}
    s = seven["vid000003"]
    assert s.peer_median is None and s.ratio is None and s.peer_ids == ()


# --- median and ratio -----------------------------------------------------------------------------
def test_median_and_ratio() -> None:
    views = [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]
    videos = [video(n, v) for n, v in enumerate(views)]
    scores = by_id(score_videos(videos, AS_OF))
    nine = scores["vid000009"]  # peers: the other nine, median of 10..90 = 50
    assert nine.peer_median == 50.0 and nine.ratio == 20.0 and nine.status == "scored"
    zero = scores["vid000000"]  # peers 20..90 and 1000: median of 9 values = 60
    assert zero.peer_median == 60.0 and zero.ratio == pytest.approx(10 / 60)
    assert isinstance(zero.ratio, float) and zero.views == 10


def test_median_of_an_even_number_of_peers_is_the_midpoint() -> None:
    videos = [video(n, v) for n, v in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])]
    row = by_id(score_videos(videos, AS_OF, peers=10))["vid000010"]  # peers 1..10 -> 5.5
    assert row.peer_median == 5.5


def test_zero_median_leaves_the_ratio_empty_but_the_row_scored() -> None:
    videos = [video(n, 0) for n in range(11)]
    videos.append(video(11, 500))
    row = by_id(score_videos(videos, AS_OF))["vid000011"]
    assert row.status == "scored" and row.peer_median == 0.0 and row.ratio is None
    assert outliers(score_videos(videos, AS_OF)) == []  # no ratio, no outlier


# --- helpers --------------------------------------------------------------------------------------
def row(video_id: str, ratio: float | None, status: str = "scored") -> OutlierScore:
    return OutlierScore(video_id, 1, "2025-01-01T00:00:00Z", 100, (), 1.0 if ratio is not None else None,
                        ratio, status)


def test_outliers_helper_threshold_and_order() -> None:
    scores = [row("a", 10.0), row("b", 9.99), row("c", 50.0), row("d", 12.0), row("e", None),
              row("f", 99.0, "too_new"), row("g", 50.0)]
    assert [s.video_id for s in outliers(scores)] == ["c", "g", "d", "a"]  # ties by video id
    assert [s.video_id for s in outliers(scores, threshold=12.0)] == ["c", "g", "d"]


def test_below_norm_helper_threshold_and_order() -> None:
    scores = [row("a", 0.1), row("b", 0.11), row("c", 0.02), row("d", 0.0), row("e", None),
              row("f", 0.01, "few_peers")]
    assert [s.video_id for s in below_norm(scores)] == ["d", "c", "a"]
    assert [s.video_id for s in below_norm(scores, threshold=0.5)] == ["d", "c", "a", "b"]


def test_typical_helper_band_and_order() -> None:
    scores = [row("a", 0.7), row("b", 1.4), row("c", 1.01), row("d", 0.95), row("e", 0.69),
              row("f", 1.41), row("g", None), row("h", 1.0, "too_new")]
    assert [s.video_id for s in typical(scores)] == ["c", "d", "a", "b"]  # 0.01, 0.05, 0.3, 0.4 away
    assert [s.video_id for s in typical(scores, low=0.9, high=1.1)] == ["c", "d"]


def test_ratio_label() -> None:
    assert ratio_label(4.24) == "4.2×" and ratio_label(0.31) == "0.3×" and ratio_label(0.0) == "0.0×"
    assert ratio_label(9.94) == "9.9×"
    assert ratio_label(10.0) == "10×" and ratio_label(12.4) == "12×" and ratio_label(12.5) == "12×"
    assert ratio_label(9.96) == "10×"  # one decimal would read 10.0, so it is a whole number
    assert ratio_label(1191.4) == "1,191×" and ratio_label(25000) == "25,000×"
    assert ratio_label(None) == "—"
    assert "×" == "×"


# --- explain: the store and the model -------------------------------------------------------------
def make_store(tmp_path: Path, videos: list[Video]) -> Store:
    store = Store(tmp_path / "c.sqlite")
    pull = store.begin_pull("chan0000001", now=NOW)
    for v in videos:
        store.upsert_video(v, pull, now=NOW)
    store.finish_pull(pull, len(videos), now=NOW)
    return store


@pytest.fixture
def store(tmp_path: Path):
    # three outliers (indexes 5, 15, 25 at 5,000 views against a norm of 100), the rest typical
    s = make_store(tmp_path, catalog(30, {5: 5000, 15: 6000, 25: 7000}))
    yield s
    s.close()


def reply(data: Any, tokens: tuple[int, int, int] = (2000, 100, 300)) -> ModelReply:
    text = data if isinstance(data, str) else json.dumps(data)
    return ModelReply(text, (), {}, *tokens)


def good_answer(**override: Any) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "patterns": [
            {"text": "The titles ask a short question.", "evidence": ["O1", "O2"], "kind": "title"},
            {"text": "Videos run long.", "evidence": ["O1", "O2", "O3"], "kind": "length"}],
        "per_video": [{"label": "O1", "why": "A clear question in the title."},
                      {"label": "O2", "why": "Close-up of one plant."}],
        "ideas": ["Ask a question in the next title."],
    }
    answer.update(override)
    return answer


def png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (32, 18), (10, 120, 30)).save(out, "PNG")
    return out.getvalue()


def user_content(call: dict[str, Any]) -> Any:
    return call["messages"][1]["content"]


def sent_text(call: dict[str, Any]) -> str:
    content = user_content(call)
    if isinstance(content, str):
        return content
    return "\n".join(p["text"] for p in content if p["type"] == "text")


def test_fewer_than_two_outliers_makes_no_model_call(tmp_path: Path) -> None:
    with make_store(tmp_path, catalog(30, {5: 5000})) as store:
        client = FakeModelClient([])
        report = explain_outliers(store, client, as_of=AS_OF, now=NOW)
        assert client.calls == []
        assert report.status == "too_few" and report.insight == {} and report.model == ""
        assert report.cost_usd is None and report.input_tokens == 0
        assert report.labels["O1"]["video_id"] == "vid000005"
        assert store.latest_outlier_report() == report and report.id > 0


def test_a_catalog_without_scores_is_too_few(tmp_path: Path) -> None:
    with Store(tmp_path / "c.sqlite") as store:
        client = FakeModelClient([])
        report = explain_outliers(store, client, as_of=AS_OF, now=NOW)
        assert report.status == "too_few" and report.pull_id is None and client.calls == []


def test_explain_stores_the_guarded_report(store: Store) -> None:
    client = FakeModelClient([reply(good_answer())])
    report = explain_outliers(store, client, as_of=AS_OF, now=NOW)
    assert report.status == "done" and report.error == "" and report.problems == []
    assert [p["text"] for p in report.insight["patterns"]] == [
        "The titles ask a short question.", "Videos run long."]
    assert report.insight["patterns"][1]["evidence"] == ["O1", "O2", "O3"]
    assert [p["label"] for p in report.insight["per_video"]] == ["O1", "O2"]
    assert report.insight["ideas"] == ["Ask a question in the next title."]
    assert report.model == "muse-spark-1.3" and report.created_at == NOW.isoformat()
    assert report.as_of == AS_OF.isoformat() and report.pull_id == store.latest_pull()["id"]
    assert (report.input_tokens, report.cached_tokens, report.output_tokens) == (2000, 100, 300)
    assert report.cost_usd == pytest.approx(estimate_cost("muse-spark-1.3", 2000, 100, 300), abs=1e-6)
    assert report.params["peers"] == 20 and report.params["outlier_threshold"] == 10.0
    # labels carry the numbers from code, highest ratio first
    assert [report.labels[k]["video_id"] for k in ("O1", "O2", "O3")] == [
        "vid000025", "vid000015", "vid000005"]
    assert report.labels["O1"]["views"] == 7000 and report.labels["O1"]["peer_median"] == 100.0
    assert report.labels["O1"]["ratio"] == 70.0
    assert report.labels["T1"]["ratio"] == 1.0
    assert sum(1 for k in report.labels if k.startswith("T")) == MAX_TYPICAL
    assert store.latest_outlier_report() == report


def test_the_model_sees_labels_and_no_video_ids(store: Store) -> None:
    client = FakeModelClient([reply(good_answer())])
    explain_outliers(store, client, as_of=AS_OF, now=NOW)
    (call,) = client.calls
    wire = json.dumps(call["messages"])
    assert "vid0000" not in wire
    text = sent_text(call)
    assert "O1 |" in text and "O3 |" in text and "T1 |" in text and "T8 |" in text and "T9 |" not in text
    assert "views: 7,000" in text and "vs the channel norm for its age: 70×" in text
    assert "vs the channel norm for its age: 1.0×" in text
    assert "length: 5:30" in text and "published: 2025" in text
    assert "tags: cactus, garden" in text and "How to repot a cactus." in text
    assert call["tools"] is None
    fmt = call["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert set(fmt["json_schema"]["schema"]["properties"]) == {"patterns", "per_video", "ideas"}
    assert user_content(call) == text  # text only without `fetch`


def test_tags_and_description_are_trimmed(tmp_path: Path) -> None:
    videos = catalog(30, {5: 5000, 15: 6000})
    videos[5] = video(5, 5000, tags=[f"tag{chr(97 + i)}" for i in range(14)],
                      description="word " * 200)
    with make_store(tmp_path, videos) as store:
        client = FakeModelClient([reply(good_answer(patterns=[], per_video=[], ideas=[]))])
        explain_outliers(store, client, as_of=AS_OF, now=NOW)
    line = next(l for l in sent_text(client.calls[0]).splitlines() if "views: 5,000" in l)
    assert "taga" in line and "tagj" in line and "tagk" not in line  # ten tags
    snippet = line.split("description (start): ")[1]
    assert len(snippet) == 300


# --- explain: the guards --------------------------------------------------------------------------
def explain_with(store: Store, answer: Any) -> Any:
    return explain_with_client(store, FakeModelClient([reply(answer)]))


def explain_with_client(store: Store, client: FakeModelClient) -> Any:
    return explain_outliers(store, client, as_of=AS_OF, now=NOW)


def test_guard_removes_unknown_and_typical_evidence(store: Store) -> None:
    answer = good_answer(patterns=[
        {"text": "Unknown labels are removed.", "evidence": ["O1", "O9", "O2", "O1"], "kind": "topic"},
        {"text": "A typical video is no evidence.", "evidence": ["O1", "T1", "T2"], "kind": "title"},
        {"text": "Only unknown.", "evidence": ["X1", "O7"], "kind": "other"},
        {"text": "No evidence at all.", "evidence": [], "kind": "other"}])
    report = explain_with(store, answer)
    assert [p["text"] for p in report.insight["patterns"]] == ["Unknown labels are removed."]
    assert report.insight["patterns"][0]["evidence"] == ["O1", "O2"]
    assert len(report.problems) == 3 and all("fewer than 2" in p for p in report.problems)


def test_guard_drops_a_pattern_with_one_outlier(store: Store) -> None:
    report = explain_with(store, good_answer(patterns=[
        {"text": "Only one video shows this.", "evidence": ["O3"], "kind": "title"}]))
    assert report.insight["patterns"] == [] and report.status == "done"
    assert any("fewer than 2" in p for p in report.problems)
    assert "The model gave no usable patterns." in report.problems


def test_guard_fixes_an_unknown_kind(store: Store) -> None:
    report = explain_with(store, good_answer(patterns=[
        {"text": "Odd kind.", "evidence": ["O1", "O2"], "kind": "vibes"}]))
    assert report.insight["patterns"][0]["kind"] == "other"


def test_guard_drops_invented_numbers_everywhere(store: Store) -> None:
    answer = {
        "patterns": [
            {"text": "Titles under 40 characters win.", "evidence": ["O1", "O2"], "kind": "title"},
            {"text": "Several run 5:30 long.", "evidence": ["O1", "O2"], "kind": "length"}],
        "per_video": [{"label": "O1", "why": "It got 999 comments."},
                      {"label": "O2", "why": "Reached 5,000 views."}],
        "ideas": ["Post every 3 days.", "Try the 2025 style again."]}
    report = explain_with(store, answer)
    assert [p["text"] for p in report.insight["patterns"]] == ["Several run 5:30 long."]
    assert report.insight["per_video"] == [{"label": "O2", "why": "Reached 5,000 views."}]
    assert report.insight["ideas"] == ["Try the 2025 style again."]
    assert sum("which was not given" in p for p in report.problems) == 3
    assert any('"40"' in p for p in report.problems)


def test_guard_accepts_a_given_number_written_either_way(store: Store) -> None:
    answer = good_answer(
        per_video=[{"label": "O1", "why": "It has 7000 views."}, {"label": "O2", "why": "6,000 views."},
                   {"label": "O3", "why": "About 70× the norm, and 5 000 views."}],
        ideas=["Repeat the 70× idea.", "The 1.0× group is flat."])
    report = explain_with(store, answer)
    assert len(report.insight["per_video"]) == 3 and len(report.insight["ideas"]) == 2
    assert report.problems == []


def test_label_digits_are_not_quotable_numbers_but_labels_may_be_named(store: Store) -> None:
    report = explain_with(store, good_answer(
        per_video=[{"label": "O1", "why": "Like O2 and O3, it asks a question."}],
        ideas=["Do it again after 3 weeks."]))
    assert report.insight["per_video"][0]["why"] == "Like O2 and O3, it asks a question."
    assert report.insight["ideas"] == []  # "3" only appears in the label O3


def test_numbers_of_the_input_are_allowed_even_from_descriptions(tmp_path: Path) -> None:
    videos = catalog(30, {5: 5000, 15: 6000})
    videos[5] = video(5, 5000, description="Model 4821 review.")
    with make_store(tmp_path, videos) as store:
        report = explain_with(store, good_answer(ideas=["Name the model 4821.", "Name the model 4822."]))
    assert report.insight["ideas"] == ["Name the model 4821."]


def test_guard_drops_unknown_and_duplicate_per_video_labels(store: Store) -> None:
    answer = good_answer(per_video=[
        {"label": "O1", "why": "First."}, {"label": "T1", "why": "A typical one."},
        {"label": "O1", "why": "Second."}, {"label": "O9", "why": "No such video."},
        {"label": "vid000025", "why": "An id."}, {"label": "O3", "why": "Third."}])
    report = explain_with(store, answer)
    assert report.insight["per_video"] == [{"label": "O1", "why": "First."},
                                           {"label": "O3", "why": "Third."}]
    assert len(report.problems) == 4


def test_lines_are_cleaned_and_limited(store: Store) -> None:
    long = "word " * 80
    answer = good_answer(
        per_video=[{"label": "O1", "why": "Uses <b>bold</b>   text"}, {"label": "O2", "why": long}],
        ideas=["a", "b", "c", "d", "e"])
    report = explain_with(store, answer)
    first, second = report.insight["per_video"]
    assert "<" not in first["why"] and ">" not in first["why"] and "  " not in first["why"]
    assert len(second["why"]) <= 240
    assert len(report.insight["ideas"]) == 3


def test_at_most_four_patterns(store: Store) -> None:
    patterns = [{"text": f"Pattern {chr(97 + i)}.", "evidence": ["O1", "O2"], "kind": "topic"}
                for i in range(6)]
    report = explain_with(store, good_answer(patterns=patterns))
    assert len(report.insight["patterns"]) == 4


# --- explain: failures ----------------------------------------------------------------------------
@pytest.mark.parametrize("content", [
    "not json at all", "", "[1, 2]", json.dumps({"patterns": []}),
    json.dumps({"patterns": [], "per_video": "x", "ideas": []})])
def test_bad_json_or_missing_keys_give_an_error_report(store: Store, content: str) -> None:
    report = explain_with_client(store, FakeModelClient([reply(content)]))
    assert report.status == "error" and report.error and report.insight == {}
    assert store.latest_outlier_report() == report
    assert report.input_tokens == 2000  # the call was made and is accounted for


def test_json_inside_a_code_fence_is_read(store: Store) -> None:
    fenced = "```json\n" + json.dumps(good_answer()) + "\n```"
    assert explain_with(store, fenced).status == "done"


def test_a_model_failure_gives_an_error_report(store: Store) -> None:
    report = explain_with_client(store, FakeModelClient([ModelError("rate_limit", "Slow down.")]))
    assert report.status == "error" and report.error == "Slow down."
    assert report.cost_usd is None


def test_a_refused_key_is_raised_and_nothing_is_stored(store: Store) -> None:
    client = FakeModelClient([ModelError("auth", "The Meta API key was refused. Check it in Configuration.")])
    with pytest.raises(ModelError) as caught:
        explain_with_client(store, client)
    assert caught.value.kind == "auth"
    assert store.latest_outlier_report() is None


# --- explain: thumbnails --------------------------------------------------------------------------
def test_thumbnails_are_attached_with_a_label_before_each(store: Store) -> None:
    fetched: list[str] = []

    def fetch(url: str) -> bytes:
        fetched.append(url)
        return png()

    client = FakeModelClient([reply(good_answer())])
    report = explain_outliers(store, client, as_of=AS_OF, fetch=fetch, now=NOW)
    assert report.status == "done" and report.problems == []
    content = user_content(client.calls[0])
    assert content[0]["type"] == "text" and "O1 |" in content[0]["text"]
    images = [i for i, p in enumerate(content) if p["type"] == "image_url"]
    assert len(images) == len(report.labels) == 11
    for i in images:
        assert content[i]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[i - 1] == {"type": "text", "text": content[i - 1]["text"]}
    assert content[images[0] - 1]["text"] == "Thumbnail of O1:"
    assert content[images[-1] - 1]["text"] == "Thumbnail of T8:"
    assert fetched[0] == "https://i.ytimg.com/vi/vid000025/mqdefault.jpg"  # O1 is the top outlier
    assert "vid0000" not in "".join(p["text"] for p in content if p["type"] == "text")


def test_one_failing_fetch_skips_that_picture_only(store: Store) -> None:
    def fetch(url: str) -> bytes:
        if "vid000015" in url:
            raise PackagingError("The thumbnail download was empty.")
        if "vid000005" in url:
            raise RuntimeError("secret-looking detail")
        return png()

    client = FakeModelClient([reply(good_answer())])
    report = explain_outliers(store, client, as_of=AS_OF, fetch=fetch, now=NOW)
    assert report.status == "done"
    content = user_content(client.calls[0])
    captions = [p["text"] for p in content if p["type"] == "text" and p["text"].startswith("Thumbnail of")]
    assert "Thumbnail of O1:" in captions and "Thumbnail of O2:" not in captions
    assert "Thumbnail of O3:" not in captions
    assert sum(p["type"] == "image_url" for p in content) == 9
    assert len(report.problems) == 2
    assert any("(O2)" in p and "download was empty" in p for p in report.problems)
    assert any("(O3)" in p and "secret" not in p for p in report.problems)
    assert "vid0000" not in json.dumps(report.problems)


def test_thumbnails_use_the_medium_size_then_the_best_other_and_only_ytimg(tmp_path: Path) -> None:
    videos = catalog(30, {5: 5000, 15: 6000, 25: 7000})
    videos[25] = video(25, 7000, thumbnails={"high": {"url": "https://i.ytimg.com/vi/x/hq.jpg"},
                                              "default": {"url": "https://i.ytimg.com/vi/x/d.jpg"}})
    videos[15] = video(15, 6000, thumbnails={"medium": {"url": "https://evil.example/m.jpg"}})
    videos[5] = video(5, 5000, thumbnails={})
    with make_store(tmp_path, videos) as store:
        fetched: list[str] = []

        def fetch(url: str) -> bytes:
            fetched.append(url)
            return png()

        client = FakeModelClient([reply(good_answer())])
        report = explain_outliers(store, client, as_of=AS_OF, fetch=fetch, now=NOW)
    assert fetched[0] == "https://i.ytimg.com/vi/x/hq.jpg"  # no medium: packaging's best size
    assert not any("evil" in u for u in fetched)
    assert len(report.problems) == 2 and any("only downloads thumbnails" in p for p in report.problems)
    assert any("no thumbnail" in p for p in report.problems)


def test_a_picture_that_is_not_an_image_is_skipped(store: Store) -> None:
    client = FakeModelClient([reply(good_answer())])
    report = explain_outliers(store, client, as_of=AS_OF, fetch=lambda url: b"not an image", now=NOW)
    assert len(report.problems) == 11
    assert user_content(client.calls[0]) == sent_text(client.calls[0])  # nothing attached: plain text
