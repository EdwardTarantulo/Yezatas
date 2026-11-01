import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fakes import FakeModelClient
from yezatas.agent import ModelError, ModelReply
from yezatas.searchgaps import (
    CHANNEL_TERMS,
    TERMS_PER_VIDEO,
    SearchGap,
    SearchTerms,
    date_range,
    fetch_terms,
    find_gaps,
    load_terms,
    missing_words,
    pick_videos,
    rank_gaps,
    rows,
    save_ranking,
    save_terms,
    latest_ranking,
    words,
)
from yezatas.store import Video

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


class FakeAnalyticsReader:
    def __init__(self, responses: dict[str | None, list[tuple[str, int]]]):
        self.responses = responses
        self.calls: list[tuple[str, str, str | None, int]] = []

    def search_terms(self, start_date, end_date, video_id, max_results):
        self.calls.append((start_date, end_date, video_id, max_results))
        lst = self.responses.get(video_id)
        if lst is None:
            return {}
        return {
            "columnHeaders": [
                {"name": "insightTrafficSourceDetail"},
                {"name": "views"},
            ],
            "rows": [[t, v] for t, v in lst],
        }


def _vid(video_id, title, views, privacy="public"):
    return Video(
        video_id=video_id,
        title=title,
        privacy_status=privacy,
        view_count=views,
    )


def test_rows_order_and_missing():
    # normal order
    r = rows(
        {
            "columnHeaders": [
                {"name": "insightTrafficSourceDetail"},
                {"name": "views"},
            ],
            "rows": [["hello world", 10], ["hello   world", 5]],
        }
    )
    assert r == (("hello world", 10),)
    # reversed columns
    r2 = rows(
        {
            "columnHeaders": [
                {"name": "views"},
                {"name": "insightTrafficSourceDetail"},
            ],
            "rows": [[20, "repot cactus winter"]],
        }
    )
    assert r2 == (("repot cactus winter", 20),)
    # missing column
    r3 = rows(
        {
            "columnHeaders": [{"name": "views"}],
            "rows": [[10, "x"]],
        }
    )
    assert r3 == ()


def test_date_range():
    assert date_range(NOW) == ("2026-03-03", "2026-05-31")


def test_pick_videos():
    v1 = _vid("a", "t", 10, "public")
    v2 = _vid("b", "t", 20, "private")
    v3 = _vid("c", "t", 10, "public")
    v4 = _vid("d", "t", 30, "public")
    picked = pick_videos([v1, v2, v3, v4], max_videos=2)
    assert [v.video_id for v in picked] == ["d", "a"]


def test_fetch_terms_calls():
    vids = [
        _vid("v1", "How to Repot a Cactus", 100),
        _vid("v2", "Cactus care", 200),
        _vid("v3", "Watering cactus", 150),
    ]
    reader = FakeAnalyticsReader(
        {
            None: [("repot cactus winter", 50)],
            "v2": [("cactus winter", 10)],
            "v3": [("water cactus", 5)],
            "v1": [("repot cactus", 7)],
        }
    )
    terms = fetch_terms(reader, vids, now=NOW, max_videos=2)
    assert reader.calls[0][2] is None and reader.calls[0][3] == 25
    assert len(reader.calls) == 3
    assert terms.calls == 3
    assert set(terms.videos.keys()) == {"v2", "v3"}


def test_save_load_roundtrip(tmp_path):
    home = tmp_path / "home"
    terms = SearchTerms(
        start_date="2026-03-03",
        end_date="2026-05-31",
        fetched_at="2026-06-01T12:00:00+00:00",
        channel=(("repot cactus winter", 50),),
        videos={"v1": (("cactus winter", 10),)},
        calls=2,
    )
    save_terms(home, terms)
    loaded = load_terms(home)
    assert loaded == terms
    (home / "search" / "terms.json").write_text("bad", encoding="utf-8")
    assert load_terms(home) is None


def test_missing_words_basic():
    assert missing_words(
        "repot cactus winter", "How to Repot a Cactus"
    ) == ("winter",)
    # plural matches
    assert missing_words("cactuses", "cactus") == ()
    # stop words and 1-char
    assert missing_words("a cactus x", "other") == ("cactus",)


def test_find_gaps_covered_and_channel():
    v1 = _vid("v1", "How to Repot a Cactus", 100)
    v2 = _vid("v2", "Cactus winter care tips", 80)
    terms = SearchTerms(
        start_date="2026-03-03",
        end_date="2026-05-31",
        fetched_at="2026-06-01T12:00:00+00:00",
        channel=(("repot cactus winter", 50), ("cactus winter", 40)),
        videos={
            "v1": (("repot cactus", 10), ("repot cactus winter", 20)),
            "v2": (("repot cactus winter", 5),),
        },
        calls=3,
    )
    gaps = find_gaps(terms, [v1, v2])
    # covered term no gap, channel term covered by one title is no gap
    # cactus winter is covered by v2, so no channel gap for it
    terms_set = [(g.term, g.video_id) for g in gaps]
    # repot cactus winter from v1 should be gap (missing winter)
    assert ("repot cactus winter", "v1") in terms_set


def test_find_gaps_sort_order():
    v1 = _vid("v1", "title one", 10)
    terms = SearchTerms(
        start_date="2026-03-03",
        end_date="2026-05-31",
        fetched_at="2026-06-01T12:00:00+00:00",
        channel=(),
        videos={"v1": (("b term", 5), ("a term", 5))},
        calls=1,
    )
    gaps = find_gaps(terms, [v1])
    assert gaps[0].term == "a term"
    assert gaps[1].term == "b term"


def test_rank_gaps_pick_handling(tmp_path):
    terms = SearchTerms(
        start_date="2026-03-03",
        end_date="2026-05-31",
        fetched_at="2026-06-01T12:00:00+00:00",
        channel=(),
        videos={},
        calls=1,
    )
    gap1 = SearchGap(
        video_id="v1",
        title="How to Repot a Cactus",
        term="repot cactus winter",
        views=1200,
        missing=("winter",),
    )
    gap2 = SearchGap(
        video_id="v2",
        title="Cactus care",
        term="water cactus",
        views=800,
        missing=("water",),
    )
    data = {
        "picks": [
            {"label": "G1", "priority": "high", "reason": "good reason"},
            {"label": "G99", "priority": "high", "reason": "unknown"},
            {"label": "G1", "priority": "high", "reason": "dup"},
            {
                "label": "G2",
                "priority": "bad",
                "reason": "needs water 9999",
            },
            {"label": "G2", "priority": "low", "reason": "ok low"},
        ]
    }
    client = FakeModelClient(
        [ModelReply(json.dumps(data), (), {}, 2000, 100, 300)]
    )
    ranking = rank_gaps(terms, [gap1, gap2], client, now=NOW)
    # unknown, duplicate, invented number dropped
    assert len(ranking.picks) == 2
    assert ranking.picks[0].label == "G1"
    assert ranking.picks[1].priority == "low"
    assert any("unknown gap" in p for p in ranking.problems)
    assert any("second pick" in p.lower() for p in ranking.problems)
    assert any("9999" in p for p in ranking.problems)
    # no video ids in user message
    user_content = ""
    for c in client.calls:
        for m in c["messages"]:
            user_content += m.get("content", "")
    assert "G1" in user_content
    assert "repot cactus winter" in user_content
    assert "v1" not in user_content
    assert ranking.cost_usd is not None and ranking.cost_usd > 0
    # save/load
    home = tmp_path / "home"
    save_ranking(home, ranking)
    loaded = latest_ranking(home)
    assert loaded == ranking


def test_rank_gaps_too_few():
    terms = SearchTerms(
        start_date="2026-03-03",
        end_date="2026-05-31",
        fetched_at="2026-06-01T12:00:00+00:00",
        channel=(),
        videos={},
        calls=1,
    )
    client = FakeModelClient([])
    ranking = rank_gaps(terms, [], client, now=NOW)
    assert ranking.status == "too_few"
    assert client.calls == []


def test_rank_gaps_auth_error():
    terms = SearchTerms(
        start_date="2026-03-03",
        end_date="2026-05-31",
        fetched_at="2026-06-01T12:00:00+00:00",
        channel=(),
        videos={},
        calls=1,
    )
    gap = SearchGap(
        video_id="v1", title="t", term="repot cactus winter", views=10, missing=("winter",)
    )
    client = FakeModelClient([ModelError("auth", "The key was refused.")])
    with pytest.raises(ModelError):
        rank_gaps(terms, [gap], client, now=NOW)


def test_rank_gaps_rate_limit_error():
    terms = SearchTerms(
        start_date="2026-03-03",
        end_date="2026-05-31",
        fetched_at="2026-06-01T12:00:00+00:00",
        channel=(),
        videos={},
        calls=1,
    )
    gap = SearchGap(
        video_id="v1", title="t", term="x y", views=10, missing=("y",)
    )
    client = FakeModelClient([ModelError("rate_limit", "Slow down.")])
    ranking = rank_gaps(terms, [gap], client, now=NOW)
    assert ranking.status == "error"


def test_rank_gaps_not_json_error():
    terms = SearchTerms(
        start_date="2026-03-03",
        end_date="2026-05-31",
        fetched_at="2026-06-01T12:00:00+00:00",
        channel=(),
        videos={},
        calls=1,
    )
    gap = SearchGap(
        video_id="v1", title="t", term="x y", views=10, missing=("y",)
    )
    client = FakeModelClient([ModelReply("not json", (), {}, 10, 0, 0)])
    ranking = rank_gaps(terms, [gap], client, now=NOW)
    assert ranking.status == "error"
