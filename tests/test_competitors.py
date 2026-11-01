"""Competitor outliers tests: parsing, reading, saving and explanation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient, make_item
from yezatas.agent import ModelError, ModelReply
from yezatas.competitors import (
    CompetitorChannel,
    CompetitorError,
    ChannelRef,
    channel_outliers,
    channel_scores,
    estimate_units,
    explain,
    fetch_channel,
    load_channels,
    parse_channel,
    remove_channel,
    save_channel,
    save_insight,
)
from yezatas.competitors import channel_count, latest_insight
from yezatas.quota import QuotaExceeded, QuotaGuard
from yezatas.store import Store, Video
from yezatas.youtube import MAX_RESULTS

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


class FakeChannelReader:
    def __init__(
        self,
        channel_items: list[dict[str, Any]],
        ids: list[str],
        views: dict[str, int],
        page_size: int = 50,
    ) -> None:
        self.channel_items = channel_items
        self.ids = ids
        self.views = views
        self.page_size = page_size
        self.calls: list[tuple[str, Any]] = []

    def channel(self, ref: ChannelRef) -> dict[str, Any]:
        self.calls.append(("channel", ref))
        if not self.channel_items:
            return {"items": []}
        return {"items": self.channel_items}

    def playlist_items(
        self, playlist_id: str, page_token: str | None
    ) -> dict[str, Any]:
        self.calls.append(("playlist_items", (playlist_id, page_token)))
        start = int(page_token) if page_token else 0
        end = min(start + self.page_size, len(self.ids))
        items = [
            {"contentDetails": {"videoId": vid}}
            for vid in self.ids[start:end]
        ]
        resp: dict[str, Any] = {"items": items}
        if end < len(self.ids):
            resp["nextPageToken"] = str(end)
        return resp

    def videos(self, ids: list[str], part: str) -> dict[str, Any]:
        self.calls.append(("videos", (list(ids), part)))
        items: list[dict[str, Any]] = []
        base = datetime(2026, 5, 1, tzinfo=timezone.utc)
        for vid in ids:
            try:
                n = int(vid[2:])
            except Exception:
                n = 0
            # find index in self.ids to get date order
            try:
                idx = self.ids.index(vid)
            except ValueError:
                idx = 0
            pub = (base - timedelta(weeks=idx)).isoformat().replace(
                "+00:00", "Z"
            )
            views = self.views.get(vid, 100)
            title = f"Video {vid}"
            item = {
                "id": vid,
                "snippet": {
                    "channelId": "UC" + "a" * 22,
                    "title": title,
                    "tags": ["t"],
                    "publishedAt": pub,
                    "thumbnails": {},
                },
                "contentDetails": {"duration": "PT8M"},
                "statistics": {
                    "viewCount": str(views),
                    "likeCount": "0",
                    "commentCount": "0",
                },
                "status": {"privacyStatus": "public"},
            }
            items.append(item)
        return {"items": items}


def _chan_item(cid: str, title: str, handle: str = "@fern") -> dict[str, Any]:
    return {
        "id": cid,
        "snippet": {"title": title, "customUrl": handle},
        "contentDetails": {"relatedPlaylists": {"uploads": "UU_uploads"}},
    }


def test_parse_channel_ok() -> None:
    assert parse_channel("@fern.corner") == ChannelRef(
        "handle", "@fern.corner"
    )
    assert parse_channel("youtube.com/@fern.corner/videos") == ChannelRef(
        "handle", "@fern.corner"
    )
    assert parse_channel(
        "https://m.youtube.com/@fern_corner?x=1"
    ) == ChannelRef("handle", "@fern_corner")
    cid = "UC" + "a" * 22
    assert parse_channel(
        "https://www.youtube.com/channel/" + cid
    ) == ChannelRef("id", cid)
    assert parse_channel(cid) == ChannelRef("id", cid)


def test_parse_channel_errors() -> None:
    for bad in [
        "",
        "fern",
        "https://www.youtube.com/watch?v=abc",
        "https://example.com/@fern",
        "youtube.com/c/fern",
        "@ab",
    ]:
        with pytest.raises(CompetitorError):
            parse_channel(bad)


def test_fetch_channel_paging(tmp_path: Path) -> None:
    cid = "UC" + "a" * 22
    ids = [f"fv{n:06d}" for n in range(120)]
    views = {vid: 100 for vid in ids}
    reader = FakeChannelReader([_chan_item(cid, "Sample Fern Corner")], ids, views, 50)
    store = Store(tmp_path / "c.sqlite")
    guard = QuotaGuard(store, 10_000, clock=lambda: NOW)
    try:
        ch = fetch_channel(reader, ChannelRef("id", cid), guard, now=NOW)
        assert ch.channel_id == cid
        assert len(ch.videos) == 120
        assert ch.videos[0].video_id == ids[0]
        assert ch.units == 7
        assert guard.used() == 7
        # check calls
        assert len([c for c in reader.calls if c[0] == "playlist_items"]) == 3
        assert len([c for c in reader.calls if c[0] == "videos"]) == 3
        assert ch.videos[0].description == ""
        assert ch.videos[0].view_count == 100
    finally:
        store.close()


def test_fetch_channel_max_videos(tmp_path: Path) -> None:
    cid = "UC" + "a" * 22
    ids = [f"fv{n:06d}" for n in range(120)]
    views = {vid: 100 for vid in ids}
    reader = FakeChannelReader([_chan_item(cid, "Sample Fern Corner")], ids, views, 50)
    store = Store(tmp_path / "c.sqlite")
    guard = QuotaGuard(store, 10_000, clock=lambda: NOW)
    try:
        ch = fetch_channel(
            reader, ChannelRef("id", cid), guard, max_videos=60, now=NOW
        )
        assert len(ch.videos) == 60
        assert len([c for c in reader.calls if c[0] == "videos"]) == 2
    finally:
        store.close()


def test_fetch_channel_no_channel(tmp_path: Path) -> None:
    cid = "UC" + "a" * 22
    reader = FakeChannelReader([], [], {}, 50)
    store = Store(tmp_path / "c.sqlite")
    guard = QuotaGuard(store, 10_000, clock=lambda: NOW)
    try:
        with pytest.raises(CompetitorError):
            fetch_channel(reader, ChannelRef("id", cid), guard, now=NOW)
    finally:
        store.close()


def test_fetch_channel_quota(tmp_path: Path) -> None:
    cid = "UC" + "a" * 22
    ids = [f"fv{n:06d}" for n in range(5)]
    reader = FakeChannelReader([_chan_item(cid, "Sample Fern Corner")], ids, {}, 50)
    store = Store(tmp_path / "c.sqlite")
    guard = QuotaGuard(store, 1, clock=lambda: NOW)
    try:
        with pytest.raises(QuotaExceeded):
            fetch_channel(reader, ChannelRef("id", cid), guard, now=NOW)
        assert not any(c[0] == "videos" for c in reader.calls)
    finally:
        store.close()


def test_estimate_units() -> None:
    assert estimate_units() == 9
    assert estimate_units(50) == 3


def test_save_load_remove(tmp_path: Path) -> None:
    # create 5 channels
    for i in range(5):
        cid = "UC" + chr(97 + i) * 22
        ch = CompetitorChannel(
            channel_id=cid,
            title=f"Chan {i}",
            handle="@h",
            fetched_at=NOW.isoformat(),
            videos=(),
            units=1,
        )
        save_channel(tmp_path, ch)
    assert channel_count(tmp_path) == 5
    # 6th should fail
    cid6 = "UC" + "z" * 22
    ch6 = CompetitorChannel(
        channel_id=cid6,
        title="Chan 6",
        handle="@h",
        fetched_at=NOW.isoformat(),
        videos=(),
        units=1,
    )
    with pytest.raises(CompetitorError):
        save_channel(tmp_path, ch6)
    # saving same replaces
    cid0 = "UC" + "a" * 22
    ch0b = CompetitorChannel(
        channel_id=cid0,
        title="Chan 0b",
        handle="@h",
        fetched_at=NOW.isoformat(),
        videos=(),
        units=2,
    )
    save_channel(tmp_path, ch0b)
    assert channel_count(tmp_path) == 5
    # bad id
    bad = CompetitorChannel(
        channel_id="../x",
        title="bad",
        handle="",
        fetched_at=NOW.isoformat(),
        videos=(),
        units=1,
    )
    with pytest.raises(CompetitorError):
        save_channel(tmp_path, bad)
    assert remove_channel(tmp_path, "../x") is False
    # malformed JSON skipped
    folder = tmp_path / "competitors" / "channels"
    (folder / ("UC" + "y" * 22 + ".json")).write_text("not json", encoding="utf-8")
    loaded = load_channels(tmp_path)
    # the malformed file is skipped
    assert len(loaded) == 5
    # remove
    assert remove_channel(tmp_path, cid0) is True
    assert channel_count(tmp_path) == 4


def _make_outlier_channel() -> CompetitorChannel:
    cid = "UC" + "a" * 22
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    vids: list[Video] = []
    for i in range(40):
        pub = (base + timedelta(days=i)).isoformat().replace("+00:00", "Z")
        views = 5000 if i in (5, 15, 25) else 100
        vids.append(
            Video(
                video_id=f"fv{i:06d}",
                channel_id=cid,
                title=f"Video {i}",
                published_at=pub,
                duration="PT8M", privacy_status="public",
                view_count=views,
            )
        )
    fetched = (base + timedelta(days=60)).isoformat()
    return CompetitorChannel(
        channel_id=cid,
        title="Sample Fern Corner",
        handle="@fern",
        fetched_at=fetched,
        videos=tuple(vids),
        units=1,
    )


def test_channel_outliers_ratio() -> None:
    ch = _make_outlier_channel()
    outs = channel_outliers(ch)
    assert len(outs) == 3
    # highest ratio first
    assert outs[0].score.ratio == 50
    assert outs[0].score.ratio == 50


def _make_explain_channels() -> list[CompetitorChannel]:
    chans: list[CompetitorChannel] = []
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for ci in range(2):
        cid = "UC" + chr(97 + ci) * 22
        vids: list[Video] = []
        for i in range(40):
            pub = (base + timedelta(days=i)).isoformat().replace("+00:00", "Z")
            views = 5000 if i in (5, 15, 25) else 100
            vids.append(
                Video(
                    video_id=f"f{ci}v{i:06d}",
                    channel_id=cid,
                    title=f"Title {ci}-{i}",
                    published_at=pub,
                    duration="PT8M", privacy_status="public",
                    view_count=views,
                )
            )
        fetched = (base + timedelta(days=60)).isoformat()
        chans.append(
            CompetitorChannel(
                channel_id=cid,
                title=f"Chan {ci}",
                handle="@h",
                fetched_at=fetched,
                videos=tuple(vids),
                units=1,
            )
        )
    return chans


def test_explain_filters_and_saves(tmp_path: Path) -> None:
    chans = _make_explain_channels()
    # need to know labels: each channel has 3 outliers, PER_CHANNEL=3, MAX_EXPLAINED=8
    # So 6 outliers O1..O6, typical T1..T4
    data = {
        "patterns": [
            {"text": "good pattern", "evidence": ["O1", "O2"], "kind": "title"},
            {"text": "bad pattern", "evidence": ["T1"], "kind": "title"},
        ],
        "per_video": [
            {"label": "O1", "why": "nice title"},
            {"label": "O2", "why": "invented 9999 views"},
        ],
        "ideas": ["try idea"],
    }
    client = FakeModelClient([ModelReply(json.dumps(data), (), {}, 2000, 100, 300)])
    insight = explain(chans, client, now=NOW)
    # one call, user message has labels but no video ids
    assert len(client.calls) == 1
    user_content = client.calls[0]["messages"][1]["content"]
    assert "O1" in user_content
    assert "Chan 0" in user_content
    assert "fv" not in user_content or "f0v" not in user_content  # no ids
    # actually check no id substring: video ids are like f0v000005
    for ch in chans:
        for v in ch.videos:
            assert v.video_id not in user_content
    # kept one pattern, dropped T evidence pattern
    assert len(insight.patterns) == 1
    # dropped why with invented number
    assert len(insight.per_video) == 1
    assert insight.ideas == ("try idea",)
    assert insight.cost_usd is not None and insight.cost_usd > 0
    # save and load
    save_insight(tmp_path, insight)
    loaded = latest_insight(tmp_path)
    assert loaded is not None
    assert loaded.patterns == insight.patterns


def test_explain_too_few() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cid = "UC" + "a" * 22
    vids: list[Video] = []
    for i in range(40):
        pub = (base + timedelta(days=i)).isoformat().replace("+00:00", "Z")
        views = 5000 if i == 5 else 100
        vids.append(
            Video(
                video_id=f"fv{i:06d}",
                channel_id=cid,
                title=f"V {i}",
                published_at=pub,
                duration="PT8M", privacy_status="public",
                view_count=views,
            )
        )
    ch = CompetitorChannel(
        channel_id=cid,
        title="Chan",
        handle="@h",
        fetched_at=(base + timedelta(days=60)).isoformat(),
        videos=tuple(vids),
        units=1,
    )
    client = FakeModelClient([])
    ins = explain([ch], client, now=NOW)
    assert ins.status == "too_few"
    assert len(client.calls) == 0


def test_explain_model_errors() -> None:
    chans = _make_explain_channels()
    client_auth = FakeModelClient([ModelError("auth", "bad key")])
    with pytest.raises(ModelError):
        explain(chans, client_auth, now=NOW)
    client_rate = FakeModelClient([ModelError("rate_limit", "slow")])
    ins = explain(chans, client_rate, now=NOW)
    assert ins.status == "error"
    client_bad = FakeModelClient([ModelReply("not json", (), {}, 10, 0, 10)])
    ins2 = explain(chans, client_bad, now=NOW)
    assert ins2.status == "error"
