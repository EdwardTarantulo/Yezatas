"""Fake YouTube reader with made-up data. No network."""

from __future__ import annotations

from typing import Any, Sequence

CHANNEL_ID = "UC_fake_channel"
UPLOADS_ID = "UU_fake_uploads"


def make_item(n: int, views: int | None = None) -> dict[str, Any]:
    """A videos.list item. Every 7th video hides likes, every 5th has a German localization."""
    stats: dict[str, str] = {"viewCount": str(views if views is not None else n * 10),
                             "commentCount": str(n % 4)}
    if n % 7 != 0:
        stats["likeCount"] = str(n)
    item: dict[str, Any] = {
        "id": f"vid{n:06d}",
        "snippet": {
            "channelId": CHANNEL_ID,
            "title": f"Repotting cactus #{n}",
            "description": f"How to repot cactus number {n}.",
            "tags": ["cactus", "garden"],
            "publishedAt": "2026-01-01T10:00:00Z",
            "defaultLanguage": "en",
            "thumbnails": {"default": {"url": f"https://example.invalid/{n}.jpg"}},
        },
        "contentDetails": {"duration": "PT5M30S"},
        "statistics": stats,
        "status": {"privacyStatus": "public"},
    }
    if n % 5 == 0:
        item["localizations"] = {"de": {"title": f"Kaktus umtopfen #{n}",
                                        "description": f"Anleitung {n}."}}
    return item


class FakeYouTubeReader:
    def __init__(self, video_count: int = 120, page_size: int = 50, has_channel: bool = True) -> None:
        self.items = [make_item(n) for n in range(1, video_count + 1)]
        self.page_size = page_size
        self.has_channel = has_channel
        self.calls: list[tuple[str, Any]] = []

    def channels_mine(self, part: str) -> dict[str, Any]:
        self.calls.append(("channels_mine", part))
        if not self.has_channel:
            return {"items": []}
        return {"items": [{"id": CHANNEL_ID,
                           "contentDetails": {"relatedPlaylists": {"uploads": UPLOADS_ID}}}]}

    def playlist_items(self, playlist_id: str, page_token: str | None) -> dict[str, Any]:
        self.calls.append(("playlist_items", (playlist_id, page_token)))
        assert playlist_id == UPLOADS_ID
        start = int(page_token) if page_token else 0
        chunk = self.items[start:start + self.page_size]
        response: dict[str, Any] = {
            "items": [{"contentDetails": {"videoId": i["id"]}} for i in chunk]}
        if start + self.page_size < len(self.items):
            response["nextPageToken"] = str(start + self.page_size)
        return response

    def videos(self, ids: Sequence[str], part: str) -> dict[str, Any]:
        self.calls.append(("videos", (tuple(ids), part)))
        assert 1 <= len(ids) <= 50
        wanted = set(ids)
        return {"items": [i for i in self.items if i["id"] in wanted]}
