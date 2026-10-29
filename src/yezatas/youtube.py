"""The YouTube Data API behind a small read-only interface. No write calls exist yet."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

MAX_RESULTS = 50


class YouTubeReader(Protocol):
    """The read calls the catalog pull needs; each returns the raw API response."""

    def channels_mine(self, part: str) -> dict[str, Any]: ...

    def playlist_items(self, playlist_id: str, page_token: str | None) -> dict[str, Any]: ...

    def videos(self, ids: Sequence[str], part: str) -> dict[str, Any]: ...


class GoogleYouTubeReader:
    """YouTubeReader backed by google-api-python-client. Thin on purpose; not unit-tested."""

    def __init__(self, credentials: Any) -> None:
        from googleapiclient.discovery import build

        self._service = build("youtube", "v3", credentials=credentials, cache_discovery=False)

    def channels_mine(self, part: str) -> dict[str, Any]:
        return self._service.channels().list(part=part, mine=True).execute()

    def playlist_items(self, playlist_id: str, page_token: str | None) -> dict[str, Any]:
        return (
            self._service.playlistItems()
            .list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=MAX_RESULTS,
                pageToken=page_token,
            )
            .execute()
        )

    def videos(self, ids: Sequence[str], part: str) -> dict[str, Any]:
        if not 1 <= len(ids) <= MAX_RESULTS:
            raise ValueError(f"videos.list takes 1 to {MAX_RESULTS} ids, got {len(ids)}")
        return self._service.videos().list(part=part, id=",".join(ids)).execute()
