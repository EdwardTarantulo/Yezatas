"""The YouTube Data API behind small interfaces: reads for the catalog pull, and one write
(videos.update) that only a push of creator-approved drafts uses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

MAX_RESULTS = 50


@dataclass(frozen=True)
class ApiErrorInfo:
    """What a failed YouTube API call said: HTTP status, machine reasons and the message."""

    status: int | None
    reasons: tuple[str, ...]
    message: str

    @property
    def reason(self) -> str:
        return self.reasons[0] if self.reasons else ""


def api_error_info(exc: BaseException) -> ApiErrorInfo:
    """Pull status, reasons and message out of a googleapiclient HttpError (duck-typed).

    Reasons come from `error.errors[].reason` (for example `quotaExceeded`) and from the
    `ErrorInfo` entries in `error.details[].reason` (for example `SERVICE_DISABLED`). Anything
    that cannot be parsed yields empty reasons rather than raising.
    """
    status = getattr(getattr(exc, "resp", None), "status", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    content = getattr(exc, "content", b"")
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    reasons: list[str] = []
    message = ""
    try:
        error = json.loads(content).get("error")
    except (ValueError, AttributeError, TypeError):
        error = None
    if isinstance(error, dict):
        message = str(error.get("message") or "")
        for entry in (error.get("errors") or []) + (error.get("details") or []):
            reason = entry.get("reason") if isinstance(entry, dict) else None
            if isinstance(reason, str) and reason not in reasons:
                reasons.append(reason)
    return ApiErrorInfo(status, tuple(reasons), message)


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


class CommentReader(Protocol):
    """The comment read the comments assistant needs; each call returns the raw API response."""

    def comment_threads(self, video_id: str, page_token: str | None) -> dict[str, Any]: ...


class GoogleCommentReader:
    """CommentReader backed by google-api-python-client. Thin on purpose; not unit-tested.

    Reading comments needs the youtube.force-ssl scope.
    """

    def __init__(self, credentials: Any) -> None:
        from googleapiclient.discovery import build

        self._service = build("youtube", "v3", credentials=credentials, cache_discovery=False)

    def comment_threads(self, video_id: str, page_token: str | None) -> dict[str, Any]:
        return (
            self._service.commentThreads()
            .list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText",
                order="time",
                pageToken=page_token,
            )
            .execute()
        )


class YouTubeWriter(Protocol):
    """What a push needs: read a batch of videos, and update one video's snippet."""

    def videos(self, ids: Sequence[str], part: str) -> dict[str, Any]: ...

    def update_video(self, body: dict[str, Any], part: str) -> dict[str, Any]: ...


class GoogleYouTubeWriter(GoogleYouTubeReader):
    """YouTubeWriter backed by google-api-python-client. Thin on purpose; not unit-tested."""

    def update_video(self, body: dict[str, Any], part: str) -> dict[str, Any]:
        return self._service.videos().update(part=part, body=body).execute()

    def set_thumbnail(self, video_id: str, image: bytes) -> dict[str, Any]:
        """Upload JPEG bytes as the video's thumbnail (thumbnails.set, 50 units)."""
        from googleapiclient.http import MediaInMemoryUpload

        media = MediaInMemoryUpload(image, mimetype="image/jpeg", resumable=False)
        return self._service.thumbnails().set(videoId=video_id, media_body=media).execute()
