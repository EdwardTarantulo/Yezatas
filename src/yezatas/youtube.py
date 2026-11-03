"""The YouTube Data API behind small interfaces: reads for the catalog pull, and the writes that
only come after the creator confirmed them (videos.update in a push, thumbnails.set, and posting a
comment or a reply)."""

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


class CaptionReader(Protocol):
    """The caption reads `captions.py` needs; both return what the API returned."""

    def caption_tracks(self, video_id: str) -> dict[str, Any]: ...

    def caption_srt(self, caption_id: str) -> bytes | str: ...


class GoogleCaptionReader:
    """CaptionReader backed by google-api-python-client. Thin on purpose; not unit-tested.

    Listing costs 50 quota units, downloading 200. Both need the youtube.force-ssl scope, and
    a download only works for the signed-in channel's own videos.
    """

    def __init__(self, credentials: Any) -> None:
        from googleapiclient.discovery import build

        self._service = build("youtube", "v3", credentials=credentials, cache_discovery=False)

    def caption_tracks(self, video_id: str) -> dict[str, Any]:
        return self._service.captions().list(part="snippet", videoId=video_id).execute()

    def caption_srt(self, caption_id: str) -> bytes | str:
        # captions.download is a media method: the client library exposes it as download_media.
        return self._service.captions().download_media(id=caption_id, tfmt="srt").execute()


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


class CommentWriter(Protocol):
    """What posting a comment needs; each call returns what the API returned."""

    def insert_reply(self, parent_id: str, text: str) -> dict[str, Any]: ...

    def insert_top_level(self, video_id: str, text: str) -> dict[str, Any]: ...


class GoogleCommentWriter:
    """CommentWriter backed by google-api-python-client. Thin on purpose; not unit-tested.

    Both calls cost 50 quota units and need the youtube.force-ssl scope. The API cannot pin a
    comment: pinning is done in YouTube Studio.
    """

    def __init__(self, credentials: Any) -> None:
        from googleapiclient.discovery import build

        self._service = build("youtube", "v3", credentials=credentials, cache_discovery=False)

    def insert_reply(self, parent_id: str, text: str) -> dict[str, Any]:
        """comments.insert: reply to the top-level comment `parent_id`."""
        body = {"snippet": {"parentId": parent_id, "textOriginal": text}}
        return self._service.comments().insert(part="snippet", body=body).execute()

    def insert_top_level(self, video_id: str, text: str) -> dict[str, Any]:
        """commentThreads.insert: a new top-level comment on a video."""
        body = {"snippet": {"videoId": video_id,
                            "topLevelComment": {"snippet": {"textOriginal": text}}}}
        return self._service.commentThreads().insert(part="snippet", body=body).execute()
