"""Catalog pull: read the channel's videos from YouTube into the local store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .quota import QuotaExceeded, QuotaGuard
from .store import Store, Video
from .youtube import MAX_RESULTS, YouTubeReader

VIDEO_PARTS = "snippet,contentDetails,statistics,status,localizations"


class NoChannelError(RuntimeError):
    """The signed-in account has no YouTube channel (or it has no uploads playlist)."""


@dataclass(frozen=True)
class PullResult:
    pull_id: int
    channel_id: str
    video_count: int
    units_used: int


def _count(stats: dict[str, Any], key: str) -> int:
    try:
        return int(stats.get(key, 0))
    except (TypeError, ValueError):
        return 0


def video_from_api(item: dict[str, Any]) -> Video:
    """Map one videos.list item to a Video. Missing statistics (e.g. hidden likes) become 0."""
    snippet = item.get("snippet") or {}
    stats = item.get("statistics") or {}
    return Video(
        video_id=item["id"],
        channel_id=snippet.get("channelId", ""),
        title=snippet.get("title", ""),
        description=snippet.get("description", ""),
        tags=list(snippet.get("tags") or []),
        privacy_status=(item.get("status") or {}).get("privacyStatus", ""),
        default_language=snippet.get("defaultLanguage"),
        default_audio_language=snippet.get("defaultAudioLanguage"),
        published_at=snippet.get("publishedAt", ""),
        duration=(item.get("contentDetails") or {}).get("duration", ""),
        view_count=_count(stats, "viewCount"),
        like_count=_count(stats, "likeCount"),
        comment_count=_count(stats, "commentCount"),
        localizations=dict(item.get("localizations") or {}),
        thumbnails=dict(snippet.get("thumbnails") or {}),
        raw=item,
    )


def pull_catalog(
    reader: YouTubeReader,
    store: Store,
    guard: QuotaGuard,
    now: datetime | None = None,
) -> PullResult:
    """Pull every upload of the signed-in channel into the store (one snapshot per video).

    Each call is charged to the quota guard before it is made. If the video batches would not
    fit in the remaining budget, QuotaExceeded is raised before any video is fetched or stored.
    """
    units = 0

    units += guard.charge("channels.list", note="catalog pull")
    channels = reader.channels_mine("id,contentDetails")
    items = channels.get("items") or []
    if not items:
        raise NoChannelError("this Google account has no YouTube channel")
    channel = items[0]
    channel_id = channel["id"]
    uploads = ((channel.get("contentDetails") or {}).get("relatedPlaylists") or {}).get("uploads")
    if not uploads:
        raise NoChannelError("the channel has no uploads playlist")

    video_ids: list[str] = []
    seen: set[str] = set()
    page_token: str | None = None
    while True:
        units += guard.charge("playlistItems.list", note="catalog pull")
        page = reader.playlist_items(uploads, page_token)
        for entry in page.get("items") or []:
            vid = (entry.get("contentDetails") or {}).get("videoId")
            if vid and vid not in seen:
                seen.add(vid)
                video_ids.append(vid)
        page_token = page.get("nextPageToken")
        if not page_token:
            break

    batches = [video_ids[i : i + MAX_RESULTS] for i in range(0, len(video_ids), MAX_RESULTS)]
    if batches:
        preview = guard.preview([("videos.list", len(batches))])
        if not preview.fits:
            raise QuotaExceeded(
                f"need {preview.total} units for {len(batches)} videos.list batches "
                f"but only {preview.remaining} remaining"
            )

    pull_id = store.begin_pull(channel_id, now)
    stored = 0
    returned: set[str] = set()
    for batch in batches:
        units += guard.charge("videos.list", note="catalog pull")
        response = reader.videos(batch, VIDEO_PARTS)
        for item in response.get("items") or []:
            video = video_from_api(item)
            store.upsert_video(video, pull_id, now)
            returned.add(video.video_id)
            stored += 1
    store.finish_pull(pull_id, stored, now)
    if stored:  # an empty answer is more likely a fault than a channel with no videos left
        store.mark_gone_except(returned, now)
    return PullResult(pull_id, channel_id, stored, units)
