"""Replace a video's thumbnail with a variant idea and keep the old one to put back (F3).
Nothing here runs unless the creator confirmed this exact swap; every YouTube call is charged first."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .packaging import SIZE_ORDER, PackagingError, fetch_thumbnail, save_image, stored_image_path
from .quota import QuotaGuard, cost_of

SWAP_PLAN: tuple[tuple[str, int], ...] = (("videos.list", 1), ("thumbnails.set", 1))
MAX_BYTES = 2 * 1024 * 1024
SWAPS_FILE = "swaps.json"
_GONE = "The saved thumbnail is gone, so it cannot be put back."


class ThumbSwapError(ValueError):
    """The message is a plain sentence."""


class ThumbnailWriter(Protocol):
    def videos(self, ids: Sequence[str], part: str) -> dict[str, Any]: ...

    def set_thumbnail(self, video_id: str, image: bytes) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ThumbnailSwap:
    swap_id: int
    video_id: str
    kind: str
    new_file: str
    backup_file: str
    variant_id: int | None
    units: int
    at: str


def swap_units() -> int:
    """Units charged for one thumbnail swap."""
    return sum(cost_of(op) * n for op, n in SWAP_PLAN)


def check_jpeg(image: bytes) -> None:
    """Validate that image is a JPEG within the size limit."""
    if not image.startswith(b"\xff\xd8\xff"):
        raise ThumbSwapError("That picture is not a JPEG file.")
    if len(image) > MAX_BYTES:
        raise ThumbSwapError("That picture is larger than YouTube's 2 MB limit for thumbnails.")


def live_thumbnail_url(item: dict[str, Any]) -> str:
    """Best https thumbnail URL from a videos.list item, or empty string."""
    snippet = item.get("snippet")
    if not isinstance(snippet, dict):
        return ""
    thumbs = snippet.get("thumbnails")
    if not isinstance(thumbs, dict):
        return ""
    for name in SIZE_ORDER:
        entry = thumbs.get(name)
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if isinstance(url, str) and url.startswith("https://"):
            return url
    return ""


def swap_thumbnail(
    writer: ThumbnailWriter,
    guard: QuotaGuard,
    images_dir: Path,
    video_id: str,
    image: bytes,
    *,
    kind: str,
    new_file: str,
    variant_id: int | None = None,
    fetch: Callable[[str], bytes] = fetch_thumbnail,
    now: datetime | None = None,
    report: Callable[[str], None] | None = None,
) -> ThumbnailSwap:
    """Charge quota, save the live thumbnail, upload the new one and record the swap.

    A failed upload does not create a swap record.
    """
    check_jpeg(image)
    if kind not in ("replace", "restore"):
        raise ValueError("kind must be 'replace' or 'restore'")
    guard.charge_plan(SWAP_PLAN, note=f"thumbnail {kind}")
    if report is not None:
        report("Reading the live thumbnail")
    data = writer.videos([video_id], "snippet")
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list) or len(items) == 0:
        raise ThumbSwapError("YouTube did not find this video on your channel. Nothing was changed.")
    item = items[0] if isinstance(items[0], dict) else {}
    url = live_thumbnail_url(item)
    if not url:
        raise ThumbSwapError(
            "YouTube gave no thumbnail for this video, so there is nothing to keep. Nothing was changed."
        )
    if report is not None:
        report("Saving the current thumbnail")
    old = fetch(url)
    if not old:
        raise ThumbSwapError(
            "YouTube gave no thumbnail for this video, so there is nothing to keep. Nothing was changed."
        )
    backup_file = save_image(images_dir, video_id, "backup", old)
    if report is not None:
        report("Uploading the new thumbnail")
    writer.set_thumbnail(video_id, image)
    existing = load_swaps(images_dir)
    max_id = max((s.swap_id for s in existing), default=0)
    swap_id = max_id + 1
    at = (now if now is not None else datetime.now(timezone.utc)).isoformat()
    units = swap_units()
    swap = ThumbnailSwap(
        swap_id=swap_id,
        video_id=video_id,
        kind=kind,
        new_file=new_file,
        backup_file=backup_file,
        variant_id=variant_id,
        units=units,
        at=at,
    )
    save_swap(images_dir, swap)
    return swap


def save_swap(images_dir: Path, swap: ThumbnailSwap) -> None:
    """Append a swap to the swaps file atomically."""
    folder = Path(images_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / SWAPS_FILE
    raw_list: list[dict[str, Any]] = []
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, list):
                raw_list = [e for e in data if isinstance(e, dict)]
        except (OSError, ValueError):
            raw_list = []
    raw_list.append(asdict(swap))
    tmp_path = folder / f".{SWAPS_FILE}.tmp"
    tmp_path.write_text(json.dumps(raw_list, indent=2), encoding="utf-8")
    tmp_path.replace(path)


_FIELDS: dict[str, type] = {
    "swap_id": int,
    "video_id": str,
    "kind": str,
    "new_file": str,
    "backup_file": str,
    "units": int,
    "at": str,
}


def _swap_from(entry: dict[str, Any]) -> ThumbnailSwap | None:
    """The swap in one saved entry, or None when a field is missing or has the wrong type."""
    for name, typ in _FIELDS.items():
        val = entry.get(name)
        if not isinstance(val, typ) or isinstance(val, bool):
            return None
    variant_id = entry.get("variant_id")
    if variant_id is not None and not (isinstance(variant_id, int) and not isinstance(variant_id, bool)):
        return None
    return ThumbnailSwap(
        swap_id=entry["swap_id"],
        video_id=entry["video_id"],
        kind=entry["kind"],
        new_file=entry["new_file"],
        backup_file=entry["backup_file"],
        variant_id=variant_id,
        units=entry["units"],
        at=entry["at"],
    )


def load_swaps(images_dir: Path) -> list[ThumbnailSwap]:
    """Swaps saved in the file, newest first."""
    path = Path(images_dir) / SWAPS_FILE
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    result: list[ThumbnailSwap] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        swap = _swap_from(entry)
        if swap is None:
            continue
        result.append(swap)
    result.sort(key=lambda s: s.swap_id, reverse=True)
    return result


def backup_image(images_dir: Path, swap: ThumbnailSwap) -> bytes:
    """Bytes of the saved backup image for a swap."""
    try:
        p = stored_image_path(images_dir, swap.video_id, swap.backup_file)
    except PackagingError as exc:
        raise ThumbSwapError(_GONE) from exc
    try:
        return p.read_bytes()
    except OSError as exc:
        raise ThumbSwapError(_GONE) from exc
