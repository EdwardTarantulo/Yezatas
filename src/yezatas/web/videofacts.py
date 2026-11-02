"""Quick info on the video page: what the stored pull already says about one video. No API calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..seo import _HASHTAG, CATEGORY_NAMES, Scorecard, parse_chapters, parse_duration
from ..store import Video
from ..subtitle_translate import language_name
from .views import format_count, format_duration

# The tile that shows a check's subject gets a warm edge while that check costs points.
FACT_CHECKS: dict[str, str] = {
    "language": "default_language", "chapters": "description_chapters",
    "hashtags": "description_hashtags", "translations": "localizations",
    "category": "category",
    "pinned": "pinned_comment",
}


# Tiles that open the place where their subject is changed: key -> (link target on the page, the place's name).
FACT_LINKS: dict[str, tuple[str, str]] = {
    "category": ("#category", "Category"),
    "pinned": ("#pinned-comment", "Pinned comment"),
    "chapters": ("#tool-transcript", "Transcript and chapters"),
    "hashtags": ("#f-description", "the description"),
    "captions": ("#tool-subtitles", "Subtitles"),
    "translations": ("#tool-translations", "Translations"),
}


@dataclass(frozen=True)
class Fact:
    key: str  # "views", "category", ...: the tile's id is "fact-<key>"
    label: str
    value: str
    tone: str = ""  # "warn" when a failed check is about this tile, else ""
    href: str = ""  # where the tile leads, or "" for a plain fact
    go: str = ""  # the link's accessible name, e.g. "Go to Subtitles"


def _part(video: Video, name: str) -> dict[str, Any]:
    part = video.raw.get(name) if isinstance(video.raw, dict) else None
    return part if isinstance(part, dict) else {}


def own_translations(video: Video) -> int:
    """Translations on YouTube, not counting the video's own language (as the SEO check counts them)."""
    own = (video.default_language or "").split("-", 1)[0]
    return sum(1 for key in video.localizations if not own or key.split("-", 1)[0] != own)


def build_facts(video: Video, card: Scorecard) -> tuple[Fact, ...]:
    failed = {c.check_id for c in card.failed}
    category_id = str(_part(video, "snippet").get("categoryId") or "")
    caption = str(_part(video, "contentDetails").get("caption") or "")
    definition = str(_part(video, "contentDetails").get("definition") or "")
    chapters = len(parse_chapters(video.description))
    hashtags = len({t.casefold() for t in _HASHTAG.findall(video.description)})
    language = video.default_language or video.default_audio_language or ""
    pinned = next((c for c in card.checks if c.check_id == "pinned_comment"), None)
    pinned_value = ("Not fetched" if pinned is None or not pinned.applicable
                    else "Yes" if pinned.passed else "None found")
    values = [
        ("views", "Views", format_count(video.view_count)),
        ("likes", "Likes", format_count(video.like_count)),
        ("comments", "Comments", format_count(video.comment_count)),
        ("duration", "Duration", format_duration(parse_duration(video.duration)) or "Not known"),
        ("category", "Category", CATEGORY_NAMES.get(category_id, "Not known")),
        ("language", "Language", language_name(language) if language else "Not set"),
        ("chapters", "Chapters", str(chapters) if chapters else "None"),
        ("hashtags", "Hashtags", str(hashtags) if hashtags else "None"),
        ("captions", "Captions", {"true": "Yes", "false": "None"}.get(caption, "Not known")),
        ("translations", "Translations", str(own_translations(video)) or "0"),
        ("quality", "Quality", {"hd": "HD", "sd": "SD"}.get(definition, "Not known")),
        ("pinned", "Pinned comment", pinned_value),
    ]
    facts = []
    for key, label, value in values:
        href, place = FACT_LINKS.get(key, ("", ""))
        facts.append(Fact(key, label, value, "warn" if FACT_CHECKS.get(key) in failed else "",
                          href, f"Go to {place}" if place else ""))
    return tuple(facts)
