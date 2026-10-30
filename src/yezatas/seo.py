"""SEO scorecard: deterministic checks per video, a 0-100 score and a "fix next" priority.

Pure code, no I/O and no model calls, so the suggestion agent can cite these facts.

INTERFACE (stable; the web UI codes against it):
- `channel_context(videos)` -> ChannelContext: facts across the catalog (duplicate titles, ...).
- `score_video(video, context)` -> Scorecard.
- `score_catalog(videos)` -> dict video_id -> Scorecard (builds the context once).
- `fix_priority(video, card)` -> float: views x missing points; higher = fix first.
- `CATEGORIES`: display order and labels of the check categories.
- `parse_duration(iso)` -> seconds or None; `CHECKS`: the ordered check functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .store import Video

CATEGORIES: dict[str, str] = {
    "title": "Title",
    "description": "Description",
    "tags": "Tags",
    "details": "Details",
}

# Thresholds (YouTube limits and friendly SEO rules of thumb)
TITLE_MIN, TITLE_GOOD_MAX = 30, 60  # phone search shows about 60 characters
TITLE_MAX = 70  # desktop search cuts the title off after about 70
TITLE_OK_MIN = 20
TITLE_CAPS_MAX_RATIO = 0.5
TITLE_EMOJI_MAX = 2
DESCRIPTION_GOOD, DESCRIPTION_OK = 500, 250
DESCRIPTION_LIMIT = 5000
HOOK_WINDOW, HOOK_MIN_TEXT = 150, 50
SEPARATOR_MAX_TEXT = 20  # letters allowed on a mostly-separator line
TITLE_WORDS_WINDOW = 200
HASHTAGS_GOOD_MAX, HASHTAGS_OK_MAX = 3, 15
CHAPTERS_MIN_DURATION = 240  # seconds
CHAPTERS_MIN_COUNT, CHAPTER_MIN_GAP = 3, 10
SHORT_MAX_DURATION = 60  # seconds
TAGS_GOOD_MIN, TAGS_GOOD_MAX = 5, 15
TAGS_CHAR_LIMIT = 500
LOCALIZATIONS_GOOD = 5

STOP_WORDS = frozenset(
    "this that with from your have will what when where which about into over more they them "
    "their there than then just like also been were here only some very much most such each "
    "every other these those would could should does make made best video videos".split()
)

_WORD = re.compile(r"[^\W\d_]{4,}")
_CAPS_WORD = re.compile(r"[^\W\d_]{3,}")
_URL = re.compile(r"(?:https?://|www\.)\S*", re.IGNORECASE)
_HASHTAG = re.compile(r"(?<!\w)#\w*[^\W\d_]\w*")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⭐⭕]")
_DURATION = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$")
_TIMESTAMP = re.compile(
    r"^[ \t]*[-*•(\[]*[ \t]*(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])", re.MULTILINE
)
_BROKEN_LINK = re.compile(
    r"https?://(?=\s|$|[)\]>\"'])|(?<![\w.-])(?:www\.)?example\.(?:com|org|net)\b|\[(?:link|url)\]"
    r"|\blink here\b|(?-i:\bTODO\b)",  # TODO in capitals only, so "todo app" videos pass
    re.IGNORECASE,
)
# Phrases, not single words: "like" or "comment" alone appear in almost every description.
_CALL_TO_ACTION = re.compile(
    r"\bsubscrib\w*|\bleave a comment\b|\bcomment below\b|\blet me know\b"
    r"|\b(?:hit|smash|tap) (?:the |that )?like\b|\blike and (?:share|subscribe)\b"
    r"|\bfollow (?:me|us)\b",
    re.IGNORECASE,
)
_LEADING_URL = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_SEPARATOR_START = re.compile(r"(?:[_\-=*~•·─━—–+]\s*){3,}")
_LETTER = re.compile(r"[^\W\d_]")
# A first line that sells or asks instead of describing the video.
_PROMO = re.compile(
    r"\bsubscrib\w*|\bsupport (?:my|the|our) channel\b|\bpatreon\b|\baffiliate\b"
    r"|\bdiscount\b|\bcoupon\b|\buse (?:my |the )?code\b|\bsponsor(?:ed|ship)?\b"
    r"|\bfollow (?:me|us)\b",  # not "buy": "Which laptop should you buy?" is a fine opening
    re.IGNORECASE,
)
_RELATED_LINK = re.compile(
    r"youtube\.com/(?:watch|playlist|shorts/)|youtu\.be/", re.IGNORECASE
)


@dataclass(frozen=True)
class CheckResult:
    check_id: str  # stable id, e.g. "title_length"
    category: str  # a key of CATEGORIES
    title: str  # what the check wants, e.g. "Title is 30 to 70 characters"
    max_points: int
    points: int  # earned, 0..max_points
    applicable: bool = True  # False: does not apply to this video (counts for nothing)
    detail: str = ""  # what was found, e.g. "The title is 18 characters."
    fix: str = ""  # how to fix it; "" when passed or not applicable

    @property
    def passed(self) -> bool:
        return self.applicable and self.points >= self.max_points


@dataclass(frozen=True)
class Scorecard:
    video_id: str
    score: int  # 0-100: earned / applicable max points, rounded
    checks: tuple[CheckResult, ...]

    @property
    def missing(self) -> int:
        return 100 - self.score

    @property
    def failed(self) -> list[CheckResult]:
        """Applicable checks that did not get full points, biggest loss first."""
        lost = [c for c in self.checks if c.applicable and not c.passed]
        return sorted(lost, key=lambda c: c.max_points - c.points, reverse=True)


@dataclass(frozen=True)
class ChannelContext:
    title_counts: dict[str, int] = field(default_factory=dict)  # normalized title -> videos


def normalize_title(title: str) -> str:
    return " ".join(title.casefold().split())


def channel_context(videos: Iterable[Video]) -> ChannelContext:
    counts: dict[str, int] = {}
    for video in videos:
        key = normalize_title(video.title)
        counts[key] = counts.get(key, 0) + 1
    return ChannelContext(title_counts=counts)


def parse_duration(duration: str) -> int | None:
    """ISO 8601 duration ("PT1H2M3S") to whole seconds; None when empty, invalid or zero."""
    match = _DURATION.match((duration or "").strip())
    if not match:
        return None
    days, hours, minutes, seconds = match.groups()
    total = (int(days or 0) * 86400 + int(hours or 0) * 3600 + int(minutes or 0) * 60
             + float(seconds or 0))
    return int(total) if total > 0 else None


def is_short(video: Video) -> bool:
    seconds = parse_duration(video.duration)
    return seconds is not None and seconds <= SHORT_MAX_DURATION


def _words(text: str) -> set[str]:
    """Meaningful words: 4+ letters, lowercase, not a stop word."""
    return {w for w in _WORD.findall(text.casefold()) if w not in STOP_WORDS}


def _result(check_id: str, category: str, title: str, max_points: int, points: int,
            detail: str = "", fix: str = "") -> CheckResult:
    """Build a result; the fix is dropped when the check passed."""
    passed = points >= max_points
    return CheckResult(check_id, category, title, max_points, points, True, detail,
                       "" if passed else fix)


def _skip(check_id: str, category: str, title: str, max_points: int,
          detail: str = "") -> CheckResult:
    return CheckResult(check_id, category, title, max_points, 0, False, detail, "")


def _half(max_points: int) -> int:
    return max_points // 2


# --- Title checks (25 points) ---

def check_title_length(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Title is 30 to 60 characters"
    n = len(video.title.strip())
    if TITLE_MIN <= n <= TITLE_GOOD_MAX:
        points = 8
    elif TITLE_OK_MIN <= n < TITLE_MIN or TITLE_GOOD_MAX < n <= TITLE_MAX:
        points = _half(8)
    else:
        points = 0
    fix = ("Add a title." if n == 0 else
           f"Aim for {TITLE_MIN} to {TITLE_GOOD_MAX} characters; phones cut titles off around "
           f"{TITLE_GOOD_MAX}, and this one is {n}.")
    return _result("title_length", "title", title, 8, points,
                   f"The title is {n} characters.", fix)


def check_title_not_truncated(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Title is not cut off in search results"
    n = len(video.title.strip())
    ok = 0 < n <= TITLE_MAX
    detail = (f"All {n} characters fit." if ok else
              "There is no title." if n == 0 else
              f"At {n} characters, the end of the title will be cut off.")
    fix = ("Add a title." if n == 0 else
           f"Search results cut titles around {TITLE_MAX} characters, so put the key words "
           "first and trim the rest.")
    return _result("title_not_truncated", "title", title, 4, 4 if ok else 0, detail, fix)


def check_title_caps(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Title is not mostly capital letters"
    letters = "".join(_CAPS_WORD.findall(video.title))
    if not letters:
        return _skip("title_caps", "title", title, 5, "The title has no words to check.")
    ratio = sum(c.isupper() for c in letters) / len(letters)
    ok = ratio <= TITLE_CAPS_MAX_RATIO
    return _result("title_caps", "title", title, 5, 5 if ok else 0,
                   f"{round(ratio * 100)}% of the letters are capitals.",
                   "Write the title in normal sentence or title case; all caps reads as shouting.")


def check_title_unique(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Title is different from your other videos"
    key = normalize_title(video.title)
    if context is None or not key:
        return _skip("title_unique", "title", title, 4)
    count = context.title_counts.get(key, 0)
    ok = count <= 1
    return _result("title_unique", "title", title, 4, 4 if ok else 0,
                   "No other video has this title." if ok else
                   f"{count} videos on the channel share this title.",
                   "Give each video its own title so viewers and search can tell them apart.")


def check_title_no_angle_brackets(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Title has no < or > characters"
    if not video.title:
        return _skip("title_no_angle_brackets", "title", title, 2)
    ok = "<" not in video.title and ">" not in video.title
    return _result("title_no_angle_brackets", "title", title, 2, 2 if ok else 0,
                   "No angle brackets." if ok else "The title contains < or >.",
                   "Remove the < and > characters; YouTube rejects them.")


def check_title_punctuation(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Title does not look like clickbait"
    text = video.title.strip()
    if not text:
        return _skip("title_punctuation", "title", title, 2)
    emojis = len(_EMOJI.findall(text))
    shouting = re.search(r"[!?]{2,}$", text) is not None
    ok = not shouting and emojis <= TITLE_EMOJI_MAX
    if shouting:
        detail = "The title ends with repeated ! or ?."
    elif emojis > TITLE_EMOJI_MAX:
        detail = f"The title has {emojis} emojis."
    else:
        detail = "No clickbait punctuation."
    return _result("title_punctuation", "title", title, 2, 2 if ok else 0, detail,
                   "Use a single ! or ? at most and keep to two emojis or fewer.")


# --- Description checks (39 points) ---

def check_description_length(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Description is at least 500 characters"
    n = len(video.description.strip())
    points = 5 if n >= DESCRIPTION_GOOD else _half(5) if n >= DESCRIPTION_OK else 0
    return _result("description_length", "description", title, 5, points,
                   f"The description is {n} characters.",
                   f"Write at least {DESCRIPTION_GOOD} characters about what the video covers; "
                   f"it is {n} now.")


def _hook_text(description: str) -> str:
    window = description[:HOOK_WINDOW]
    window = _HASHTAG.sub(" ", _URL.sub(" ", window))
    return " ".join(window.split())


def _first_line_problem(description: str) -> str:
    """What is wrong with the first non-empty line, or "" when it is fine."""
    first = next((line.strip() for line in description.splitlines() if line.strip()), "")
    if _LEADING_URL.match(first):
        return "The description starts with a link."
    if (_SEPARATOR_START.match(first)
            and len(_LETTER.findall(first)) <= SEPARATOR_MAX_TEXT):
        return "The description starts with a separator line."
    if _PROMO.search(first):
        return "The description starts with a promotion or call to action."
    return ""


def check_description_hook(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Description opens with a real sentence"
    if is_short(video):
        return _skip("description_hook", "description", title, 8,
                     "Shorts do not show a description opening.")
    problem = _first_line_problem(video.description)
    n = len(_hook_text(video.description))
    if not problem and n < HOOK_MIN_TEXT:
        problem = f"The first {HOOK_WINDOW} characters hold only {n} characters of plain text."
    return _result("description_hook", "description", title, 8, 0 if problem else 8,
                   problem or f"The first {HOOK_WINDOW} characters hold {n} characters of "
                   "plain text.",
                   "Start with one or two sentences that say what the video is about; move "
                   "links and promos lower. This text shows above \"more\" and in search "
                   "results.")


def check_description_title_words(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Description mentions the title's key words early"
    title_words = _words(video.title)
    if not title_words:
        return _skip("description_title_words", "description", title, 4)
    shared = title_words & _words(video.description[:TITLE_WORDS_WINDOW])
    return _result("description_title_words", "description", title, 4, 4 if shared else 0,
                   f"Shared words: {', '.join(sorted(shared))}." if shared else
                   f"None of the title's key words are in the first {TITLE_WORDS_WINDOW} "
                   "characters.",
                   "Use the main words from the title in the first two lines of the description.")


def check_description_links_ok(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Description has no placeholder or broken links"
    if not video.description.strip():
        return _skip("description_links_ok", "description", title, 2)
    match = _BROKEN_LINK.search(video.description)
    return _result("description_links_ok", "description", title, 2, 0 if match else 2,
                   f"Found \"{match.group(0).strip()}\"." if match else
                   "No placeholder links found.",
                   "Replace the placeholder with the real link or delete it.")


def check_description_related_links(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Description links to another video or playlist"
    if is_short(video) or not video.description.strip():
        return _skip("description_related_links", "description", title, 3)
    ok = _RELATED_LINK.search(video.description) is not None
    return _result("description_related_links", "description", title, 3, 3 if ok else 0,
                   "A link to another video or playlist is there." if ok else
                   "No link to another video or playlist found.",
                   "Link a related video or playlist so viewers keep watching your channel.")


def parse_chapters(description: str) -> list[int]:
    """Seconds of every timestamp that starts a line."""
    return [int(h or 0) * 3600 + int(m) * 60 + int(s)
            for h, m, s in _TIMESTAMP.findall(description)]


def check_description_chapters(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Long video has chapters starting at 0:00"
    seconds = parse_duration(video.duration)
    if seconds is None or seconds < CHAPTERS_MIN_DURATION:
        return _skip("description_chapters", "description", title, 8,
                     "Chapters only matter for videos of 4 minutes or more.")
    stamps = parse_chapters(video.description)
    if len(stamps) < CHAPTERS_MIN_COUNT:
        problem = f"Found {len(stamps)} timestamps; chapters need at least {CHAPTERS_MIN_COUNT}."
    elif stamps[0] != 0:
        problem = "The first timestamp is not 0:00."
    elif any(b - a < CHAPTER_MIN_GAP for a, b in zip(stamps, stamps[1:])):
        problem = f"Timestamps must go up, at least {CHAPTER_MIN_GAP} seconds apart."
    else:
        problem = ""
    return _result("description_chapters", "description", title, 8, 0 if problem else 8,
                   problem or f"{len(stamps)} chapters found.",
                   "Add chapters, one per line like \"0:00 Intro\", at least 3, starting at "
                   "0:00 and matching what is really in the video.")


def check_description_hashtags(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Description has 1 to 3 hashtags"
    n = len({t.casefold() for t in _HASHTAG.findall(video.description)})
    if 1 <= n <= HASHTAGS_GOOD_MAX:
        points = 3
    elif n <= HASHTAGS_OK_MAX:
        points = _half(3)
    else:
        points = 0
    fix = ("Add 1 to 3 hashtags at the end of the description." if n == 0 else
           f"Keep to 1 to {HASHTAGS_GOOD_MAX} hashtags; YouTube ignores all of them when "
           f"there are more than {HASHTAGS_OK_MAX}.")
    return _result("description_hashtags", "description", title, 3, points,
                   f"The description has {n} hashtags.", fix)


def check_description_no_angle_brackets(video: Video,
                                        context: ChannelContext | None) -> CheckResult:
    title = "Description has no < or > characters"
    if not video.description:
        return _skip("description_no_angle_brackets", "description", title, 1)
    ok = "<" not in video.description and ">" not in video.description
    return _result("description_no_angle_brackets", "description", title, 1, 1 if ok else 0,
                   "No angle brackets." if ok else "The description contains < or >.",
                   "Remove the < and > characters; YouTube rejects them.")


def check_description_limit(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Description fits the 5,000 character limit"
    n = len(video.description)
    if n == 0:
        return _skip("description_limit", "description", title, 1)
    ok = n <= DESCRIPTION_LIMIT
    return _result("description_limit", "description", title, 1, 1 if ok else 0,
                   f"The description is {n} characters.",
                   f"Cut {n - DESCRIPTION_LIMIT} characters; YouTube allows "
                   f"{DESCRIPTION_LIMIT:,}.")


def check_description_not_title_copy(video: Video,
                                     context: ChannelContext | None) -> CheckResult:
    title = "Description says more than the title"
    desc, name = normalize_title(video.description), normalize_title(video.title)
    if not desc or not name:
        return _skip("description_not_title_copy", "description", title, 2)
    copy = desc == name or (len(desc) < len(name) + 20 and name in desc)
    return _result("description_not_title_copy", "description", title, 2, 0 if copy else 2,
                   "The description only repeats the title." if copy else
                   "The description adds new text.",
                   "Say what happens in the video; do not just repeat the title.")


def check_engagement_prompt(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Description invites viewers to subscribe, like or comment"
    ok = _CALL_TO_ACTION.search(video.description) is not None
    return _result("engagement_prompt", "description", title, 2, 2 if ok else 0,
                   "A call to action is there." if ok else "No call to action found.",
                   "Add a short line asking viewers to subscribe, like or comment.")


# --- Tag checks (14 points) ---

def check_tags_present(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Video has 5 to 15 tags"
    n = len([t for t in video.tags if t.strip()])
    points = 5 if n >= TAGS_GOOD_MIN else _half(5) if n >= 1 else 0
    fix = (f"Add {TAGS_GOOD_MIN} to {TAGS_GOOD_MAX} tags, starting with the main topic."
           if n == 0 else
           f"Add a few more tags; {TAGS_GOOD_MIN} to {TAGS_GOOD_MAX} works well.")
    return _result("tags_present", "tags", title, 5, points, f"The video has {n} tags.", fix)


def tags_char_count(tags: Iterable[str]) -> int:
    """Characters YouTube counts: quotes around tags with spaces, plus commas between tags."""
    tags = list(tags)
    return sum(len(t) + (2 if " " in t else 0) for t in tags) + max(len(tags) - 1, 0)


def check_tags_length(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Tags fit the 500 character limit"
    if not video.tags:
        return _skip("tags_length", "tags", title, 2)
    n = tags_char_count(video.tags)
    return _result("tags_length", "tags", title, 2, 2 if n <= TAGS_CHAR_LIMIT else 0,
                   f"The tags use {n} of {TAGS_CHAR_LIMIT} characters.",
                   f"Remove tags until they fit; they are {n - TAGS_CHAR_LIMIT} characters over.")


def check_tags_title_match(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "At least one tag matches the title"
    title_words = _words(video.title)
    if not video.tags or not title_words:
        return _skip("tags_title_match", "tags", title, 4)
    shared = title_words & _words(" ".join(video.tags))
    return _result("tags_title_match", "tags", title, 4, 4 if shared else 0,
                   f"Shared words: {', '.join(sorted(shared))}." if shared else
                   "No tag shares a key word with the title.",
                   "Add a tag that uses the main words from the title.")


def check_tags_unique(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "No tag is repeated"
    if not video.tags:
        return _skip("tags_unique", "tags", title, 3)
    seen: set[str] = set()
    repeated: list[str] = []
    for tag in video.tags:
        key = " ".join(tag.casefold().split())
        if key in seen and key not in repeated:
            repeated.append(key)
        seen.add(key)
    return _result("tags_unique", "tags", title, 3, 0 if repeated else 3,
                   f"Repeated: {', '.join(repeated)}." if repeated else "All tags are different.",
                   "Delete the repeated tags.")


# --- Detail checks (22 points) ---

def check_default_language(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Video language is set"
    ok = bool(video.default_language)
    return _result("default_language", "details", title, 5, 5 if ok else 0,
                   f"The video language is {video.default_language}." if ok else
                   "No video language is set.",
                   "Set the video language in YouTube Studio under Advanced settings.")


def check_audio_language(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Spoken (audio) language is set"
    ok = bool(video.default_audio_language)
    return _result("audio_language", "details", title, 5, 5 if ok else 0,
                   f"The audio language is {video.default_audio_language}." if ok else
                   "No audio language is set.",
                   "Set the audio language so YouTube can offer the right captions and dubbing.")


def _is_own_language(key: str, language: str | None) -> bool:
    """Studio lists the video's own language among its translations ("en", or "en" on "en-GB")."""
    if not language:
        return False
    return (key == language or ("-" not in key and language.split("-", 1)[0] == key)
            or ("-" not in language and key.split("-", 1)[0] == language))


def check_localizations(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "Title and description are translated into 5+ languages"
    n = sum(1 for key in video.localizations
            if not _is_own_language(key, video.default_language))
    points = 8 if n >= LOCALIZATIONS_GOOD else _half(8) if n >= 1 else 0
    fix = ("Add translated titles and descriptions; Yezatas can translate them for you."
           if n == 0 else
           f"Add more languages; {LOCALIZATIONS_GOOD} or more reach more viewers, and Yezatas "
           "can translate them for you.")
    return _result("localizations", "details", title, 8, points,
                   f"The video has {n} translation{'' if n == 1 else 's'}.", fix)


def check_thumbnail_hd(video: Video, context: ChannelContext | None) -> CheckResult:
    title = "A sharp thumbnail is available"
    if not video.thumbnails:
        return _skip("thumbnail_hd", "details", title, 4, "No thumbnail information found.")
    ok = "maxres" in video.thumbnails or "standard" in video.thumbnails
    return _result("thumbnail_hd", "details", title, 4, 4 if ok else 0,
                   "A high-resolution thumbnail exists." if ok else "Only small thumbnails exist.",
                   "Upload a 1280 x 720 thumbnail.")


CheckFn = Callable[[Video, "ChannelContext | None"], CheckResult]

CHECKS: tuple[CheckFn, ...] = (
    check_title_length,
    check_title_not_truncated,
    check_title_caps,
    check_title_unique,
    check_title_no_angle_brackets,
    check_title_punctuation,
    check_description_length,
    check_description_hook,
    check_description_title_words,
    check_description_links_ok,
    check_description_related_links,
    check_description_chapters,
    check_description_hashtags,
    check_description_no_angle_brackets,
    check_description_limit,
    check_description_not_title_copy,
    check_engagement_prompt,
    check_tags_present,
    check_tags_length,
    check_tags_title_match,
    check_tags_unique,
    check_default_language,
    check_audio_language,
    check_localizations,
    check_thumbnail_hd,
)


def score_video(video: Video, context: ChannelContext | None = None) -> Scorecard:
    """Run every check. Without a context the "title is unique" check is not applicable."""
    checks = tuple(check(video, context) for check in CHECKS)
    earned = sum(c.points for c in checks if c.applicable)
    total = sum(c.max_points for c in checks if c.applicable)
    return Scorecard(video.video_id, round(100 * earned / total) if total else 100, checks)


def score_catalog(videos: Iterable[Video]) -> dict[str, Scorecard]:
    videos = list(videos)
    context = channel_context(videos)
    return {v.video_id: score_video(v, context) for v in videos}


def fix_priority(video: Video, card: Scorecard) -> float:
    return float(video.view_count * card.missing)
