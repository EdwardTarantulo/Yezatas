"""Page models for the dashboard, the content table, the video page and the changes pages.

Pure functions: they take videos and scorecards and return plain data for the templates, so
nothing here touches the disk or YouTube. Formatting (view counts, dates) lives here too.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence
from urllib.parse import quote, urlencode

from markupsafe import Markup, escape

from .. import adcheck as adcheck_mod
from .. import edits
from ..agent import estimate_cost
from ..bulk import MAX_BULK, BulkChange, BulkResult, BulkSpec
from ..copilot import REF, catalog_refs
from ..diffs import DraftDiff, diff_draft, diff_row
from ..links import SKIPPED_MESSAGE, LinkCheckSummary, LinkEntry
from ..outliers import (BELOW_THRESHOLD, DESCRIPTION_CHARS, MAX_EXPLAINED, MAX_TYPICAL, MIN_AGE_DAYS,
                        OUTLIER_THRESHOLD, PEERS, OutlierScore, below_norm, outliers, ratio_label)
from ..push import PushItem
from ..seo import (CATEGORIES, CHAPTERS_MIN_DURATION, CheckResult, Scorecard, chapter_problem,
                   fix_priority, parse_chapters, parse_duration)
from ..store import (AdCheck, AgentRun, CopilotProposal, CopilotTurn, DraftRow, OutlierReport, ShortsFind,
                     PackagingReview, PushRecord, PushRun, SubtitleTranslationRecord,
                     ThumbnailVariant, TranscriptRecord, TranslationRecord, Video, Comment,
                     CommentReply, ReplyTemplate, Store)
from ..comments import CATEGORIES as COMMENT_CATEGORIES
from ..postreply import posted_url
from .. import affiliate as affiliate_mod
from .. import bulkagent as bulkagent_mod
from .. import customchecks as customchecks_mod
from .. import glossary as glossary_mod
from .. import replytemplates as replytemplates_mod
from ..captions import MODEL_NAME as CAPTIONS_MODEL, language_name as caption_language_name
from ..soundtags import MODEL_NAME as SOUNDS_MODEL
from ..transcribe import ASR_LANGUAGES, ASR_MODEL, ASR_PRICE_PER_HOUR, clock
from ..translate import DEFAULT_TARGETS, LANGUAGES, same_language, source_hash, youtube_key
from ..strategy import StrategyReport
from ..validate import Draft, Issue, has_errors

PAGE_SIZE = 50
MAX_QUERY_CHARS = 200
FIX_NEXT_ROWS = 8
PROBLEM_ROWS = 6
DESCRIPTION_SNIPPET_CHARS = 90

GOOD_FROM = 80
OK_FROM = 50

# (key, label, range text) in display order.
BANDS: tuple[tuple[str, str, str], ...] = (
    ("good", "Good", "80+"),
    ("ok", "OK", "50-79"),
    ("poor", "Needs work", "under 50"),
)
BAND_KEYS = tuple(b[0] for b in BANDS)
# the sortable columns: key, title, the first click's direction, and how each direction reads
SORTS: tuple[tuple[str, str, str, str, str], ...] = (
    ("title", "Video", "asc", "A to Z", "Z to A"),
    ("visibility", "Visibility", "asc", "A to Z", "Z to A"),
    ("date", "Date", "desc", "oldest first", "newest first"),
    ("views", "Views", "desc", "fewest first", "most first"),
    ("norm", "vs norm", "desc", "lowest first", "highest first"),
    ("score", "SEO score", "asc", "lowest first", "highest first"),
)
DEFAULT_SORT = "date"
SORT_KEYS = ("fix", *(s[0] for s in SORTS))
SORT_ALIASES = {"newest": "date"}  # links from before the columns were sortable
FIRST_DIRECTION = {s[0]: s[2] for s in SORTS}
DIRECTIONS = ("asc", "desc")
VISIBILITIES: tuple[tuple[str, str], ...] = (
    ("all", "All"),
    ("public", "Public"),
    ("unlisted", "Unlisted"),
    ("private", "Private"),
)
VISIBILITY_KEYS = tuple(v[0] for v in VISIBILITIES)

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# --- formatting -------------------------------------------------------------------------------
def band_of(score: int) -> str:
    """'good' (80+), 'ok' (50-79) or 'poor' (under 50)."""
    if score >= GOOD_FROM:
        return "good"
    return "ok" if score >= OK_FROM else "poor"


def format_count(n: int) -> str:
    """Like YouTube: 999, 1.5K, 21.1K, 1.2M. Rounds down, never up to the next unit."""
    if n < 1000:
        return str(max(n, 0))
    for size, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if n >= size:
            whole, frac = divmod(n * 10 // size, 10)
            return f"{whole}{suffix}" if frac == 0 or whole >= 100 else f"{whole}.{frac}{suffix}"
    return str(n)


def format_date(iso: str) -> str:
    """'2026-10-30T10:00:00Z' -> 'Oct 30, 2026'. Anything unreadable gives ''."""
    try:
        d = date.fromisoformat((iso or "")[:10])
    except ValueError:
        return ""
    return f"{_MONTHS[d.month - 1]} {d.day}, {d.year}"


def thumb_url(video: Video, prefer: Sequence[str] = ("medium", "default")) -> str:
    """The first usable https thumbnail URL among `prefer`, or '' (the page shows a grey box)."""
    thumbs = video.thumbnails if isinstance(video.thumbnails, dict) else {}
    for name in prefer:
        entry = thumbs.get(name)
        url = entry.get("url") if isinstance(entry, dict) else None
        if isinstance(url, str) and url.startswith("https://"):
            return url
    return ""


def bar_class(count: int, total: int) -> str:
    """A width class (w-0 ... w-100, 5% steps; at least w-5 when count > 0)."""
    if total <= 0 or count <= 0:
        return "w-0"
    return f"w-{min(100, max(5, round(100 * count / total / 5) * 5))}"


# why a video has no "vs norm" ratio, by OutlierScore.status
RATIO_WHY = {
    "too_new": f"Too new to compare: videos need {MIN_AGE_DAYS} days of views first",
    "not_public": "Only public videos are compared",
    "no_duration": "Live streams are not compared",
    "few_peers": "Not enough other public videos from around the same time to compare with",
    "no_views": "The videos around it have no views, so there is no norm to compare with",
}


# --- entries ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Entry:
    """A video with its scorecard and the display strings the templates need."""

    video: Video
    card: Scorecard
    score: OutlierScore | None = None  # how its views compare with its peers; set for the content table

    @property
    def is_outlier(self) -> bool:
        s = self.score
        return s is not None and s.status == "scored" and s.ratio is not None and s.ratio >= OUTLIER_THRESHOLD

    @property
    def ratio_text(self) -> str:
        """The views against the norm, as outliers.ratio_label writes it ("—" when there is none)."""
        s = self.score
        return ratio_label(s.ratio if s is not None and s.status == "scored" else None)

    @property
    def ratio_title(self) -> str:
        """The tooltip of the vs norm cell: why there is no ratio, or what the ratio means."""
        s = self.score
        if s is None:
            return ""
        if s.status == "scored" and s.ratio is not None:
            return (f"Views divided by the median views of the {len(s.peer_ids)} public videos "
                    "published nearest to this one")
        return RATIO_WHY.get(s.status if s.status != "scored" else "no_views", "")

    @property
    def band(self) -> str:
        return band_of(self.card.score)

    @property
    def views_text(self) -> str:
        return format_count(self.video.view_count)

    @property
    def date_text(self) -> str:
        return format_date(self.video.published_at)

    @property
    def visibility(self) -> str:
        return (self.video.privacy_status or "").capitalize()

    @property
    def thumb(self) -> str:
        return thumb_url(self.video)

    @property
    def snippet(self) -> str:
        text = " ".join((self.video.description or "").split())
        if len(text) <= DESCRIPTION_SNIPPET_CHARS:
            return text
        return text[:DESCRIPTION_SNIPPET_CHARS].rstrip() + "…"

    @property
    def top_failed(self) -> str:
        failed = self.card.failed
        return failed[0].title if failed else ""

    @property
    def priority(self) -> float:
        return fix_priority(self.video, self.card)


def make_entries(videos: Sequence[Video], cards: Mapping[str, Scorecard]) -> list[Entry]:
    return [Entry(v, cards[v.video_id]) for v in videos if v.video_id in cards]


# --- dashboard --------------------------------------------------------------------------------
def dashboard_model(entries: Sequence[Entry]) -> dict[str, object]:
    public = [e for e in entries if e.video.privacy_status == "public"]
    average: int | None = None
    if public:
        average = int(sum(e.card.score for e in public) / len(public) + 0.5)

    bands = []
    for key, label, rng in BANDS:
        count = sum(1 for e in public if e.band == key)
        bands.append({"key": key, "label": label, "range": rng, "count": count,
                      "bar": bar_class(count, len(public))})

    fixable = [e for e in entries if e.card.failed]
    fixable.sort(key=lambda e: (-e.priority, -e.card.missing, e.video.video_id))

    counts: dict[str, int] = {}
    titles: dict[str, str] = {}
    for e in public:
        for c in e.card.checks:
            if c.applicable and not c.passed:
                counts[c.check_id] = counts.get(c.check_id, 0) + 1
                titles.setdefault(c.check_id, c.title)
    ranked = sorted(counts, key=lambda cid: (-counts[cid], titles[cid], cid))[:PROBLEM_ROWS]
    problems = [{"check_id": cid, "title": titles[cid], "count": counts[cid],
                 "href": "/videos?" + urlencode({"check": cid})} for cid in ranked]

    def by_status(status: str) -> int:
        return sum(1 for e in entries if e.video.privacy_status == status)

    return {
        "average": average,
        "public_count": len(public),
        "bands": bands,
        "fix_next": fixable[:FIX_NEXT_ROWS],
        "problems": problems,
        "totals": {"all": len(entries), "public": len(public), "unlisted": by_status("unlisted"),
                   "private": by_status("private")},
    }


# --- content table ----------------------------------------------------------------------------
def _pick(value: str | None, allowed: Sequence[str], default: str) -> str:
    return value if value in allowed else default


def _page_number(raw: str | None) -> int:
    try:
        return int((raw or "").strip())
    except ValueError:
        return 1


def listing_href(q: str = "", sort: str = DEFAULT_SORT, visibility: str = "all", band: str = "",
                 check: str = "", page: int = 1, outliers: bool = False, direction: str = "") -> str:
    """A /videos URL that carries only the non-default parameters."""
    params: list[tuple[str, str]] = []
    if q:
        params.append(("q", q))
    if sort != DEFAULT_SORT:
        params.append(("sort", sort))
    if sort in FIRST_DIRECTION and direction in DIRECTIONS and direction != FIRST_DIRECTION[sort]:
        params.append(("dir", direction))
    if visibility != "all":
        params.append(("visibility", visibility))
    if band:
        params.append(("band", band))
    if check:
        params.append(("check", check))
    if outliers:
        params.append(("outliers", "1"))
    if page > 1:
        params.append(("page", str(page)))
    return "/videos" + (f"?{urlencode(params)}" if params else "")


@dataclass
class Chip:
    label: str
    href: str
    active: bool


@dataclass
class SortColumn:
    """A column title that sorts the table: `href` sorts by it, or reverses it when it is the active one."""
    key: str
    label: str
    href: str
    active: bool
    direction: str  # the table's direction when active, else ""
    arrow: str  # the direction the arrow shows: the current one, or the one a click gives
    title: str

    @property
    def aria_sort(self) -> str:
        return ("ascending" if self.direction == "asc" else "descending") if self.active else ""


@dataclass
class Listing:
    entries: list[Entry]
    total: int
    page: int
    pages: int
    q: str
    sort: str
    visibility: str
    band: str
    check: str
    check_title: str
    visibility_chips: list[Chip] = field(default_factory=list)
    band_chips: list[Chip] = field(default_factory=list)
    direction: str = ""
    columns: dict[str, SortColumn] = field(default_factory=dict)
    fix_chip: Chip | None = None
    outliers: bool = False
    outlier_chip: Chip | None = None
    prev_href: str = ""
    next_href: str = ""
    clear_check_href: str = ""
    filtered: bool = False

    @property
    def first(self) -> int:
        return (self.page - 1) * PAGE_SIZE + 1 if self.total else 0

    @property
    def last(self) -> int:
        return min(self.page * PAGE_SIZE, self.total)


def _matches(entry: Entry, needle: str) -> bool:
    v = entry.video
    haystack = [v.title or "", v.description or "", *(str(t) for t in (v.tags or []))]
    return any(needle in part.casefold() for part in haystack)


def _failed_check(entry: Entry, check_id: str) -> CheckResult | None:
    for c in entry.card.checks:
        if c.check_id == check_id and c.applicable and not c.passed:
            return c
    return None


def _ratio(entry: Entry) -> float | None:
    s = entry.score
    return s.ratio if s is not None and s.status == "scored" else None


_SORT_KEYS: dict[str, Callable[[Entry], tuple[Any, str]]] = {
    "title": lambda e: ((e.video.title or "").casefold(), e.video.video_id),
    "visibility": lambda e: (e.video.privacy_status or "", e.video.video_id),
    "date": lambda e: (e.video.published_at or "", e.video.video_id),
    "views": lambda e: (e.video.view_count, e.video.video_id),
    "score": lambda e: (e.card.score, e.video.video_id),
}


def _sort(entries: list[Entry], sort: str, direction: str = "") -> list[Entry]:
    if sort == "fix":
        return sorted(entries, key=lambda e: (-e.priority, -e.card.missing, e.video.video_id))
    desc = (direction or FIRST_DIRECTION.get(sort, "desc")) == "desc"
    if sort == "norm":  # videos without a ratio come last either way, in a stable order
        scored = [e for e in entries if _ratio(e) is not None]
        rest = [e for e in entries if _ratio(e) is None]
        scored.sort(key=lambda e: (_ratio(e) or 0.0, e.video.video_id), reverse=desc)
        return scored + sorted(rest, key=lambda e: e.video.video_id)
    return sorted(entries, key=_SORT_KEYS.get(sort, _SORT_KEYS[DEFAULT_SORT]), reverse=desc)


def _column(key: str, label: str, first: str, words: tuple[str, str], sort: str, direction: str,
            href: Callable[..., str]) -> SortColumn:
    active = key == sort
    target = ("asc" if direction == "desc" else "desc") if active else first
    reads = words[0] if target == "asc" else words[1]
    return SortColumn(key=key, label=label, href=href(sort=key, direction=target), active=active,
                      direction=direction if active else "", arrow=direction if active else first,
                      title=f"Order by {label}, {reads}")


def build_listing(entries: Sequence[Entry], params: Mapping[str, str],
                  scores: Mapping[str, OutlierScore] | None = None) -> Listing:
    """Filter, sort and paginate. Unknown values fall back to the defaults; never raises.

    `scores` (video id -> OutlierScore) feeds the vs norm column, its sort and the Outliers filter."""
    q = (params.get("q") or "").strip()[:MAX_QUERY_CHARS]
    raw_sort = params.get("sort")
    sort = _pick(SORT_ALIASES.get(raw_sort or "", raw_sort), SORT_KEYS, DEFAULT_SORT)
    direction = "" if sort == "fix" else _pick(params.get("dir"), DIRECTIONS, FIRST_DIRECTION[sort])
    visibility = _pick(params.get("visibility"), VISIBILITY_KEYS, "all")
    band = _pick(params.get("band"), BAND_KEYS, "")
    check = (params.get("check") or "").strip()[:100]
    only_outliers = (params.get("outliers") or "").strip() == "1"

    rows = list(entries)
    if scores:
        rows = [replace(e, score=scores.get(e.video.video_id)) for e in rows]
    if q:
        needle = q.casefold()
        rows = [e for e in rows if _matches(e, needle)]
    if visibility != "all":
        rows = [e for e in rows if e.video.privacy_status == visibility]
    if band:
        rows = [e for e in rows if e.band == band]
    if only_outliers:
        rows = [e for e in rows if e.is_outlier]
    check_title = ""
    if check:
        for e in entries:
            for c in e.card.checks:
                if c.check_id == check:
                    check_title = c.title
                    break
            if check_title:
                break
        rows = [e for e in rows if _failed_check(e, check) is not None]
    rows = _sort(rows, sort, direction)

    total = len(rows)
    pages = max(1, -(-total // PAGE_SIZE))
    page = min(max(_page_number(params.get("page")), 1), pages)
    shown = rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    def href(**changes: object) -> str:
        args: dict[str, object] = {"q": q, "sort": sort, "visibility": visibility, "band": band,
                                   "check": check, "outliers": only_outliers, "page": 1,
                                   "direction": direction}
        args.update(changes)
        return listing_href(**args)  # type: ignore[arg-type]

    listing = Listing(
        entries=shown, total=total, page=page, pages=pages, q=q, sort=sort, direction=direction,
        visibility=visibility, band=band, check=check, check_title=check_title,
        visibility_chips=[Chip(label, href(visibility=key), key == visibility)
                          for key, label in VISIBILITIES],
        band_chips=[Chip("Any score", href(band=""), band == "")]
        + [Chip(f"{label} ({rng})", href(band=key), key == band) for key, label, rng in BANDS],
        columns={key: _column(key, label, first, (asc, desc), sort, direction, href)
                 for key, label, first, asc, desc in SORTS},
        fix_chip=Chip("Fix next first", href(sort=DEFAULT_SORT, direction="") if sort == "fix"
                      else href(sort="fix", direction=""), sort == "fix"),
        outliers=only_outliers,
        outlier_chip=Chip(f"Outliers ({ratio_label(OUTLIER_THRESHOLD)}+)",
                          href(outliers=not only_outliers), only_outliers),
        prev_href=href(page=page - 1) if page > 1 else "",
        next_href=href(page=page + 1) if page < pages else "",
        clear_check_href=href(check=""),
        filtered=bool(q or check or band or only_outliers or visibility != "all"),
    )
    return listing


# --- video page -------------------------------------------------------------------------------
@dataclass
class CheckGroup:
    key: str
    label: str
    checks: list[CheckResult]
    earned: int
    possible: int


def group_checks(card: Scorecard) -> list[CheckGroup]:
    """Checks grouped in CATEGORIES order (unknown categories last); failed ones first."""
    keys = list(CATEGORIES) + sorted({c.category for c in card.checks} - set(CATEGORIES))

    def rank(c: CheckResult) -> int:
        if not c.applicable:
            return 2
        return 1 if c.passed else 0

    groups: list[CheckGroup] = []
    for key in keys:
        members = [c for c in card.checks if c.category == key]
        if not members:
            continue
        members.sort(key=rank)  # stable: keeps the engine's order inside each rank
        live = [c for c in members if c.applicable]
        groups.append(CheckGroup(
            key=key, label=CATEGORIES.get(key, key.capitalize()), checks=members,
            earned=sum(c.points for c in live), possible=sum(c.max_points for c in live),
        ))
    return groups


# --- changes (drafts), push confirmation and history ------------------------------------------
ORIGIN_LABELS = {"manual": "Edited by you", "restore": "Restored text", "bulk-agent": "Bulk edit agent",
                 "bulk-ask": "Bulk edit by asking"}

# operation -> plain label, for the quota table on the push page
QUOTA_LABELS = {
    "videos.list": "Read the live text of your videos first",
    "videos.update": "Update one video",
    "thumbnails.set": "Set the new thumbnail",
}

# push item status -> (label, pill class)
PUSH_STATUS_LABELS: dict[str, tuple[str, str]] = {
    "pushed": ("Pushed", "done"),
    "conflict": ("Changed in Studio, not pushed", "warn"),
    "failed": ("Failed", "problem"),
    "skipped": ("Skipped", "warn"),
    "missing": ("Missing on YouTube", "problem"),
    "not_attempted": ("Not attempted", ""),
}
HISTORY_STATUS_LABELS: dict[str, tuple[str, str]] = {
    "pushed": ("Pushed", "done"),
    "conflict": ("Changed in Studio, not pushed", "warn"),
    "failed": ("Failed", "problem"),
    "pending": ("Unfinished", "warn"),
}


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def summarize_diff(diff: DraftDiff) -> str:
    """One line on what a draft changes, for example 'title; description (2 lines added)'."""
    parts: list[str] = []
    if diff.title_changed:
        parts.append("title")
    if diff.description.changed:
        bits = []
        if diff.description.added:
            bits.append(f"{_count(diff.description.added, 'line')} added")
        if diff.description.removed:
            bits.append(f"{diff.description.removed} removed")
        parts.append("description" + (f" ({', '.join(bits)})" if bits else ""))
    if diff.tags.changed:
        bits = []
        if diff.tags.added:
            bits.append(f"{len(diff.tags.added)} added")
        if diff.tags.removed:
            bits.append(f"{len(diff.tags.removed)} removed")
        if diff.tags.reordered:
            bits.append("new order")
        parts.append("tags" + (f" ({', '.join(bits)})" if bits else ""))
    if diff.language is not None:
        parts.append("video language")
    loc = diff.localizations
    if loc.changed:
        bits = [f"{len(names)} {word}" for names, word in
                ((loc.added, "added"), (loc.changed_langs, "changed"), (loc.removed, "removed"))
                if names]
        parts.append(f"translations ({', '.join(bits)})")
    return "Changes: " + "; ".join(parts) if parts else "No changes."


@dataclass(frozen=True)
class ChangeCard:
    """A draft with everything the changes page and the push page show about it."""

    row: DraftRow
    video: Video
    diff: DraftDiff
    errors: tuple[Issue, ...]
    warnings: tuple[Issue, ...]
    stale: bool
    ready: bool  # approved, current and error-free: this one would be pushed
    approve_block: str  # why Approve is unavailable ("" when it is available)
    fingerprint: str
    summary: str
    origin: str
    unchecked: tuple[str, ...] = ()  # translations going out that no check passed

    @property
    def video_id(self) -> str:
        return self.row.video_id

    @property
    def thumb(self) -> str:
        return thumb_url(self.video)

    @property
    def views_text(self) -> str:
        return format_count(self.video.view_count)

    @property
    def issues(self) -> tuple[Issue, ...]:
        return self.errors + self.warnings

    @property
    def approved(self) -> bool:
        return self.ready

    @property
    def sends_translations(self) -> bool:
        return self.diff.localizations.changed

    @property
    def sends_language(self) -> bool:
        return self.diff.language is not None


def unchecked_translations(row: DraftRow, records: Iterable[Any]) -> tuple[str, ...]:
    """Sentences for each translation this draft would send that no check passed: the text in the
    draft matches a stored translation marked flagged or unverified. Text the creator typed has no record and no sentence."""
    sent = row.localizations or {}
    found = []
    for r in records:
        key = next((k for k in sent if same_language(k, r.language)), None)
        if key is None or r.status not in ("flagged", "unverified"):
            continue
        entry = sent[key]
        if entry.get("title") == r.title and entry.get("description") == r.description \
                and row.base_localizations.get(key) != entry:
            found.append(f"{lang_name(key)}: {'failed a check' if r.status == 'flagged' else 'never checked'}")
    return tuple(found)


def build_change_card(row: DraftRow, video: Video) -> ChangeCard:
    issues = edits.issues_for(row)
    errors = tuple(i for i in issues if i.severity == "error")
    warnings = tuple(i for i in issues if i.severity != "error")
    stale = edits.is_stale(row, video)
    block = ""
    if stale:
        block = "Update the draft to YouTube's new text first."
    elif has_errors(issues):
        block = "Fix the errors first."
    diff = diff_row(row)
    # the video's own language that Studio lists among the translations is not a translation
    own = row.default_language or row.base_default_language
    if own is not None and diff.localizations.kept:
        kept = tuple(k for k in diff.localizations.kept if not same_language(k, own))
        diff = replace(diff, localizations=replace(diff.localizations, kept=kept))
    if diff.title_changed or diff.description.changed:
        untouched = list(row.base_localizations) if row.localizations is None else list(diff.localizations.kept)
        untouched = [k for k in untouched if own is None or not same_language(k, own)]
        if untouched:
            n = len(untouched)
            warnings = warnings + (Issue(
                "localizations", "translations_out_of_date", "warning",
                f"The new title or description goes to YouTube with {n} translation{'' if n == 1 else 's'} "
                f"made from the old text: {', '.join(lang_name(k) for k in untouched)}. Translate "
                f"{'it' if n == 1 else 'them'} again on the video page first, or "
                f"{'it stays' if n == 1 else 'they stay'} out of date."),)
    origin = row.note or ORIGIN_LABELS.get(row.source, row.source.capitalize())
    return ChangeCard(
        row=row, video=video, diff=diff, errors=errors, warnings=warnings, stale=stale,
        ready=row.approved_at is not None and not block, approve_block=block,
        fingerprint=edits.fingerprint(row), summary=summarize_diff(diff), origin=origin,
    )


def change_cards(pairs: Iterable[tuple[DraftRow, Video]],
                 translations: Callable[[str], Iterable[Any]] | None = None) -> list[ChangeCard]:
    """`translations` gives a video's latest stored translations, to warn about unchecked ones."""
    cards = []
    for row, video in pairs:
        card = build_change_card(row, video)
        unchecked = unchecked_translations(row, translations(row.video_id)) if translations else ()
        if unchecked:
            card = replace(card, unchecked=unchecked, warnings=card.warnings + (Issue(
                "localizations", "translations_unchecked", "warning",
                "You ticked translations that no check passed: " + "; ".join(unchecked)
                + ". They go to YouTube as they are."),))
        cards.append(card)
    return cards


def plan_lines(lines: Iterable[tuple[str, int, int]]) -> list[dict[str, Any]]:
    return [{"label": QUOTA_LABELS.get(op, op), "operation": op, "count": count, "units": units}
            for op, count, units in lines]


def push_results(items: Iterable[PushItem], titles: Mapping[str, str]) -> list[dict[str, Any]]:
    """Rows for the result list of a finished push."""
    rows = []
    for item in items:
        label, tone = PUSH_STATUS_LABELS.get(item.status, (item.status, ""))
        rows.append({"video_id": item.video_id, "title": titles.get(item.video_id) or item.video_id,
                     "status": item.status, "label": label, "tone": tone,
                     "message": item.message})
    return rows


@dataclass(frozen=True)
class Backup:
    title: str
    description: str
    tags: tuple[str, ...]
    languages: tuple[str, ...] = ()  # translations the backup holds (not the video's own language)


def backup_of_record(record: PushRecord) -> Backup | None:
    """The saved text of a push record, or None when it has no title to restore."""
    before = record.before
    title = before.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    description = before.get("description")
    tags = before.get("tags")
    localizations = before.get("localizations")
    own = before.get("defaultLanguage")
    languages = tuple(
        sorted(k for k in localizations
               if not (isinstance(own, str) and same_language(k, own)))
    ) if isinstance(localizations, dict) else ()
    return Backup(
        title,
        description if isinstance(description, str) else "",
        tuple(t for t in tags if isinstance(t, str)) if isinstance(tags, list) else (),
        languages,
    )


@dataclass(frozen=True)
class HistoryRow:
    record: PushRecord
    title: str
    when: str
    label: str
    tone: str
    backup: Backup | None
    has_draft: bool  # restore is refused while this video has a draft


def history_rows(records: Iterable[PushRecord], titles: Mapping[str, str],
                 fmt_time: Callable[[str], str],
                 draft_ids: Collection[str] = ()) -> list[HistoryRow]:
    rows = []
    for record in records:
        label, tone = HISTORY_STATUS_LABELS.get(record.status, (record.status, ""))
        rows.append(HistoryRow(
            record=record, title=titles.get(record.video_id) or record.video_id,
            when=fmt_time(record.started_at), label=label, tone=tone,
            backup=backup_of_record(record), has_draft=record.video_id in draft_ids,
        ))
    return rows


@dataclass(frozen=True)
class HistoryGroup:
    """The History rows of one push run, or a single older push (run_id None, no header)."""

    run_id: int | None
    header: str  # "Push of <time> · N videos · K changed"; "" without a run
    can_undo: bool  # more than one row of the run reached YouTube (one video has its own Restore)
    rows: list[HistoryRow]


def _videos(n: int) -> str:
    return f"{n} video" + ("" if n == 1 else "s")


def history_groups(records: Iterable[PushRecord], runs: Mapping[int, PushRun],
                   run_records: Mapping[int, Sequence[PushRecord]],
                   titles: Mapping[str, str], fmt_time: Callable[[str], str],
                   draft_ids: Collection[str] = ()) -> list[HistoryGroup]:
    """Group the newest-first push records by run, in the order their newest record appears.

    A run shows all its records (`run_records`, oldest first), even those beyond the page limit.
    Pushes without a run stay one group each: old groups are never guessed.
    """
    groups: list[HistoryGroup] = []
    seen: set[int] = set()
    for record in records:
        if record.run_id is None:
            groups.append(HistoryGroup(
                None, "", False, history_rows([record], titles, fmt_time, draft_ids)))
            continue
        if record.run_id in seen:
            continue
        seen.add(record.run_id)
        members = list(reversed(run_records.get(record.run_id) or [record]))  # newest first
        changed = sum(1 for m in members if m.status == "pushed")
        run = runs.get(record.run_id)
        started = fmt_time(run.started_at if run is not None else members[-1].started_at)
        count = run.video_count if run is not None and run.video_count else len(members)
        groups.append(HistoryGroup(
            record.run_id, f"Push of {started} · {_videos(count)} · {changed} changed",
            changed > 1, history_rows(members, titles, fmt_time, draft_ids)))
    return groups


MAX_LISTED_SKIPS = 5


def run_restore_summary(result: edits.RunRestore, titles: Mapping[str, str]) -> str:
    """One flash sentence for an undone push run: how many drafts, and which videos were skipped."""
    drafted = result.count("drafted")
    text = (f"Restore drafts made for {_videos(drafted)}." if drafted
            else "No restore drafts were made.")
    same = result.count("unchanged")
    if same:
        text += f" {_videos(same)} already had that text on YouTube."
    skipped = [i for i in result.items if i.status == "skipped"]
    if skipped:
        listed = [f"{titles.get(i.video_id) or i.video_id}: {i.message}"
                  for i in skipped[:MAX_LISTED_SKIPS]]
        if len(skipped) > MAX_LISTED_SKIPS:
            listed.append(f"and {len(skipped) - MAX_LISTED_SKIPS} more")
        text += f" {len(skipped)} skipped: " + "; ".join(listed) + "."
    return text


# --- AI suggestions: agent runs, the fix-next batch -------------------------------------------
# run status -> (label, pill class)
RUN_STATUS_LABELS: dict[str, tuple[str, str]] = {
    "drafted": ("Draft made", "done"),
    "no_change": ("No change needed", ""),
    "rejected": ("Stopped by the checks", "warn"),
    "failed": ("Failed", "problem"),
    "running": ("Running", "working"),
}
# batch item status -> (label, pill class): the run statuses plus two that never reach a run
ITEM_STATUS_LABELS: dict[str, tuple[str, str]] = {
    **RUN_STATUS_LABELS,
    "skipped": ("Skipped", "warn"),
    "not_attempted": ("Not attempted", ""),
}
FIELD_LABELS = {"title": "Title", "description": "Description", "tags": "Tags"}
STOP_ERROR_KINDS = ("auth", "rate_limit")


def format_cost(cost: float | None) -> str:
    """'$0.012', or "unknown" for a model without a known price."""
    return "unknown" if cost is None else f"${cost:.3f}"


@dataclass(frozen=True)
class RunChange:
    field: str
    check: str
    reason: str


@dataclass(frozen=True)
class RunToolCall:
    name: str
    arguments: str  # compact JSON (or the raw text when the model sent something unparsable)
    result_chars: int


@dataclass(frozen=True)
class RunView:
    """An agent run with everything the video page, the Changes page and the Agent log show."""

    run: AgentRun
    title: str
    when: str
    label: str
    tone: str
    changes: tuple[RunChange, ...]
    not_fixed: tuple[RunChange, ...]
    tool_calls: tuple[RunToolCall, ...]
    problems: tuple[str, ...]
    errors: tuple[str, ...]
    tokens: str
    cost: str

    @property
    def has_details(self) -> bool:
        return bool(self.tool_calls or self.problems or self.errors)


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def run_view(run: AgentRun, titles: Mapping[str, str], check_titles: Mapping[str, str],
             fmt_time: Callable[[str], str], *, interrupted: bool = False) -> RunView:
    """A run as plain display data. `interrupted`: it says running but no job is running."""
    label, tone = RUN_STATUS_LABELS.get(run.status, (run.status, ""))
    if interrupted and run.status == "running":
        label, tone = "Interrupted", "warn"

    def check_title(check_id: Any) -> str:
        cid = str(check_id or "")
        return check_titles.get(cid) or cid or "General"

    result = run.result if isinstance(run.result, dict) else {}
    changes = tuple(
        RunChange(FIELD_LABELS.get(str(c.get("field")), str(c.get("field") or "")),
                  check_title(c.get("check_id")), str(c.get("reason") or ""))
        for c in _dicts(result.get("changes")))
    not_fixed = tuple(RunChange("", check_title(c.get("check_id")), str(c.get("reason") or ""))
                      for c in _dicts(result.get("not_fixed")))
    tool_calls: list[RunToolCall] = []
    problems: list[str] = []
    errors: list[str] = []
    for event in _dicts(run.log):
        kind = event.get("type")
        if kind == "tool":
            args = event.get("arguments")
            text = args if isinstance(args, str) else json.dumps(
                args if args is not None else {}, separators=(",", ":"), ensure_ascii=False)
            chars = event.get("result_chars")
            tool_calls.append(RunToolCall(str(event.get("name") or ""), text,
                                          chars if isinstance(chars, int) else 0))
        elif kind == "problems":
            problems.extend(str(p) for p in (event.get("problems") or []))
        elif kind == "error":
            errors.append(str(event.get("message") or event.get("kind") or "Unknown error"))
    cost = estimate_cost(run.model, run.input_tokens, run.cached_tokens, run.output_tokens)
    return RunView(
        run=run, title=titles.get(run.video_id) or run.video_id,
        when=fmt_time(run.started_at), label=label, tone=tone, changes=changes,
        not_fixed=not_fixed, tool_calls=tuple(tool_calls), problems=tuple(problems),
        errors=tuple(errors),
        tokens=(f"{run.input_tokens:,} tokens in ({run.cached_tokens:,} cached), "
                f"{run.output_tokens:,} out"),
        cost=format_cost(cost),
    )


def stops_batch(run: AgentRun | None) -> bool:
    """True when a failed run ended on an auth or rate-limit error: more runs would fail too."""
    events = _dicts(run.log) if run is not None else []
    last = events[-1] if events else {}
    return last.get("type") == "error" and last.get("kind") in STOP_ERROR_KINDS


@dataclass(frozen=True)
class SuggestItem:
    """One video of a suggestion job and what became of it."""

    video_id: str
    title: str
    status: str  # a key of ITEM_STATUS_LABELS
    message: str = ""

    @property
    def label(self) -> str:
        return ITEM_STATUS_LABELS.get(self.status, (self.status, ""))[0]

    @property
    def tone(self) -> str:
        return ITEM_STATUS_LABELS.get(self.status, (self.status, ""))[1]


@dataclass(frozen=True)
class SuggestBatch:
    """The result of a suggestion job: one item per video, and why it stopped early (or '')."""

    items: tuple[SuggestItem, ...]
    stopped: str = ""

    def count(self, status: str) -> int:
        return sum(1 for i in self.items if i.status == status)


# --- translations: the video page card ----------------------------------------------------------
# record status -> (label, pill class)
TRANSLATION_STATUS_LABELS: dict[str, tuple[str, str]] = {
    "verified": ("Verified", "done"),
    "flagged": ("Flagged", "warn"),
    "unverified": ("Unverified", "unverified"),
    "failed": ("Failed", "problem"),
}


def translation_label(record: TranslationRecord | SubtitleTranslationRecord) -> tuple[str, str]:
    """(label, pill class) of a stored translation. One the same model translated and checked (Muse
    Glimmer when Primary AI is Local) is "Self-checked", not "Verified"."""
    if record.status == "verified" and record.verifier and record.verifier == record.translator:
        return "Self-checked", "working"
    return TRANSLATION_STATUS_LABELS.get(record.status, (record.status, ""))


def lang_name(code: str | None) -> str:
    """'Spanish' for es; 'German (de-DE)' for a regional code; the code itself when unknown."""
    if not code:
        return "Not set"
    if code in LANGUAGES:
        return LANGUAGES[code]
    base = code.split("-", 1)[0]
    return f"{LANGUAGES[base]} ({code})" if base in LANGUAGES else code


@dataclass(frozen=True)
class SourceOption:
    code: str
    name: str
    selected: bool


@dataclass(frozen=True)
class TargetOption:
    code: str
    name: str
    ticked: bool
    on_youtube: str  # "", "on YouTube" or "on YouTube as de-DE"
    state: str = ""  # "own" (the video's language), "translated", "stale", "youtube", "failed", "" (none)
    chip: str = ""  # the state as a pill: "Translated", "Out of date", "On YouTube", "Check failed"
    tone: str = ""  # the pill's class


# A target language's state on the Translations card: its pill text and class.
TARGET_STATES: dict[str, tuple[str, str]] = {
    "translated": ("Translated", "done"), "stale": ("Out of date", "warn"),
    "youtube": ("On YouTube", "working"), "failed": ("Check failed", "problem"),
}


@dataclass(frozen=True)
class TranslationView:
    """The latest translation into one language, ready to show."""

    record: TranslationRecord
    name: str
    label: str
    tone: str
    score: str  # "4/5", or "" when the checker did not score it
    outdated: bool  # made from other text than the video's current text
    addable: bool
    ticked: bool
    needs_check: bool  # flagged or unverified: the creator must say "I checked this translation"
    key: str  # the YouTube key it would be saved under (de-DE when Studio already has that)
    in_draft: bool
    note: str  # why the checker could not check it ("" when it could)
    back_title: str
    back_description: str


@dataclass(frozen=True)
class DraftTranslation:
    key: str
    name: str
    state: str  # "New in this draft", "Changed in this draft" or "On YouTube"


@dataclass(frozen=True)
class TranslateView:
    video_id: str
    language_code: str  # the video's language ("" when not set)
    language_label: str
    sources: tuple[SourceOption, ...]
    language_known: bool  # the video language is one of the choices, so it is preselected
    targets: tuple[TargetOption, ...]
    results: tuple[TranslationView, ...]
    addable_count: int
    in_draft: tuple[DraftTranslation, ...]
    stale_names: tuple[str, ...] = ()  # target languages whose translation is out of date
    stale_codes: tuple[str, ...] = ()
    failed_names: tuple[str, ...] = ()  # target languages whose last translation failed its check
    failed_codes: tuple[str, ...] = ()
    summary: str = ""  # "2 translated · 1 out of date · 3 on YouTube · 18 not translated"


def _same_text(loc: Mapping[str, str] | None, record: TranslationRecord) -> bool:
    return loc is not None and loc == {
        "title": record.title.strip(),
        "description": record.description.replace("\r\n", "\n").replace("\r", "\n"),
    }


def stale_youtube_keys(video: Video, records: Mapping[str, TranslationRecord]) -> frozenset[str]:
    """Translations on YouTube that are Yezatas' text made from another title or description than the
    one on YouTube now: the title or description changed after they were pushed."""
    current = source_hash(video.title, video.description)
    youtube = edits.plain_localizations(video.localizations)
    keys = list(youtube)
    stale: set[str] = set()
    for code, record in records.items():
        if record.status == "failed" or record.source_hash == current:
            continue
        key = youtube_key(code, keys)
        if _same_text(youtube.get(key), record):
            stale.add(key)
    return frozenset(stale)


def build_translate_view(video: Video, row: DraftRow | None,
                         records: Mapping[str, TranslationRecord],
                         stale_keys: frozenset[str] = frozenset()) -> TranslateView:
    """What the translation card shows: the language choice, the results and the draft's translations."""
    language = (row.default_language if row is not None and row.default_language else None) \
        or video.default_language
    youtube = edits.plain_localizations(video.localizations)
    own = (lambda key: bool(language) and same_language(key, language))
    youtube_keys = [k for k in youtube if not own(k)]
    in_draft_map = (row.localizations if row is not None and row.localizations is not None
                    else None)
    pool = list(in_draft_map if in_draft_map is not None else youtube)

    selected = next((code for code in LANGUAGES if language and same_language(code, language)), "")
    sources = tuple(SourceOption(code, name, code == selected) for code, name in LANGUAGES.items())
    text = (row.title, row.description) if row is not None else (video.title, video.description)
    current = source_hash(*text)
    # A draft with another title or description makes every translation on YouTube out of date too.
    text_changed = row is not None and current != source_hash(video.title, video.description)
    targets = []
    for code, name in LANGUAGES.items():
        key = youtube_key(code, youtube_keys)
        on_youtube = key in youtube_keys
        mark = "" if not on_youtube else ("on YouTube" if key == code else f"on YouTube as {key}")
        record = records.get(code)
        if code == selected:
            state = "own"
        elif record is not None and record.status != "failed":
            state = "translated" if record.source_hash == current else "stale"
        elif on_youtube:
            state = "stale" if text_changed or key in stale_keys else "youtube"
        elif record is not None:
            state = "failed"
        else:
            state = ""
        chip, tone = TARGET_STATES.get(state, ("", ""))
        targets.append(TargetOption(code, name, code in DEFAULT_TARGETS and code != selected, mark,
                                    state, chip, tone))
    counted = [(n, label) for n, label in (
        (sum(1 for t in targets if t.state == s), label) for s, label in (
            ("translated", "translated"), ("stale", "out of date"), ("youtube", "on YouTube"),
            ("failed", "check failed"), ("", "not translated"))) if n]
    ordered = [c for c in LANGUAGES if c in records] + sorted(c for c in records if c not in LANGUAGES)
    results = []
    for code in ordered:
        record = records[code]
        label, tone = translation_label(record)
        outdated = record.source_hash != current
        addable = not outdated and record.status != "failed"
        key = youtube_key(code, pool)
        present = in_draft_map is not None and _same_text(in_draft_map.get(key), record)
        back = record.back_translation if isinstance(record.back_translation, dict) else {}
        results.append(TranslationView(
            record=record, name=lang_name(code), label=label, tone=tone,
            score=f"{record.score}/5" if record.score is not None else "",
            outdated=outdated, addable=addable,
            ticked=addable and record.status == "verified" and not present,
            needs_check=record.status in ("flagged", "unverified"), key=key, in_draft=present,
            note=str(back.get("note") or ""),
            back_title=str(back.get("title") or ""),
            back_description=str(back.get("description") or ""),
        ))

    listed = []
    if row is not None and row.localizations is not None:
        loc = diff_row(row).localizations
        for key in row.localizations:
            if own(key):
                continue
            state = ("New in this draft" if key in loc.added
                     else "Changed in this draft" if key in loc.changed_langs else "On YouTube")
            listed.append(DraftTranslation(key, lang_name(key), state))
    return TranslateView(
        video_id=video.video_id, language_code=language or "", language_label=lang_name(language),
        sources=sources, language_known=bool(selected), targets=tuple(targets),
        results=tuple(results), addable_count=sum(1 for r in results if r.addable),
        in_draft=tuple(listed),
        stale_names=tuple(t.name for t in targets if t.state == "stale"),
        stale_codes=tuple(t.code for t in targets if t.state == "stale"),
        failed_names=tuple(t.name for t in targets if t.state == "failed"),
        failed_codes=tuple(t.code for t in targets if t.state == "failed"),
        summary=" · ".join(f"{n} {label}" for n, label in counted),
    )


# --- transcripts and chapters ---------------------------------------------------------------------
@dataclass(frozen=True)
class TranscriptView:
    """What the "Transcript and chapters" card shows besides the job state."""

    video_language_label: str  # the video's own language ("Not set" when it has none)
    language_known: bool  # that language is one transcription supports, so it is preselected
    language_set: bool  # the video has a language at all
    languages: tuple[SourceOption, ...]
    price: str  # "$0.18"
    asr_model: str  # the speech model, whatever made the latest transcript
    model: str  # what made the latest transcript (the speech model when there is none)
    record: TranscriptRecord | None
    free: bool  # the latest transcript came from YouTube's captions or the sound tagger: no cost
    spoken: str  # the language the transcript was made in, by name
    when: str
    length: str
    cost: str
    text: str  # the m:ss lines
    line_count: int
    blocker: str  # why "Make chapters" is off ("" when it is on)


def _base_code(code: str | None) -> str:
    return (code or "").replace("_", "-").split("-", 1)[0].lower()


def chapters_blocker(video: Video, row: DraftRow | None, record: TranscriptRecord | None) -> str:
    """Why chapters cannot be made now, in a sentence, or "" when they can."""
    if record is None:
        return "Transcribe the video first."
    if record.model == SOUNDS_MODEL:
        return "This transcript has sound labels, not speech, so there is nothing to make chapters from."
    seconds = parse_duration(video.duration)
    if seconds is None:
        seconds = record.duration_ms // 1000
    if seconds < CHAPTERS_MIN_DURATION:
        return "Chapters only help on videos longer than a minute; this one is shorter."
    text = row.description if row is not None else video.description
    stamps = parse_chapters(text)
    if stamps:
        problem = chapter_problem(stamps)
        if problem:
            return (f"The description already has timestamps, but YouTube will not show them as "
                    f"chapters. {problem} Yezatas never rewrites your own chapters: fix them in "
                    "the description, or remove them to make new ones here.")
        return ("The description already has timestamps. Yezatas never rewrites your own "
                "chapters; remove them first if you want new ones.")
    if not any(str(s.get("text") or "").strip() for s in record.segments):
        return "The transcript has no text to make chapters from."
    return ""


def _spoken(record: TranscriptRecord | None) -> str:
    """The language of a transcript by name; a sound-only transcript has none."""
    if record is None:
        return ""
    if not record.language:
        return "Sounds, no speech"
    return ASR_LANGUAGES.get(record.language) or caption_language_name(record.language)


def build_transcript_view(video: Video, row: DraftRow | None, record: TranscriptRecord | None,
                          when: str = "") -> TranscriptView:
    """The language choice, the latest transcript as `m:ss text` lines and the chapters check."""
    code = _base_code(video.default_audio_language) or _base_code(video.default_language)
    selected = code if code in ASR_LANGUAGES else ""
    languages = tuple(SourceOption(c, n, c == selected) for c, n in ASR_LANGUAGES.items())
    lines = []
    if record is not None:
        for segment in record.segments:
            text = " ".join(str(segment.get("text") or "").split())
            if text:
                lines.append(f"{clock(int(segment['start_ms']))} {text}")
    return TranscriptView(
        video_language_label=lang_name(video.default_audio_language or video.default_language),
        language_known=bool(selected), language_set=bool(code), languages=languages,
        price=f"${ASR_PRICE_PER_HOUR:.2f}", asr_model=ASR_MODEL,
        model=record.model if record is not None else ASR_MODEL,
        record=record, free=record is not None and record.model in (CAPTIONS_MODEL, SOUNDS_MODEL),
        spoken=_spoken(record),
        when=when, length=clock(record.duration_ms) if record is not None else "",
        cost=(format_cost(record.duration_ms / 3_600_000 * ASR_PRICE_PER_HOUR)
              if record is not None else ""),
        text="\n".join(lines), line_count=len(lines),
        blocker=chapters_blocker(video, row, record),
    )


# --- translated subtitles -------------------------------------------------------------------------
@dataclass(frozen=True)
class SubtitleResult:
    """The newest subtitles in one language, ready to show."""

    record: SubtitleTranslationRecord
    name: str
    label: str
    tone: str
    score: str  # "4/5", or "" when the checker did not score it
    problems: tuple[str, ...]  # found by the code checks
    issues: tuple[str, ...]  # found by the checker
    note: str  # why the checker could not check it ("" when it could)
    back: tuple[tuple[int, str], ...]  # the blind back-translation: (segment number, text)
    models: str  # "spark-x, checked by glimmer-y"
    tokens: str  # "1,200 in, 3,400 out", or "" when none were counted
    url: str  # where to download the .srt ("" for a failed language)


@dataclass(frozen=True)
class SubtitlesView:
    """What the "Subtitles in other languages" block shows besides the job state."""

    source_label: str  # the transcript's language, by name
    targets: tuple[TargetOption, ...]
    results: tuple[SubtitleResult, ...]


def build_subtitles_view(video_id: str, transcript: TranscriptRecord | None,
                         records: Mapping[str, SubtitleTranslationRecord]) -> SubtitlesView:
    """The language choice and the stored results for the newest transcript, in LANGUAGES order.

    The boxes start ticked as the default languages when nothing was translated yet, and as the
    languages that have a result otherwise.
    """
    source = transcript.language if transcript is not None else ""
    options = [(code, name) for code, name in LANGUAGES.items()
               if not (source and same_language(code, source))]
    ticked = ({code for code in records} if records
              else {code for code in DEFAULT_TARGETS if code in dict(options)})
    targets = tuple(TargetOption(code, name, code in ticked, "") for code, name in options)
    results = []
    for code in LANGUAGES:
        record = records.get(code)
        if record is None:
            continue
        label, tone = translation_label(record)
        back = record.back_translation if isinstance(record.back_translation, dict) else {}
        pieces = back.get("segments")
        results.append(SubtitleResult(
            record=record, name=lang_name(code), label=label, tone=tone,
            score=f"{record.score}/5" if record.score is not None else "",
            problems=tuple(record.problems), issues=tuple(record.issues),
            note=str(back.get("note") or ""),
            back=tuple((int(p["n"]), str(p["text"])) for p in pieces
                       if isinstance(p, dict) and "n" in p and "text" in p)
            if isinstance(pieces, list) else (),
            models=record.translator + (f", checked by {record.verifier}" if record.verifier else ""),
            tokens=(f"{record.input_tokens:,} in, {record.output_tokens:,} out"
                    if record.input_tokens or record.output_tokens else ""),
            url="" if record.status == "failed" else (
                f"/videos/{quote(video_id)}/subtitles/{quote(code)}.srt"),
        ))
    return SubtitlesView(
        source_label=ASR_LANGUAGES.get(source, lang_name(source)) if source else "Not set",
        targets=targets, results=tuple(results))


# --- packaging check ------------------------------------------------------------------------------
VARIANT_LABELS = {"ok": ("Made", "done"), "flagged": ("Text to check", "warn"),
                  "failed": ("Failed", "problem")}
PREVIEW_OTHERS = 7  # the channel's own top videos shown next to the one being judged
PREVIEW_SEARCH_ROWS = 3
PREVIEW_PHONE_ROWS = 3
DESCRIPTION_PREVIEW_CHARS = 150


@dataclass(frozen=True)
class PackagingFact:
    label: str
    value: str


@dataclass(frozen=True)
class PackagingIdea:
    instruction: str
    overlay: str  # the words to put on the picture; "" keeps the current text


@dataclass(frozen=True)
class VariantView:
    """One variant idea, ready to show. The URLs are empty for a variant that has no picture."""

    id: int
    label: str
    tone: str
    instruction: str
    overlay: str
    image_url: str
    download_url: str
    replace_url: str
    preview_url: str
    text_badge: str  # "Text matches", 'Text differs: reads "..."' or "Text not checked"
    text_tone: str
    subject_change: str  # "Subject share: 12% -> 31%", or ""
    error: str
    cost: str
    model: str


@dataclass(frozen=True)
class PackagingView:
    """What the Packaging card shows besides the job state."""

    title: str  # the title next to the thumbnail
    thumb: str
    review: PackagingReview | None
    when: str
    failed: str  # the error sentence of a failed check, "" otherwise
    facts: tuple[PackagingFact, ...]  # measured by code
    reading: tuple[PackagingFact, ...]  # read by the model
    points: tuple[str, ...]
    ideas: tuple[PackagingIdea, ...]
    problems: tuple[str, ...]  # what the fact guard dropped
    cost: str
    models: str
    tokens: str
    can_variants: bool
    variants: tuple[VariantView, ...]
    preview_url: str


def _whole(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _share_fact(found: Any) -> str:
    """The share of the frame a segmenter box takes, or why there is no number."""
    if not isinstance(found, dict):
        return "not measured"
    share = _whole(found.get("share_pct"))
    return f"{share}% of the frame" if found.get("found") and share is not None else "not found"


def _packaging_facts(review: PackagingReview) -> tuple[tuple[PackagingFact, ...],
                                                       tuple[PackagingFact, ...]]:
    """(what code measured, what the model read), as plain facts."""
    m = review.measurements if isinstance(review.measurements, dict) else {}
    r = review.reading if isinstance(review.reading, dict) else {}
    measured: list[PackagingFact] = []
    width, height = _whole(m.get("width")), _whole(m.get("height"))
    if width is not None and height is not None:
        measured.append(PackagingFact("Size", f"{width} x {height} pixels"))
    brightness = _whole(m.get("brightness"))
    if brightness is not None:
        measured.append(PackagingFact("Brightness", f"{brightness} out of 255 on average"))
    contrast = _whole(m.get("contrast"))
    if contrast is not None:
        measured.append(PackagingFact("Contrast", f"{contrast} (the spread of brightness; 0 is flat)"))
    measured.append(PackagingFact("Subject share", _share_fact(m.get("subject"))))
    if "face" in m:
        measured.append(PackagingFact("Face share", _share_fact(m.get("face"))))
    else:
        measured.append(PackagingFact("Face share", "not measured" if r.get("face") is True
                                      else "no face seen"))
    if isinstance(m.get("badge_risk"), bool):
        measured.append(PackagingFact("Text or subject in the duration-badge corner",
                                      "yes" if m["badge_risk"] else "no"))
    zone = str(m.get("text_zone") or r.get("text_zone") or "none")
    measured.append(PackagingFact("Text zone", "no text" if zone == "none" else zone))
    text = str(r.get("text") or "")
    subject = str(r.get("subject") or "")
    read = (PackagingFact("Text on the thumbnail", f"“{text}”" if text else "none"),
            PackagingFact("Subject", subject or "none found"),
            PackagingFact("Face", "yes" if r.get("face") is True else "no"))
    return tuple(measured), read


def _variant_view(video_id: str, variant: ThumbnailVariant) -> VariantView:
    label, tone = VARIANT_LABELS.get(variant.status, (variant.status, ""))
    made = variant.status != "failed"
    base = f"/videos/{quote(video_id)}/packaging/variants/{variant.id}.jpg"
    if variant.text_ok is True:
        badge, text_tone = "Text matches", "done"
    elif variant.text_ok is False:
        read = f"reads “{variant.text_read}”" if variant.text_read else "no text was read"
        badge, text_tone = f"Text differs: {read}", "warn"
    else:
        badge, text_tone = "Text not checked", "unverified"
    change = ""
    subject = variant.measurements.get("subject") if isinstance(variant.measurements, dict) else None
    if isinstance(subject, dict):
        before = _whole(subject.get("before_share_pct"))
        after = subject.get("after")
        share = _whole(after.get("share_pct")) if isinstance(after, dict) and after.get("found") else None
        was = "not measured" if before is None else f"{before}%"
        if share is None:
            change = f"Subject share: {was} before; not found in the variant"
        else:
            change = f"Subject share: {was} → {share}%"
    return VariantView(
        id=variant.id, label=label, tone=tone, instruction=variant.instruction,
        overlay=variant.overlay_text, image_url=base if made else "",
        download_url=base + "?download=1" if made else "",
        replace_url=f"/videos/{quote(video_id)}/packaging/variants/{variant.id}/replace" if made else "",
        preview_url=f"/videos/{quote(video_id)}/packaging?variant={variant.id}" if made else "",
        text_badge=badge if made else "", text_tone=text_tone, subject_change=change if made else "",
        error=variant.error, cost=format_cost(variant.cost_usd), model=variant.model)


def build_packaging_view(video_id: str, title: str, thumb: str, review: PackagingReview | None,
                         variants: Sequence[ThumbnailVariant],
                         when: Callable[[str], str] = lambda iso: iso) -> PackagingView:
    """The Packaging card: the newest check of the video and the variants made from it."""
    preview_url = f"/videos/{quote(video_id)}/packaging"
    if review is None:
        return PackagingView(title, thumb, None, "", "", (), (), (), (), (), "", "", "", False, (),
                             preview_url)
    done = review.status == "done"
    facts, read = _packaging_facts(review) if done else ((), ())
    critique = review.critique if isinstance(review.critique, dict) else {}
    points = tuple(str(p) for p in critique.get("points") or () if isinstance(p, str))
    ideas = tuple(PackagingIdea(str(i.get("instruction") or ""), str(i.get("overlay_text") or ""))
                  for i in critique.get("ideas") or () if isinstance(i, dict))
    models = review.models if isinstance(review.models, dict) else {}
    return PackagingView(
        title=title, thumb=thumb, review=review, when=when(review.created_at),
        failed=review.error if not done else "", facts=facts, reading=read,
        points=points if done else (), ideas=ideas if done else (),
        problems=tuple(review.problems) if done else (), cost=format_cost(review.cost_usd),
        models=", ".join(str(v) for v in (models.get("vision"), models.get("segmenter")) if v),
        tokens=(f"{review.input_tokens:,} in, {review.output_tokens:,} out"
                if review.input_tokens or review.output_tokens else ""),
        can_variants=done and bool(review.image_file) and bool(ideas),
        variants=tuple(_variant_view(video_id, v) for v in variants), preview_url=preview_url)


# --- packaging preview (the feed mock-ups) --------------------------------------------------------
@dataclass(frozen=True)
class PreviewCard:
    """One video in a mock feed."""

    video_id: str
    title: str
    thumb: str
    views: str  # "1.5K views"
    date: str
    badge: str  # the duration badge, "" when the catalog has no duration
    mine: bool
    description: str  # the start of the description, for the search row


@dataclass(frozen=True)
class PreviewChoice:
    label: str
    href: str
    active: bool


@dataclass(frozen=True)
class PreviewView:
    desktop: tuple[PreviewCard, ...]
    phone: tuple[PreviewCard, ...]
    search: tuple[PreviewCard, ...]
    others: int  # how many of the channel's other videos are shown


def format_duration(seconds: int | None) -> str:
    """'4:07' or '1:02:03'; '' when there is no duration."""
    if not seconds or seconds < 0:
        return ""
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def preview_card(video: Video, *, mine: bool = False, title: str | None = None,
                 thumb: str | None = None) -> PreviewCard:
    views = "1 view" if video.view_count == 1 else f"{format_count(video.view_count)} views"
    return PreviewCard(
        video_id=video.video_id, title=video.title if title is None else title,
        thumb=thumb if thumb is not None else thumb_url(video), views=views,
        date=format_date(video.published_at), badge=format_duration(parse_duration(video.duration)),
        mine=mine, description=" ".join(video.description.split())[:DESCRIPTION_PREVIEW_CHARS])


def top_other_videos(video_id: str, catalog: Iterable[Video],
                     limit: int = PREVIEW_OTHERS) -> list[Video]:
    """The channel's other public videos with the most views (ties by id, so the order is stable)."""
    others = [v for v in catalog if v.video_id != video_id and v.privacy_status == "public"]
    others.sort(key=lambda v: (-v.view_count, v.video_id))
    return others[:limit]


def build_preview(mine: PreviewCard, others: Sequence[Video]) -> PreviewView:
    """The same video in a desktop grid, a phone list and a search result, among the others."""
    cards = [preview_card(v) for v in others]

    def among(rows: list[PreviewCard]) -> tuple[PreviewCard, ...]:
        return tuple(rows[:1] + [mine] + rows[1:])  # second place: it has neighbours on both sides

    return PreviewView(desktop=among(cards[:PREVIEW_OTHERS]), phone=among(cards[:PREVIEW_PHONE_ROWS]),
                       search=(mine, *cards[:PREVIEW_SEARCH_ROWS]), others=len(cards))


# --- outliers --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OutlierRow:
    """One video of the outlier or below-the-norm table. Numbers are formatted by code."""

    video_id: str
    title: str
    thumb: str
    published: str
    views: str  # "12,345"
    median: str  # the peers' median views, rounded
    ratio: str  # outliers.ratio_label, "4.2×"
    why: str  # the model's line about this video; "" when there is none or it is out of date


@dataclass(frozen=True)
class OutlierChip:
    """A label in a pattern's evidence: the video it stands for, linked when it is still in the catalog."""

    label: str
    title: str
    href: str


@dataclass(frozen=True)
class OutlierPattern:
    text: str
    kind: str
    chips: tuple[OutlierChip, ...]


@dataclass(frozen=True)
class OutlierReportView:
    """The newest explanation, ready to show."""

    status: str  # "done" | "too_few" | "error"
    when: str
    old: bool  # made from an older pull than the one the table is built from
    error: str
    patterns: tuple[OutlierPattern, ...]
    ideas: tuple[str, ...]
    problems: tuple[str, ...]
    cost: str
    model: str
    tokens: str


@dataclass(frozen=True)
class OutlierView:
    """The Outliers page besides the job state."""

    left_out: int  # videos that could not be scored (too new, not public, live streams, ...)
    outliers: tuple[OutlierRow, ...]
    below: tuple[OutlierRow, ...]
    report: OutlierReportView | None
    limits: Mapping[str, Any]  # the constants the page text quotes, so the text never drifts from the code


def pull_moment(row: Any) -> datetime:
    """The time of the views: when the pull finished (it started, if that is missing). Never "now"."""
    for key in ("finished_at", "started_at"):
        text = str(row[key] or "")
        if text:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
    raise ValueError("The pull has no time.")


def _outlier_row(video: Video, score: OutlierScore, why: str = "") -> OutlierRow:
    return OutlierRow(
        video_id=video.video_id, title=video.title, thumb=thumb_url(video),
        published=format_date(video.published_at), views=f"{score.views:,}",
        median=f"{round(score.peer_median or 0):,}", ratio=ratio_label(score.ratio), why=why)


def outlier_line(score: OutlierScore | None) -> dict[str, str] | None:
    """The one line of the video page: how the video compares with its norm. None when it has no ratio."""
    if score is None:
        return None
    if score.status == "too_new":
        return {"text": f"Too new to compare (under {MIN_AGE_DAYS} days)", "link": ""}
    if score.status != "scored" or score.ratio is None:
        return None
    return {"text": f"{ratio_label(score.ratio)} the channel norm for its age (median "
                    f"{round(score.peer_median or 0):,} views of {len(score.peer_ids)} videos "
                    "published around it)", "link": "/outliers"}


def outlier_count(scores: Sequence[OutlierScore]) -> int:
    return len(outliers(scores))


def _outlier_report_view(report: OutlierReport, titles: Mapping[str, str], old: bool,
                         when: Callable[[str], str]) -> OutlierReportView:
    labels = report.labels if isinstance(report.labels, dict) else {}
    insight = report.insight if isinstance(report.insight, dict) else {}

    def chip(label: str) -> OutlierChip:
        found = labels.get(label)
        video_id = found.get("video_id") if isinstance(found, dict) else None
        if not isinstance(video_id, str) or video_id not in titles:
            return OutlierChip(label, "", "")
        return OutlierChip(label, titles[video_id], f"/videos/{quote(video_id)}")

    patterns: list[OutlierPattern] = []
    for item in _dicts(insight.get("patterns")):
        evidence = item.get("evidence")
        patterns.append(OutlierPattern(
            text=str(item.get("text") or ""), kind=str(item.get("kind") or "other"),
            chips=tuple(chip(x) for x in evidence if isinstance(x, str))
            if isinstance(evidence, list) else ()))
    ideas = insight.get("ideas")
    shown = report.status == "done"
    return OutlierReportView(
        status=report.status, when=when(report.created_at), old=old, error=report.error,
        patterns=tuple(patterns) if shown else (),
        ideas=tuple(str(i) for i in ideas if isinstance(i, str)) if shown and isinstance(ideas, list) else (),
        problems=tuple(report.problems), cost=format_cost(report.cost_usd), model=report.model,
        tokens=(f"{report.input_tokens:,} in, {report.output_tokens:,} out"
                if report.input_tokens or report.output_tokens else ""))


def build_outlier_view(videos: Sequence[Video], scores: Sequence[OutlierScore],
                       report: OutlierReport | None, current_pull_id: int | None,
                       when: Callable[[str], str] = lambda iso: iso) -> OutlierView:
    """The Outliers page: the table of outliers and of videos below the norm, and the newest explanation.

    The model's per-video lines only show when the report was made from the current pull; an older
    report is still shown, with a note, because its patterns may still be useful.
    """
    by_id = {v.video_id: v for v in videos}
    old = report is not None and report.pull_id != current_pull_id
    whys: dict[str, str] = {}
    if report is not None and report.status == "done" and not old:
        labels = report.labels if isinstance(report.labels, dict) else {}
        insight = report.insight if isinstance(report.insight, dict) else {}
        for item in _dicts(insight.get("per_video")):
            found = labels.get(str(item.get("label")))
            video_id = found.get("video_id") if isinstance(found, dict) else None
            if isinstance(video_id, str) and item.get("why"):
                whys[video_id] = str(item["why"])
    return OutlierView(
        left_out=sum(1 for s in scores if s.status != "scored"),
        outliers=tuple(_outlier_row(by_id[s.video_id], s, whys.get(s.video_id, ""))
                       for s in outliers(scores)),
        below=tuple(_outlier_row(by_id[s.video_id], s) for s in below_norm(scores)),
        report=(_outlier_report_view(report, {k: v.title for k, v in by_id.items()}, old, when)
                if report is not None else None),
        limits={"peers": PEERS, "min_age": MIN_AGE_DAYS, "high": ratio_label(OUTLIER_THRESHOLD),
                "low": ratio_label(BELOW_THRESHOLD), "explained": MAX_EXPLAINED,
                "typical": MAX_TYPICAL, "chars": DESCRIPTION_CHARS})


# --- bulk edit (find & replace, footer block) ---------------------------------------------------
BULK_EXCERPT_CHARS = 90
BULK_LEAD_CHARS = 30


@dataclass(frozen=True)
class BulkForm:
    """The values of the bulk form as typed (kept on the page when something is wrong)."""

    kind: str = "replace"
    find: str = ""
    replace_with: str = ""
    fields: tuple[str, ...] = ("description",)
    footer: str = ""
    marker: str = ""
    visibility: str = "public"
    date_from: str = ""
    date_to: str = ""
    submitted: bool = False  # True when the Preview button was used

    def spec(self, video_ids: Sequence[str] = ()) -> BulkSpec:
        return BulkSpec(kind=self.kind, find=self.find, replace_with=self.replace_with,
                        fields=self.fields, footer=self.footer, footer_marker=self.marker,
                        visibility=self.visibility, published_from=self.date_from,
                        published_to=self.date_to, video_ids=tuple(video_ids))

    def pairs(self) -> list[tuple[str, str]]:
        """The form values as name/value pairs: hidden fields and the preview query string."""
        out = [("kind", self.kind), ("find", self.find), ("replace_with", self.replace_with)]
        out += [("field", f) for f in self.fields]
        out += [("footer", self.footer), ("marker", self.marker), ("visibility", self.visibility),
                ("from", self.date_from), ("to", self.date_to)]
        return out

    def preview_url(self) -> str:
        return "/bulk?" + urlencode([*self.pairs(), ("preview", "1")])


def parse_bulk_form(params: Any, *, submitted: bool) -> BulkForm:
    """Read the bulk form from a query string or a posted form (anything with get/getlist)."""
    if not submitted:
        return BulkForm()
    return BulkForm(
        kind=str(params.get("kind") or "replace"),
        find=str(params.get("find") or ""),
        replace_with=str(params.get("replace_with") or ""),
        fields=tuple(str(f) for f in params.getlist("field")),
        footer=str(params.get("footer") or ""),
        marker=str(params.get("marker") or ""),
        visibility=str(params.get("visibility") or "public"),
        date_from=str(params.get("from") or "").strip(),
        date_to=str(params.get("to") or "").strip(),
        submitted=True,
    )


def _window(text: str, start: int) -> str:
    chunk = text[start:start + BULK_EXCERPT_CHARS]
    return ("…" if start > 0 else "") + chunk + ("…" if start + BULK_EXCERPT_CHARS < len(text) else "")


def _excerpt(label: str, old: str, new: str) -> tuple[str, str, str] | None:
    """(label, before, after): the first changed line of `old` -> `new`, cut around the change."""
    old_lines, new_lines = old.split("\n"), new.split("\n")
    i = 0
    while i < len(old_lines) and i < len(new_lines) and old_lines[i] == new_lines[i]:
        i += 1
    if i >= len(old_lines) and i >= len(new_lines):
        return None
    before = old_lines[i] if i < len(old_lines) else ""
    after = new_lines[i] if i < len(new_lines) else ""
    while not before.strip() and not after.strip() and i < max(len(old_lines), len(new_lines)) - 1:
        i += 1  # a blank line is no help: show the first line with text
        before = old_lines[i] if i < len(old_lines) else ""
        after = new_lines[i] if i < len(new_lines) else ""
    common = 0
    while common < len(before) and common < len(after) and before[common] == after[common]:
        common += 1
    start = max(0, common - BULK_LEAD_CHARS)
    return label, _window(before, start), _window(after, start)


@dataclass(frozen=True)
class BulkRow:
    video_id: str
    title: str
    count: int
    excerpts: tuple[tuple[str, str, str], ...]  # (label, before, after)
    warnings: tuple[str, ...]
    blocked: str


def build_bulk_rows(plan: Sequence[BulkChange], videos: Mapping[str, Video]) -> list[BulkRow]:
    rows: list[BulkRow] = []
    for change in plan:
        video = videos.get(change.video_id)
        found: list[tuple[str, str, str]] = []
        if video is not None:
            text = video.description.replace("\r\n", "\n").replace("\r", "\n")
            for label, old, now in (("Title", video.title, change.title),
                                    ("Description", text, change.description)):
                excerpt = _excerpt(label, old, now)
                if excerpt is not None:
                    found.append(excerpt)
        rows.append(BulkRow(
            video_id=change.video_id,
            title=(video.title if video is not None else change.title) or change.video_id,
            count=change.count, excerpts=tuple(found),
            warnings=tuple(i.message for i in change.issues if i.severity != "error"),
            blocked=change.blocked))
    return rows


@dataclass(frozen=True)
class BulkPreview:
    rows: tuple[BulkRow, ...]
    ready: int  # rows that can be drafted
    blocked: int
    summary: str
    too_many: str  # "" or the sentence about MAX_BULK


def build_bulk_preview(plan: Sequence[BulkChange], videos: Mapping[str, Video]) -> BulkPreview:
    rows = build_bulk_rows(plan, videos)
    blocked = sum(1 for r in rows if r.blocked)
    ready = len(rows) - blocked
    if not rows:
        summary = "No video matches: nothing would change."
    else:
        summary = f"{_videos(ready)} would change" + (f", {blocked} blocked" if blocked else "") + "."
    too_many = (f"This would touch {len(rows)} videos, but at most {MAX_BULK} can be changed at "
                "once, so Make drafts will refuse. Narrow the dates or untick videos."
                if len(rows) > MAX_BULK else "")
    return BulkPreview(tuple(rows), ready, blocked, summary, too_many)


def bulk_apply_message(result: BulkResult) -> str:
    """The flash text after Make drafts."""
    text = (f"Drafts made for {_videos(result.drafted)}." if result.drafted
            else "No drafts were made.")
    if result.blocked:
        text += f" {_videos(result.blocked)} blocked."
    if result.not_in_plan:
        text += f" {_videos(result.not_in_plan)} no longer matched."
    if result.drafted:
        text += (" Review them on the Changes page; nothing changes on YouTube until you "
                 "approve and push.")
    return text


# --- the bulk edit agent -------------------------------------------------------------------------
BULK_EXAMPLES = (
    "Shorten every title over 60 characters",
    "Swap my old Discord link for a new one on videos from 2024",
    "Add a line at the top of every description about my new channel",
)
AGENT_ARGUMENT_CHARS = 300


@dataclass(frozen=True)
class AgentReviewRow:
    video_id: str
    title: str
    thumb: str
    reason: str
    fields: tuple[str, ...]  # labels: "Title", "Description", "Tags"
    blocked: str
    warnings: tuple[str, ...]
    diff: DraftDiff


@dataclass(frozen=True)
class AgentReview:
    rows: tuple[AgentReviewRow, ...]  # the videos that can be drafted first, then the blocked ones
    ready: int
    blocked: int
    summary: str  # "N videos ready, M blocked", counted by code
    too_many: str  # "" or the sentence about MAX_BULK


def build_agent_review(reviewed: Sequence[bulkagent_mod.Reviewed]) -> AgentReview:
    rows: list[AgentReviewRow] = []
    for item in reviewed:
        change, video = item.staged, item.video
        rows.append(AgentReviewRow(
            video_id=change.video_id,
            title=(video.title if video is not None else change.title0) or change.video_id,
            thumb=thumb_url(video) if video is not None else "",
            reason=change.reason, fields=tuple(FIELD_LABELS[f] for f in change.fields),
            blocked=item.blocked, warnings=item.warnings,
            diff=diff_draft(Draft(change.title0, change.description0.replace("\r\n", "\n"),
                                  change.tags0),
                            Draft(change.title, change.description, change.tags))))
    rows.sort(key=lambda r: bool(r.blocked))
    blocked = sum(1 for r in rows if r.blocked)
    ready = len(rows) - blocked
    too_many = (f"{len(rows)} videos are staged, but at most {MAX_BULK} can be changed at once, so "
                "Make drafts will refuse. Untick videos." if len(rows) > MAX_BULK else "")
    return AgentReview(tuple(rows), ready, blocked, f"{_videos(ready)} ready, {blocked} blocked",
                       too_many)


@dataclass(frozen=True)
class AgentToolCall:
    name: str
    arguments: str
    result_chars: int


@dataclass(frozen=True)
class AgentSessionView:
    requests: tuple[str, ...]  # what the creator wrote, oldest first (words just sent come last)
    log: tuple[str, ...]  # what it did, one sentence per line, written by code
    closing: Markup  # the model's closing message after the number guard, escaped
    notes: tuple[str, ...]
    error: str
    model: str
    tokens: str
    cost: str
    tool_calls: tuple[AgentToolCall, ...]
    staged: int
    stopping: bool
    running: bool


def build_agent_session(session: bulkagent_mod.BulkSession, titles: Mapping[str, str],
                        running: bool) -> AgentSessionView:
    with session.lock:
        requests = (*session.requests, *([session.pending] if session.pending else []))
        log, notes, error = tuple(session.log), tuple(session.notes), session.error
        closing, refs, model = session.closing, dict(session.refs), session.model
        calls = [dict(c) for c in session.calls]
        staged = len(session.staged)
    used, cached, out = session.tokens()
    tool_calls = []
    for call in calls:
        args = call.get("arguments")
        text = args if isinstance(args, str) else json.dumps(
            args if args is not None else {}, separators=(",", ":"), ensure_ascii=False)
        if len(text) > AGENT_ARGUMENT_CHARS:
            text = text[:AGENT_ARGUMENT_CHARS - 1] + "…"
        tool_calls.append(AgentToolCall(str(call.get("name") or ""), text,
                                        int(call.get("result_chars") or 0)))
    return AgentSessionView(
        requests=requests, log=log,
        closing=render_answer(closing, refs, titles) if closing else Markup(""),
        notes=notes, error=error, model=model,
        tokens=f"{used:,} tokens in ({cached:,} cached), {out:,} out",
        cost=format_cost(estimate_cost(model, used, cached, out)), tool_calls=tuple(tool_calls),
        staged=staged, stopping=session.stop.is_set(), running=running)


# --- the copilot chat -----------------------------------------------------------------------------
STARTER_QUESTIONS = (
    "What should I fix this week?",
    "Which of my popular videos have no chapters?",
    "What do my best videos have in common?",
)
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_BULLET = re.compile(r"^\s*[-*]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*\d{1,2}\.\s+(.*)$")


def _inline(text: str, refs: Mapping[str, str], titles: Mapping[str, str]) -> str:
    """One line of an answer as safe HTML: escaped first, then **bold** and ref links."""
    safe = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", str(escape(text)))

    def link(found: "re.Match[str]") -> str:
        ref = found.group(0)
        video_id = refs.get(ref)
        title = titles.get(video_id) if isinstance(video_id, str) else None
        if not title or not isinstance(video_id, str):
            return ref  # a ref the turn does not know, or a video that is gone, stays text
        return (f'<a href="/videos/{escape(quote(video_id))}">{escape(title)}</a> '
                f'<small class="cp-ref">{ref}</small>')

    return REF.sub(link, safe)


def render_answer(answer: str, refs: Mapping[str, str], titles: Mapping[str, str]) -> Markup:
    """A model answer as safe HTML.

    Everything is escaped first. Then: paragraphs on blank lines, "- " or "* " bullets, "1." lists
    and **bold**. A ref like V12 that is in `refs` and whose video is in `titles` (video id -> its
    current title) becomes a link to the video page; any other ref stays plain text.
    """
    out: list[str] = []
    paragraph: list[str] = []
    items: list[str] = []
    tag = ""

    def close() -> None:
        nonlocal tag
        if paragraph:
            out.append("<p>" + "<br>".join(paragraph) + "</p>")
            paragraph.clear()
        if items:
            out.append(f"<{tag}>" + "".join(f"<li>{i}</li>" for i in items) + f"</{tag}>")
            items.clear()
        tag = ""

    for line in answer.splitlines():
        if not line.strip():
            close()
            continue
        bullet = _BULLET.match(line)
        marker = bullet or _NUMBERED.match(line)
        if marker:
            kind = "ul" if bullet else "ol"
            if paragraph or (tag and tag != kind):
                close()
            tag = kind
            items.append(_inline(marker.group(1), refs, titles))
        elif items and line[0].isspace():
            items[-1] += "<br>" + _inline(line.strip(), refs, titles)  # a wrapped list item
        else:
            if items:
                close()
            paragraph.append(_inline(line.strip(), refs, titles))
    close()
    return Markup("".join(out))


@dataclass(frozen=True)
class CopilotProposalView:
    """An edit the copilot offered, next to the video's current text."""

    id: int
    video_id: str
    ref: str
    video_title: str  # the video's current title ("" when it is gone)
    title_after: str
    excerpt: tuple[str, str, str] | None  # (label, before, after) of the first changed line
    tags_note: str
    reason: str
    status: str  # "open" | "drafted" | "dismissed"
    gone: bool
    has_draft: bool  # the video already has a draft, so Make draft would be refused


@dataclass(frozen=True)
class CopilotToolCall:
    name: str
    arguments: str
    result_chars: int


@dataclass(frozen=True)
class CopilotTurnView:
    """One question and answer of the chat, ready for the page."""

    id: int
    question: str
    when: str
    status: str  # "answered" | "failed" | "running" | "interrupted"
    answer: Markup
    failed: str  # the plain sentence when the turn failed
    notes: tuple[str, ...]
    proposals: tuple[CopilotProposalView, ...]
    model: str
    tokens: str
    cost: str
    tool_calls: tuple[CopilotToolCall, ...]
    events: tuple[str, ...]  # the other log lines (checks, errors)


def _tags_note(old: Sequence[str], new: Sequence[str]) -> str:
    before = {t.casefold() for t in old}
    after = {t.casefold() for t in new}
    added, removed = len(after - before), len(before - after)
    if not added and not removed:
        return "The order of the tags changes." if list(old) != list(new) else ""
    parts = ([f"{added} added"] if added else []) + ([f"{removed} removed"] if removed else [])
    return "Tags: " + ", ".join(parts)


def _proposal_view(proposal: CopilotProposal, turn: CopilotTurn, videos: Mapping[str, Video],
                   drafts: Collection[str]) -> CopilotProposalView:
    video = videos.get(proposal.video_id)
    ref = next((r for r, v in turn.refs.items() if v == proposal.video_id), "")
    if video is None:
        return CopilotProposalView(proposal.id, proposal.video_id, ref, "", proposal.title, None,
                                   "", proposal.reason, proposal.status, True, False)
    old = video.description.replace("\r\n", "\n").replace("\r", "\n")
    new = proposal.description.replace("\r\n", "\n").replace("\r", "\n")
    return CopilotProposalView(
        proposal.id, proposal.video_id, ref, video.title, proposal.title,
        _excerpt("Description", old, new), _tags_note(video.tags, proposal.tags), proposal.reason,
        proposal.status, False, proposal.video_id in drafts)


def _turn_view(turn: CopilotTurn, proposals: Sequence[CopilotProposal],
               videos: Mapping[str, Video], drafts: Collection[str], running: bool,
               when: Callable[[str], str]) -> CopilotTurnView:
    status = "interrupted" if turn.status == "running" and not running else turn.status
    tool_calls: list[CopilotToolCall] = []
    events: list[str] = []
    for event in _dicts(turn.log):
        kind = event.get("type")
        if kind == "tool":
            args = event.get("arguments")
            text = args if isinstance(args, str) else json.dumps(
                args if args is not None else {}, separators=(",", ":"), ensure_ascii=False)
            chars = event.get("result_chars")
            tool_calls.append(CopilotToolCall(str(event.get("name") or ""), text,
                                              chars if isinstance(chars, int) else 0))
        elif kind == "problems":
            events.extend(str(p) for p in (event.get("problems") or []))
        elif kind == "guard" and isinstance(event.get("numbers"), list):
            numbers = ", ".join(str(n) for n in event["numbers"])
            events.append(f"Number check: {numbers} came from no tool; "
                          + ("the model was asked to fix it." if event.get("round") == 1
                             else "the sentence was removed."))
        elif isinstance(event.get("message"), str):
            events.append(event["message"])
    titles = {v.video_id: v.title for v in videos.values()}
    cost = estimate_cost(turn.model, turn.input_tokens, turn.cached_tokens, turn.output_tokens)
    return CopilotTurnView(
        id=turn.id, question=turn.question, when=when(turn.asked_at), status=status,
        answer=(render_answer(turn.answer, turn.refs, titles) if status == "answered"
                else Markup("")),
        failed=turn.answer if status == "failed" else "", notes=tuple(turn.notes),
        proposals=tuple(_proposal_view(p, turn, videos, drafts) for p in proposals),
        model=turn.model,
        tokens=(f"{turn.input_tokens:,} tokens in ({turn.cached_tokens:,} cached), "
                f"{turn.output_tokens:,} out"),
        cost=format_cost(cost), tool_calls=tuple(tool_calls), events=tuple(events))


def build_copilot_turns(turns: Sequence[CopilotTurn], proposals: Sequence[CopilotProposal],
                        videos: Mapping[str, Video], drafts: Collection[str], running: bool,
                        when: Callable[[str], str] = lambda iso: iso) -> list[CopilotTurnView]:
    """The turns of one chat, oldest first, each with its proposals. `running`: a copilot job is
    running, so a turn that still says "running" is not an interrupted one."""
    return [_turn_view(t, [p for p in proposals if p.turn_id == t.id], videos, drafts, running,
                       when) for t in turns]


# --- links: every link in the descriptions, ranked by reach ---------------------------------
LINKS_PAGE_SIZE = 200
LINK_SHORT_CHARS = 60
# (key, label) in display order; the key goes into the `show` query parameter.
LINK_FILTERS: tuple[tuple[str, str], ...] = (
    ("all", "All"),
    ("broken", "Broken"),
    ("redirected", "Redirected"),
    ("trouble", "Could not be checked"),
    ("own", "Own private or unknown videos"),
    ("unchecked", "Not checked"),
)
LINK_FILTER_KEYS = tuple(k for k, _ in LINK_FILTERS)
LINK_OWN_PROBLEMS = ("private", "not_in_catalog")
# own-video state -> (label, pill class)
LINK_OWN_STATES: dict[str, tuple[str, str]] = {
    "public": ("Your public video", "done"),
    "unlisted": ("Your unlisted video", "done"),
    "private": ("Your private video", "problem"),
    "not_in_catalog": ("Not in your catalog", "warn"),
}
# check status -> (label, pill class)
LINK_CHECK_STATES: dict[str, tuple[str, str]] = {
    "ok": ("Working", "done"),
    "redirected": ("Moves", "warn"),
    "broken": ("Broken", "problem"),
    "error": ("Could not be checked", "warn"),
}


def shorten_link(url: str, limit: int = LINK_SHORT_CHARS) -> str:
    """The link as shown in a table: no scheme, no "www.", cut with an ellipsis. The full text
    goes into a title attribute."""
    text = url.split("://", 1)[1] if "://" in url else url
    if text.lower().startswith("www."):
        text = text[4:]
    return text if len(text) <= limit else text[:limit - 1] + "…"


def link_href(url: str) -> str:
    """Something a browser can open: a link written as `www.…` gets https."""
    return url if "://" in url else "https://" + url


def _external_state(entry: LinkEntry) -> str:
    """The check status that counts for a link: "" when it was never visited (or only skipped)."""
    check = entry.check
    return check.status if check is not None and check.status in LINK_CHECK_STATES else ""


@dataclass(frozen=True)
class LinkVideoRef:
    video_id: str
    title: str


@dataclass(frozen=True)
class LinkRow:
    url: str
    href: str  # what the browser opens
    short: str
    domain: str
    kind: str  # "video" | "youtube" | "external"
    videos: tuple[LinkVideoRef, ...]  # most viewed first
    reach: str  # "12.3K"
    reach_title: str  # "12,345 views in total"
    status: str  # a word for the status pill
    status_class: str  # "done" | "warn" | "problem" | "unverified"
    detail: str  # the plain sentence behind the status ("Not found (404).")
    when: str  # when it was checked ("" when it was not)
    replace_href: str


@dataclass(frozen=True)
class LinksSummary:
    links: int
    videos: int  # videos that carry at least one link
    broken: int
    redirected: int
    trouble: int  # could not be checked
    own: int  # links to the creator's own private or unknown videos
    unchecked: int  # external links never visited
    external: int  # all external links (what a check visits)
    line: str


@dataclass(frozen=True)
class LinksView:
    rows: tuple[LinkRow, ...]
    summary: LinksSummary
    show: str
    chips: tuple[Chip, ...]
    total: int  # rows matching the filter
    page: int
    pages: int
    first: int
    last: int
    prev_href: str
    next_href: str


def _links_line(s: LinksSummary) -> str:
    if not s.links:
        return "No links in your descriptions."
    parts = [f"{s.broken} broken", f"{s.redirected} redirected"]
    if s.trouble:
        parts.append(f"{s.trouble} could not be checked")
    parts.append(f"{s.own} to your private or unknown videos")
    parts.append(f"{s.unchecked} not checked")
    return f"{_count(s.links, 'link')} in {_count(s.videos, 'video')}; " + ", ".join(parts) + "."


def links_page_href(show: str = "all", page: int = 1) -> str:
    args: list[tuple[str, str]] = []
    if show != "all":
        args.append(("show", show))
    if page > 1:
        args.append(("page", str(page)))
    return "/links" + ("?" + urlencode(args) if args else "")


def _link_matches(entry: LinkEntry, show: str) -> bool:
    state = _external_state(entry)
    if show == "broken":
        return state == "broken"
    if show == "redirected":
        return state == "redirected"
    if show == "trouble":
        return state == "error"
    if show == "own":
        return entry.target_state in LINK_OWN_PROBLEMS
    if show == "unchecked":
        return entry.kind == "external" and not state
    return True


def _link_row(entry: LinkEntry, titles: Mapping[str, str],
              when: Callable[[str], str]) -> LinkRow:
    state = _external_state(entry)
    check = entry.check
    checked = ""
    if entry.kind == "video":
        status, css = LINK_OWN_STATES[entry.target_state]
        detail = entry.finding
    elif entry.kind == "youtube":
        status, css, detail = "Not visited", "unverified", SKIPPED_MESSAGE
    elif state and check is not None:
        status, css = LINK_CHECK_STATES[state]
        detail, checked = check.message, when(check.checked_at)
    else:
        status, css, detail = "Not checked yet", "unverified", ""
    return LinkRow(
        url=entry.url, href=link_href(entry.url), short=shorten_link(entry.url),
        domain=entry.domain, kind=entry.kind,
        videos=tuple(LinkVideoRef(v, titles.get(v) or v) for v in entry.video_ids),
        reach=format_count(entry.reach), reach_title=f"{entry.reach:,} views in total",
        status=status, status_class=css, detail=detail, when=checked,
        replace_href="/bulk?" + urlencode([("kind", "replace"), ("find", entry.url),
                                          ("field", "description"), ("visibility", "all")]))


def build_links_view(entries: Sequence[LinkEntry], videos: Sequence[Video],
                     params: Mapping[str, str],
                     when: Callable[[str], str] = lambda iso: iso) -> LinksView:
    """Filter and page the inventory (already ranked by reach). Unknown values fall back to the
    defaults; never raises."""
    show = _pick(params.get("show"), LINK_FILTER_KEYS, "all")
    states = [_external_state(e) for e in entries]
    summary = LinksSummary(
        links=len(entries), videos=len({v for e in entries for v in e.video_ids}),
        broken=states.count("broken"), redirected=states.count("redirected"),
        trouble=states.count("error"),
        own=sum(1 for e in entries if e.target_state in LINK_OWN_PROBLEMS),
        unchecked=sum(1 for e, st in zip(entries, states) if e.kind == "external" and not st),
        external=sum(1 for e in entries if e.kind == "external"), line="")
    summary = replace(summary, line=_links_line(summary))
    counts = {"all": len(entries), "broken": summary.broken, "redirected": summary.redirected,
              "trouble": summary.trouble, "own": summary.own, "unchecked": summary.unchecked}

    rows = [e for e in entries if _link_matches(e, show)]
    total = len(rows)
    pages = max(1, -(-total // LINKS_PAGE_SIZE))
    page = min(max(_page_number(params.get("page")), 1), pages)
    shown = rows[(page - 1) * LINKS_PAGE_SIZE: page * LINKS_PAGE_SIZE]
    titles = {v.video_id: v.title for v in videos}
    return LinksView(
        rows=tuple(_link_row(e, titles, when) for e in shown), summary=summary, show=show,
        chips=tuple(Chip(f"{label} ({counts[key]})", links_page_href(key), key == show)
                    for key, label in LINK_FILTERS),
        total=total, page=page, pages=pages,
        first=(page - 1) * LINKS_PAGE_SIZE + 1 if total else 0,
        last=min(page * LINKS_PAGE_SIZE, total),
        prev_href=links_page_href(show, page - 1) if page > 1 else "",
        next_href=links_page_href(show, page + 1) if page < pages else "")


def links_check_line(summary: LinkCheckSummary) -> str:
    """The one line shown when a link check ends, such as "Checked 212 links: 3 broken."."""
    problems = [f"{summary.count(status)} {label}" for status, label in
                (("broken", "broken"), ("redirected", "redirected"),
                 ("error", "could not be checked")) if summary.count(status)]
    text = f"Checked {_count(summary.checked, 'link')}"
    if problems:
        text += ": " + ", ".join(problems) + "."
    else:
        text += ": all answered." if summary.checked else "."
    if summary.not_checked:
        text += (f" {_count(summary.not_checked, 'link')} with the least reach "
                 f"{'was' if summary.not_checked == 1 else 'were'} left out.")
    return text


# --- the advertiser-friendly check card -----------------------------------------------------------
AD_VERDICT_TONES = {"no_concerns": "done", "worth_a_look": "warn", "likely_limited": "problem"}
AD_CHANGED_LINE = "The title, description, tags or transcript changed since this check."
AD_NO_TRANSCRIPT_LINE = "No transcript: only the title, description and tags were read."


@dataclass(frozen=True)
class AdFindingView:
    category: str  # the plain label
    where: str  # "Title", "Description", "Tags" or "Transcript at 1:05"
    quote: str
    why: str
    fix: str


@dataclass(frozen=True)
class AdCheckView:
    """What the Advertiser-friendly check card shows besides the job state."""

    check: AdCheck | None
    failed: bool
    error: str
    verdict: str  # the plain label, "" for a failed check
    tone: str
    summary: str
    findings: tuple[AdFindingView, ...]
    notes: tuple[str, ...]
    changed_line: str  # AD_CHANGED_LINE when the text differs from what was read, else ""
    no_transcript_line: str
    when: str
    model: str
    tokens: str
    cost: str
    advice: str


def _ad_finding(item: Any) -> AdFindingView | None:
    if not isinstance(item, dict):
        return None
    category = str(item.get("category") or "")
    source = str(item.get("source") or "")
    where = adcheck_mod.SOURCE_LABELS.get(source, source)
    if source == "transcript" and item.get("time"):
        where += f" at {item['time']}"
    return AdFindingView(
        category=adcheck_mod.CATEGORIES.get(category, category), where=where,
        quote=str(item.get("quote") or ""), why=str(item.get("why") or ""),
        fix=str(item.get("fix") or ""))


SHORTS_STALE_LINE = "The transcript changed after this run. Find the Shorts again to use the new one."


@dataclass(frozen=True)
class ShortsClipView:
    """A clip that could stand alone as a Short for the template."""

    title: str
    why: str
    quote: str
    start: str
    end: str
    length: str
    watch_url: str


@dataclass(frozen=True)
class ShortsView:
    """The newest Shorts run as plain text for the template."""

    find: ShortsFind | None
    failed: bool
    error: str
    message: str
    clips: tuple[ShortsClipView, ...]
    notes: tuple[str, ...]
    stale_line: str
    when: str
    model: str
    tokens: str


def build_shorts_view(
    video_id: str,
    find: ShortsFind | None,
    transcript_id: int,
    when: Callable[[str], str] = lambda iso: iso,
) -> ShortsView:
    """The newest Shorts run as plain text for the template. Everything in it is escaped there."""
    if find is None:
        return ShortsView(None, False, "", "", (), (), "", "", "", "")
    failed = find.status == "failed"
    tokens_total = find.input_tokens + find.output_tokens
    tokens = format_count(tokens_total) if tokens_total else ""
    stale_line = SHORTS_STALE_LINE if find.transcript_id != transcript_id else ""
    error = find.error or "The Shorts finder did not finish. Try again."
    message = find.message or ""
    clips: list[ShortsClipView] = []
    if not failed:
        for c in find.clips:
            start_ms = int(c.get("start_ms", 0))
            end_ms = int(c.get("end_ms", 0))
            start = clock(start_ms)
            end = clock(end_ms)
            length = f"{round((end_ms - start_ms) / 1000)} s"
            watch_url = f"https://www.youtube.com/watch?v={quote(video_id)}&t={start_ms // 1000}s"
            clips.append(
                ShortsClipView(
                    title=str(c.get("title") or ""),
                    why=str(c.get("why") or ""),
                    quote=str(c.get("quote") or ""),
                    start=start,
                    end=end,
                    length=length,
                    watch_url=watch_url,
                )
            )
    return ShortsView(
        find=find,
        failed=failed,
        error=error,
        message=message,
        clips=tuple(clips),
        notes=tuple(str(n) for n in find.notes),
        stale_line=stale_line,
        when=when(find.created_at),
        model=find.model,
        tokens=tokens,
    )

def build_adcheck_view(check: AdCheck | None, current_hash: str | None,
                       when: Callable[[str], str] = lambda iso: iso) -> AdCheckView:
    """The newest check of a video as plain text for the template. Everything in it is escaped there."""
    advice = adcheck_mod.ADVICE_LINE
    if check is None:
        return AdCheckView(None, False, "", "", "", "", (), (), "", "", "", "", "", "", advice)
    failed = check.status != "done"
    tokens = check.input_tokens + check.output_tokens
    cost = estimate_cost(check.model, check.input_tokens, check.cached_tokens, check.output_tokens)
    found = (_ad_finding(item) for item in check.findings) if not failed else ()
    return AdCheckView(
        check=check, failed=failed, error=check.error or "The check did not finish. Try again.",
        verdict="" if failed else adcheck_mod.VERDICTS.get(check.verdict, check.verdict),
        tone="problem" if failed else AD_VERDICT_TONES.get(check.verdict, ""),
        summary=check.summary,
        findings=tuple(f for f in found if f is not None),
        notes=tuple(str(n) for n in check.notes),
        changed_line=(AD_CHANGED_LINE if not failed and current_hash is not None
                      and current_hash != check.source_hash else ""),
        no_transcript_line="" if failed or check.had_transcript else AD_NO_TRANSCRIPT_LINE,
        when=when(check.created_at), model=check.model,
        tokens=f"{tokens:,}" if tokens else "", cost=format_cost(cost) if tokens else "",
        advice=advice)


# --- the comments card (G4, G5) -------------------------------------------------------------------
COMMENTS_LIMIT = 100
NOT_SORTED_KEY = "not_sorted"
NOT_SORTED_LABEL = "Not sorted"


@dataclass(frozen=True)
class CommentChipView:
    key: str
    label: str
    count: int
    href: str
    active: bool


@dataclass(frozen=True)
class CommentTemplateView:
    name: str
    text: str


@dataclass(frozen=True)
class CommentRowView:
    comment: Comment
    category_label: str
    date_text: str
    likes_text: str
    draft: CommentReply | None
    templates: tuple[CommentTemplateView, ...]
    youtube_url: str
    posted: CommentReply | None = None  # the reply Yezatas posted to this comment, if any
    posted_url: str = ""  # that reply on YouTube


@dataclass(frozen=True)
class CommentsView:
    video_id: str
    chips: tuple[CommentChipView, ...]
    pinned: CommentReply | None
    rows: tuple[CommentRowView, ...]
    total: int
    filtered_total: int
    filter_key: str
    pinned_posted: CommentReply | None = None  # the pinned-comment draft Yezatas posted (pin it in Studio)
    pinned_posted_url: str = ""


def _posted_replies(replies: Sequence[CommentReply]) -> tuple[dict[str, CommentReply], CommentReply | None]:
    """The newest posted reply of each comment, and the newest posted pinned-comment draft."""
    by_comment: dict[str, CommentReply] = {}
    pinned: CommentReply | None = None
    for reply in sorted((r for r in replies if r.status == "posted"),
                        key=lambda r: (r.posted_at, r.id), reverse=True):
        if not reply.comment_id:
            pinned = pinned or reply
        elif reply.comment_id not in by_comment:
            by_comment[reply.comment_id] = reply
    return by_comment, pinned


def _posted_link(video_id: str, reply: CommentReply | None) -> str:
    return posted_url(video_id, reply.posted_comment_id) if reply is not None else ""


def build_comments_view(video_id: str, comments: Sequence[Comment], replies: Sequence[CommentReply], templates: Sequence[ReplyTemplate], params: Mapping[str, str] | None = None) -> CommentsView:
    ordered = sorted(comments, key=lambda c: (c.published_at or "", c.comment_id), reverse=True)
    raw = str((params or {}).get("c") or "").strip()
    if raw in ("", "all"):
        selected = "all"
    elif raw in COMMENT_CATEGORIES:
        selected = raw
    elif raw in (NOT_SORTED_KEY, "unsorted", "not-sorted"):
        selected = NOT_SORTED_KEY
    else:
        selected = "all"
    counts: dict[str, int] = {key: 0 for key in COMMENT_CATEGORIES}
    not_sorted = 0
    for comment in ordered:
        if comment.category in counts:
            counts[comment.category] += 1
        else:
            not_sorted += 1
    base = f"/videos/{quote(video_id)}"
    chips = [CommentChipView("all", "All", len(ordered), f"{base}#comments", selected == "all")]
    chips += [CommentChipView(key, label, counts[key], f"{base}?c={key}#comments", selected == key)
                for key, label in COMMENT_CATEGORIES.items()]
    chips.append(CommentChipView(NOT_SORTED_KEY, NOT_SORTED_LABEL, not_sorted, f"{base}?c={NOT_SORTED_KEY}#comments", selected == NOT_SORTED_KEY))
    if selected == "all":
        filtered = ordered
    elif selected == NOT_SORTED_KEY:
        filtered = [c for c in ordered if c.category not in COMMENT_CATEGORIES]
    else:
        filtered = [c for c in ordered if c.category == selected]
    open_replies = [r for r in replies if r.status == "open"]
    pinned = next((r for r in open_replies if r.comment_id == ""), None)
    drafts: dict[str, CommentReply] = {}
    for reply in open_replies:
        if reply.comment_id and reply.comment_id not in drafts:
            drafts[reply.comment_id] = reply
    posted, pinned_posted = _posted_replies(replies)
    rows: list[CommentRowView] = []
    for comment in filtered[:COMMENTS_LIMIT]:
        matched: tuple[CommentTemplateView, ...] = ()
        if comment.category in ("question", "feedback"):
            matched = tuple(
                CommentTemplateView(t.name, replytemplates_mod.fill(t.body, comment.author))
                for t in replytemplates_mod.suggest(comment.text, templates)[:3])
        rows.append(CommentRowView(
            comment=comment,
            category_label=COMMENT_CATEGORIES.get(
                comment.category,
                NOT_SORTED_LABEL if not comment.category else comment.category),
            date_text=format_date(comment.published_at),
            likes_text=format_count(comment.like_count),
            draft=drafts.get(comment.comment_id), templates=matched,
            youtube_url="https://www.youtube.com/watch?"
            + urlencode({"v": video_id, "lc": comment.comment_id}),
            posted=posted.get(comment.comment_id),
            posted_url=_posted_link(video_id, posted.get(comment.comment_id))))
    return CommentsView(video_id=video_id, chips=tuple(chips), pinned=pinned, rows=tuple(rows), total=len(ordered), filtered_total=len(filtered), filter_key=selected,
                        pinned_posted=pinned_posted, pinned_posted_url=_posted_link(video_id, pinned_posted))


CHANNEL_COMMENTS_PATH = "/comments"


@dataclass(frozen=True)
class PostedReplyRow:
    """One reply or comment Yezatas posted, for the History page."""

    reply: CommentReply
    title: str  # the video's title
    when: str
    url: str  # the new comment on YouTube, "" when YouTube sent no id
    answers: str  # who it answered ("Alice"), "" for a new top-level comment
    pinned_draft: bool


def build_posted_rows(replies: Sequence[CommentReply], titles: Mapping[str, str],
                      authors: Mapping[str, str], when: Callable[[str], str]) -> tuple[PostedReplyRow, ...]:
    return tuple(PostedReplyRow(
        reply=r, title=titles.get(r.video_id, r.video_id), when=when(r.posted_at),
        url=posted_url(r.video_id, r.posted_comment_id),
        answers=authors.get(r.comment_id, ""), pinned_draft=not r.comment_id) for r in replies)


@dataclass(frozen=True)
class ChannelCommentRow:
    comment: Comment
    video_id: str
    video_title: str
    video_href: str  # the video's page in Yezatas
    category_label: str
    date_text: str
    likes_text: str
    draft: CommentReply | None
    templates: tuple[CommentTemplateView, ...]
    youtube_url: str
    posted: CommentReply | None = None  # the reply Yezatas posted to this comment, if any
    posted_url: str = ""  # that reply on YouTube


@dataclass(frozen=True)
class ChannelVideoRow:
    video_id: str
    title: str
    href: str  # /comments?v=<id>, keeping the other filters
    meta: str  # e.g. "3 questions · 12 comments"
    unsorted: int
    active: bool


@dataclass(frozen=True)
class ChannelCommentsView:
    chips: tuple[CommentChipView, ...]
    rows: tuple[ChannelCommentRow, ...]
    videos: tuple[ChannelVideoRow, ...]
    total: int  # stored comments given to the builder
    filtered_total: int
    unsorted: int  # over all comments given
    video_count: int  # videos that have comments
    fetched_text: str  # format_date of the newest fetched_at, "" when none
    filter_key: str
    video_filter: str  # "" or a video id that has comments
    draft_only: bool
    draft_only_href: str  # the same page with "only with a reply draft" switched
    next_path: str  # this page's own path with its filters, for forms to come back to


def build_channel_comments_view(comments: Sequence[Comment], replies: Sequence[CommentReply], videos: Mapping[str, Video], templates: Sequence[ReplyTemplate], params: Mapping[str, str] | None = None) -> ChannelCommentsView:
    ordered = sorted(comments, key=lambda c: (c.published_at or "", c.comment_id), reverse=True)
    raw = str((params or {}).get("c") or "").strip()
    if raw in ("", "all"):
        selected = "all"
    elif raw in COMMENT_CATEGORIES:
        selected = raw
    elif raw in (NOT_SORTED_KEY, "unsorted", "not-sorted"):
        selected = NOT_SORTED_KEY
    else:
        selected = "all"
    vf_raw = str((params or {}).get("v") or "").strip()
    comment_video_ids = {c.video_id for c in ordered}
    video_filter = vf_raw if vf_raw in comment_video_ids else ""
    draft_only = str((params or {}).get("draft") or "") == "1"

    def _channel_href(category: str, video: str, draft: bool) -> str:
        pairs: list[tuple[str, str]] = []
        if category != "all":
            pairs.append(("c", category))
        if video:
            pairs.append(("v", video))
        if draft:
            pairs.append(("draft", "1"))
        if pairs:
            return f"{CHANNEL_COMMENTS_PATH}?{urlencode(pairs)}"
        return CHANNEL_COMMENTS_PATH

    open_replies = [r for r in replies if r.status == "open"]
    drafts: dict[str, CommentReply] = {}
    for reply in open_replies:
        if reply.comment_id and reply.comment_id not in drafts:
            drafts[reply.comment_id] = reply
    posted, _ = _posted_replies(replies)

    if video_filter:
        scope = [c for c in ordered if c.video_id == video_filter]
    else:
        scope = list(ordered)

    counts: dict[str, int] = {key: 0 for key in COMMENT_CATEGORIES}
    not_sorted_scope = 0
    for comment in scope:
        if comment.category in counts:
            counts[comment.category] += 1
        else:
            not_sorted_scope += 1
    chips: list[CommentChipView] = []
    chips.append(CommentChipView("all", "All", len(scope), _channel_href("all", video_filter, draft_only), selected == "all"))
    for key, label in COMMENT_CATEGORIES.items():
        chips.append(CommentChipView(key, label, counts[key], _channel_href(key, video_filter, draft_only), selected == key))
    chips.append(CommentChipView(NOT_SORTED_KEY, NOT_SORTED_LABEL, not_sorted_scope, _channel_href(NOT_SORTED_KEY, video_filter, draft_only), selected == NOT_SORTED_KEY))

    if selected == "all":
        filtered = list(scope)
    elif selected == NOT_SORTED_KEY:
        filtered = [c for c in scope if c.category not in COMMENT_CATEGORIES]
    else:
        filtered = [c for c in scope if c.category == selected]
    if draft_only:
        filtered = [c for c in filtered if c.comment_id in drafts]

    rows: list[ChannelCommentRow] = []
    for comment in filtered[:COMMENTS_LIMIT]:
        vid = comment.video_id
        vobj = videos.get(vid)
        if vobj is not None and getattr(vobj, "title", ""):
            video_title = vobj.title
        else:
            video_title = vid
        video_href = f"/videos/{quote(vid)}"
        category_label = COMMENT_CATEGORIES.get(comment.category, NOT_SORTED_LABEL)
        date_text = format_date(comment.published_at)
        likes_text = format_count(comment.like_count)
        draft = drafts.get(comment.comment_id)
        matched: tuple[CommentTemplateView, ...] = ()
        if comment.category in ("question", "feedback"):
            matched = tuple(
                CommentTemplateView(t.name, replytemplates_mod.fill(t.body, comment.author))
                for t in replytemplates_mod.suggest(comment.text, templates)[:3]
            )
        youtube_url = f"https://www.youtube.com/watch?v={quote(vid)}&lc={quote(comment.comment_id)}"
        rows.append(ChannelCommentRow(
            comment=comment,
            video_id=vid,
            video_title=video_title,
            video_href=video_href,
            category_label=category_label,
            date_text=date_text,
            likes_text=likes_text,
            draft=draft,
            templates=matched,
            youtube_url=youtube_url,
            posted=posted.get(comment.comment_id),
            posted_url=_posted_link(vid, posted.get(comment.comment_id)),
        ))

    # rail over all ordered
    seen: list[str] = []
    seen_set: set[str] = set()
    for c in ordered:
        if c.video_id not in seen_set:
            seen_set.add(c.video_id)
            seen.append(c.video_id)
    video_rows: list[ChannelVideoRow] = []
    for vid in seen:
        v_comments = [c for c in ordered if c.video_id == vid]
        n = len(v_comments)
        q = sum(1 for c in v_comments if c.category == "question")
        unsorted_v = sum(1 for c in v_comments if c.category not in COMMENT_CATEGORIES)
        if q:
            meta = f"{q} question{'' if q == 1 else 's'} \u00b7 {n} comment{'' if n == 1 else 's'}"
        else:
            meta = f"{n} comment{'' if n == 1 else 's'}"
        vobj = videos.get(vid)
        title = vobj.title if vobj is not None and getattr(vobj, "title", "") else vid
        href = _channel_href(selected, vid, draft_only)
        active = vid == video_filter
        video_rows.append(ChannelVideoRow(video_id=vid, title=title, href=href, meta=meta, unsorted=unsorted_v, active=active))

    unsorted_total = sum(1 for c in ordered if c.category not in COMMENT_CATEGORIES)
    if ordered:
        newest_fetched = max(c.fetched_at for c in ordered)
        fetched_text = format_date(newest_fetched)
    else:
        fetched_text = ""
    video_count = len(seen)
    draft_only_href = _channel_href(selected, video_filter, not draft_only)
    next_path = _channel_href(selected, video_filter, draft_only)
    return ChannelCommentsView(
        chips=tuple(chips),
        rows=tuple(rows),
        videos=tuple(video_rows),
        total=len(ordered),
        filtered_total=len(filtered),
        unsorted=unsorted_total,
        video_count=video_count,
        fetched_text=fetched_text,
        filter_key=selected,
        video_filter=video_filter,
        draft_only=draft_only,
        draft_only_href=draft_only_href,
        next_path=next_path,
    )


@dataclass(frozen=True)
class GlossaryRow:
    index: int
    term: str
    keep: bool
    translations: str


def build_glossary_rows(terms: Sequence["glossary_mod.GlossaryTerm"]) -> tuple[GlossaryRow, ...]:
    rows: list[GlossaryRow] = []
    for idx, gt in enumerate(terms):
        if gt.keep:
            trans_str = ""
        else:
            trans_str = ", ".join(f"{code}: {text}" for code, text in gt.translations)
        rows.append(GlossaryRow(index=idx, term=gt.term, keep=gt.keep, translations=trans_str))
    return tuple(rows)


@dataclass(frozen=True)
class OutdatedRow:
    video_id: str
    title: str
    href: str
    languages: tuple[str, ...]
    subtitles: tuple[str, ...]
    made: str


@dataclass(frozen=True)
class OutdatedView:
    rows: tuple[OutdatedRow, ...]
    more: int


def build_outdated_view(store: Store, limit: int = 100) -> OutdatedView:
    rows: list[OutdatedRow] = []
    for video in store.list_videos():
        draft = store.get_draft(video.video_id)
        if draft is not None:
            text = (draft.title, draft.description)
        else:
            text = (video.title, video.description)
        current_hash = source_hash(*text)

        latest_trans = store.latest_translations(video.video_id)
        outdated_langs: list[str] = []
        dates: list[str] = []
        for code, rec in latest_trans.items():
            if rec.status == "failed":
                continue
            if rec.source_hash != current_hash:
                outdated_langs.append(lang_name(code))
                dates.append(rec.created_at)

        latest = store.latest_transcript(video.video_id)
        # languages that already have a non-failed translation from the latest transcript
        all_subs = list(store.list_subtitle_translations(video.video_id))
        latest_lang_has: set[str] = set()
        if latest is not None:
            for rec in all_subs:
                if rec.transcript_id == latest.id and rec.status != "failed":
                    latest_lang_has.add(rec.language)

        outdated_subs: list[str] = []
        seen: set[str] = set()
        for rec in all_subs:
            if rec.language in seen:
                continue
            seen.add(rec.language)
            if rec.status == "failed":
                continue
            if latest is None:
                continue
            if rec.transcript_id == latest.id:
                continue
            if rec.language in latest_lang_has:
                continue
            outdated_subs.append(lang_name(rec.language))
            dates.append(rec.created_at)

        if outdated_langs or outdated_subs:
            newest = max(dates) if dates else ""
            made = format_date(newest)
            href = f"/videos/{quote(video.video_id)}#translations"
            rows.append(OutdatedRow(
                video_id=video.video_id,
                title=video.title,
                href=href,
                languages=tuple(outdated_langs),
                subtitles=tuple(outdated_subs),
                made=made,
            ))
    more = max(0, len(rows) - limit)
    kept = tuple(rows[:limit])
    return OutdatedView(rows=kept, more=more)


def parse_glossary_form(params: Mapping[str, Any]) -> "glossary_mod.GlossaryTerm":
    term = str(params.get("term") or "")
    keep = "keep" in params
    raw = str(params.get("translations") or "")
    translations: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            raise glossary_mod.GlossaryError(
                'Write each fixed translation as "de = Wort", one per line.'
            )
        code_part, trans_part = line.split("=", 1)
        code = code_part.strip()
        trans = trans_part.strip()
        translations[code] = trans
    return glossary_mod.make_term(term, keep, translations)


# --- affiliate links (G2) -------------------------------------------------------------------------
def _clean_rule_domain(domain: str) -> str:
    """A rule domain as the core stores it: lowercase, no leading "www.", no trailing dot."""
    key = domain.strip().lower()
    if key.startswith("www."):
        key = key[4:]
    return key.rstrip(".")


def parse_affiliate_rule_form(params: Mapping[str, Any]) -> "affiliate_mod.AffiliateRule":
    """One rule from the add-rule form, or AffiliateError with a plain sentence."""
    name = str(params.get("name") or "").strip()
    if not name:
        raise affiliate_mod.AffiliateError("Give the rule a name.")
    domains: list[str] = []
    for part in str(params.get("domains") or "").split(","):
        cleaned = _clean_rule_domain(part)
        if cleaned and cleaned not in domains:
            domains.append(cleaned)
    if not domains:
        raise affiliate_mod.AffiliateError(
            "List at least one domain, separated by commas, like example.com.")
    param = str(params.get("param") or "").strip()
    if not param:
        raise affiliate_mod.AffiliateError("Give the rule a parameter, like tag.")
    return affiliate_mod.AffiliateRule(name, tuple(domains), param,
                                       str(params.get("value") or ""))


@dataclass(frozen=True)
class AffiliateRow:
    video_id: str
    title: str
    tagged: int
    issues: tuple[str, ...]
    blocked: str


@dataclass(frozen=True)
class AffiliatePreview:
    rows: tuple[AffiliateRow, ...]
    ready: int
    blocked: int
    summary: str


def build_affiliate_preview(plan: Sequence["affiliate_mod.AffiliateChange"],
                            videos: Mapping[str, Video]) -> AffiliatePreview:
    rows = tuple(AffiliateRow(
        video_id=change.video_id,
        title=(videos[change.video_id].title if change.video_id in videos
               else change.video_id),
        tagged=change.tagged,
        issues=tuple(issue.message for issue in change.issues),
        blocked=change.blocked) for change in plan)
    blocked = sum(1 for row in rows if row.blocked)
    ready = len(rows) - blocked
    if not rows:
        summary = "No video matches: nothing would change."
    else:
        summary = (f"{_videos(ready)} would change"
                   + (f", {blocked} blocked" if blocked else "") + ".")
    return AffiliatePreview(rows, ready, blocked, summary)


def affiliate_apply_message(result: BulkResult) -> str:
    """The flash text after Make drafts on the Affiliate page."""
    text = (f"Drafts made for {_videos(result.drafted)}." if result.drafted
            else "No drafts were made.")
    if result.blocked:
        text += f" {_videos(result.blocked)} blocked."
    if result.not_in_plan:
        text += f" {_videos(result.not_in_plan)} no longer matched."
    if result.drafted:
        text += (" Review them on the Changes page; nothing changes on YouTube until you "
                 "approve and push.")
    return text


# --- custom checks (G7) -----------------------------------------------------------------------------
FAILING_LIMIT = 20


def explain_rule(rule: Any) -> str:
    """What a stored rule means, in the core's own sentence (never model text)."""
    data = rule.rule if not isinstance(rule, dict) else rule
    if not isinstance(data, dict) or data.get("kind") not in customchecks_mod.KINDS:
        return "This check cannot be run."
    return customchecks_mod.explain(data)


def parse_rule_proposal(raw: str, request: str) -> tuple[dict[str, Any], str]:
    """(clean rule, explain) re-validated from the posted proposal, or RuleError.

    The page never trusts the hidden fields: the rule is checked again in the
    core, and the domain, text or number must still be in the creator's words,
    exactly as `rule_from_request` checks them.
    """
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        raise customchecks_mod.RuleError(
            "That check cannot be saved. Ask for it again.") from None
    if not isinstance(data, dict):
        raise customchecks_mod.RuleError("That check cannot be saved. Ask for it again.")
    return customchecks_mod.validate_rule(data, request)


@dataclass(frozen=True)
class CheckFailing:
    video_id: str
    title: str


@dataclass(frozen=True)
class CheckRowView:
    rule: Any  # a store CustomRule
    explain: str
    has_result: bool
    passing: int
    not_applicable: int
    failing: tuple[CheckFailing, ...]
    more: int


def build_checks_view(rules: Sequence[Any], results: Mapping[int, Any],
                      titles: Mapping[str, str]) -> tuple[CheckRowView, ...]:
    """Every stored rule with its result. `results` maps rule id -> RuleResult (enabled only)."""
    rows: list[CheckRowView] = []
    for rule in rules:
        result = results.get(rule.id)
        if result is None:
            rows.append(CheckRowView(rule, explain_rule(rule), False, 0, 0, (), 0))
            continue
        failing = tuple(CheckFailing(video_id, titles.get(video_id) or video_id)
                        for video_id in result.failing[:FAILING_LIMIT])
        rows.append(CheckRowView(
            rule, explain_rule(rule), True, result.passing, result.not_applicable,
            failing, max(0, len(result.failing) - FAILING_LIMIT)))
    return tuple(rows)


# --- strategy --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyRef:
    """One video a strategy point names. `ratio` is the code-made label, or ""."""

    video_id: str
    title: str
    href: str
    ratio: str


@dataclass(frozen=True)
class StrategyPointView:
    text: str
    refs: tuple[StrategyRef, ...]


@dataclass(frozen=True)
class StrategyIdeaView:
    title: str
    why: str
    refs: tuple[StrategyRef, ...]


@dataclass(frozen=True)
class StrategyReportView:
    """The newest saved strategy report, ready to show. All model text stays escaped."""

    works: tuple[StrategyPointView, ...]
    gaps: tuple[StrategyPointView, ...]
    ideas: tuple[StrategyIdeaView, ...]
    notes: tuple[str, ...]
    model: str
    tokens: str
    cost: str
    when: str
    stale: bool


def _strategy_moment(text: str) -> datetime | None:
    """Parse an ISO time (a trailing Z counts as UTC); None when it cannot be read."""
    try:
        moment = datetime.fromisoformat((text or "").strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def build_strategy_view(report: StrategyReport | None, videos: Sequence[Video],
                        scores: Sequence[OutlierScore], current_as_of: datetime | None,
                        when: Callable[[str], str] = lambda iso: iso) -> StrategyReportView | None:
    """The saved strategy report with current titles and code-made ratio labels.

    A video that left the catalog shows its id as its title. The ratio shows only for
    outliers and videos below the norm, and every number in it comes from code. Refs like V12
    in the model's text become the title of that video when the point names it.
    """
    if report is None:
        return None
    titles = {v.video_id: v.title for v in videos}
    score_by_id = {s.video_id: s for s in scores}
    marked = {s.video_id for s in outliers(scores)} | {s.video_id for s in below_norm(scores)}
    ref_ids = catalog_refs(videos)

    def named(text: str, ids: Sequence[str]) -> str:
        def title_of(m: re.Match[str]) -> str:
            vid_id = ref_ids.get(m.group(0))
            if vid_id is not None and vid_id in ids:
                return f'"{titles.get(vid_id) or vid_id}"'
            return m.group(0)
        return REF.sub(title_of, text)

    def ref_for(video_id: str) -> StrategyRef:
        score = score_by_id.get(video_id)
        ratio = (ratio_label(score.ratio)
                 if video_id in marked and score is not None and score.ratio is not None else "")
        return StrategyRef(video_id, titles.get(video_id) or video_id,
                           f"/videos/{quote(video_id)}", ratio)

    def refs_of(ids: Sequence[str]) -> tuple[StrategyRef, ...]:
        return tuple(ref_for(video_id) for video_id in ids)

    stale = False
    if current_as_of is not None:
        created = _strategy_moment(report.created_at)
        current = (current_as_of.replace(tzinfo=timezone.utc)
                   if current_as_of.tzinfo is None else current_as_of)
        stale = created is not None and created < current
    tokens = (f"{report.input_tokens:,} in, {report.output_tokens:,} out"
              if report.input_tokens or report.output_tokens else "")
    cost = format_cost(estimate_cost(report.model, report.input_tokens,
                                     report.cached_tokens, report.output_tokens))
    return StrategyReportView(
        works=tuple(StrategyPointView(named(p.text, p.video_ids), refs_of(p.video_ids))
                   for p in report.works),
        gaps=tuple(StrategyPointView(named(p.text, p.video_ids), refs_of(p.video_ids))
                   for p in report.gaps),
        ideas=tuple(StrategyIdeaView(named(i.title, i.video_ids), named(i.why, i.video_ids),
                                    refs_of(i.video_ids))
                   for i in report.ideas),
        notes=tuple(report.notes), model=report.model, tokens=tokens, cost=cost,
        when=when(report.created_at), stale=stale)
