"""Page models for the dashboard, the content table, the video page and the changes pages.

Pure functions: they take videos and scorecards and return plain data for the templates, so
nothing here touches the disk or YouTube. Formatting (view counts, dates) lives here too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence
from urllib.parse import urlencode

from .. import edits
from ..agent import estimate_cost
from ..diffs import DraftDiff, diff_draft
from ..push import PushItem
from ..seo import CATEGORIES, CheckResult, Scorecard, fix_priority
from ..store import AgentRun, DraftRow, PushRecord, Video
from ..validate import Issue, has_errors

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
SORTS: tuple[tuple[str, str], ...] = (
    ("fix", "Fix next"),
    ("score", "Lowest score"),
    ("views", "Most views"),
    ("newest", "Newest"),
)
SORT_KEYS = tuple(s[0] for s in SORTS)
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


# --- entries ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Entry:
    """A video with its scorecard and the display strings the templates need."""

    video: Video
    card: Scorecard

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


def listing_href(q: str = "", sort: str = "fix", visibility: str = "all", band: str = "",
                 check: str = "", page: int = 1) -> str:
    """A /videos URL that carries only the non-default parameters."""
    params: list[tuple[str, str]] = []
    if q:
        params.append(("q", q))
    if sort != "fix":
        params.append(("sort", sort))
    if visibility != "all":
        params.append(("visibility", visibility))
    if band:
        params.append(("band", band))
    if check:
        params.append(("check", check))
    if page > 1:
        params.append(("page", str(page)))
    return "/videos" + (f"?{urlencode(params)}" if params else "")


@dataclass
class Chip:
    label: str
    href: str
    active: bool


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
    sort_chips: list[Chip] = field(default_factory=list)
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


def _sort(entries: list[Entry], sort: str) -> list[Entry]:
    if sort == "score":
        return sorted(entries, key=lambda e: (e.card.score, -e.video.view_count, e.video.video_id))
    if sort == "views":
        return sorted(entries, key=lambda e: (-e.video.view_count, e.video.video_id))
    if sort == "newest":
        return sorted(entries, key=lambda e: (e.video.published_at or "", e.video.video_id),
                      reverse=True)
    return sorted(entries, key=lambda e: (-e.priority, -e.card.missing, e.video.video_id))


def build_listing(entries: Sequence[Entry], params: Mapping[str, str]) -> Listing:
    """Filter, sort and paginate. Unknown values fall back to the defaults; never raises."""
    q = (params.get("q") or "").strip()[:MAX_QUERY_CHARS]
    sort = _pick(params.get("sort"), SORT_KEYS, "fix")
    visibility = _pick(params.get("visibility"), VISIBILITY_KEYS, "all")
    band = _pick(params.get("band"), BAND_KEYS, "")
    check = (params.get("check") or "").strip()[:100]

    rows = list(entries)
    if q:
        needle = q.casefold()
        rows = [e for e in rows if _matches(e, needle)]
    if visibility != "all":
        rows = [e for e in rows if e.video.privacy_status == visibility]
    if band:
        rows = [e for e in rows if e.band == band]
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
    rows = _sort(rows, sort)

    total = len(rows)
    pages = max(1, -(-total // PAGE_SIZE))
    page = min(max(_page_number(params.get("page")), 1), pages)
    shown = rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    def href(**changes: object) -> str:
        args: dict[str, object] = {"q": q, "sort": sort, "visibility": visibility, "band": band,
                                   "check": check, "page": 1}
        args.update(changes)
        return listing_href(**args)  # type: ignore[arg-type]

    listing = Listing(
        entries=shown, total=total, page=page, pages=pages, q=q, sort=sort,
        visibility=visibility, band=band, check=check, check_title=check_title,
        visibility_chips=[Chip(label, href(visibility=key), key == visibility)
                          for key, label in VISIBILITIES],
        band_chips=[Chip("Any score", href(band=""), band == "")]
        + [Chip(f"{label} ({rng})", href(band=key), key == band) for key, label, rng in BANDS],
        sort_chips=[Chip(label, href(sort=key), key == sort) for key, label in SORTS],
        prev_href=href(page=page - 1) if page > 1 else "",
        next_href=href(page=page + 1) if page < pages else "",
        clear_check_href=href(check=""),
        filtered=bool(q or check or band or visibility != "all"),
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
ORIGIN_LABELS = {"manual": "Edited by you", "restore": "Restored text"}

# operation -> plain label, for the quota table on the push page
QUOTA_LABELS = {
    "videos.list": "Read the live text of your videos first",
    "videos.update": "Update one video",
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
    diff = diff_draft(row.base, row.draft)
    origin = row.note or ORIGIN_LABELS.get(row.source, row.source.capitalize())
    return ChangeCard(
        row=row, video=video, diff=diff, errors=errors, warnings=warnings, stale=stale,
        ready=row.approved_at is not None and not block, approve_block=block,
        fingerprint=edits.fingerprint(row), summary=summarize_diff(diff), origin=origin,
    )


def change_cards(pairs: Iterable[tuple[DraftRow, Video]]) -> list[ChangeCard]:
    return [build_change_card(row, video) for row, video in pairs]


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


def backup_of_record(record: PushRecord) -> Backup | None:
    """The saved text of a push record, or None when it has no title to restore."""
    before = record.before
    title = before.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    description = before.get("description")
    tags = before.get("tags")
    return Backup(
        title,
        description if isinstance(description, str) else "",
        tuple(t for t in tags if isinstance(t, str)) if isinstance(tags, list) else (),
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
