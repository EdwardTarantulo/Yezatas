"""SQLite store: the local video catalog, per-pull snapshots and the quota ledger."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # validate imports seo, which imports this module
    from .validate import Draft

_SCHEMA_V1: tuple[str, ...] = (
    """CREATE TABLE pulls (
        id INTEGER PRIMARY KEY,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        channel_id TEXT NOT NULL,
        video_count INTEGER
    )""",
    """CREATE TABLE videos (
        video_id TEXT PRIMARY KEY,
        channel_id TEXT,
        title TEXT,
        description TEXT,
        tags_json TEXT,
        privacy_status TEXT,
        default_language TEXT,
        default_audio_language TEXT,
        published_at TEXT,
        duration TEXT,
        view_count INTEGER,
        like_count INTEGER,
        comment_count INTEGER,
        localizations_json TEXT,
        thumbnails_json TEXT,
        raw_json TEXT,
        updated_at TEXT NOT NULL,
        last_pull_id INTEGER REFERENCES pulls(id)
    )""",
    """CREATE TABLE snapshots (
        id INTEGER PRIMARY KEY,
        pull_id INTEGER NOT NULL REFERENCES pulls(id),
        video_id TEXT NOT NULL,
        taken_at TEXT NOT NULL,
        data_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_snapshots_video_taken ON snapshots(video_id, taken_at)",
    """CREATE TABLE quota_ledger (
        id INTEGER PRIMARY KEY,
        ts_utc TEXT NOT NULL,
        quota_day TEXT NOT NULL,
        operation TEXT NOT NULL,
        units INTEGER NOT NULL,
        count INTEGER NOT NULL,
        note TEXT
    )""",
    "CREATE INDEX idx_quota_ledger_day ON quota_ledger(quota_day)",
)

_SCHEMA_V2: tuple[str, ...] = (
    """CREATE TABLE drafts (
        video_id TEXT PRIMARY KEY REFERENCES videos(video_id),
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        tags_json TEXT NOT NULL,
        base_title TEXT NOT NULL,
        base_description TEXT NOT NULL,
        base_tags_json TEXT NOT NULL,
        source TEXT NOT NULL,
        note TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        approved_at TEXT
    )""",
    """CREATE TABLE pushes (
        id INTEGER PRIMARY KEY,
        video_id TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        status TEXT NOT NULL,
        source TEXT NOT NULL,
        before_json TEXT NOT NULL,
        after_json TEXT NOT NULL DEFAULT '{}',
        message TEXT NOT NULL DEFAULT ''
    )""",
    "CREATE INDEX idx_pushes_video ON pushes(video_id, id)",
)

_SCHEMA_V3: tuple[str, ...] = (
    """CREATE TABLE agent_runs (
        id INTEGER PRIMARY KEY,
        video_id TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        model TEXT NOT NULL,
        status TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        log_json TEXT NOT NULL DEFAULT '[]',
        result_json TEXT NOT NULL DEFAULT '{}',
        input_tokens INTEGER NOT NULL DEFAULT 0,
        cached_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX idx_agent_runs_video ON agent_runs(video_id, id)",
)

# v4: drafts may also set the video language and the translations (NULL: leave YouTube's as is),
# and every machine translation is kept with its checks.
_SCHEMA_V4: tuple[str, ...] = (
    "ALTER TABLE drafts ADD COLUMN default_language TEXT",
    "ALTER TABLE drafts ADD COLUMN localizations_json TEXT",
    "ALTER TABLE drafts ADD COLUMN base_default_language TEXT",
    "ALTER TABLE drafts ADD COLUMN base_localizations_json TEXT NOT NULL DEFAULT '{}'",
    # Older drafts were made from the catalog's current language and translations: record them as
    # the base, or every such draft would look stale.
    """UPDATE drafts SET
        base_default_language = (SELECT default_language FROM videos
                                 WHERE videos.video_id = drafts.video_id),
        base_localizations_json = COALESCE((SELECT localizations_json FROM videos
                                            WHERE videos.video_id = drafts.video_id), '{}')""",
    """CREATE TABLE translations (
        id INTEGER PRIMARY KEY,
        video_id TEXT NOT NULL,
        language TEXT NOT NULL,
        source_language TEXT NOT NULL,
        source_hash TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        translator TEXT NOT NULL,
        verifier TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL,
        score INTEGER,
        issues_json TEXT NOT NULL DEFAULT '[]',
        problems_json TEXT NOT NULL DEFAULT '[]',
        back_translation_json TEXT NOT NULL DEFAULT '{}',
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX idx_translations_video ON translations(video_id, language, id)",
)

_SCHEMA_V5: tuple[str, ...] = (
    """CREATE TABLE transcripts (
        id INTEGER PRIMARY KEY,
        video_id TEXT NOT NULL,
        language TEXT NOT NULL DEFAULT '',
        model TEXT NOT NULL,
        source_name TEXT NOT NULL DEFAULT '',
        duration_ms INTEGER NOT NULL DEFAULT 0,
        segments_json TEXT NOT NULL DEFAULT '[]',
        problems_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX idx_transcripts_video ON transcripts(video_id, id)",
)

# v6: translated subtitles. Every row holds all of a transcript's segments with the transcript's own
# times; only the text is translated.
_SCHEMA_V6: tuple[str, ...] = (
    """CREATE TABLE subtitle_translations (
        id INTEGER PRIMARY KEY,
        video_id TEXT NOT NULL,
        transcript_id INTEGER NOT NULL,
        language TEXT NOT NULL,
        source_language TEXT NOT NULL,
        segments_json TEXT NOT NULL DEFAULT '[]',
        translator TEXT NOT NULL,
        verifier TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL,
        score INTEGER,
        issues_json TEXT NOT NULL DEFAULT '[]',
        problems_json TEXT NOT NULL DEFAULT '[]',
        back_translation_json TEXT NOT NULL DEFAULT '{}',
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX idx_subtitle_translations_video "
    "ON subtitle_translations(video_id, transcript_id, language, id)",
)

# v7: packaging checks and thumbnail variant ideas. Image files are kept by bare file name under
# <home>/thumbnails/<video_id>/; the name is generated by code, never taken from a model.
_SCHEMA_V7: tuple[str, ...] = (
    """CREATE TABLE packaging_reviews (
        id INTEGER PRIMARY KEY,
        video_id TEXT NOT NULL,
        title TEXT NOT NULL,
        thumbnail_url TEXT NOT NULL,
        image_file TEXT NOT NULL DEFAULT '',
        measurements_json TEXT NOT NULL DEFAULT '{}',
        reading_json TEXT NOT NULL DEFAULT '{}',
        critique_json TEXT NOT NULL DEFAULT '{}',
        problems_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL,
        error TEXT NOT NULL DEFAULT '',
        models_json TEXT NOT NULL DEFAULT '{}',
        input_tokens INTEGER NOT NULL DEFAULT 0,
        cached_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        cost_usd REAL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE thumbnail_variants (
        id INTEGER PRIMARY KEY,
        review_id INTEGER NOT NULL,
        video_id TEXT NOT NULL,
        instruction TEXT NOT NULL,
        overlay_text TEXT NOT NULL DEFAULT '',
        image_file TEXT NOT NULL DEFAULT '',
        text_read TEXT NOT NULL DEFAULT '',
        text_ok INTEGER,
        measurements_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL,
        error TEXT NOT NULL DEFAULT '',
        model TEXT NOT NULL DEFAULT '',
        cost_usd REAL,
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX idx_packaging_reviews_video ON packaging_reviews(video_id, id)",
    "CREATE INDEX idx_thumbnail_variants_review ON thumbnail_variants(review_id, id)",
)

# v8: own-channel outlier reports (one Muse Spark explanation of what the best videos share). The
# scores themselves are not stored: they are recomputed from the catalog.
_SCHEMA_V8: tuple[str, ...] = (
    """CREATE TABLE outlier_reports (
        id INTEGER PRIMARY KEY,
        pull_id INTEGER,
        as_of TEXT NOT NULL,
        params_json TEXT NOT NULL DEFAULT '{}',
        labels_json TEXT NOT NULL DEFAULT '{}',
        insight_json TEXT NOT NULL DEFAULT '{}',
        problems_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL,
        error TEXT NOT NULL DEFAULT '',
        model TEXT NOT NULL DEFAULT '',
        input_tokens INTEGER NOT NULL DEFAULT 0,
        cached_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        cost_usd REAL,
        created_at TEXT NOT NULL
    )""",
)

# v9: push runs. One push of several drafts is one run, so History can undo it in one step. Older
# pushes keep run_id NULL (they are restored one by one; groups are never guessed).
_SCHEMA_V9: tuple[str, ...] = (
    "ALTER TABLE pushes ADD COLUMN run_id INTEGER",
    """CREATE TABLE push_runs (
        id INTEGER PRIMARY KEY,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        video_count INTEGER NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX idx_pushes_run ON pushes(run_id, id)",
)

# v10: the copilot chat. One row per question and its answer; proposals are edits the model
# offered in an answer, which become drafts only when the creator asks.
_SCHEMA_V10: tuple[str, ...] = (
    """CREATE TABLE copilot_turns (
        id INTEGER PRIMARY KEY,
        chat_id INTEGER NOT NULL,
        asked_at TEXT NOT NULL,
        finished_at TEXT,
        question TEXT NOT NULL,
        model TEXT NOT NULL,
        status TEXT NOT NULL,
        answer TEXT NOT NULL DEFAULT '',
        refs_json TEXT NOT NULL DEFAULT '{}',
        notes_json TEXT NOT NULL DEFAULT '[]',
        log_json TEXT NOT NULL DEFAULT '[]',
        input_tokens INTEGER NOT NULL DEFAULT 0,
        cached_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX idx_copilot_turns_chat ON copilot_turns(chat_id, id)",
    """CREATE TABLE copilot_proposals (
        id INTEGER PRIMARY KEY,
        turn_id INTEGER NOT NULL,
        video_id TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        tags_json TEXT NOT NULL DEFAULT '[]',
        reason TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'open',
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX idx_copilot_proposals_turn ON copilot_proposals(turn_id, id)",
)

# v11: link checks (G1: is each description link still alive) and advertiser-friendly checks (C9:
# advice from Muse Spark, with every quote checked against the video's own text).
_SCHEMA_V11: tuple[str, ...] = (
    """CREATE TABLE link_checks (
        id INTEGER PRIMARY KEY,
        url TEXT NOT NULL,
        checked_at TEXT NOT NULL,
        status TEXT NOT NULL,
        http_status INTEGER,
        final_url TEXT NOT NULL DEFAULT '',
        message TEXT NOT NULL DEFAULT ''
    )""",
    "CREATE INDEX idx_link_checks_url ON link_checks(url, id)",
    """CREATE TABLE ad_checks (
        id INTEGER PRIMARY KEY,
        video_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        model TEXT NOT NULL,
        status TEXT NOT NULL,
        verdict TEXT NOT NULL DEFAULT '',
        summary TEXT NOT NULL DEFAULT '',
        findings_json TEXT NOT NULL DEFAULT '[]',
        notes_json TEXT NOT NULL DEFAULT '[]',
        source_hash TEXT NOT NULL DEFAULT '',
        had_transcript INTEGER NOT NULL DEFAULT 0,
        error TEXT NOT NULL DEFAULT '',
        input_tokens INTEGER NOT NULL DEFAULT 0,
        cached_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX idx_ad_checks_video ON ad_checks(video_id, id)",
)

# v12: comments (G4: fetched comments and their categories, reply and pinned-comment drafts,
# reply templates) and custom checks (G7: rules the creator described in plain words).
_SCHEMA_V12: tuple[str, ...] = (
    """CREATE TABLE comments (
        comment_id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        author TEXT NOT NULL DEFAULT '',
        text TEXT NOT NULL,
        published_at TEXT NOT NULL DEFAULT '',
        like_count INTEGER NOT NULL DEFAULT 0,
        reply_count INTEGER NOT NULL DEFAULT 0,
        fetched_at TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT '',
        classified_at TEXT NOT NULL DEFAULT ''
    )""",
    "CREATE INDEX idx_comments_video_published ON comments(video_id, published_at)",
    """CREATE TABLE comment_replies (
        id INTEGER PRIMARY KEY,
        video_id TEXT NOT NULL,
        comment_id TEXT NOT NULL DEFAULT '',
        text TEXT NOT NULL,
        model TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'open',
        notes_json TEXT NOT NULL DEFAULT '[]'
    )""",
    "CREATE INDEX idx_comment_replies_video ON comment_replies(video_id, id)",
    """CREATE TABLE reply_templates (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        body TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE custom_rules (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        request TEXT NOT NULL DEFAULT '',
        rule_json TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    )""",
)

# v13: Shorts finder runs (D5).
_SCHEMA_V13: tuple[str, ...] = (
    """CREATE TABLE shorts_finds (
        id INTEGER PRIMARY KEY,
        video_id TEXT NOT NULL,
        transcript_id INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        model TEXT NOT NULL,
        status TEXT NOT NULL,
        message TEXT NOT NULL DEFAULT '',
        clips_json TEXT NOT NULL DEFAULT '[]',
        notes_json TEXT NOT NULL DEFAULT '[]',
        error TEXT NOT NULL DEFAULT '',
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX idx_shorts_finds_video ON shorts_finds(video_id, id)",
)

# v14: whether the channel itself replied to a comment (Today page).
_SCHEMA_V14: tuple[str, ...] = (
    "ALTER TABLE comments ADD COLUMN owner_replied INTEGER NOT NULL DEFAULT 0",
)

# v15: the category check's verdict per video, and the pinned comment (mark and suggestion) per video.
_SCHEMA_V15: tuple[str, ...] = (
    """CREATE TABLE category_checks (
        video_id TEXT PRIMARY KEY,
        checked_at TEXT NOT NULL,
        model TEXT NOT NULL,
        status TEXT NOT NULL,
        current_id TEXT NOT NULL DEFAULT '',
        suggested_id TEXT NOT NULL DEFAULT '',
        reason TEXT NOT NULL DEFAULT '',
        confirmed_at TEXT NOT NULL DEFAULT '',
        error TEXT NOT NULL DEFAULT '',
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE pinned_comments (
        video_id TEXT PRIMARY KEY,
        marked_at TEXT NOT NULL DEFAULT '',
        suggestion TEXT NOT NULL DEFAULT '',
        suggested_at TEXT NOT NULL DEFAULT '',
        model TEXT NOT NULL DEFAULT '',
        error TEXT NOT NULL DEFAULT '',
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0
    )""",
)

# MIGRATIONS[i] upgrades the database from user_version i to i + 1.
MIGRATIONS: tuple[tuple[str, ...], ...] = (
    _SCHEMA_V1, _SCHEMA_V2, _SCHEMA_V3, _SCHEMA_V4, _SCHEMA_V5, _SCHEMA_V6, _SCHEMA_V7,
    _SCHEMA_V8, _SCHEMA_V9, _SCHEMA_V10, _SCHEMA_V11, _SCHEMA_V12, _SCHEMA_V13, _SCHEMA_V14, _SCHEMA_V15,
)

_ORDER_COLUMNS = frozenset(
    {"view_count", "like_count", "comment_count", "published_at", "title", "updated_at", "video_id"}
)


def _iso(now: datetime | None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return now.astimezone(timezone.utc).isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass
class Video:
    video_id: str
    channel_id: str = ""
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    privacy_status: str = ""
    default_language: str | None = None
    default_audio_language: str | None = None
    published_at: str = ""
    duration: str = ""
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    localizations: dict[str, Any] = field(default_factory=dict)
    thumbnails: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Video":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass(frozen=True)
class DraftRow:
    """A proposed edit of one video, with the catalog text it was made from (its base)."""

    video_id: str
    title: str
    description: str
    tags: tuple[str, ...]
    base_title: str
    base_description: str
    base_tags: tuple[str, ...]
    source: str
    note: str
    created_at: str
    updated_at: str
    approved_at: str | None
    # None: the draft leaves YouTube's value alone. Otherwise the value to send: the video
    # language, and the complete translations map {language: {"title", "description"}}.
    default_language: str | None = None
    localizations: dict[str, dict[str, str]] | None = None
    base_default_language: str | None = None
    base_localizations: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def draft(self) -> "Draft":
        from .validate import Draft

        return Draft(self.title, self.description, self.tags)

    @property
    def base(self) -> "Draft":
        from .validate import Draft

        return Draft(self.base_title, self.base_description, self.base_tags)


@dataclass(frozen=True)
class PushRecord:
    """One push attempt; `before` is YouTube's live text right before it (the backup)."""

    id: int
    video_id: str
    started_at: str
    finished_at: str | None
    status: str  # "pending" | "pushed" | "conflict" | "failed"
    source: str
    before: dict[str, Any]
    after: dict[str, Any]
    message: str
    run_id: int | None = None  # the push run it belongs to; None for pushes made before v9


@dataclass(frozen=True)
class PushRun:
    """One call of push_drafts: the pushes it made share its id."""

    id: int
    started_at: str
    finished_at: str | None
    video_count: int


@dataclass(frozen=True)
class AgentRun:
    """One run of the suggestion agent on one video: what it called, what it proposed, what it cost."""

    id: int
    video_id: str
    started_at: str
    finished_at: str | None
    model: str
    status: str  # "running" | "drafted" | "no_change" | "rejected" | "failed"
    summary: str
    log: list[Any]  # events in order: tool calls, guard problems, repair rounds
    result: dict[str, Any]  # the suggestion as returned (title, description, tags, changes, summary)
    input_tokens: int
    cached_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class CopilotTurn:
    """One question to the copilot and its answer, with what it looked at and what it cost."""

    id: int
    chat_id: int
    asked_at: str
    finished_at: str | None
    question: str
    model: str
    status: str  # "running" | "answered" | "failed"
    answer: str  # the checked answer; videos appear as refs like "V12" (see `refs`)
    refs: dict[str, str]  # ref -> video id, for the refs the answer and its proposals use
    notes: list[str]  # what the checks removed or changed, shown under the answer
    log: list[Any]  # events in order: tool calls, guard problems, errors
    input_tokens: int
    cached_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class CopilotProposal:
    """An edit the copilot offered for one video. It becomes a draft only when the creator asks."""

    id: int
    turn_id: int
    video_id: str
    title: str
    description: str
    tags: tuple[str, ...]
    reason: str
    status: str  # "open" | "drafted" | "dismissed"
    created_at: str


@dataclass(frozen=True)
class LinkCheck:
    """One check of one link: whether it still answers, as of `checked_at`."""

    id: int
    url: str
    checked_at: str
    status: str  # "ok" | "redirected" | "broken" | "error" | "skipped"
    http_status: int | None
    final_url: str  # where a redirect ended ("" when it did not move)
    message: str  # a plain sentence for the creator ("Not found (404).")


@dataclass(frozen=True)
class AdCheck:
    """One advertiser-friendly check of one video: advice only, with quotes checked by code."""

    id: int  # 0 before it is stored
    video_id: str
    created_at: str
    model: str
    status: str  # "done" | "failed"
    verdict: str  # "no_concerns" | "worth_a_look" | "likely_limited" ("" when failed)
    summary: str
    findings: list[dict[str, Any]]  # each with category, source, quote, (time), why, fix
    notes: list[str]  # what the checks dropped ("A finding was dropped: its quote is not in ...")
    source_hash: str  # the text it read (title, description, tags, transcript id)
    had_transcript: bool
    error: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class CategoryCheck:
    """Muse Spark's view of which YouTube category fits one video (the newest check only)."""

    video_id: str
    checked_at: str
    model: str
    status: str  # "done" | "failed"
    current_id: str  # the video's category id when it was checked ("" when it had none)
    suggested_id: str  # the category id Muse Spark named ("" when failed)
    reason: str
    confirmed_at: str  # when the user said the current category is correct ("" when not)
    error: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ShortsFind:
    """One Shorts finder run on one video: moments picked by segment number, times and quotes from the transcript."""

    id: int
    video_id: str
    transcript_id: int
    created_at: str
    model: str
    status: str  # "found" | "none" | "failed"
    message: str
    clips: list[dict[str, Any]]
    notes: list[str]
    error: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Comment:
    """One top-level comment on a video, as fetched from YouTube, with its category ("" until classified)."""

    comment_id: str
    video_id: str
    author: str = ""
    text: str = ""
    published_at: str = ""
    like_count: int = 0
    reply_count: int = 0
    fetched_at: str = ""
    category: str = ""  # a key of comments.CATEGORIES, or "" when not classified yet
    classified_at: str = ""
    owner_replied: bool = False  # the channel itself replied on YouTube (seen at the last fetch)


@dataclass(frozen=True)
class CommentReply:
    """One reply draft for a comment (or, with an empty `comment_id`, the pinned-comment draft
    for the video). A draft only: the creator posts it themself; nothing is sent to YouTube."""

    id: int
    video_id: str
    comment_id: str  # "" is the pinned-comment draft for the video
    text: str
    model: str
    created_at: str
    status: str  # "open" | "used" | "dismissed"
    notes: tuple[str, ...]  # what the guards dropped or changed, in plain words


@dataclass(frozen=True)
class ReplyTemplate:
    """One saved reply text the creator can reuse; `{name}` in the body is the commenter's name."""

    id: int
    name: str
    body: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CustomRule:
    """One custom check the creator described in plain words, turned into a fixed rule by code."""

    id: int
    name: str
    request: str  # the creator's own words the rule was made from
    rule: dict[str, Any]  # the fixed rule: {"kind": ..., ...} (see customchecks.KINDS)
    enabled: bool
    created_at: str


@dataclass(frozen=True)
class TranslationRecord:
    """One machine translation of a video's title and description, with its checks."""

    id: int  # 0 before it is stored
    video_id: str
    language: str  # YouTube language code of the translation, e.g. "es"
    source_language: str  # the language of the text that was translated
    source_hash: str  # identifies the exact source title + description (see translate.py)
    title: str
    description: str
    translator: str  # model id that translated
    verifier: str  # model id that checked it ("" when it could not be checked)
    status: str  # "verified" | "flagged" | "unverified" | "failed"
    score: int | None  # the verifier's 1-5 score
    issues: list[str]  # the verifier's issues
    problems: list[str]  # the code checks' problems (links, times, limits)
    back_translation: dict[str, str]  # the verifier's blind back-translation: title, description
    input_tokens: int
    output_tokens: int
    created_at: str


@dataclass(frozen=True)
class TranscriptRecord:
    """A timed transcript of a video's audio. Every time comes from the audio itself: the segment
    boundaries are where the audio was cut, so chapters and subtitles never invent a time."""

    id: int  # 0 before it is stored
    video_id: str
    language: str  # the language hint it was transcribed with, e.g. "en" ("" for none)
    model: str  # model id that transcribed it
    source_name: str  # the audio or video file's name (not its full path)
    duration_ms: int  # length of the audio
    segments: list[dict[str, Any]]  # [{"start_ms", "end_ms", "text"}], in order
    problems: list[str]  # pieces that could not be transcribed, as short sentences
    created_at: str


@dataclass(frozen=True)
class SubtitleTranslationRecord:
    """A transcript's subtitles translated into one language, with their checks. `segments` has ALL
    the transcript's segments in order; the times are copied from the transcript, only the text is
    translated (an empty segment stays empty)."""

    id: int  # 0 before it is stored
    video_id: str
    transcript_id: int  # the transcript these subtitles were made from
    language: str  # YouTube language code of the subtitles, e.g. "es"
    source_language: str  # the transcript's language
    segments: list[dict[str, Any]]  # [{"start_ms", "end_ms", "text"}], in order
    translator: str  # model id that translated
    verifier: str  # model id that checked it ("" when it could not be checked)
    status: str  # "verified" | "flagged" | "unverified" | "failed"
    score: int | None  # the verifier's 1-5 score (lowest over its chunks)
    issues: list[str]  # the verifier's issues
    problems: list[str]  # the code checks' problems
    back_translation: dict[str, Any]  # {"segments": [...]} or {"note": "..."} or {}
    input_tokens: int
    output_tokens: int
    created_at: str


@dataclass(frozen=True)
class PackagingReview:
    """One packaging check of a video: the title and thumbnail judged together. Numbers in
    `measurements` come from code (Pillow, SAM boxes); the models only wrote `reading` and
    `critique`. `image_file` is a bare file name under <home>/thumbnails/<video_id>/ ("" if none)."""

    id: int  # 0 before it is stored
    video_id: str
    title: str  # the title that was reviewed
    thumbnail_url: str
    image_file: str
    measurements: dict[str, Any]
    reading: dict[str, Any]  # what the vision model read: text, text_zone, subject, face
    critique: dict[str, Any]  # {"points": [...], "ideas": [{"instruction", "overlay_text"}]}
    problems: list[str]  # things the fact guard dropped, as short sentences
    status: str  # "done" | "failed"
    error: str
    models: dict[str, str]  # {"vision": ..., "segmenter": ...}
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    cost_usd: float | None
    created_at: str


@dataclass(frozen=True)
class ThumbnailVariant:
    """One variant idea made from a review's thumbnail by Muse Image (an AI-made image to download,
    and to upload only when the creator confirms it on its own page). `text_ok` is None when the text
    was not checked."""

    id: int  # 0 before it is stored
    review_id: int
    video_id: str
    instruction: str
    overlay_text: str
    image_file: str  # 1280x720 JPEG, a bare file name under <home>/thumbnails/<video_id>/
    text_read: str  # what the vision model reads on the variant
    text_ok: bool | None
    measurements: dict[str, Any]
    status: str  # "ok" | "flagged" | "failed"
    error: str
    model: str
    cost_usd: float | None
    created_at: str


@dataclass(frozen=True)
class OutlierReport:
    """One outlier explanation. Numbers in `labels` come from code; the model only wrote `insight`
    (patterns, per_video, ideas), after the guards. Labels (O1.., T1..) map to videos in `labels`:
    {label: {"video_id", "ratio", "views", "peer_median"}}. `pull_id` is the pull the views came from."""

    id: int  # 0 before it is stored
    pull_id: int | None
    as_of: str  # the time of the views
    params: dict[str, Any]
    labels: dict[str, Any]
    insight: dict[str, Any]
    problems: list[str]
    status: str  # "done" | "too_few" | "error"
    error: str
    model: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    cost_usd: float | None
    created_at: str


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    @property
    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def _migrate(self) -> None:
        for target in range(self.schema_version, len(MIGRATIONS)):
            # sqlite3 starts a transaction before DML only; DDL is run inside an explicit one
            # so a failed migration leaves the version untouched.
            self._conn.isolation_level = None
            try:
                self._conn.execute("BEGIN")
                for statement in MIGRATIONS[target]:
                    self._conn.execute(statement)
                self._conn.execute(f"PRAGMA user_version = {target + 1}")
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            finally:
                self._conn.isolation_level = ""

    # --- pulls -----------------------------------------------------------
    def begin_pull(self, channel_id: str, now: datetime | None = None) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO pulls (started_at, channel_id) VALUES (?, ?)",
                (_iso(now), channel_id),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def finish_pull(self, pull_id: int, video_count: int, now: datetime | None = None) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE pulls SET finished_at = ?, video_count = ? WHERE id = ?",
                (_iso(now), video_count, pull_id),
            )

    def get_pull(self, pull_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM pulls WHERE id = ?", (pull_id,)).fetchone()

    def latest_pull(self) -> sqlite3.Row | None:
        """The newest finished pull, or None."""
        return self._conn.execute(
            "SELECT * FROM pulls WHERE finished_at IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()

    # --- videos ----------------------------------------------------------
    def upsert_video(self, video: Video, pull_id: int, now: datetime | None = None) -> None:
        """Update the current state of a video and add a snapshot, in one transaction."""
        ts = _iso(now)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO videos (video_id, channel_id, title, description, tags_json,
                    privacy_status, default_language, default_audio_language, published_at,
                    duration, view_count, like_count, comment_count, localizations_json,
                    thumbnails_json, raw_json, updated_at, last_pull_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    channel_id=excluded.channel_id, title=excluded.title,
                    description=excluded.description, tags_json=excluded.tags_json,
                    privacy_status=excluded.privacy_status,
                    default_language=excluded.default_language,
                    default_audio_language=excluded.default_audio_language,
                    published_at=excluded.published_at, duration=excluded.duration,
                    view_count=excluded.view_count, like_count=excluded.like_count,
                    comment_count=excluded.comment_count,
                    localizations_json=excluded.localizations_json,
                    thumbnails_json=excluded.thumbnails_json, raw_json=excluded.raw_json,
                    updated_at=excluded.updated_at, last_pull_id=excluded.last_pull_id
                """,
                (
                    video.video_id, video.channel_id, video.title, video.description,
                    _dumps(video.tags), video.privacy_status, video.default_language,
                    video.default_audio_language, video.published_at, video.duration,
                    video.view_count, video.like_count, video.comment_count,
                    _dumps(video.localizations), _dumps(video.thumbnails), _dumps(video.raw),
                    ts, pull_id,
                ),
            )
            self._conn.execute(
                "INSERT INTO snapshots (pull_id, video_id, taken_at, data_json) "
                "VALUES (?, ?, ?, ?)",
                (pull_id, video.video_id, ts, _dumps(video.to_dict())),
            )

    @staticmethod
    def _row_to_video(row: sqlite3.Row) -> Video:
        return Video(
            video_id=row["video_id"],
            channel_id=row["channel_id"],
            title=row["title"],
            description=row["description"],
            tags=json.loads(row["tags_json"] or "[]"),
            privacy_status=row["privacy_status"],
            default_language=row["default_language"],
            default_audio_language=row["default_audio_language"],
            published_at=row["published_at"],
            duration=row["duration"],
            view_count=row["view_count"],
            like_count=row["like_count"],
            comment_count=row["comment_count"],
            localizations=json.loads(row["localizations_json"] or "{}"),
            thumbnails=json.loads(row["thumbnails_json"] or "{}"),
            raw=json.loads(row["raw_json"] or "{}"),
        )

    def get_video(self, video_id: str) -> Video | None:
        row = self._conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
        return self._row_to_video(row) if row else None

    def list_videos(self, order_by: str = "view_count desc") -> list[Video]:
        parts = order_by.split()
        if len(parts) not in (1, 2) or parts[0] not in _ORDER_COLUMNS:
            raise ValueError(f"invalid order_by: {order_by!r}")
        direction = parts[1].lower() if len(parts) == 2 else "asc"
        if direction not in ("asc", "desc"):
            raise ValueError(f"invalid order_by: {order_by!r}")
        rows = self._conn.execute(
            f"SELECT * FROM videos ORDER BY {parts[0]} {direction}, video_id"
        ).fetchall()
        return [self._row_to_video(r) for r in rows]

    def snapshots_for(self, video_id: str) -> list[tuple[int, str, Video]]:
        """Snapshots of one video, oldest first, as (pull_id, taken_at, Video)."""
        rows = self._conn.execute(
            "SELECT pull_id, taken_at, data_json FROM snapshots WHERE video_id = ? "
            "ORDER BY taken_at, id",
            (video_id,),
        ).fetchall()
        return [
            (r["pull_id"], r["taken_at"], Video.from_dict(json.loads(r["data_json"])))
            for r in rows
        ]

    def previous_view_counts(self) -> dict[str, tuple[str, int]]:
        """Each video's view count in its second-newest snapshot, as video id -> (taken_at, view_count).

        Videos with fewer than two snapshots are left out: there is no earlier pull to compare with.
        """
        rows = self._conn.execute(
            "SELECT video_id, taken_at, json_extract(data_json, '$.view_count') AS views FROM ("
            " SELECT video_id, taken_at, data_json, ROW_NUMBER() OVER ("
            "  PARTITION BY video_id ORDER BY taken_at DESC, id DESC) AS n FROM snapshots"
            ") WHERE n = 2"
        ).fetchall()
        return {r["video_id"]: (r["taken_at"], int(r["views"] or 0)) for r in rows}

    # --- quota ledger ----------------------------------------------------
    def add_quota_entries(
        self,
        entries: Sequence[tuple[str, int, int, str]],
        quota_day: str,
        now: datetime | None = None,
    ) -> None:
        """Write (operation, units, count, note) rows in one transaction."""
        ts = _iso(now)
        with self._conn:
            self._conn.executemany(
                "INSERT INTO quota_ledger (ts_utc, quota_day, operation, units, count, note) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(ts, quota_day, op, units, count, note) for op, units, count, note in entries],
            )

    def add_quota_entry(
        self,
        operation: str,
        units: int,
        count: int = 1,
        note: str = "",
        *,
        quota_day: str,
        now: datetime | None = None,
    ) -> None:
        self.add_quota_entries([(operation, units, count, note)], quota_day, now)

    def quota_used(self, quota_day: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(units), 0) FROM quota_ledger WHERE quota_day = ?", (quota_day,)
        ).fetchone()
        return int(row[0])

    def quota_entries(self, quota_day: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM quota_ledger WHERE quota_day = ? ORDER BY id", (quota_day,)
        ).fetchall()

    # --- drafts ----------------------------------------------------------
    @staticmethod
    def _row_to_draft(row: sqlite3.Row) -> DraftRow:
        return DraftRow(
            video_id=row["video_id"],
            title=row["title"],
            description=row["description"],
            tags=tuple(json.loads(row["tags_json"])),
            base_title=row["base_title"],
            base_description=row["base_description"],
            base_tags=tuple(json.loads(row["base_tags_json"])),
            source=row["source"],
            note=row["note"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            approved_at=row["approved_at"],
            default_language=row["default_language"],
            localizations=(json.loads(row["localizations_json"])
                           if row["localizations_json"] is not None else None),
            base_default_language=row["base_default_language"],
            base_localizations=json.loads(row["base_localizations_json"] or "{}"),
        )

    def put_draft(self, row: DraftRow) -> None:
        """Insert or replace the draft of a video (every column)."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO drafts (video_id, title, description, tags_json, base_title,
                    base_description, base_tags_json, source, note, created_at, updated_at,
                    approved_at, default_language, localizations_json, base_default_language,
                    base_localizations_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    title=excluded.title, description=excluded.description,
                    tags_json=excluded.tags_json, base_title=excluded.base_title,
                    base_description=excluded.base_description,
                    base_tags_json=excluded.base_tags_json, source=excluded.source,
                    note=excluded.note, created_at=excluded.created_at,
                    updated_at=excluded.updated_at, approved_at=excluded.approved_at,
                    default_language=excluded.default_language,
                    localizations_json=excluded.localizations_json,
                    base_default_language=excluded.base_default_language,
                    base_localizations_json=excluded.base_localizations_json
                """,
                (
                    row.video_id, row.title, row.description, _dumps(list(row.tags)),
                    row.base_title, row.base_description, _dumps(list(row.base_tags)),
                    row.source, row.note, row.created_at, row.updated_at, row.approved_at,
                    row.default_language,
                    _dumps(row.localizations) if row.localizations is not None else None,
                    row.base_default_language, _dumps(row.base_localizations),
                ),
            )

    def get_draft(self, video_id: str) -> DraftRow | None:
        row = self._conn.execute("SELECT * FROM drafts WHERE video_id = ?", (video_id,)).fetchone()
        return self._row_to_draft(row) if row else None

    def list_drafts(self) -> list[DraftRow]:
        rows = self._conn.execute("SELECT * FROM drafts ORDER BY video_id").fetchall()
        return [self._row_to_draft(r) for r in rows]

    def delete_draft(self, video_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute("DELETE FROM drafts WHERE video_id = ?", (video_id,))
        return cur.rowcount > 0

    def set_draft_approval(self, video_id: str, approved_at: str | None) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE drafts SET approved_at = ? WHERE video_id = ?", (approved_at, video_id)
            )

    def update_video_text(
        self,
        video_id: str,
        title: str,
        description: str,
        tags: Sequence[str],
        now: datetime | None = None,
        *,
        default_language: str | None = None,
        localizations: dict[str, Any] | None = None,
    ) -> None:
        """Set the catalog text of a video (also inside raw_json). No snapshot.

        `default_language` and `localizations` are only changed when given (None: leave them).
        """
        row = self._conn.execute(
            "SELECT raw_json, default_language, localizations_json FROM videos WHERE video_id = ?",
            (video_id,),
        ).fetchone()
        if row is None:
            raise KeyError(video_id)
        raw = json.loads(row["raw_json"] or "{}")
        snippet = raw.get("snippet")
        if isinstance(snippet, dict):
            snippet.update(title=title, description=description, tags=list(tags))
            if default_language is not None:
                snippet["defaultLanguage"] = default_language
        if localizations is not None and isinstance(raw, dict):
            raw["localizations"] = localizations
        language = default_language if default_language is not None else row["default_language"]
        localized = (_dumps(localizations) if localizations is not None
                     else row["localizations_json"])
        with self._conn:
            self._conn.execute(
                "UPDATE videos SET title = ?, description = ?, tags_json = ?, raw_json = ?, "
                "default_language = ?, localizations_json = ?, updated_at = ? WHERE video_id = ?",
                (title, description, _dumps(list(tags)), _dumps(raw), language, localized,
                 _iso(now), video_id),
            )

    # --- pushes ----------------------------------------------------------
    @staticmethod
    def _row_to_push(row: sqlite3.Row) -> PushRecord:
        return PushRecord(
            id=row["id"],
            video_id=row["video_id"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            status=row["status"],
            source=row["source"],
            before=json.loads(row["before_json"]),
            after=json.loads(row["after_json"]),
            message=row["message"],
            run_id=row["run_id"],
        )

    def begin_push(
        self,
        video_id: str,
        source: str,
        before: dict[str, Any],
        now: datetime | None = None,
        run_id: int | None = None,
    ) -> int:
        """Record the backup of YouTube's live text before a push; status "pending"."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO pushes (video_id, started_at, status, source, before_json, run_id) "
                "VALUES (?, ?, 'pending', ?, ?, ?)",
                (video_id, _iso(now), source, _dumps(before), run_id),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def begin_push_run(self, now: datetime | None = None) -> int:
        """Open a push run; the pushes of one push_drafts call carry its id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO push_runs (started_at) VALUES (?)", (_iso(now),)
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def finish_push_run(self, run_id: int, video_count: int, now: datetime | None = None) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE push_runs SET finished_at = ?, video_count = ? WHERE id = ?",
                (_iso(now), video_count, run_id),
            )

    def get_push_run(self, run_id: int) -> PushRun | None:
        row = self._conn.execute("SELECT * FROM push_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return PushRun(row["id"], row["started_at"], row["finished_at"], row["video_count"])

    def list_pushes_of_run(self, run_id: int) -> list[PushRecord]:
        """The push records of one run, in push order (oldest first)."""
        rows = self._conn.execute(
            "SELECT * FROM pushes WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [self._row_to_push(r) for r in rows]

    def get_push_runs(self, run_ids: Sequence[int]) -> dict[int, PushRun]:
        """Several runs at once, by id."""
        found: dict[int, PushRun] = {}
        for run_id in dict.fromkeys(run_ids):
            run = self.get_push_run(run_id)
            if run is not None:
                found[run_id] = run
        return found

    def later_pushed(self, video_id: str, push_id: int) -> PushRecord | None:
        """The first push of this video after `push_id` that reached YouTube, if any."""
        row = self._conn.execute(
            "SELECT * FROM pushes WHERE video_id = ? AND id > ? AND status = 'pushed' "
            "ORDER BY id LIMIT 1",
            (video_id, push_id),
        ).fetchone()
        return self._row_to_push(row) if row else None

    def finish_push(
        self,
        push_id: int,
        status: str,
        after: dict[str, Any],
        message: str,
        now: datetime | None = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE pushes SET finished_at = ?, status = ?, after_json = ?, message = ? "
                "WHERE id = ?",
                (_iso(now), status, _dumps(after), message, push_id),
            )

    def get_push(self, push_id: int) -> PushRecord | None:
        row = self._conn.execute("SELECT * FROM pushes WHERE id = ?", (push_id,)).fetchone()
        return self._row_to_push(row) if row else None

    def list_pushes(self, video_id: str | None = None, limit: int = 100) -> list[PushRecord]:
        """Push records, newest first, optionally for one video."""
        if video_id is None:
            rows = self._conn.execute(
                "SELECT * FROM pushes ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM pushes WHERE video_id = ? ORDER BY id DESC LIMIT ?",
                (video_id, limit),
            ).fetchall()
        return [self._row_to_push(r) for r in rows]

    # --- agent runs ------------------------------------------------------
    @staticmethod
    def _row_to_agent_run(row: sqlite3.Row) -> AgentRun:
        return AgentRun(
            id=row["id"],
            video_id=row["video_id"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            model=row["model"],
            status=row["status"],
            summary=row["summary"],
            log=json.loads(row["log_json"]),
            result=json.loads(row["result_json"]),
            input_tokens=row["input_tokens"],
            cached_tokens=row["cached_tokens"],
            output_tokens=row["output_tokens"],
        )

    def begin_agent_run(self, video_id: str, model: str, now: datetime | None = None) -> int:
        """Record the start of a suggestion run; status "running"."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO agent_runs (video_id, started_at, model, status) "
                "VALUES (?, ?, ?, 'running')",
                (video_id, _iso(now), model),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def finish_agent_run(
        self,
        run_id: int,
        status: str,
        summary: str,
        log: list[Any],
        result: dict[str, Any],
        tokens: tuple[int, int, int] = (0, 0, 0),
        now: datetime | None = None,
    ) -> None:
        """Close a run. `tokens` is (input, cached, output)."""
        with self._conn:
            self._conn.execute(
                "UPDATE agent_runs SET finished_at = ?, status = ?, summary = ?, log_json = ?, "
                "result_json = ?, input_tokens = ?, cached_tokens = ?, output_tokens = ? WHERE id = ?",
                (_iso(now), status, summary, _dumps(log), _dumps(result), *tokens, run_id),
            )

    def get_agent_run(self, run_id: int) -> AgentRun | None:
        row = self._conn.execute("SELECT * FROM agent_runs WHERE id = ?", (run_id,)).fetchone()
        return self._row_to_agent_run(row) if row else None

    def list_agent_runs(self, video_id: str | None = None, limit: int = 100) -> list[AgentRun]:
        """Agent runs, newest first, optionally for one video."""
        if video_id is None:
            rows = self._conn.execute(
                "SELECT * FROM agent_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM agent_runs WHERE video_id = ? ORDER BY id DESC LIMIT ?",
                (video_id, limit),
            ).fetchall()
        return [self._row_to_agent_run(r) for r in rows]

    # --- copilot -----------------------------------------------------------
    @staticmethod
    def _row_to_copilot_turn(row: sqlite3.Row) -> CopilotTurn:
        return CopilotTurn(
            id=row["id"],
            chat_id=row["chat_id"],
            asked_at=row["asked_at"],
            finished_at=row["finished_at"],
            question=row["question"],
            model=row["model"],
            status=row["status"],
            answer=row["answer"],
            refs=json.loads(row["refs_json"]),
            notes=json.loads(row["notes_json"]),
            log=json.loads(row["log_json"]),
            input_tokens=row["input_tokens"],
            cached_tokens=row["cached_tokens"],
            output_tokens=row["output_tokens"],
        )

    def begin_copilot_turn(self, chat_id: int, question: str, model: str,
                           now: datetime | None = None) -> int:
        """Record a question; status "running"."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO copilot_turns (chat_id, asked_at, question, model, status) "
                "VALUES (?, ?, ?, ?, 'running')",
                (chat_id, _iso(now), question, model),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def finish_copilot_turn(
        self,
        turn_id: int,
        status: str,
        answer: str,
        refs: dict[str, str],
        notes: Sequence[str],
        log: list[Any],
        tokens: tuple[int, int, int] = (0, 0, 0),
        now: datetime | None = None,
    ) -> None:
        """Close a turn. `tokens` is (input, cached, output)."""
        with self._conn:
            self._conn.execute(
                "UPDATE copilot_turns SET finished_at = ?, status = ?, answer = ?, refs_json = ?, "
                "notes_json = ?, log_json = ?, input_tokens = ?, cached_tokens = ?, "
                "output_tokens = ? WHERE id = ?",
                (_iso(now), status, answer, _dumps(refs), _dumps(list(notes)), _dumps(log),
                 *tokens, turn_id),
            )

    def get_copilot_turn(self, turn_id: int) -> CopilotTurn | None:
        row = self._conn.execute("SELECT * FROM copilot_turns WHERE id = ?", (turn_id,)).fetchone()
        return self._row_to_copilot_turn(row) if row else None

    def list_copilot_turns(self, chat_id: int) -> list[CopilotTurn]:
        """The turns of one chat, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM copilot_turns WHERE chat_id = ? ORDER BY id", (chat_id,)).fetchall()
        return [self._row_to_copilot_turn(r) for r in rows]

    def latest_copilot_chat_id(self) -> int | None:
        """The chat of the newest turn, or None before the first question."""
        row = self._conn.execute(
            "SELECT chat_id FROM copilot_turns ORDER BY id DESC LIMIT 1").fetchone()
        return int(row["chat_id"]) if row else None

    def next_copilot_chat_id(self) -> int:
        """An id no turn uses yet, for a new chat."""
        row = self._conn.execute("SELECT MAX(chat_id) AS top FROM copilot_turns").fetchone()
        return int(row["top"] or 0) + 1

    @staticmethod
    def _row_to_copilot_proposal(row: sqlite3.Row) -> CopilotProposal:
        return CopilotProposal(
            id=row["id"],
            turn_id=row["turn_id"],
            video_id=row["video_id"],
            title=row["title"],
            description=row["description"],
            tags=tuple(json.loads(row["tags_json"])),
            reason=row["reason"],
            status=row["status"],
            created_at=row["created_at"],
        )

    def add_copilot_proposal(self, turn_id: int, video_id: str, title: str, description: str,
                             tags: Sequence[str], reason: str, now: datetime | None = None) -> int:
        """Store an edit the copilot offered; status "open"."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO copilot_proposals (turn_id, video_id, title, description, tags_json, "
                "reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (turn_id, video_id, title, description, _dumps(list(tags)), reason, _iso(now)),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_copilot_proposal(self, proposal_id: int) -> CopilotProposal | None:
        row = self._conn.execute(
            "SELECT * FROM copilot_proposals WHERE id = ?", (proposal_id,)).fetchone()
        return self._row_to_copilot_proposal(row) if row else None

    def list_copilot_proposals(self, turn_ids: Sequence[int]) -> list[CopilotProposal]:
        """The proposals of these turns, in the order they were made."""
        if not turn_ids:
            return []
        marks = ", ".join("?" * len(turn_ids))
        rows = self._conn.execute(
            f"SELECT * FROM copilot_proposals WHERE turn_id IN ({marks}) ORDER BY id",
            tuple(turn_ids)).fetchall()
        return [self._row_to_copilot_proposal(r) for r in rows]

    def set_copilot_proposal_status(self, proposal_id: int, status: str) -> None:
        with self._conn:
            self._conn.execute("UPDATE copilot_proposals SET status = ? WHERE id = ?",
                               (status, proposal_id))

    # --- link checks -------------------------------------------------------
    def save_link_check(self, url: str, status: str, http_status: int | None = None,
                        final_url: str = "", message: str = "",
                        now: datetime | None = None) -> int:
        """Record one check of one link; returns its id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO link_checks (url, checked_at, status, http_status, final_url, message) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (url, _iso(now), status, http_status, final_url, message),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def latest_link_checks(self) -> dict[str, LinkCheck]:
        """url -> its newest check, for every link ever checked."""
        rows = self._conn.execute(
            "SELECT * FROM link_checks WHERE id IN (SELECT MAX(id) FROM link_checks GROUP BY url)"
        ).fetchall()
        return {r["url"]: LinkCheck(id=r["id"], url=r["url"], checked_at=r["checked_at"],
                                    status=r["status"], http_status=r["http_status"],
                                    final_url=r["final_url"], message=r["message"])
                for r in rows}

    # --- advertiser-friendly checks ----------------------------------------
    @staticmethod
    def _row_to_ad_check(row: sqlite3.Row) -> AdCheck:
        return AdCheck(
            id=row["id"],
            video_id=row["video_id"],
            created_at=row["created_at"],
            model=row["model"],
            status=row["status"],
            verdict=row["verdict"],
            summary=row["summary"],
            findings=json.loads(row["findings_json"]),
            notes=json.loads(row["notes_json"]),
            source_hash=row["source_hash"],
            had_transcript=bool(row["had_transcript"]),
            error=row["error"],
            input_tokens=row["input_tokens"],
            cached_tokens=row["cached_tokens"],
            output_tokens=row["output_tokens"],
        )

    def save_ad_check(self, record: AdCheck, now: datetime | None = None) -> int:
        """Store a check (`id` and `created_at` are ignored); returns the new id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO ad_checks (video_id, created_at, model, status, verdict, summary, "
                "findings_json, notes_json, source_hash, had_transcript, error, input_tokens, "
                "cached_tokens, output_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.video_id, _iso(now), record.model, record.status, record.verdict,
                 record.summary, _dumps(record.findings), _dumps(record.notes), record.source_hash,
                 int(record.had_transcript), record.error, record.input_tokens,
                 record.cached_tokens, record.output_tokens),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def latest_ad_check(self, video_id: str) -> AdCheck | None:
        """The video's newest check, or None."""
        row = self._conn.execute(
            "SELECT * FROM ad_checks WHERE video_id = ? ORDER BY id DESC LIMIT 1", (video_id,)
        ).fetchone()
        return self._row_to_ad_check(row) if row else None

    # --- category checks (V7) ---
    @staticmethod
    def _row_to_category_check(row: sqlite3.Row) -> CategoryCheck:
        return CategoryCheck(
            video_id=row["video_id"], checked_at=row["checked_at"], model=row["model"],
            status=row["status"], current_id=row["current_id"], suggested_id=row["suggested_id"],
            reason=row["reason"], confirmed_at=row["confirmed_at"], error=row["error"],
            input_tokens=row["input_tokens"], output_tokens=row["output_tokens"],
        )

    def save_category_check(self, record: CategoryCheck, now: datetime | None = None) -> None:
        """Store the video's check in place of the one before (`checked_at` is set here; a new check
        clears an earlier "It's correct")."""
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO category_checks (video_id, checked_at, model, status, current_id, "
                "suggested_id, reason, confirmed_at, error, input_tokens, output_tokens) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?)",
                (record.video_id, _iso(now), record.model, record.status, record.current_id,
                 record.suggested_id, record.reason, record.error, record.input_tokens,
                 record.output_tokens),
            )

    def category_check(self, video_id: str) -> CategoryCheck | None:
        row = self._conn.execute("SELECT * FROM category_checks WHERE video_id = ?", (video_id,)).fetchone()
        return self._row_to_category_check(row) if row else None

    def category_checks(self) -> dict[str, CategoryCheck]:
        """Every video's check, by video id."""
        rows = self._conn.execute("SELECT * FROM category_checks").fetchall()
        return {row["video_id"]: self._row_to_category_check(row) for row in rows}

    def confirm_category(self, video_id: str, now: datetime | None = None) -> bool:
        """The user says the video's category is correct; False when there is no finished check."""
        with self._conn:
            cur = self._conn.execute(
                "UPDATE category_checks SET confirmed_at = ? WHERE video_id = ? AND status = 'done'",
                (_iso(now), video_id))
        return cur.rowcount > 0

    # --- Shorts finder (D5) ---
    @staticmethod
    def _row_to_shorts_find(row: sqlite3.Row) -> ShortsFind:
        return ShortsFind(
            id=row["id"],
            video_id=row["video_id"],
            transcript_id=row["transcript_id"],
            created_at=row["created_at"],
            model=row["model"],
            status=row["status"],
            message=row["message"],
            clips=json.loads(row["clips_json"]),
            notes=json.loads(row["notes_json"]),
            error=row["error"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
        )

    def save_shorts_find(self, record: ShortsFind, now: datetime | None = None) -> int:
        """Store a Shorts finder run (id and created_at are ignored); returns the new id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO shorts_finds (video_id, transcript_id, created_at, model, status, message, "
                "clips_json, notes_json, error, input_tokens, output_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.video_id, record.transcript_id, _iso(now), record.model, record.status,
                 record.message, _dumps(record.clips), _dumps(record.notes), record.error,
                 record.input_tokens, record.output_tokens),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def latest_shorts_find(self, video_id: str) -> ShortsFind | None:
        """The video's newest Shorts finder run, or None."""
        row = self._conn.execute(
            "SELECT * FROM shorts_finds WHERE video_id = ? ORDER BY id DESC LIMIT 1", (video_id,)
        ).fetchone()
        return self._row_to_shorts_find(row) if row else None

    # --- translations ----------------------------------------------------
    @staticmethod
    def _row_to_translation(row: sqlite3.Row) -> TranslationRecord:
        return TranslationRecord(
            id=row["id"],
            video_id=row["video_id"],
            language=row["language"],
            source_language=row["source_language"],
            source_hash=row["source_hash"],
            title=row["title"],
            description=row["description"],
            translator=row["translator"],
            verifier=row["verifier"],
            status=row["status"],
            score=row["score"],
            issues=json.loads(row["issues_json"]),
            problems=json.loads(row["problems_json"]),
            back_translation=json.loads(row["back_translation_json"]),
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            created_at=row["created_at"],
        )

    def add_translation(self, record: TranslationRecord, now: datetime | None = None) -> int:
        """Store a translation (its `id` and `created_at` are ignored); returns the new id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO translations (video_id, language, source_language, source_hash, "
                "title, description, translator, verifier, status, score, issues_json, "
                "problems_json, back_translation_json, input_tokens, output_tokens, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.video_id, record.language, record.source_language, record.source_hash,
                 record.title, record.description, record.translator, record.verifier,
                 record.status, record.score, _dumps(record.issues), _dumps(record.problems),
                 _dumps(record.back_translation), record.input_tokens, record.output_tokens,
                 _iso(now)),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_translation(self, translation_id: int) -> TranslationRecord | None:
        row = self._conn.execute(
            "SELECT * FROM translations WHERE id = ?", (translation_id,)
        ).fetchone()
        return self._row_to_translation(row) if row else None

    def list_translations(self, video_id: str) -> list[TranslationRecord]:
        """Every translation of a video, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM translations WHERE video_id = ? ORDER BY id DESC", (video_id,)
        ).fetchall()
        return [self._row_to_translation(r) for r in rows]

    @staticmethod
    def _row_to_transcript(row: sqlite3.Row) -> TranscriptRecord:
        return TranscriptRecord(
            id=row["id"],
            video_id=row["video_id"],
            language=row["language"],
            model=row["model"],
            source_name=row["source_name"],
            duration_ms=row["duration_ms"],
            segments=json.loads(row["segments_json"]),
            problems=json.loads(row["problems_json"]),
            created_at=row["created_at"],
        )

    def add_transcript(self, record: TranscriptRecord, now: datetime | None = None) -> int:
        """Store a transcript (its `id` and `created_at` are ignored); returns the new id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO transcripts (video_id, language, model, source_name, duration_ms, "
                "segments_json, problems_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (record.video_id, record.language, record.model, record.source_name,
                 record.duration_ms, _dumps(record.segments), _dumps(record.problems), _iso(now)),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_transcript(self, transcript_id: int) -> TranscriptRecord | None:
        row = self._conn.execute(
            "SELECT * FROM transcripts WHERE id = ?", (transcript_id,)
        ).fetchone()
        return self._row_to_transcript(row) if row else None

    def latest_transcript(self, video_id: str) -> TranscriptRecord | None:
        """The newest transcript of a video, or None."""
        row = self._conn.execute(
            "SELECT * FROM transcripts WHERE video_id = ? ORDER BY id DESC LIMIT 1", (video_id,)
        ).fetchone()
        return self._row_to_transcript(row) if row else None

    def latest_translations(self, video_id: str) -> dict[str, TranslationRecord]:
        """The newest translation of a video per language."""
        latest: dict[str, TranslationRecord] = {}
        for record in self.list_translations(video_id):
            latest.setdefault(record.language, record)
        return latest

    # --- subtitle translations -------------------------------------------
    @staticmethod
    def _row_to_subtitle_translation(row: sqlite3.Row) -> SubtitleTranslationRecord:
        return SubtitleTranslationRecord(
            id=row["id"],
            video_id=row["video_id"],
            transcript_id=row["transcript_id"],
            language=row["language"],
            source_language=row["source_language"],
            segments=json.loads(row["segments_json"]),
            translator=row["translator"],
            verifier=row["verifier"],
            status=row["status"],
            score=row["score"],
            issues=json.loads(row["issues_json"]),
            problems=json.loads(row["problems_json"]),
            back_translation=json.loads(row["back_translation_json"]),
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            created_at=row["created_at"],
        )

    def add_subtitle_translation(self, record: SubtitleTranslationRecord,
                                 now: datetime | None = None) -> int:
        """Store translated subtitles (`id` and `created_at` are ignored); returns the new id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO subtitle_translations (video_id, transcript_id, language, "
                "source_language, segments_json, translator, verifier, status, score, "
                "issues_json, problems_json, back_translation_json, input_tokens, "
                "output_tokens, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.video_id, record.transcript_id, record.language, record.source_language,
                 _dumps(record.segments), record.translator, record.verifier, record.status,
                 record.score, _dumps(record.issues), _dumps(record.problems),
                 _dumps(record.back_translation), record.input_tokens, record.output_tokens,
                 _iso(now)),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_subtitle_translation(self, record_id: int) -> SubtitleTranslationRecord | None:
        row = self._conn.execute(
            "SELECT * FROM subtitle_translations WHERE id = ?", (record_id,)
        ).fetchone()
        return self._row_to_subtitle_translation(row) if row else None

    def list_subtitle_translations(self, video_id: str) -> list[SubtitleTranslationRecord]:
        """Every subtitle translation of a video, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM subtitle_translations WHERE video_id = ? ORDER BY id DESC", (video_id,)
        ).fetchall()
        return [self._row_to_subtitle_translation(r) for r in rows]

    def latest_subtitle_translations(
        self, video_id: str, transcript_id: int
    ) -> dict[str, SubtitleTranslationRecord]:
        """The newest subtitle translation per language, made from that transcript only."""
        latest: dict[str, SubtitleTranslationRecord] = {}
        for record in self.list_subtitle_translations(video_id):
            if record.transcript_id == transcript_id:
                latest.setdefault(record.language, record)
        return latest

    # --- packaging reviews and thumbnail variants -------------------------
    @staticmethod
    def _row_to_packaging_review(row: sqlite3.Row) -> PackagingReview:
        return PackagingReview(
            id=row["id"],
            video_id=row["video_id"],
            title=row["title"],
            thumbnail_url=row["thumbnail_url"],
            image_file=row["image_file"],
            measurements=json.loads(row["measurements_json"]),
            reading=json.loads(row["reading_json"]),
            critique=json.loads(row["critique_json"]),
            problems=json.loads(row["problems_json"]),
            status=row["status"],
            error=row["error"],
            models=json.loads(row["models_json"]),
            input_tokens=row["input_tokens"],
            cached_tokens=row["cached_tokens"],
            output_tokens=row["output_tokens"],
            cost_usd=row["cost_usd"],
            created_at=row["created_at"],
        )

    def add_packaging_review(self, record: PackagingReview, now: datetime | None = None) -> int:
        """Store a packaging check (`id` and `created_at` are ignored); returns the new id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO packaging_reviews (video_id, title, thumbnail_url, image_file, "
                "measurements_json, reading_json, critique_json, problems_json, status, error, "
                "models_json, input_tokens, cached_tokens, output_tokens, cost_usd, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.video_id, record.title, record.thumbnail_url, record.image_file,
                 _dumps(record.measurements), _dumps(record.reading), _dumps(record.critique),
                 _dumps(record.problems), record.status, record.error, _dumps(record.models),
                 record.input_tokens, record.cached_tokens, record.output_tokens, record.cost_usd,
                 _iso(now)),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_packaging_review(self, review_id: int) -> PackagingReview | None:
        row = self._conn.execute(
            "SELECT * FROM packaging_reviews WHERE id = ?", (review_id,)).fetchone()
        return self._row_to_packaging_review(row) if row else None

    def latest_packaging_review(self, video_id: str) -> PackagingReview | None:
        """The newest packaging check of a video, or None."""
        row = self._conn.execute(
            "SELECT * FROM packaging_reviews WHERE video_id = ? ORDER BY id DESC LIMIT 1",
            (video_id,)).fetchone()
        return self._row_to_packaging_review(row) if row else None

    @staticmethod
    def _row_to_thumbnail_variant(row: sqlite3.Row) -> ThumbnailVariant:
        return ThumbnailVariant(
            id=row["id"],
            review_id=row["review_id"],
            video_id=row["video_id"],
            instruction=row["instruction"],
            overlay_text=row["overlay_text"],
            image_file=row["image_file"],
            text_read=row["text_read"],
            text_ok=None if row["text_ok"] is None else bool(row["text_ok"]),
            measurements=json.loads(row["measurements_json"]),
            status=row["status"],
            error=row["error"],
            model=row["model"],
            cost_usd=row["cost_usd"],
            created_at=row["created_at"],
        )

    def add_thumbnail_variant(self, record: ThumbnailVariant, now: datetime | None = None) -> int:
        """Store a thumbnail variant (`id` and `created_at` are ignored); returns the new id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO thumbnail_variants (review_id, video_id, instruction, overlay_text, "
                "image_file, text_read, text_ok, measurements_json, status, error, model, "
                "cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.review_id, record.video_id, record.instruction, record.overlay_text,
                 record.image_file, record.text_read,
                 None if record.text_ok is None else int(record.text_ok),
                 _dumps(record.measurements), record.status, record.error, record.model,
                 record.cost_usd, _iso(now)),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def list_thumbnail_variants(self, review_id: int) -> list[ThumbnailVariant]:
        """The variants of a review, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM thumbnail_variants WHERE review_id = ? ORDER BY id", (review_id,)
        ).fetchall()
        return [self._row_to_thumbnail_variant(r) for r in rows]

    def get_thumbnail_variant(self, variant_id: int) -> ThumbnailVariant | None:
        row = self._conn.execute(
            "SELECT * FROM thumbnail_variants WHERE id = ?", (variant_id,)).fetchone()
        return self._row_to_thumbnail_variant(row) if row else None

    # --- outlier reports --------------------------------------------------
    @staticmethod
    def _row_to_outlier_report(row: sqlite3.Row) -> OutlierReport:
        return OutlierReport(
            id=row["id"],
            pull_id=row["pull_id"],
            as_of=row["as_of"],
            params=json.loads(row["params_json"]),
            labels=json.loads(row["labels_json"]),
            insight=json.loads(row["insight_json"]),
            problems=json.loads(row["problems_json"]),
            status=row["status"],
            error=row["error"],
            model=row["model"],
            input_tokens=row["input_tokens"],
            cached_tokens=row["cached_tokens"],
            output_tokens=row["output_tokens"],
            cost_usd=row["cost_usd"],
            created_at=row["created_at"],
        )

    def save_outlier_report(self, record: OutlierReport, now: datetime | None = None) -> int:
        """Store an outlier report (`id` and `created_at` are ignored); returns the new id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO outlier_reports (pull_id, as_of, params_json, labels_json, insight_json, "
                "problems_json, status, error, model, input_tokens, cached_tokens, output_tokens, "
                "cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.pull_id, record.as_of, _dumps(record.params), _dumps(record.labels),
                 _dumps(record.insight), _dumps(record.problems), record.status, record.error,
                 record.model, record.input_tokens, record.cached_tokens, record.output_tokens,
                 record.cost_usd, _iso(now)),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def latest_outlier_report(self) -> OutlierReport | None:
        """The newest outlier report, or None."""
        row = self._conn.execute(
            "SELECT * FROM outlier_reports ORDER BY id DESC LIMIT 1").fetchone()
        return self._row_to_outlier_report(row) if row else None

    # --- comments, reply drafts and templates (G4, G5) --------------------
    @staticmethod
    def _row_to_comment(row: sqlite3.Row) -> Comment:
        return Comment(
            comment_id=row["comment_id"],
            video_id=row["video_id"],
            author=row["author"],
            text=row["text"],
            published_at=row["published_at"],
            like_count=row["like_count"],
            reply_count=row["reply_count"],
            fetched_at=row["fetched_at"],
            category=row["category"],
            classified_at=row["classified_at"],
            owner_replied=bool(row["owner_replied"]),
        )

    def upsert_comments(self, comments: Sequence[Comment], now: datetime | None = None) -> int:
        """Store fetched comments; returns how many were given.

        A comment that is already stored keeps its category and classified_at: only its text,
        author, published time, likes and reply count are updated.
        """
        items = list(comments)
        ts = _iso(now)
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO comments (comment_id, video_id, author, text, published_at,
                    like_count, reply_count, fetched_at, category, classified_at, owner_replied)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(comment_id) DO UPDATE SET
                    video_id=excluded.video_id, author=excluded.author, text=excluded.text,
                    published_at=excluded.published_at, like_count=excluded.like_count,
                    reply_count=excluded.reply_count, fetched_at=excluded.fetched_at, owner_replied=excluded.owner_replied
                """,
                [(c.comment_id, c.video_id, c.author, c.text, c.published_at, c.like_count,
                  c.reply_count, c.fetched_at or ts, c.category, c.classified_at, int(c.owner_replied)) for c in items],
            )
        return len(items)

    def list_comments(self, video_id: str) -> list[Comment]:
        """The stored comments of a video, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM comments WHERE video_id = ? ORDER BY published_at DESC, comment_id",
            (video_id,),
        ).fetchall()
        return [self._row_to_comment(r) for r in rows]

    def set_comment_categories(self, categories: dict[str, str], now: datetime | None = None) -> None:
        """Set the category of each comment in the {comment_id: category} mapping given."""
        ts = _iso(now)
        with self._conn:
            self._conn.executemany(
                "UPDATE comments SET category = ?, classified_at = ? WHERE comment_id = ?",
                [(category, ts, comment_id) for comment_id, category in dict(categories).items()],
            )

    @staticmethod
    def _row_to_comment_reply(row: sqlite3.Row) -> CommentReply:
        return CommentReply(
            id=row["id"],
            video_id=row["video_id"],
            comment_id=row["comment_id"],
            text=row["text"],
            model=row["model"],
            created_at=row["created_at"],
            status=row["status"],
            notes=tuple(json.loads(row["notes_json"])),
        )

    def add_comment_reply(self, video_id: str, comment_id: str, text: str, model: str,
                          notes: Sequence[str], now: datetime | None = None) -> int:
        """Store a reply draft (an empty `comment_id` is the pinned-comment draft); returns its id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO comment_replies (video_id, comment_id, text, model, created_at, "
                "notes_json) VALUES (?, ?, ?, ?, ?, ?)",
                (video_id, comment_id, text, model, _iso(now), _dumps(list(notes))),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def list_comment_replies(self, video_id: str, status: str | None = None) -> list[CommentReply]:
        """The reply drafts of a video in the order they were made, optionally only one status."""
        if status is None:
            rows = self._conn.execute(
                "SELECT * FROM comment_replies WHERE video_id = ? ORDER BY id", (video_id,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM comment_replies WHERE video_id = ? AND status = ? ORDER BY id",
                (video_id, status),
            ).fetchall()
        return [self._row_to_comment_reply(r) for r in rows]

    def set_comment_reply_status(self, id: int, status: str) -> None:
        """Set a reply draft's status: "open", "used" or "dismissed"."""
        if status not in ("open", "used", "dismissed"):
            raise ValueError(f'Unknown reply status "{status}": use open, used or dismissed.')
        with self._conn:
            self._conn.execute(
                "UPDATE comment_replies SET status = ? WHERE id = ?", (status, id))

    def list_recent_comments(self, limit: int = 500) -> list[Comment]:
        """The stored comments of every video, newest first, at most `limit`."""
        rows = self._conn.execute(
            "SELECT * FROM comments ORDER BY published_at DESC, comment_id LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_comment(r) for r in rows]

    def comments_to_answer(self, limit: int = 50) -> list[Comment]:
        """Questions the channel has not replied to and whose reply draft was not used or dismissed, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM comments c WHERE c.category = 'question' AND c.owner_replied = 0 "
            "AND NOT EXISTS (SELECT 1 FROM comment_replies r WHERE r.comment_id = c.comment_id "
            "AND r.status IN ('used', 'dismissed')) "
            "ORDER BY c.published_at DESC, c.comment_id LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_comment(r) for r in rows]

    def list_open_comment_replies(self) -> list[CommentReply]:
        """The open reply drafts of every video, in the order they were made."""
        rows = self._conn.execute(
            "SELECT * FROM comment_replies WHERE status = 'open' ORDER BY id"
        ).fetchall()
        return [self._row_to_comment_reply(r) for r in rows]

    def dismiss_open_comment_replies(self, video_id: str) -> int:
        """Dismiss a video's open reply drafts; returns how many were dismissed."""
        with self._conn:
            cur = self._conn.execute(
                "UPDATE comment_replies SET status = 'dismissed' "
                "WHERE video_id = ? AND status = 'open'", (video_id,))
        return cur.rowcount

    @staticmethod
    def _row_to_reply_template(row: sqlite3.Row) -> ReplyTemplate:
        return ReplyTemplate(
            id=row["id"],
            name=row["name"],
            body=row["body"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def save_reply_template(self, name: str, body: str, now: datetime | None = None,
                            template_id: int | None = None) -> int:
        """Save a reply template; with `template_id`, update that template. Returns its id."""
        ts = _iso(now)
        with self._conn:
            if template_id is None:
                cur = self._conn.execute(
                    "INSERT INTO reply_templates (name, body, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (name, body, ts, ts),
                )
                return int(cur.lastrowid)  # type: ignore[arg-type]
            self._conn.execute(
                "UPDATE reply_templates SET name = ?, body = ?, updated_at = ? WHERE id = ?",
                (name, body, ts, template_id),
            )
        return template_id

    def list_reply_templates(self) -> list[ReplyTemplate]:
        """Every reply template, by name."""
        rows = self._conn.execute(
            "SELECT * FROM reply_templates ORDER BY name COLLATE NOCASE, id").fetchall()
        return [self._row_to_reply_template(r) for r in rows]

    def delete_reply_template(self, id: int) -> bool:
        """Delete a reply template. True when there was one."""
        with self._conn:
            cur = self._conn.execute("DELETE FROM reply_templates WHERE id = ?", (id,))
        return cur.rowcount > 0

    # --- custom rules (G7) --------------------------------------------------
    @staticmethod
    def _row_to_custom_rule(row: sqlite3.Row) -> CustomRule:
        return CustomRule(
            id=row["id"],
            name=row["name"],
            request=row["request"],
            rule=json.loads(row["rule_json"]),
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
        )

    def add_custom_rule(self, name: str, request: str, rule: dict[str, Any],
                        now: datetime | None = None) -> int:
        """Store a custom rule; returns its id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO custom_rules (name, request, rule_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (name, request, _dumps(rule), _iso(now)),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def list_custom_rules(self, enabled_only: bool = False) -> list[CustomRule]:
        """Every custom rule in the order they were made, or only the enabled ones."""
        if enabled_only:
            rows = self._conn.execute(
                "SELECT * FROM custom_rules WHERE enabled != 0 ORDER BY id").fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM custom_rules ORDER BY id").fetchall()
        return [self._row_to_custom_rule(r) for r in rows]

    def set_custom_rule_enabled(self, id: int, enabled: bool) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE custom_rules SET enabled = ? WHERE id = ?", (int(bool(enabled)), id))

    def delete_custom_rule(self, id: int) -> bool:
        """Delete a custom rule. True when there was one."""
        with self._conn:
            cur = self._conn.execute("DELETE FROM custom_rules WHERE id = ?", (id,))
        return cur.rowcount > 0
