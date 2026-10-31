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

# MIGRATIONS[i] upgrades the database from user_version i to i + 1.
MIGRATIONS: tuple[tuple[str, ...], ...] = (
    _SCHEMA_V1, _SCHEMA_V2, _SCHEMA_V3, _SCHEMA_V4, _SCHEMA_V5
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
        )

    def begin_push(
        self, video_id: str, source: str, before: dict[str, Any], now: datetime | None = None
    ) -> int:
        """Record the backup of YouTube's live text before a push; status "pending"."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO pushes (video_id, started_at, status, source, before_json) "
                "VALUES (?, ?, 'pending', ?, ?)",
                (video_id, _iso(now), source, _dumps(before)),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

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
