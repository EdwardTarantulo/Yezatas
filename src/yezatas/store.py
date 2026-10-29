"""SQLite store: the local video catalog, per-pull snapshots and the quota ledger."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

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

# MIGRATIONS[i] upgrades the database from user_version i to i + 1.
MIGRATIONS: tuple[tuple[str, ...], ...] = (_SCHEMA_V1,)

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
