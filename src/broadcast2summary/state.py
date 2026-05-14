from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS feeds_meta (
    feed_name TEXT PRIMARY KEY,
    last_run_at TEXT,
    last_success_at TEXT
);

CREATE TABLE IF NOT EXISTS processed_episodes (
    guid TEXT PRIMARY KEY,
    feed_name TEXT NOT NULL,
    title TEXT,
    pub_date TEXT,
    processed_at TEXT,
    status TEXT,
    transcript_chars INTEGER,
    summary_model TEXT,
    quality_pass_level INTEGER,
    output_local_path TEXT,
    output_wiki_token TEXT,
    duration_seconds INTEGER
);

CREATE TABLE IF NOT EXISTS failed_queue (
    guid TEXT PRIMARY KEY,
    feed_name TEXT NOT NULL,
    title TEXT,
    audio_url TEXT,
    failed_stage TEXT,
    error TEXT,
    attempts INTEGER DEFAULT 1,
    last_attempt_at TEXT,
    mp3_path TEXT
);
"""


@dataclass(frozen=True)
class EpisodeRecord:
    guid: str
    feed_name: str
    title: str
    pub_date: str
    processed_at: str
    status: str
    transcript_chars: int
    summary_model: str
    quality_pass_level: int
    output_local_path: str | None
    output_wiki_token: str | None
    duration_seconds: int


@dataclass(frozen=True)
class FailedRecord:
    guid: str
    feed_name: str
    title: str
    audio_url: str
    failed_stage: str
    error: str
    attempts: int
    last_attempt_at: str
    mp3_path: str | None


@dataclass(frozen=True)
class FeedMeta:
    feed_name: str
    last_run_at: str | None
    last_success_at: str | None


class State:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    def is_processed(self, guid: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM processed_episodes WHERE guid = ? AND status = 'success'",
                (guid,),
            ).fetchone()
        return row is not None

    def record_episode(self, rec: EpisodeRecord) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO processed_episodes
                (guid, feed_name, title, pub_date, processed_at, status,
                 transcript_chars, summary_model, quality_pass_level,
                 output_local_path, output_wiki_token, duration_seconds)
                VALUES (:guid, :feed_name, :title, :pub_date, :processed_at,
                        :status, :transcript_chars, :summary_model,
                        :quality_pass_level, :output_local_path,
                        :output_wiki_token, :duration_seconds)""",
                asdict(rec),
            )

    def enqueue_failed(self, rec: FailedRecord) -> None:
        with self._conn() as c:
            existing = c.execute(
                "SELECT attempts FROM failed_queue WHERE guid = ?", (rec.guid,)
            ).fetchone()
            attempts = (existing["attempts"] + 1) if existing else rec.attempts
            data = asdict(rec) | {"attempts": attempts}
            c.execute(
                """INSERT OR REPLACE INTO failed_queue
                (guid, feed_name, title, audio_url, failed_stage, error,
                 attempts, last_attempt_at, mp3_path)
                VALUES (:guid, :feed_name, :title, :audio_url, :failed_stage,
                        :error, :attempts, :last_attempt_at, :mp3_path)""",
                data,
            )

    def dequeue_failed(self, guid: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM failed_queue WHERE guid = ?", (guid,))

    def list_failed(self) -> list[FailedRecord]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM failed_queue ORDER BY last_attempt_at DESC").fetchall()
        return [FailedRecord(**dict(r)) for r in rows]

    def get_failed(self, guid: str) -> FailedRecord | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM failed_queue WHERE guid = ?", (guid,)).fetchone()
        return FailedRecord(**dict(row)) if row else None

    def touch_feed_run(self, feed_name: str, *, success: bool, at: str) -> None:
        with self._conn() as c:
            existing = c.execute(
                "SELECT last_success_at FROM feeds_meta WHERE feed_name = ?", (feed_name,)
            ).fetchone()
            last_success = at if success else (existing["last_success_at"] if existing else None)
            c.execute(
                """INSERT OR REPLACE INTO feeds_meta
                   (feed_name, last_run_at, last_success_at)
                   VALUES (?, ?, ?)""",
                (feed_name, at, last_success),
            )

    def get_feed_meta(self, feed_name: str) -> FeedMeta | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM feeds_meta WHERE feed_name = ?", (feed_name,)
            ).fetchone()
        return FeedMeta(**dict(row)) if row else None
