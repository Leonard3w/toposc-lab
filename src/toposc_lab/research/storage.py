"""Transactional SQLite journal, checksummed payloads, and atomic readable exports."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from toposc_lab._file_io import remove_temporary, replace_atomic


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        replace_atomic(name, path)
    finally:
        remove_temporary(Path(name))


class ResearchStore:
    """Each method opens a short-lived connection, including dashboard reads.

    SQLite is authoritative. File exports can be reconstructed after interruption.
    WAL + FULL synchronization and a single engine lease protect stage accounting.
    UI control requests use their own short write transaction.
    """

    def __init__(self, directory: str | Path, *, create: bool = False) -> None:
        self.directory = Path(directory)
        self.path = self.directory / "research.sqlite3"
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self.connect() as db:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS objects (
                        kind TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                        checksum TEXT NOT NULL, updated TEXT NOT NULL,
                        PRIMARY KEY(kind,id));
                    CREATE TABLE IF NOT EXISTS attempts (
                        number INTEGER PRIMARY KEY AUTOINCREMENT,
                        candidate TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
                        started TEXT NOT NULL, finished TEXT, seconds REAL, error TEXT);
                    CREATE TABLE IF NOT EXISTS events (
                        number INTEGER PRIMARY KEY AUTOINCREMENT,
                        created TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS controls (
                        number INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL,
                        created TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0);
                """)
        elif not self.path.is_file():
            raise FileNotFoundError(f"No research experiment at {self.directory}")

    @contextmanager
    def connect(self, *, readonly: bool = False) -> Iterator[sqlite3.Connection]:
        if readonly:
            db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=15)
        else:
            db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=15000")
            if not readonly:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def put(db: sqlite3.Connection, kind: str, identity: str, value: Any) -> None:
        payload = dumps(value)
        db.execute("INSERT OR REPLACE INTO objects VALUES(?,?,?,?,?)",
                   (kind, identity, payload, sha256(payload.encode()).hexdigest(), utc_now()))

    @staticmethod
    def decode(row: sqlite3.Row) -> Any:
        if sha256(row["payload"].encode()).hexdigest() != row["checksum"]:
            raise ValueError(f"Corrupt database payload {row['kind']}/{row['id']}")
        return json.loads(row["payload"])

    def get(self, kind: str, identity: str = "current", default: Any = None) -> Any:
        with self.connect(readonly=True) as db:
            row = db.execute("SELECT * FROM objects WHERE kind=? AND id=?", (kind, identity)).fetchone()
            return self.decode(row) if row else default

    def all(self, kind: str) -> list[Any]:
        with self.connect(readonly=True) as db:
            return [self.decode(r) for r in db.execute(
                "SELECT * FROM objects WHERE kind=? ORDER BY rowid", (kind,))]

    def save(self, kind: str, value: Any, identity: str = "current") -> None:
        with self.connect() as db:
            self.put(db, kind, identity, value)

    @staticmethod
    def event(db: sqlite3.Connection, kind: str, payload: Any) -> None:
        db.execute("INSERT INTO events(created,kind,payload) VALUES(?,?,?)",
                   (utc_now(), kind, dumps(payload)))

    def events(self) -> list[dict[str, Any]]:
        with self.connect(readonly=True) as db:
            return [{**dict(r), "payload": json.loads(r["payload"])}
                    for r in db.execute("SELECT * FROM events ORDER BY number")]

    def attempts(self) -> list[dict[str, Any]]:
        with self.connect(readonly=True) as db:
            return [dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY number")]

    def request(self, action: str) -> None:
        if action not in ("pause", "stop", "checkpoint", "resume", "archive"):
            raise ValueError("unsupported control action")
        with self.connect() as db:
            db.execute("INSERT INTO controls(action,created) VALUES(?,?)", (action, utc_now()))
            self.event(db, "control_requested", {"action": action})

    def integrity_check(self) -> None:
        with self.connect(readonly=True) as db:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed")
            for row in db.execute("SELECT * FROM objects"):
                self.decode(row)
