"""SQLite-backed memory store for recent observations."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone

from config import DB_PATH


_conn: sqlite3.Connection | None = None
_conn_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    global _conn
    with _conn_lock:
        if _conn is None:
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        return _conn


def init_db() -> None:
    conn = _get_conn()
    with _conn_lock:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp TEXT,
              observation TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS known_objects (
              label TEXT PRIMARY KEY,
              first_seen TEXT,
              last_seen TEXT,
              times_seen INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.commit()


def add_memory(observation: str) -> None:
    conn = _get_conn()
    with _conn_lock:
        conn.execute(
            "INSERT INTO memory (timestamp, observation) VALUES (?, ?)",
            (datetime.now(timezone.utc).isoformat(), observation),
        )
        conn.commit()


def get_recent_memory(n: int) -> str:
    conn = _get_conn()
    with _conn_lock:
        rows = conn.execute(
            "SELECT observation FROM memory ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()

    observations = [row[0] for row in reversed(rows)]
    return "\n".join(observations)


def count_memories() -> int:
    conn = _get_conn()
    with _conn_lock:
        row = conn.execute("SELECT COUNT(*) FROM memory").fetchone()
    return int(row[0] if row else 0)


def get_known_labels() -> set[str]:
    """Return the set of every object label ever seen, for grounded novelty checks."""
    conn = _get_conn()
    with _conn_lock:
        rows = conn.execute("SELECT label FROM known_objects").fetchall()
    return {row[0] for row in rows}


def touch_label(label: str) -> bool:
    """Record a sighting of `label`. Returns True if this is the FIRST time it's been seen."""
    if not label:
        return False
    normalized = label.strip().lower()
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    with _conn_lock:
        row = conn.execute(
            "SELECT times_seen FROM known_objects WHERE label = ?", (normalized,)
        ).fetchone()
        is_new = row is None
        if is_new:
            conn.execute(
                "INSERT INTO known_objects (label, first_seen, last_seen, times_seen) VALUES (?, ?, ?, 1)",
                (normalized, now, now),
            )
        else:
            conn.execute(
                "UPDATE known_objects SET last_seen = ?, times_seen = times_seen + 1 WHERE label = ?",
                (now, normalized),
            )
        conn.commit()
    return is_new


def close_db() -> None:
    global _conn
    with _conn_lock:
        if _conn is not None:
            _conn.close()
            _conn = None
