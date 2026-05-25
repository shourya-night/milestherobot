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


def close_db() -> None:
    global _conn
    with _conn_lock:
        if _conn is not None:
            _conn.close()
            _conn = None
