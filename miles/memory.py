"""SQLite-backed memory store for recent observations."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from config import DB_PATH


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp TEXT,
              observation TEXT
            );
            """
        )


def add_memory(observation: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO memory (timestamp, observation) VALUES (?, ?)",
            (datetime.now(timezone.utc).isoformat(), observation),
        )


def get_recent_memory(n: int) -> str:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT observation FROM memory ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()

    observations = [row[0] for row in reversed(rows)]
    return "\n".join(observations)


def count_memories() -> int:
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) FROM memory").fetchone()
    return int(row[0] if row else 0)
