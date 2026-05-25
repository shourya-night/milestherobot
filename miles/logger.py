"""Reliable CSV cycle logging for Miles research output."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from config import LOG_PATH

_HEADERS = [
    "cycle",
    "timestamp",
    "human_speech",
    "move",
    "arm",
    "say",
    "mem_update",
    "raw_response",
]
_initialized = False


def _ensure_header() -> None:
    global _initialized
    if _initialized:
        return
    path = Path(LOG_PATH)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(_HEADERS)
    _initialized = True


def init_logger() -> None:
    """Initialize log file/header once at startup."""
    _ensure_header()


def log_cycle(cycle, human_speech, move, arm, say, mem, raw) -> None:
    with Path(LOG_PATH).open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                cycle,
                datetime.now(timezone.utc).isoformat(),
                human_speech or "",
                move,
                arm,
                say,
                mem or "",
                raw,
            ]
        )
