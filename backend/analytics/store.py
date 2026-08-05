"""Write path and schema init for detection records."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone

from backend import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name  TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    source      TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_created ON detections (created_at);
CREATE INDEX IF NOT EXISTS idx_detections_class   ON detections (class_name);
"""


def init_schema() -> None:
    """Create detections table and indexes if they do not exist."""
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.executescript(_SCHEMA)


def record_detections(
    items: Iterable[Mapping[str, object]],
    source: str,
) -> int:
    """Persist detection results. Returns number of rows written."""
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (str(it["class_name"]), float(it["confidence"]), source, now) for it in items
    ]
    if not rows:
        return 0
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.executemany(
            "INSERT INTO detections (class_name, confidence, source, created_at)"
            " VALUES (?, ?, ?, ?)",
            rows,
        )
    return len(rows)
