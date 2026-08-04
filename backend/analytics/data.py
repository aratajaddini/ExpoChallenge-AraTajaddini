"""Read-only analytics queries over the detections table."""

import sqlite3

from backend.config import DB_PATH


def _conn() -> sqlite3.Connection:
    """Open a read-only connection to the analytics database."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def count_by_class() -> dict[str, int]:
    """Total detections grouped by class name."""
    with _conn() as c:
        rows = c.execute(
            "SELECT class_name, COUNT(*) AS n FROM detections GROUP BY class_name"
        ).fetchall()
    return {r["class_name"]: r["n"] for r in rows}


def total_count() -> int:
    """Total number of detections recorded."""
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM detections").fetchone()[0]


def most_common_class() -> str | None:
    """Class with the highest detection count, or None if empty."""
    counts = count_by_class()
    return max(counts, key=counts.get) if counts else None
