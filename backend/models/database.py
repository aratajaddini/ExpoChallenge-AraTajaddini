"""SQLite access layer: schema creation, lightweight migrations, connection helper."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from backend.config import DB_PATH


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection, creating the parent directory if needed."""
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    """Existing column names of a table."""
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive migrations for databases created by earlier versions."""
    cols = _column_names(conn, "predictions")
    if "source" not in cols:
        conn.execute(
            "ALTER TABLE predictions ADD COLUMN source TEXT NOT NULL DEFAULT 'image'"
        )
    if "frames_analyzed" not in cols:
        conn.execute(
            "ALTER TABLE predictions ADD COLUMN frames_analyzed INTEGER NOT NULL DEFAULT 1"
        )


def init_db() -> None:
    """Create tables and indexes if missing, then apply migrations."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                top_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'image'
                    CHECK (source IN ('image', 'video')),
                frames_analyzed INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                correct_class TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_created_at "
            "ON predictions(created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_prediction_id "
            "ON feedback(prediction_id)"
        )
        _migrate(conn)