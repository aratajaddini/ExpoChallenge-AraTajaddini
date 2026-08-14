"""Shift-scoped API key issuing and verification (SQLite, hash-only storage)."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from backend.config import DB_PATH

PREFIX = "swr_"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash(raw: str) -> str:
    """SHA-256 of the raw key; only the digest is stored."""
    return hashlib.sha256(raw.encode()).hexdigest()


def _parse_ts(value: str) -> datetime:
    """Parse an ISO timestamp and ensure it is timezone-aware (UTC)."""
    ts = datetime.fromisoformat(value)
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success, always close."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:  # auto‑commits on success, rolls back on exception
            yield conn
    finally:
        conn.close()  # ✅ always close the connection


def init_schema() -> None:
    """Create the api_keys table if missing. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL UNIQUE,
                preview TEXT NOT NULL,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0,
                last_used_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys (key_hash);
            """
        )


def issue_key(label: str, hours: int = 8) -> tuple[str, str]:
    """Mint a key valid for `hours`. Returns (raw_key, expires_at_iso)."""
    raw = PREFIX + secrets.token_urlsafe(32)
    now = _utc_now()
    expires = now + timedelta(hours=hours)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO api_keys (key_hash, preview, label, created_at, expires_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (_hash(raw), raw[:8], label, now.isoformat(), expires.isoformat()),
        )
    return raw, expires.isoformat()


def verify_key(raw: str) -> sqlite3.Row | None:
    """Return the key row if it exists, is not revoked and not expired."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (_hash(raw),)
        ).fetchone()
        if row is None or row["revoked"]:
            return None
        # ✅ Normalise expiry to aware datetime before comparison
        if _parse_ts(row["expires_at"]) <= _utc_now():
            return None
        conn.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
            (_utc_now().isoformat(), row["id"]),
        )
        return row


def revoke_key(key_id: int) -> bool:
    """Revoke a key by id. True if a row changed."""
    with _connect() as conn:
        cur = conn.execute("UPDATE api_keys SET revoked = 1 WHERE id = ?", (key_id,))
        return cur.rowcount > 0


def list_keys(active_only: bool = False) -> list[sqlite3.Row]:
    """List issued keys (metadata and preview only, never raw keys)."""
    sql = (
        "SELECT id, preview, label, created_at, expires_at, revoked, last_used_at"
        " FROM api_keys"
    )
    params: tuple = ()
    if active_only:
        sql += " WHERE revoked = 0 AND expires_at > ?"
        params = (_utc_now().isoformat(),)
    sql += " ORDER BY id DESC"
    with _connect() as conn:
        return conn.execute(sql, params).fetchall()
