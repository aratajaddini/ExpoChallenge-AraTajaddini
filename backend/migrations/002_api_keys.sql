-- backend/migrations/002_api_keys.sql
-- Reference only: the table is created at runtime by backend.keys.init_schema().
CREATE TABLE IF NOT EXISTS api_keys (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash     TEXT    NOT NULL UNIQUE,
    preview      TEXT    NOT NULL,          -- first 12 chars, for admin matching
    label        TEXT    NOT NULL,          -- "shift-A / Ali"
    created_at   TEXT    NOT NULL,
    expires_at   TEXT    NOT NULL,
    revoked      INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys (key_hash);
