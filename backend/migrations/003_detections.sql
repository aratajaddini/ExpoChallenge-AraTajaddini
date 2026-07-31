-- backend/migrations/003_detections.sql
-- Reference only: the table is created at runtime by backend.analytics.store.init_schema().
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name  TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    source      TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_detections_created
    ON detections (created_at);

CREATE INDEX IF NOT EXISTS idx_detections_class
    ON detections (class_name);
