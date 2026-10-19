"""
prediction_log.py -- store every prediction in SQLite (a SQL database that lives in one file; built into Python).

Why log predictions?  Debugging ("what did the model say for this input?"), monitoring (drift, latency),
auditing, and building the NEXT training set from real traffic + user feedback.
"""

import sqlite3                       # built-in: no server to install
import time
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,   -- unique row number, assigned by the database
    ts            TEXT    NOT NULL,                    -- when (UTC, ISO format)
    model_version TEXT    NOT NULL,                    -- WHICH model answered (needed for A/B and rollbacks)
    input_text    TEXT    NOT NULL,
    label         TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    latency_ms    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_version ON predictions(model_version);  -- makes filtering by version fast
"""


class PredictionLog:
    def __init__(self, path: str = ":memory:"):          # ":memory:" = temporary DB that vanishes on exit
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row               # rows behave like dicts: row["label"]
        self.conn.executescript(SCHEMA)

    def record(self, model_version, text, label, confidence, latency_ms) -> None:
        # The ? placeholders are PARAMETERIZED QUERIES: values are passed separately from the SQL text,
        # which prevents SQL injection. NEVER build SQL with f-strings from user input.
        self.conn.execute(
            "INSERT INTO predictions (ts, model_version, input_text, label, confidence, latency_ms) VALUES (?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), model_version, text, label, confidence, latency_ms),
        )
        self.conn.commit()                                # commit = make the change permanent

    def label_counts(self, model_version: str | None = None) -> dict:
        sql = "SELECT label, COUNT(*) AS n FROM predictions"
        args: tuple = ()
        if model_version:
            sql += " WHERE model_version = ?"
            args = (model_version,)
        sql += " GROUP BY label ORDER BY n DESC"
        return {row["label"]: row["n"] for row in self.conn.execute(sql, args)}

    def latency_stats(self) -> dict:
        lat = sorted(r[0] for r in self.conn.execute("SELECT latency_ms FROM predictions"))
        if not lat:
            return {}
        pick = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 3)     # nearest-rank percentile
        return {"count": len(lat), "avg": round(sum(lat) / len(lat), 3), "p50": pick(0.5), "p95": pick(0.95), "max": round(lat[-1], 3)}

    def low_confidence(self, below: float = 0.6, limit: int = 5) -> list:
        """Rows worth a human look -- candidates for relabelling and the next training set."""
        cur = self.conn.execute(
            "SELECT id, input_text, label, confidence FROM predictions WHERE confidence < ? ORDER BY confidence LIMIT ?",
            (below, limit))
        return [dict(r) for r in cur]
