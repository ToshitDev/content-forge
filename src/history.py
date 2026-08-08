"""SQLite-backed run history: every completed pipeline run logged as one row.

Kept alongside the per-run JSON in examples/ (full detail, one file per
run) rather than replacing it — history.db is for QUERYING across runs
(averages, trends over time), which a folder of JSON files is bad at and
a database is exactly built for. See scripts/analyze_history.py.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "history.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    niche TEXT,
    audience TEXT,
    platform TEXT,
    clarity INTEGER,
    retention INTEGER,
    save_potential INTEGER,
    shareability INTEGER,
    audience_fit INTEGER,
    cta_strength INTEGER,
    final_call TEXT,
    latency_seconds REAL,
    cached INTEGER
)
"""

_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs (timestamp)"

_INSERT_RUN = """
INSERT INTO runs (
    timestamp, niche, audience, platform,
    clarity, retention, save_potential, shareability, audience_fit, cta_strength,
    final_call, latency_seconds, cached
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def init_db(path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the "runs" table and its timestamp index if they don't exist.

    CREATE TABLE/INDEX IF NOT EXISTS makes this idempotent — safe to
    call every time, never errors or duplicates anything if the table
    is already there.
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)
        conn.commit()
    finally:
        conn.close()


def log_run(
    run_result: dict[str, Any],
    latency_seconds: float,
    cached: bool,
    path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Insert one row summarizing a completed pipeline run.

    Calls init_db() first, so this works even if the caller never set
    up the table separately.

    Args:
        run_result: The dict returned by run_pipeline_async — reads its
            "profile" (niche/audience/platform) and "growth" (scores,
            final_call) entries.
        latency_seconds: Wall-clock time the run took, in seconds.
        cached: Whether the run had caching enabled (the use_cache
            passed to run_pipeline_async) — this reflects whether
            caching was *allowed*, not a guarantee every one of the 5
            steps was actually a cache hit rather than a real API call.
        path: SQLite database file. Defaults to history.db at the
            project root.
    """
    init_db(path)

    profile = run_result["profile"]
    growth = run_result["growth"]
    scores = growth.scores

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            _INSERT_RUN,
            (
                _now_iso(),
                profile["niche"],
                profile["audience"],
                profile["platform"],
                scores.clarity,
                scores.retention,
                scores.save_potential,
                scores.shareability,
                scores.audience_fit,
                scores.cta_strength,
                growth.final_call,
                latency_seconds,
                int(cached),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string, for the timestamp column."""
    return datetime.now(timezone.utc).isoformat()
