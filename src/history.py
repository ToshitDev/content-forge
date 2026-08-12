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
    cached INTEGER,
    voice_cost_estimate REAL
)
"""

_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs (timestamp)"

_INSERT_RUN = """
INSERT INTO runs (
    timestamp, niche, audience, platform,
    clarity, retention, save_potential, shareability, audience_fit, cta_strength,
    final_call, latency_seconds, cached, voice_cost_estimate
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPDATE_VOICE_COST = "UPDATE runs SET voice_cost_estimate = ? WHERE id = ?"


def init_db(path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the "runs" table and its timestamp index if they don't exist.

    CREATE TABLE/INDEX IF NOT EXISTS makes this idempotent — safe to
    call every time, never errors or duplicates anything if the table
    is already there. Also backfills voice_cost_estimate onto a runs
    table created before that column existed, via ALTER TABLE — CREATE
    TABLE IF NOT EXISTS is a no-op on an existing table, so it alone
    wouldn't add a new column to anyone's pre-Phase-13 history.db.
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)
        _ensure_voice_cost_estimate_column(conn)
        conn.commit()
    finally:
        conn.close()


def _ensure_voice_cost_estimate_column(conn: sqlite3.Connection) -> None:
    """Add voice_cost_estimate to an existing "runs" table that predates it."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    if "voice_cost_estimate" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN voice_cost_estimate REAL")


def update_voice_cost(
    run_id: int, voice_cost_estimate: float, path: str | Path = DEFAULT_DB_PATH
) -> None:
    """Attach a voiceover cost estimate to an already-logged run.

    Voice generation happens later than the pipeline run itself (a
    separate, optional UI action), so it can't be included in the
    original log_run() insert — this updates that row in place instead
    of writing a second one.
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute(_UPDATE_VOICE_COST, (voice_cost_estimate, run_id))
        conn.commit()
    finally:
        conn.close()


def log_run(
    run_result: dict[str, Any],
    latency_seconds: float,
    cached: bool,
    path: str | Path = DEFAULT_DB_PATH,
    voice_cost_estimate: float | None = None,
) -> int:
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
        voice_cost_estimate: Almost always None here — voice generation
            is a separate, later UI action, so most rows get this via
            update_voice_cost() afterward rather than at insert time.
            The parameter exists for a caller that already has the
            figure up front.

    Returns:
        The inserted row's id, for a later update_voice_cost() call.
    """
    init_db(path)

    profile = run_result["profile"]
    growth = run_result["growth"]
    scores = growth.scores

    conn = sqlite3.connect(path)
    try:
        cursor = conn.execute(
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
                voice_cost_estimate,
            ),
        )
        conn.commit()
        # lastrowid is only None for a statement that isn't an INSERT, or
        # an INSERT into a WITHOUT ROWID table — neither applies here.
        assert cursor.lastrowid is not None
        return cursor.lastrowid
    finally:
        conn.close()


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string, for the timestamp column."""
    return datetime.now(timezone.utc).isoformat()
