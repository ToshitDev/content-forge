"""Tests for src/history.py's SQLite-backed run history.

Every test uses its own throwaway database file (pytest's tmp_path),
so nothing here ever touches the real history.db.
"""

import sqlite3

import pytest

from src.history import init_db, log_run
from src.models import GrowthReview, Scores


@pytest.fixture
def db_path(tmp_path):
    """A throwaway database file path, fresh for each test."""
    return tmp_path / "test_history.db"


def _table_names(db_path) -> list[str]:
    """List every table name in the database."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


def _fake_run_result() -> dict:
    """Build a run_result dict shaped like run_pipeline_async's real output,
    using the actual dataclasses log_run reads from it."""
    scores = Scores(
        clarity=8,
        retention=7,
        save_potential=8,
        shareability=6,
        audience_fit=9,
        cta_strength=7,
    )
    growth = GrowthReview(
        scores=scores,
        justifications={},
        weaknesses=[],
        captions=[],
        final_call="POST",
        final_call_reason="Strong hook, clear CTA.",
    )
    return {
        "profile": {
            "niche": "student productivity",
            "audience": "college students who procrastinate",
            "platform": "Instagram",
        },
        "growth": growth,
    }


def test_init_db_creates_the_runs_table(db_path):
    """init_db() creates a "runs" table."""
    init_db(db_path)

    assert "runs" in _table_names(db_path)


def test_init_db_is_idempotent(db_path):
    """Calling init_db() twice doesn't error or duplicate the table."""
    init_db(db_path)
    init_db(db_path)  # must not raise

    assert _table_names(db_path).count("runs") == 1


def test_log_run_inserts_a_row_with_the_right_data(db_path):
    """log_run() inserts exactly one row, with values read correctly
    from the run_result dict."""
    log_run(_fake_run_result(), latency_seconds=12.5, cached=False, path=db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM runs").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    row = rows[0]
    assert row["niche"] == "student productivity"
    assert row["audience"] == "college students who procrastinate"
    assert row["platform"] == "Instagram"
    assert row["clarity"] == 8
    assert row["retention"] == 7
    assert row["save_potential"] == 8
    assert row["shareability"] == 6
    assert row["audience_fit"] == 9
    assert row["cta_strength"] == 7
    assert row["final_call"] == "POST"
    assert row["latency_seconds"] == 12.5
    assert row["cached"] == 0
    assert row["timestamp"]  # a non-empty, real ISO timestamp string


def test_log_run_works_without_a_separate_init_db_call(db_path):
    """log_run() creates the table itself if it doesn't already exist."""
    log_run(_fake_run_result(), latency_seconds=1.0, cached=True, path=db_path)  # must not raise

    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    finally:
        conn.close()
    assert count == 1
