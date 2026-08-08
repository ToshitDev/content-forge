"""Analyze src/history.py's run history: score averages, trends over
time, and cached-vs-uncached latency.

Run with: python scripts/analyze_history.py
Needs pandas (see requirements-analysis.txt); the chart step is skipped
gracefully if matplotlib isn't installed.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Lets this run as `python scripts/analyze_history.py` from anywhere,
# without needing PYTHONPATH set manually — puts the repo root (this
# file's parent's parent) on sys.path so `from src...` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.history import DEFAULT_DB_PATH

SCORE_COLUMNS = [
    "clarity",
    "retention",
    "save_potential",
    "shareability",
    "audience_fit",
    "cta_strength",
]

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHART_PATH = DOCS_DIR / "score_trends.png"


def load_runs(path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Load every row from the runs table into a DataFrame."""
    conn = sqlite3.connect(path)
    try:
        return pd.read_sql_query("SELECT * FROM runs", conn, parse_dates=["timestamp"])
    finally:
        conn.close()


def print_average_scores(df: pd.DataFrame) -> None:
    """Print the mean of each score column across all runs."""
    print("=== Average scores (all runs) ===")
    print(df[SCORE_COLUMNS].mean().round(2).to_string())
    print()


def print_score_trend(df: pd.DataFrame) -> None:
    """Print average scores grouped by day."""
    print("=== Score trend by day ===")
    daily = df.groupby(df["timestamp"].dt.date)[SCORE_COLUMNS].mean().round(2)
    print(daily.to_string())
    print()


def print_latency_by_cache(df: pd.DataFrame) -> None:
    """Print average latency for cached vs. uncached runs."""
    print("=== Average latency: cached vs. uncached ===")
    latency = df.groupby("cached")["latency_seconds"].mean().round(2)
    latency.index = latency.index.map({0: "uncached", 1: "cached"})
    print(latency.to_string())
    print()


def save_score_trend_chart(df: pd.DataFrame) -> None:
    """Save a PNG line chart of daily average scores to docs/score_trends.png.

    Skipped (with a message, not an error) if matplotlib isn't installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart (see requirements-analysis.txt).")
        return

    daily = df.groupby(df["timestamp"].dt.date)[SCORE_COLUMNS].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    for column in SCORE_COLUMNS:
        ax.plot(daily.index, daily[column], marker="o", label=column)
    ax.set_xlabel("Date")
    ax.set_ylabel("Average score (1-10)")
    ax.set_title("ContentForge — Growth agent scores over time")
    ax.set_ylim(0, 10)
    ax.legend()
    fig.autofmt_xdate()

    DOCS_DIR.mkdir(exist_ok=True)
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart to {CHART_PATH}")


if __name__ == "__main__":
    runs = load_runs()

    if runs.empty:
        print("No runs recorded yet in history.db — run the pipeline first.")
    else:
        print_average_scores(runs)
        print_score_trend(runs)
        print_latency_by_cache(runs)
        save_score_trend_chart(runs)
