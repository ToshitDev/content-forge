"""Benchmark: sequential vs. concurrent execution of independent pipeline runs.

Demonstrates the actual concurrency opportunity in this codebase — see
src/pipeline.py's module docstring for why the 5 stages *within* one run
can't be parallelized (Amdahl's law), but separate, independent runs can.

Cache is disabled for both arms so the comparison reflects real API
latency, not cache hits — this makes real API calls and costs money.
"""

import sys
import time
from pathlib import Path

# Lets this run as `python scripts/benchmark_concurrency.py` from anywhere,
# without needing PYTHONPATH set manually — puts the repo root (this
# file's parent's parent) on sys.path so `from src...` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline, run_pipeline_batch

SAMPLE_JOBS = [
    (
        {
            "niche": "student productivity",
            "audience": "college students who procrastinate",
            "platform": "Instagram",
            "format": "reel",
            "brand_voice": "casual, direct, slightly funny",
            "cta": "follow for the full system",
        },
        (
            "I make a timetable every Sunday and quit by Tuesday\n"
            "how do you study when your phone is right there"
        ),
    ),
    (
        {
            "niche": "personal finance for Gen Z",
            "audience": "22-year-olds who just started their first job",
            "platform": "TikTok",
            "format": "reel",
            "brand_voice": "blunt, a little sarcastic",
            "cta": "save this before payday",
        },
        (
            "nobody taught me how to budget\n"
            "I'm always broke by the 15th and I don't know why"
        ),
    ),
    (
        {
            "niche": "home cooking for beginners",
            "audience": "people who just moved out for the first time",
            "platform": "Instagram",
            "format": "carousel",
            "brand_voice": "warm, encouraging, no jargon",
            "cta": "save this for your next grocery run",
        },
        (
            "I don't know how to cook rice without a rice cooker\n"
            "every recipe assumes I already know what a roux is"
        ),
    ),
]


def run_sequential() -> float:
    """Run every sample job one after another; return elapsed seconds."""
    start = time.perf_counter()
    for profile, research_material in SAMPLE_JOBS:
        run_pipeline(profile, research_material, use_cache=False)
    return time.perf_counter() - start


def run_concurrent() -> float:
    """Run every sample job at once via run_pipeline_batch; return elapsed seconds."""
    start = time.perf_counter()
    run_pipeline_batch(SAMPLE_JOBS, use_cache=False)
    return time.perf_counter() - start


if __name__ == "__main__":
    print(f"Benchmarking {len(SAMPLE_JOBS)} independent pipeline runs (cache disabled).\n")

    print("--- Sequential: one job after another ---")
    sequential_time = run_sequential()
    print(f"Sequential total: {sequential_time:.1f}s\n")

    print("--- Concurrent: asyncio.gather via run_pipeline_batch ---")
    concurrent_time = run_concurrent()
    print(f"Concurrent total: {concurrent_time:.1f}s\n")

    print(f"Speedup: {sequential_time / concurrent_time:.2f}x")
