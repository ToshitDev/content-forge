"""Runs the full 5-agent content pipeline: Research → Hook → Script → Visual → Growth.

WHY THE 5 STAGES CAN'T RUN IN PARALLEL (Amdahl's law)
-------------------------------------------------------
Research → Hook → Script → Visual → Growth is a true sequential data
dependency, not just a scheduling choice: Hook needs Research's chosen
idea, Script needs Hook's winning line, Visual needs Script's finished
script, and Growth reviews the finished hook+script+visual package.
Stage N+1 literally cannot start until stage N's output exists, so
there is no way to run two of these stages concurrently within a
single run — the data simply doesn't exist yet.

Amdahl's law puts a number on why that matters. If P is the fraction
of a program that CAN be parallelized and N is the number of workers,
the maximum possible speedup from parallelizing is:

    speedup <= 1 / ((1 - P) + P/N)

For one pipeline run, P (the parallelizable fraction of the 5-stage
chain) is effectively 0 — every stage sits on the critical path.
Plugging P=0 into that formula caps speedup at 1x regardless of N: 10
workers or 100, it doesn't matter, because there's nothing to hand to
the extra workers. Throwing concurrency at a serial chain doesn't make
it faster; it just adds scheduling overhead for no benefit.

WHERE THE REAL CONCURRENCY OPPORTUNITY IS
-------------------------------------------------------
The chain is serial WITHIN one run, but separate runs (different
topics/profiles) share no dependency at all — job A's Research call
never waits on job B's Hook call. That's the case Amdahl's law doesn't
constrain: run_pipeline_batch() below runs N independent pipelines
concurrently via asyncio.gather, so N runs' worth of network-I/O wait
time overlaps instead of stacking up serially. See
scripts/benchmark_concurrency.py for the measured speedup.
"""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Self, TypeVar

from src.agents.base import BaseAgent
from src.agents.growth import GrowthAgent
from src.agents.hook import HookAgent
from src.agents.research import ResearchAgent
from src.agents.script import ScriptAgent
from src.agents.visual import VisualAgent
from src.history import log_run
from src.models import GrowthReview, HookOutput, ResearchOutput, ScriptOutput, VisualOutput

logger = logging.getLogger(__name__)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
TOTAL_STEPS = 5

# Called after each step as on_progress(step_num, total_steps, agent_name).
ProgressCallback = Callable[[int, int, str], None]

# A (profile, research_material) pair — everything one pipeline run needs.
PipelineJob = tuple[dict[str, str], str]


class _FromDict(Protocol):
    """Anything with a from_dict(cls, d) classmethod — every output dataclass."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self: ...


# Ties _run_step's output_cls argument to its return type, so e.g. passing
# ResearchOutput makes the call site's result infer as ResearchOutput, not
# a vague Any that would let a typo like `.top_pick` pass mypy silently.
# Bound to _FromDict so mypy also knows output_cls.from_dict(...) is valid.
T = TypeVar("T", bound=_FromDict)


async def run_pipeline_async(
    profile: dict[str, str],
    research_material: str,
    on_progress: ProgressCallback | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Run the full content pipeline end to end and save the result.

    This is the real implementation. run_pipeline() below is a thin sync
    wrapper around it for non-async callers (the Streamlit app, the CLI
    entry point); run_pipeline_batch() runs several of these
    concurrently. The 5-step chain itself is unavoidably sequential —
    see the module docstring for why.

    Args:
        profile: Shared context used across agents. Must include "niche",
            "audience", "platform", "format", "brand_voice", and "cta".
        research_material: Raw audience material (comments, questions,
            notes) for the Research Agent to work from.
        on_progress: Optional callback invoked after each step completes,
            as on_progress(step_num, total_steps, agent_name). Lets a UI
            (or anything else) report progress; CLI usage can ignore it —
            the step still prints, same as before.
        use_cache: Passed to every agent. False forces every step to call
            the API fresh instead of reusing a cached identical prompt.

    Returns:
        A dict with keys "research", "hook", "script", "visual", "growth",
        "saved_path" (where the run's JSON record was written), and
        "profile" (the profile dict this run was given).
    """
    start_time = time.perf_counter()

    research = await _run_step(
        1,
        "Research",
        ResearchAgent(use_cache=use_cache),
        {
            "niche": profile["niche"],
            "audience": profile["audience"],
            "platform": profile["platform"],
            "brand_voice": profile["brand_voice"],
            "research_material": research_material,
        },
        ResearchOutput,
        on_progress,
    )

    chosen_idea = research.top_picks[0].idea
    hook = await _run_step(
        2,
        "Hook",
        HookAgent(use_cache=use_cache),
        {
            "niche": profile["niche"],
            "audience": profile["audience"],
            "platform": profile["platform"],
            "brand_voice": profile["brand_voice"],
            "chosen_idea": chosen_idea,
        },
        HookOutput,
        on_progress,
    )

    script = await _run_step(
        3,
        "Script",
        ScriptAgent(use_cache=use_cache),
        {
            "niche": profile["niche"],
            "audience": profile["audience"],
            "platform": profile["platform"],
            "format": profile["format"],
            "brand_voice": profile["brand_voice"],
            "cta": profile["cta"],
            "winning_hook": hook.winner.text,
        },
        ScriptOutput,
        on_progress,
    )

    visual = await _run_step(
        4,
        "Visual",
        VisualAgent(use_cache=use_cache),
        {
            "platform": profile["platform"],
            "format": profile["format"],
            "brand_voice": profile["brand_voice"],
            "script": script.script,
        },
        VisualOutput,
        on_progress,
    )

    growth = await _run_step(
        5,
        "Growth",
        GrowthAgent(use_cache=use_cache),
        {
            "niche": profile["niche"],
            "audience": profile["audience"],
            "platform": profile["platform"],
            "cta": profile["cta"],
            "hook": hook.winner.text,
            "script": script.script,
            "visual_plan": _format_visual_plan(visual),
        },
        GrowthReview,
        on_progress,
    )

    outputs = {
        "research": research,
        "hook": hook,
        "script": script,
        "visual": visual,
        "growth": growth,
    }
    saved_path = _save_run(profile, research_material, outputs)
    outputs["saved_path"] = str(saved_path)
    outputs["profile"] = profile

    latency_seconds = time.perf_counter() - start_time
    try:
        log_run(outputs, latency_seconds, use_cache)
    except sqlite3.Error as error:
        # History logging is analytics, not the actual deliverable — a
        # DB-level hiccup (locked file, disk full, etc.) shouldn't cost
        # the user a run that otherwise completed and is already saved
        # to examples/. Deliberately NOT catching Exception broadly: a
        # KeyError/AttributeError here would mean our own data
        # extraction is broken, which is a real bug worth surfacing
        # loudly, not something to paper over.
        logger.error("Failed to log run to history.db: %s", error)

    return outputs


def run_pipeline(
    profile: dict[str, str],
    research_material: str,
    on_progress: ProgressCallback | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Sync wrapper around run_pipeline_async, for non-async callers.

    Same signature, same behavior, same return value as
    run_pipeline_async — this just blocks until it's done instead of
    being awaited. Used by app.py (Streamlit isn't async) and by this
    module's own CLI entry point below.
    """
    return asyncio.run(run_pipeline_async(profile, research_material, on_progress, use_cache))


async def _run_pipeline_batch_async(
    jobs: list[PipelineJob], use_cache: bool
) -> list[dict[str, Any]]:
    """Run every job concurrently; see run_pipeline_batch for the public API."""
    tasks = [
        run_pipeline_async(profile, research_material, use_cache=use_cache)
        for profile, research_material in jobs
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        {"error": str(result)} if isinstance(result, BaseException) else result
        for result in raw_results
    ]


def run_pipeline_batch(jobs: list[PipelineJob], use_cache: bool = True) -> list[dict[str, Any]]:
    """Run multiple INDEPENDENT pipeline jobs concurrently.

    This is the actual concurrency opportunity in this codebase — see
    the module docstring for why the 5 stages *within* one run can't be
    parallelized, but separate runs can. Each job runs through
    run_pipeline_async at the same time via asyncio.gather, so N jobs'
    worth of network-wait time overlaps instead of stacking up serially.

    Args:
        jobs: A (profile, research_material) tuple per pipeline run.
        use_cache: Passed to every job.

    Returns:
        One result dict per job, in the same order as `jobs` (asyncio.
        gather preserves input order — not completion order). A job
        that raises never crashes the others: its slot in the returned
        list is {"error": "<message>"} instead of a normal outputs
        dict, so callers can always index results[i] without a
        try/except around the whole batch.
    """
    return asyncio.run(_run_pipeline_batch_async(jobs, use_cache))


async def _run_step(
    step_num: int,
    name: str,
    agent: BaseAgent,
    inputs: dict[str, Any],
    output_cls: type[T],
    on_progress: ProgressCallback | None = None,
) -> T:
    """Run one pipeline step, parse its result, and report progress.

    Prints a progress line unconditionally (CLI usage relies on this) —
    that's user-facing UX, so it stays a print rather than a log call.
    The failure path below is logged, though: it's the kind of thing
    worth having in logs/contentforge.log even though the caller also
    sees it via the raised RuntimeError.

    Raises:
        RuntimeError: If the API call or parsing fails, naming which
            agent failed and including the underlying error message.
    """
    try:
        data = await agent.run_parsed(inputs)
        result = output_cls.from_dict(data)
    except Exception as error:
        logger.error("%s agent failed: %s", name, error)
        raise RuntimeError(f"{name} agent failed: {error}") from error
    print(f"[{step_num}/{TOTAL_STEPS}] {name} agent... done")
    if on_progress:
        on_progress(step_num, TOTAL_STEPS, name)
    return result


def _format_visual_plan(visual: VisualOutput) -> str:
    """Render a VisualOutput as readable text for the Growth Agent's prompt."""
    lines = ["Cover options: " + " | ".join(visual.cover_options)]
    for frame in visual.frame_plan:
        # frame.on_screen_text can legitimately be "" (a pure b-roll/demo
        # beat). Say so explicitly rather than rendering bare "" — an
        # empty pair of quotes reads as a formatting glitch, not "no
        # text overlay," and Growth is asked to judge clarity from this.
        text = f'"{frame.on_screen_text}"' if frame.on_screen_text else "no text overlay"
        lines.append(f"- {frame.position}: {text} ({frame.visual_direction})")
    lines.append("Assets needed: " + ", ".join(visual.assets_needed))
    return "\n".join(lines)


def _save_run(profile: dict[str, str], research_material: str, outputs: dict[str, Any]) -> Path:
    """Save a full pipeline run as JSON to examples/run_<timestamp>_<id>.json.

    The trailing random id (not just the second-precision timestamp)
    matters once runs happen concurrently: two batch jobs can easily
    finish in the same wall-clock second, and without it the second
    one to finish would silently overwrite the first one's saved file.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    record = {
        "timestamp": timestamp,
        "profile": profile,
        "research_material": research_material,
        "outputs": {name: asdict(result) for name, result in outputs.items()},
    }
    EXAMPLES_DIR.mkdir(exist_ok=True)
    path = EXAMPLES_DIR / f"run_{timestamp}_{unique_id}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"Saved run to {path}")
    return path


if __name__ == "__main__":
    import argparse

    from src.logging_config import configure_logging

    configure_logging()

    parser = argparse.ArgumentParser(description="Run the pipeline with sample data.")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the response cache and call the API fresh for every step.",
    )
    args = parser.parse_args()

    sample_profile = {
        "niche": "student productivity",
        "audience": "college students who procrastinate",
        "platform": "Instagram",
        "format": "reel",
        "brand_voice": "casual, direct, slightly funny",
        "cta": "follow for the full system",
    }
    sample_research_material = (
        "I make a timetable every Sunday and quit by Tuesday\n"
        "how do you study when your phone is right there\n"
        "I always plan too much and then feel behind by Wednesday\n"
        "does anyone actually stick to a study schedule or is it a scam\n"
        "I redo my planner every week instead of actually studying"
    )

    result = run_pipeline(sample_profile, sample_research_material, use_cache=not args.no_cache)

    growth = result["growth"]
    print("\n=== Growth scores ===")
    for field_name, value in asdict(growth.scores).items():
        print(f"{field_name}: {value}/10")
    print(f"\nFinal call: {growth.final_call} — {growth.final_call_reason}")
