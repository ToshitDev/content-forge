"""Runs the full 5-agent content pipeline: Research → Hook → Script → Visual → Growth."""

import json
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
from src.models import GrowthReview, HookOutput, ResearchOutput, ScriptOutput, VisualOutput

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
TOTAL_STEPS = 5

# Called after each step as on_progress(step_num, total_steps, agent_name).
ProgressCallback = Callable[[int, int, str], None]

class _FromDict(Protocol):
    """Anything with a from_dict(cls, d) classmethod — every output dataclass."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self: ...


# Ties _run_step's output_cls argument to its return type, so e.g. passing
# ResearchOutput makes the call site's result infer as ResearchOutput, not
# a vague Any that would let a typo like `.top_pick` pass mypy silently.
# Bound to _FromDict so mypy also knows output_cls.from_dict(...) is valid.
T = TypeVar("T", bound=_FromDict)


def run_pipeline(
    profile: dict[str, str],
    research_material: str,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the full content pipeline end to end and save the result.

    Args:
        profile: Shared context used across agents. Must include "niche",
            "audience", "platform", "format", "brand_voice", and "cta".
        research_material: Raw audience material (comments, questions,
            notes) for the Research Agent to work from.
        on_progress: Optional callback invoked after each step completes,
            as on_progress(step_num, total_steps, agent_name). Lets a UI
            (or anything else) report progress; CLI usage can ignore it —
            the step still prints, same as before.

    Returns:
        A dict with keys "research", "hook", "script", "visual", "growth",
        and "saved_path" (where the run's JSON record was written).
    """
    research = _run_step(
        1,
        "Research",
        ResearchAgent(),
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
    hook = _run_step(
        2,
        "Hook",
        HookAgent(),
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

    script = _run_step(
        3,
        "Script",
        ScriptAgent(),
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

    visual = _run_step(
        4,
        "Visual",
        VisualAgent(),
        {
            "platform": profile["platform"],
            "format": profile["format"],
            "brand_voice": profile["brand_voice"],
            "script": script.script,
        },
        VisualOutput,
        on_progress,
    )

    growth = _run_step(
        5,
        "Growth",
        GrowthAgent(),
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
    return outputs


def _run_step(
    step_num: int,
    name: str,
    agent: BaseAgent,
    inputs: dict[str, Any],
    output_cls: type[T],
    on_progress: ProgressCallback | None = None,
) -> T:
    """Run one pipeline step, parse its result, and report progress.

    Prints a progress line unconditionally (CLI usage relies on this),
    then also calls on_progress(step_num, TOTAL_STEPS, name) if given.

    Raises:
        RuntimeError: If the API call or parsing fails, naming which
            agent failed and including the underlying error message.
    """
    try:
        data = agent.run_parsed(inputs)
        result = output_cls.from_dict(data)
    except Exception as error:
        raise RuntimeError(f"{name} agent failed: {error}") from error
    print(f"[{step_num}/{TOTAL_STEPS}] {name} agent... done")
    if on_progress:
        on_progress(step_num, TOTAL_STEPS, name)
    return result


def _format_visual_plan(visual: VisualOutput) -> str:
    """Render a VisualOutput as readable text for the Growth Agent's prompt."""
    lines = ["Cover options: " + " | ".join(visual.cover_options)]
    for frame in visual.frame_plan:
        lines.append(
            f'- {frame.position}: "{frame.on_screen_text}" ({frame.visual_direction})'
        )
    lines.append("Assets needed: " + ", ".join(visual.assets_needed))
    return "\n".join(lines)


def _save_run(profile: dict[str, str], research_material: str, outputs: dict[str, Any]) -> Path:
    """Save a full pipeline run as JSON to examples/run_<timestamp>.json."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    record = {
        "timestamp": timestamp,
        "profile": profile,
        "research_material": research_material,
        "outputs": {name: asdict(result) for name, result in outputs.items()},
    }
    EXAMPLES_DIR.mkdir(exist_ok=True)
    path = EXAMPLES_DIR / f"run_{timestamp}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"Saved run to {path}")
    return path


if __name__ == "__main__":
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

    result = run_pipeline(sample_profile, sample_research_material)

    growth = result["growth"]
    print("\n=== Growth scores ===")
    for field_name, value in asdict(growth.scores).items():
        print(f"{field_name}: {value}/10")
    print(f"\nFinal call: {growth.final_call} — {growth.final_call_reason}")
