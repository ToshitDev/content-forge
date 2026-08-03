"""Runs the full 5-agent content pipeline: Research → Hook → Script → Visual → Growth."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.agents.growth import GrowthAgent
from src.agents.hook import HookAgent
from src.agents.research import ResearchAgent
from src.agents.script import ScriptAgent
from src.agents.visual import VisualAgent
from src.models import GrowthReview, HookOutput, ResearchOutput, ScriptOutput, VisualOutput

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
TOTAL_STEPS = 5


def run_pipeline(profile: dict, research_material: str) -> dict:
    """Run the full content pipeline end to end and save the result.

    Args:
        profile: Shared context used across agents. Must include "niche",
            "audience", "platform", "format", "brand_voice", and "cta".
        research_material: Raw audience material (comments, questions,
            notes) for the Research Agent to work from.

    Returns:
        A dict with keys "research", "hook", "script", "visual", "growth",
        each holding the corresponding parsed output object.
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
    )

    outputs = {
        "research": research,
        "hook": hook,
        "script": script,
        "visual": visual,
        "growth": growth,
    }
    _save_run(profile, research_material, outputs)
    return outputs


def _run_step(step_num: int, name: str, agent, inputs: dict, output_cls):
    """Run one pipeline step, parse its result, and print a progress line.

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


def _save_run(profile: dict, research_material: str, outputs: dict) -> Path:
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
    sample_research_material = "\n".join(
        [
            "I make a timetable every Sunday and quit by Tuesday",
            "how do you study when your phone is right there",
            "I always plan too much and then feel behind by Wednesday",
            "does anyone actually stick to a study schedule or is it a scam",
            "I redo my planner every week instead of actually studying",
        ]
    )

    result = run_pipeline(sample_profile, sample_research_material)

    growth = result["growth"]
    print("\n=== Growth scores ===")
    for field_name, value in asdict(growth.scores).items():
        print(f"{field_name}: {value}/10")
    print(f"\nFinal call: {growth.final_call} — {growth.final_call_reason}")
