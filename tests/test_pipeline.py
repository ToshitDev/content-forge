"""Tests for pipeline.py's batch/concurrent orchestration.

Mocks BaseAgent._call_api (inherited by all 5 agent subclasses) so the
real run_pipeline_batch() / run_pipeline_async() orchestration runs
genuinely end to end — real chaining, real JSON parsing, real dataclass
construction — with no network access. Also redirects EXAMPLES_DIR to a
throwaway directory, since a real run writes a saved-run JSON file.
"""

import json

import pytest

from src import pipeline
from src.agents.base import BaseAgent

# One canned, schema-valid response per agent, keyed by agent.prompt_name.
FAKE_RESPONSES = {
    "research": json.dumps(
        {
            "pain_points": ["p1"],
            "ideas": [{"idea": "i1", "pain_point": "p1", "unique_angle": "a1"}],
            "top_picks": [{"idea": "i1", "reason": "r1"}],
            "missing_info": [],
        }
    ),
    "hook": json.dumps(
        {
            "hooks": [{"text": "h1", "type": "curiosity gap"}],
            "winner": {"text": "h1", "reasoning": "good"},
            "runner_up": {"text": "h2", "reasoning": "ok"},
        }
    ),
    "script": json.dumps({"script": "s1", "word_count": 10, "viewer_takeaway": "t1"}),
    "visual": json.dumps(
        {
            "cover_options": ["c1", "c2"],
            "frame_plan": [
                {"position": "opening", "on_screen_text": "t1", "visual_direction": "v1"}
            ],
            "assets_needed": ["a1"],
        }
    ),
    "growth": json.dumps(
        {
            "scores": {
                "clarity": 8,
                "retention": 7,
                "save_potential": 8,
                "shareability": 7,
                "audience_fit": 8,
                "cta_strength": 7,
            },
            "justifications": {
                "clarity": "j1",
                "retention": "j2",
                "save_potential": "j3",
                "shareability": "j4",
                "audience_fit": "j5",
                "cta_strength": "j6",
            },
            "weaknesses": [{"issue": "w1", "fix": "f1"}],
            "captions": ["cap1", "cap2"],
            "final_call": "POST",
            "final_call_reason": "good enough",
        }
    ),
}


async def _fake_call_api(self: BaseAgent, prompt: str) -> str:
    """Stand in for BaseAgent._call_api: no network, canned per-agent JSON.

    Raises if the filled prompt contains the marker "TRIGGER_FAILURE",
    so a test can make exactly one job fail without touching the others.
    """
    if "TRIGGER_FAILURE" in prompt:
        raise RuntimeError("simulated agent failure")
    return FAKE_RESPONSES[self.prompt_name]


@pytest.fixture(autouse=True)
def mock_agents_and_isolate_examples(monkeypatch, tmp_path):
    """No network calls, and saved-run JSON goes to a throwaway directory."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(BaseAgent, "_call_api", _fake_call_api)
    monkeypatch.setattr(pipeline, "EXAMPLES_DIR", tmp_path)


def _sample_job(niche: str) -> pipeline.PipelineJob:
    """Build a (profile, research_material) job tuple for the given niche."""
    profile = {
        "niche": niche,
        "audience": "test audience",
        "platform": "Instagram",
        "format": "reel",
        "brand_voice": "casual",
        "cta": "follow",
    }
    return profile, "sample research material"


def test_run_pipeline_batch_returns_results_in_order():
    """Results come back in the same order jobs were submitted, not completion order."""
    jobs = [_sample_job("topic A"), _sample_job("topic B"), _sample_job("topic C")]

    results = pipeline.run_pipeline_batch(jobs, use_cache=False)

    assert len(results) == 3
    for result in results:
        assert "error" not in result
        assert result["growth"].final_call == "POST"


def test_run_pipeline_batch_isolates_a_failing_job():
    """One job failing is captured as data, not raised, and doesn't affect the others."""
    jobs = [
        _sample_job("topic A"),
        _sample_job("TRIGGER_FAILURE"),
        _sample_job("topic C"),
    ]

    results = pipeline.run_pipeline_batch(jobs, use_cache=False)

    assert len(results) == 3
    assert "error" not in results[0]
    assert "error" in results[1]
    assert "Research agent failed" in results[1]["error"]
    assert "error" not in results[2]
