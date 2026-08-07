"""Tests for BaseAgent's JSON parsing helpers and HookOutput.from_dict.

No network access or real API key is required. parse_json() only works
on text already in memory, and creating a BaseAgent instance just needs
*some* string in ANTHROPIC_API_KEY to satisfy its startup check — we
never call the API in these tests, so the value doesn't matter.
"""

import pytest

from src.agents.base import BaseAgent
from src.models import HookOutput


@pytest.fixture
def agent(monkeypatch) -> BaseAgent:
    """A BaseAgent instance that can't reach the network, and doesn't need to."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    return BaseAgent(prompt_name="hook")


def test_parse_json_clean(agent):
    """Plain JSON with no code fence parses as-is."""
    assert agent.parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_with_json_fence(agent):
    """JSON wrapped in ```json ... ``` fences parses correctly."""
    raw = '```json\n{"a": 1}\n```'
    assert agent.parse_json(raw) == {"a": 1}


def test_parse_json_with_plain_fence(agent):
    """JSON wrapped in plain ``` ... ``` fences (no language tag) parses correctly."""
    raw = '```\n{"a": 1}\n```'
    assert agent.parse_json(raw) == {"a": 1}


def test_parse_json_fenced_truncated_no_closing_fence(agent):
    """A response cut off mid-stream (opening fence, no closing fence) still
    gets its leading fence stripped, instead of failing to parse at char 0.

    This is the actual bug: a truncated response can leave a complete,
    valid JSON body with the opening ```json fence still glued to the
    front and no closing ``` ever written.
    """
    raw = '```json\n{"a": 1, "b": 2}'
    assert agent.parse_json(raw) == {"a": 1, "b": 2}


def test_parse_json_strips_leading_and_trailing_fences_independently(agent):
    """Leading and trailing fences are stripped independently of each other."""
    # Trailing fence only — no opening fence present.
    assert agent.parse_json('{"a": 1}\n```') == {"a": 1}
    # Leading fence only — no closing fence present (same shape as the
    # truncation case above, just without the "json" language tag).
    assert agent.parse_json('```\n{"a": 1}') == {"a": 1}


def test_parse_json_repairs_object_closed_as_array(agent):
    """A `{...}` object closed with `]` instead of `}` is repaired and parsed.

    Reproduced from a real Growth-agent failure: "justifications": {...}
    was closed with a stray "]" instead of "}", immediately before the
    sibling key "weaknesses": [...] — an array. The model appears to have
    anticipated the wrong closer from what came right after.
    """
    raw = (
        '{"justifications": {"clarity": "Good hook, lands fast."]'
        ', "weaknesses": [{"issue": "CTA is vague", "fix": "Be specific"}]}'
    )
    result = agent.parse_json(raw)
    assert result["justifications"]["clarity"] == "Good hook, lands fast."
    assert result["weaknesses"] == [{"issue": "CTA is vague", "fix": "Be specific"}]


def test_parse_json_repair_does_not_touch_brackets_inside_strings(agent):
    """Bracket-shaped characters that are part of a string's actual content
    are left alone — only structural, mismatched closers get fixed."""
    raw = '{"caption": "Use the [PAUSE] marker, then {relax}.", "score": 5}'
    result = agent.parse_json(raw)
    assert result["caption"] == "Use the [PAUSE] marker, then {relax}."
    assert result["score"] == 5


def test_parse_json_handles_em_dash_and_smart_quotes_in_string_value(agent):
    """Typographic punctuation (em dash, smart quotes) inside a JSON string
    value parses as-is, unescaped — no repair needed.

    None of these characters are the JSON string delimiter (") or a
    control character, so they were never invalid JSON in the first
    place. This guards against a plausible-sounding but incorrect
    assumption that em dashes/smart quotes need escaping — if a future
    change tried to "fix" this, this test would catch it.
    """
    raw = '{"caption": "Sunday you — organized. Tuesday you — “winging it.”"}'
    result = agent.parse_json(raw)
    assert result["caption"] == "Sunday you — organized. Tuesday you — “winging it.”"


def test_parse_json_malformed_raises_with_raw_text(agent):
    """Malformed JSON raises ValueError whose message includes the raw text."""
    raw = "this is not json at all"
    with pytest.raises(ValueError) as exc_info:
        agent.parse_json(raw)
    assert raw in str(exc_info.value)


def test_hook_output_from_dict_valid():
    """HookOutput.from_dict builds a HookOutput from a well-formed dict."""
    data = {
        "hooks": [{"text": "Stop scrolling", "type": "pattern interrupt"}],
        "winner": {"text": "Stop scrolling", "reasoning": "Short and direct."},
        "runner_up": {"text": "You're doing this wrong", "reasoning": "Safer bet."},
    }
    result = HookOutput.from_dict(data)
    assert result.winner.text == "Stop scrolling"
    assert result.winner.reasoning == "Short and direct."
    assert len(result.hooks) == 1


def test_hook_output_from_dict_missing_key_raises():
    """HookOutput.from_dict raises a clear error when a required key is missing."""
    data = {
        "hooks": [{"text": "Stop scrolling", "type": "pattern interrupt"}],
        "winner": {"text": "Stop scrolling", "reasoning": "Short and direct."},
        # "runner_up" intentionally missing
    }
    with pytest.raises(ValueError, match="runner_up"):
        HookOutput.from_dict(data)
