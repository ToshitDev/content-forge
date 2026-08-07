"""Tests for BaseAgent._fill_template's placeholder-validation logic.

No network access or real API key is required — _fill_template is pure
text processing. Creating a BaseAgent instance just needs *some* string
in ANTHROPIC_API_KEY to satisfy its startup check.
"""

import pytest

from src.agents.base import BaseAgent


@pytest.fixture
def agent(monkeypatch) -> BaseAgent:
    """A BaseAgent instance that can't reach the network, and doesn't need to."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    return BaseAgent(prompt_name="hook")


def test_fill_template_ignores_bracket_text_from_substituted_values(agent):
    """A substituted value's own bracket-shaped text must never be mistaken
    for one of *this* template's own placeholders.

    Reproduces a real bug: visual.txt's [SCRIPT] placeholder gets filled
    with the Script agent's generated text, and the model habitually
    writes its own "[PROBLEM]"/"[VALUE]" section headers into that text
    (the same habit that produces the already-tolerated "[PAUSE]"/"[CUT]"
    markers). Neither "PROBLEM" nor "VALUE" is one of visual.txt's own
    placeholders — they only appear here because they're baked into the
    *value* being substituted in, not the template itself — so filling
    it must not raise.
    """
    template = "Platform: [PLATFORM]\n\nScript:\n[SCRIPT]\n"
    inputs = {
        "platform": "Instagram",
        "script": (
            "[HOOK]\nYou're not lazy.\n"
            "[PROBLEM]\nYou keep procrastinating.\n"
            "[VALUE]\nTry this instead.\n"
            "[CTA]\nFollow for more."
        ),
    }

    filled = agent._fill_template(template, inputs)

    assert "Instagram" in filled
    # The script's own bracket text is preserved verbatim, not stripped
    # or rejected — it just isn't treated as a placeholder of this template.
    assert "[PROBLEM]" in filled
    assert "[VALUE]" in filled


def test_fill_template_still_catches_a_genuinely_missing_placeholder(agent):
    """A real, template-declared placeholder with no matching input key
    still raises — the dynamic-content fix above doesn't weaken genuine
    missing-input detection (e.g. growth.txt's [CTA]/[HOOK], which ARE
    legitimately required placeholders elsewhere)."""
    template = "Platform: [PLATFORM]\nCTA: [CTA]\n"

    with pytest.raises(ValueError, match=r"\[CTA\]"):
        agent._fill_template(template, {"platform": "Instagram"})


def test_fill_template_still_tolerates_static_pause_cut_markers(agent):
    """[PAUSE]/[CUT] present in the STATIC template text itself (not via
    a substituted value) are still tolerated, same as before this fix —
    this is the Phase 4 case, distinct from the dynamic-echo case above."""
    template = "Script: [SCRIPT]\nAdd [PAUSE] or [CUT] markers where useful.\n"

    filled = agent._fill_template(template, {"script": "some script text"})

    assert "some script text" in filled
    assert "[PAUSE]" in filled
    assert "[CUT]" in filled
