"""Tests for src/models.py's dataclasses — FramePlanItem, SuggestOutput,
and StyleOutput.

Pure data-shape tests: no network access or API key needed.
"""

import pytest

from src.models import FramePlanItem, StyleOutput, SuggestOutput


def test_frame_plan_item_missing_on_screen_text_defaults_to_empty_string():
    """on_screen_text is optional — a pure b-roll/demo beat can omit it
    entirely, and from_dict should default it to "" rather than raise."""
    data = {"position": "middle", "visual_direction": "wide shot of desk setup"}

    result = FramePlanItem.from_dict(data)

    assert result.on_screen_text == ""
    assert result.position == "middle"
    assert result.visual_direction == "wide shot of desk setup"


def test_frame_plan_item_explicit_null_on_screen_text_also_defaults_to_empty_string():
    """An explicit JSON null (not just a missing key) is tolerated too."""
    data = {"position": "middle", "visual_direction": "b-roll", "on_screen_text": None}

    result = FramePlanItem.from_dict(data)

    assert result.on_screen_text == ""


def test_frame_plan_item_present_on_screen_text_is_kept():
    """A real on_screen_text value is used as-is, not overridden."""
    data = {
        "position": "opening",
        "on_screen_text": "You're not lazy",
        "visual_direction": "text overlay on phone screen",
    }

    result = FramePlanItem.from_dict(data)

    assert result.on_screen_text == "You're not lazy"


def test_frame_plan_item_missing_position_still_raises():
    """position is still required — the fix only relaxes on_screen_text."""
    data = {"on_screen_text": "some text", "visual_direction": "close-up"}

    with pytest.raises(ValueError, match="position"):
        FramePlanItem.from_dict(data)


def test_frame_plan_item_missing_visual_direction_still_raises():
    """visual_direction is still required — the fix only relaxes on_screen_text."""
    data = {"position": "opening", "on_screen_text": "some text"}

    with pytest.raises(ValueError, match="visual_direction"):
        FramePlanItem.from_dict(data)


def test_suggest_output_parses_valid_response():
    """A well-formed {"suggestions": [...]} dict parses into the list as-is."""
    data = {"suggestions": ["comment one", "comment two", "comment three"]}

    result = SuggestOutput.from_dict(data)

    assert result.suggestions == ["comment one", "comment two", "comment three"]


def test_suggest_output_missing_suggestions_key_raises():
    """suggestions is required — no default, since there's nothing sensible
    to fall back to if the model didn't return any."""
    with pytest.raises(ValueError, match="suggestions"):
        SuggestOutput.from_dict({})


def test_style_output_parses_valid_response():
    """A well-formed style kit dict parses into matching fields."""
    data = {
        "colors": ["warm terracotta (#C97C5D)", "cream (#F4F1EA)"],
        "font_mood": "clean and minimal",
        "layout_tendency": "centered, lots of whitespace",
        "vibe": "calm, editorial, unhurried",
    }

    result = StyleOutput.from_dict(data)

    assert result.colors == ["warm terracotta (#C97C5D)", "cream (#F4F1EA)"]
    assert result.font_mood == "clean and minimal"
    assert result.layout_tendency == "centered, lots of whitespace"
    assert result.vibe == "calm, editorial, unhurried"


def test_style_output_missing_colors_key_raises():
    """Each field is required — no defaults to silently fall back to."""
    data = {
        "font_mood": "clean and minimal",
        "layout_tendency": "centered, lots of whitespace",
        "vibe": "calm, editorial, unhurried",
    }

    with pytest.raises(ValueError, match="colors"):
        StyleOutput.from_dict(data)
