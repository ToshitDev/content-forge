"""Tests for src/models.py's dataclasses — currently just FramePlanItem.

Pure data-shape tests: no network access or API key needed.
"""

import pytest

from src.models import FramePlanItem


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
