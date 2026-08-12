"""Tests for src/agents/voice.py's retry/error-handling logic.

Calls VoiceAgent._call_with_retry directly with a fake callable — no
real network access, no real ElevenLabs API key (just a fake env var to
get past the constructor's fail-fast check).
"""

from unittest.mock import MagicMock

import pytest
from elevenlabs.core.api_error import ApiError

from src.agents import voice


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    """VoiceAgent() fails fast without an API key — give it a fake one."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key-not-real")


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Retry backoff shouldn't actually slow the test suite down."""
    monkeypatch.setattr(voice.time, "sleep", lambda seconds: None)


def test_call_with_retry_succeeds_on_first_try():
    agent = voice.VoiceAgent()

    result = agent._call_with_retry(lambda: "ok")

    assert result == "ok"


def test_call_with_retry_retries_on_retryable_error_then_succeeds():
    """A 503 (overloaded) is retried, and a later success is returned."""
    agent = voice.VoiceAgent()
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ApiError(status_code=503, body={"detail": "overloaded"})
        return "ok"

    result = agent._call_with_retry(flaky)

    assert result == "ok"
    assert attempts["count"] == 3


def test_call_with_retry_raises_immediately_on_non_retryable_error():
    """A 401 (bad key) fails identically every time, so it isn't retried."""
    agent = voice.VoiceAgent()
    attempts = {"count": 0}

    def always_unauthorized():
        attempts["count"] += 1
        raise ApiError(status_code=401, body={"detail": "invalid key"})

    with pytest.raises(ApiError):
        agent._call_with_retry(always_unauthorized)

    assert attempts["count"] == 1


def test_call_with_retry_gives_up_after_exhausting_retries():
    """A persistently retryable error still eventually raises, not loops forever."""
    agent = voice.VoiceAgent()
    attempts = {"count": 0}

    def always_overloaded():
        attempts["count"] += 1
        raise ApiError(status_code=503, body={"detail": "still overloaded"})

    with pytest.raises(ApiError):
        agent._call_with_retry(always_overloaded)

    assert attempts["count"] == 1 + len(voice.RETRY_DELAYS)


def test_estimate_cost_scales_with_text_length():
    assert voice.estimate_cost("") == 0.0
    assert voice.estimate_cost("a" * 1000) > voice.estimate_cost("a" * 10)


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    with pytest.raises(OSError, match="ELEVENLABS_API_KEY"):
        voice.VoiceAgent()


def test_clean_script_for_voice_strips_production_markers():
    """A full production script's stage directions — HOOK/VALUE/CTA
    section labels, an inline PAUSE marker, an ON-SCREEN TEXT callout,
    and a bare CUT marker — are all removed, leaving only what should
    actually be spoken."""
    script = (
        "[HOOK - 0-3 sec]\n"
        "Ever notice how the best study sessions happen when you stop "
        "trying to plan the perfect schedule?\n\n"
        "[VALUE]\n"
        "Here's the trick: set a timer for 20 minutes, pick ONE task, "
        "and go. [PAUSE 1 sec]\n"
        "No planner. No app. Just a timer and your notes.\n\n"
        '[ON-SCREEN TEXT: "20 minutes. One task. Go."]\n\n'
        "[CUT]\n\n"
        "[CTA]\n"
        "Follow for more no-nonsense study tips."
    )

    cleaned = voice.clean_script_for_voice(script)

    for marker in ("[HOOK", "[VALUE]", "[PAUSE", "[ON-SCREEN TEXT", "[CUT]", "[CTA]"):
        assert marker not in cleaned
    assert "[" not in cleaned and "]" not in cleaned
    assert "Ever notice how the best study sessions" in cleaned
    assert "Here's the trick: set a timer for 20 minutes" in cleaned
    assert "No planner. No app. Just a timer and your notes." in cleaned
    assert "Follow for more no-nonsense study tips." in cleaned
    # No run of 3+ blank lines left behind by a removed standalone marker.
    assert "\n\n\n" not in cleaned


def test_clean_script_for_voice_strips_bare_unbracketed_section_labels():
    """The Script Agent sometimes writes section headers as bare words on
    their own line instead of bracketing them — "PROBLEM" and "VALUE"
    with no brackets at all, as opposed to "[HOOK - 0-3 sec]". Those
    bare labels must be stripped too, or they get read aloud literally
    ("...PROBLEM...You fill it out...")."""
    script = (
        "[HOOK - 0-3 sec]\n"
        "Your planner is lying to you.\n\n"
        "PROBLEM\n"
        "You fill it out like it's gospel. Then Wednesday hits and "
        "you've done none of it. That's not a planning problem. "
        "That's a you problem.\n\n"
        "VALUE\n"
        "Here's what actually works: plan for procrastination, not "
        "perfection.\n\n"
        "CTA\n"
        "Follow for the full system."
    )

    cleaned = voice.clean_script_for_voice(script)
    lines = cleaned.splitlines()

    for bare_label in ("PROBLEM", "VALUE", "CTA", "HOOK"):
        assert bare_label not in lines
    assert "Your planner is lying to you." in cleaned
    # "problem" as an ordinary word inside a real sentence must survive —
    # only a line that's NOTHING BUT the bare label gets stripped.
    assert "That's not a planning problem. That's a you problem." in cleaned
    assert "Here's what actually works: plan for procrastination, not perfection." in cleaned
    assert "Follow for the full system." in cleaned


def test_clean_script_for_voice_keeps_emotional_tags_but_strips_markers():
    """Production markers (including PROBLEM, not covered by the other
    test) are stripped, but ElevenLabs v3 emotional audio tags survive
    intact so they reach ElevenLabs."""
    script = (
        "[HOOK - 0-3 sec]\n"
        "[excited] Ever notice how the best study sessions happen when "
        "you stop trying to plan the perfect schedule?\n\n"
        "[PROBLEM]\n"
        "[sighs] We've all been there — [whispers] the planner guilt is real.\n\n"
        "[VALUE]\n"
        "Here's the trick: set a timer for 20 minutes, pick ONE task, "
        "and go. [PAUSE 1 sec]\n"
        "[laughs] No planner. No app. Just a timer and your notes.\n\n"
        '[ON-SCREEN TEXT: "20 minutes. One task. Go."]\n\n'
        "[CUT]\n\n"
        "[CTA]\n"
        "[serious] Follow for more no-nonsense study tips."
    )

    cleaned = voice.clean_script_for_voice(script)

    for marker in (
        "[HOOK",
        "[PROBLEM]",
        "[VALUE]",
        "[PAUSE",
        "[ON-SCREEN TEXT",
        "[CUT]",
        "[CTA]",
    ):
        assert marker not in cleaned
    for tag in ("[excited]", "[sighs]", "[whispers]", "[laughs]", "[serious]"):
        assert tag in cleaned
    assert "Ever notice how the best study sessions" in cleaned
    assert "Follow for more no-nonsense study tips." in cleaned


def test_generate_uses_v3_and_low_stability_when_tags_present(tmp_path, monkeypatch):
    """A script with a surviving emotional tag routes to eleven_v3 with
    the low "Creative"-equivalent stability — v2 ignores tags entirely."""
    monkeypatch.setattr(voice, "AUDIO_DIR", tmp_path)
    agent = voice.VoiceAgent()
    agent.client.text_to_speech.convert = MagicMock(return_value=iter([b"fake-mp3-bytes"]))

    agent.generate("[excited] Set a timer and go.")

    _, kwargs = agent.client.text_to_speech.convert.call_args
    assert kwargs["model_id"] == voice.V3_MODEL_ID
    assert kwargs["voice_settings"].stability == voice.V3_TAG_STABILITY


def test_generate_uses_v2_when_no_tags_present(tmp_path, monkeypatch):
    """A plain script (production markers only, no emotional tags left
    after cleaning) stays on the cheaper, non-alpha v2 model."""
    monkeypatch.setattr(voice, "AUDIO_DIR", tmp_path)
    agent = voice.VoiceAgent()
    agent.client.text_to_speech.convert = MagicMock(return_value=iter([b"fake-mp3-bytes"]))

    agent.generate("[HOOK - 0-3 sec] Set a timer and go.")

    _, kwargs = agent.client.text_to_speech.convert.call_args
    assert kwargs["model_id"] == voice.V2_MODEL_ID
    assert kwargs["voice_settings"] is agent.voice_settings


def test_clone_voice_passes_a_valid_dict_for_labels():
    """labels must always be an actual dict — never omitted, None, or a
    string. The installed ElevenLabs SDK serializes an omitted `labels`
    as the literal JSON string "null" (jsonable_encoder(OMIT) -> None,
    then json.dumps(None) -> "null"), which the API rejects with 400
    invalid_labels ("Labels must be a serialized dictionary object")."""
    agent = voice.VoiceAgent()
    fake_response = MagicMock(voice_id="cloned-voice-123")
    agent.client.voices.ivc.create = MagicMock(return_value=fake_response)

    voice_id = agent.clone_voice(b"fake audio bytes", "sample.mp3", name="Test Voice")

    assert voice_id == "cloned-voice-123"
    _, kwargs = agent.client.voices.ivc.create.call_args
    assert "labels" in kwargs
    labels = kwargs["labels"]
    assert labels is not None
    assert not isinstance(labels, str)
    assert isinstance(labels, dict)
