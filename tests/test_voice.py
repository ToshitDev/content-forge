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
