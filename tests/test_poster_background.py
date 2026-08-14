"""Tests for src/poster_background.py's BFL API integration.

Mocks requests.post/requests.get — no real network calls, no real
BFL_API_KEY needed beyond a fake env var to get past the fail-fast check.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.models import StyleOutput
from src.poster_background import _build_prompt, generate_ai_background

SAMPLE_STYLE_KIT = StyleOutput(
    colors=["warm terracotta (#C97C5D)", "sage green (#7A9B8E)"],
    font_mood="clean and minimal",
    layout_tendency="centered, lots of whitespace",
    vibe="calm and understated",
)


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    """generate_ai_background fails fast without a key — give it a fake one."""
    monkeypatch.setenv("BFL_API_KEY", "test-key-not-real")


def test_build_prompt_always_includes_no_text_instruction():
    prompt = _build_prompt(SAMPLE_STYLE_KIT)
    lowered = prompt.lower()
    for phrase in ("no text", "no words", "no letters", "no numbers", "no typography"):
        assert phrase in lowered


def test_generate_ai_background_sends_a_prompt_with_no_text_instruction():
    """End to end through generate_ai_background (not just _build_prompt
    directly) — confirms the actual POST body BFL receives carries the
    no-text instruction, with both the initial POST and the polling GET
    mocked, plus the final image download."""
    post_response = MagicMock(status_code=200)
    post_response.json.return_value = {
        "id": "req-123",
        "polling_url": "https://api.us2.bfl.ai/v1/get_result?id=req-123",
    }

    poll_response = MagicMock(status_code=200)
    poll_response.json.return_value = {
        "id": "req-123",
        "status": "Ready",
        "result": {"sample": "https://delivery.example.com/sample.jpg"},
    }

    download_response = MagicMock(status_code=200, content=b"fake-image-bytes")

    with (
        patch("src.poster_background.requests.post", return_value=post_response) as mock_post,
        patch(
            "src.poster_background.requests.get", side_effect=[poll_response, download_response]
        ),
    ):
        result = generate_ai_background(SAMPLE_STYLE_KIT, (1080, 1350))

    assert result == b"fake-image-bytes"

    _, kwargs = mock_post.call_args
    sent_prompt = kwargs["json"]["prompt"]
    assert "no text" in sent_prompt.lower()
    assert "no words" in sent_prompt.lower()
    # Width/height sent to BFL must be valid (multiple of 32, in
    # [256, 1440]) even though the caller asked for 1080x1350.
    assert kwargs["json"]["width"] % 32 == 0
    assert kwargs["json"]["height"] % 32 == 0


def test_generate_ai_background_retries_transient_failure_then_succeeds():
    """A 503 on the initial request is retried, not treated as fatal."""
    failing_response = MagicMock(status_code=503, text="overloaded")
    ok_response = MagicMock(status_code=200)
    ok_response.json.return_value = {"id": "req-456", "polling_url": "https://x/get_result?id=req-456"}

    poll_response = MagicMock(status_code=200)
    poll_response.json.return_value = {
        "id": "req-456",
        "status": "Ready",
        "result": {"sample": "https://delivery.example.com/sample.jpg"},
    }
    download_response = MagicMock(status_code=200, content=b"fake-image-bytes")

    with (
        patch(
            "src.poster_background.requests.post", side_effect=[failing_response, ok_response]
        ) as mock_post,
        patch("src.poster_background.time.sleep", return_value=None),
        patch(
            "src.poster_background.requests.get", side_effect=[poll_response, download_response]
        ),
    ):
        result = generate_ai_background(SAMPLE_STYLE_KIT, (1080, 1350))

    assert result == b"fake-image-bytes"
    assert mock_post.call_count == 2


def test_generate_ai_background_times_out_if_never_ready():
    """The polling loop gives up after POLL_TIMEOUT_SECONDS rather than
    looping forever, raising a clear TimeoutError."""
    post_response = MagicMock(status_code=200)
    post_response.json.return_value = {"id": "req-789", "polling_url": "https://x/get_result?id=req-789"}

    pending_response = MagicMock(status_code=200)
    pending_response.json.return_value = {"id": "req-789", "status": "Pending"}

    # A fake monotonic clock that advances well past the timeout after
    # a couple of calls, so the test doesn't actually wait 30s.
    fake_time = {"now": 0.0}

    def fake_monotonic():
        fake_time["now"] += 20
        return fake_time["now"]

    with (
        patch("src.poster_background.requests.post", return_value=post_response),
        patch("src.poster_background.requests.get", return_value=pending_response),
        patch("src.poster_background.time.sleep", return_value=None),
        patch("src.poster_background.time.monotonic", side_effect=fake_monotonic),
        pytest.raises(TimeoutError),
    ):
        generate_ai_background(SAMPLE_STYLE_KIT, (1080, 1350))


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("BFL_API_KEY", raising=False)

    with pytest.raises(OSError, match="BFL_API_KEY"):
        generate_ai_background(SAMPLE_STYLE_KIT, (1080, 1350))
