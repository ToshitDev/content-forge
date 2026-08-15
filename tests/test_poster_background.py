"""Tests for src/poster_background.py's BFL API integration.

Mocks requests.post/requests.get — no real network calls, no real
BFL_API_KEY needed beyond a fake env var to get past the fail-fast check.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.models import StyleOutput
from src.poster_background import (
    IMAGE_EDIT_MODEL_ENDPOINT,
    MAX_GENERATION_ATTEMPTS,
    TEXT_ONLY_MODEL_ENDPOINT,
    _build_accent_prompt,
    _build_background_prompt,
    _build_reference_prompt,
    _parse_content_check_response,
    generate_accent_element,
    generate_ai_background,
)

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


@pytest.fixture(autouse=True)
def fake_content_check(monkeypatch):
    """generate_ai_background runs a Claude vision content safety check
    after every generation (see _check_background_is_clean) — stub it to
    always report "clean" so tests that aren't specifically exercising
    that check don't need a real ANTHROPIC_API_KEY or hit the network.
    Tests for the check itself override this via monkeypatch locally.
    """
    monkeypatch.setattr(
        "src.poster_background._check_background_is_clean", lambda image_bytes: (True, None)
    )


def _mocked_job(sample_bytes: bytes = b"fake-image-bytes"):
    """Build the post/poll/download mock chain shared by most tests here:
    a successful job submission, an immediately-Ready poll, and a
    download returning `sample_bytes`."""
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
    download_response = MagicMock(status_code=200, content=sample_bytes)
    return post_response, poll_response, download_response


def test_build_background_prompt_always_includes_no_text_instruction():
    prompt = _build_background_prompt(SAMPLE_STYLE_KIT)
    lowered = prompt.lower()
    for phrase in ("no text", "no words", "no letters", "no numbers"):
        assert phrase in lowered


def test_build_reference_prompt_always_includes_no_text_instruction():
    prompt = _build_reference_prompt(SAMPLE_STYLE_KIT)
    lowered = prompt.lower()
    for phrase in ("no text", "no words", "no letters", "no numbers"):
        assert phrase in lowered


def test_build_background_prompt_includes_strengthened_no_logo_instruction():
    prompt = _build_background_prompt(SAMPLE_STYLE_KIT).lower()
    for phrase in ("zero logos", "emblems", "badges", "brand marks", "shield-shaped"):
        assert phrase in prompt


def test_build_reference_prompt_includes_strengthened_no_logo_instruction():
    """Most important on this path: conditioning on a real reference
    image is what let the model reproduce an actual logo out of it, not
    just describe one in words — this is the leak the fix targets."""
    prompt = _build_reference_prompt(SAMPLE_STYLE_KIT).lower()
    for phrase in ("zero logos", "emblems", "badges", "brand marks", "shield-shaped"):
        assert phrase in prompt


def test_build_accent_prompt_includes_no_text_instruction_and_element_type():
    prompt = _build_accent_prompt(SAMPLE_STYLE_KIT, "badge")
    assert "badge" in prompt.lower()
    assert "no text" in prompt.lower()


def test_generate_ai_background_sends_a_prompt_with_no_text_instruction():
    """End to end through generate_ai_background (not just the prompt
    builder directly) — confirms the actual POST body BFL receives
    carries the no-text instruction, with both the initial POST and the
    polling GET mocked, plus the final image download."""
    post_response, poll_response, download_response = _mocked_job()

    with (
        patch("src.poster_background.requests.post", return_value=post_response) as mock_post,
        patch(
            "src.poster_background.requests.get", side_effect=[poll_response, download_response]
        ),
    ):
        result = generate_ai_background(SAMPLE_STYLE_KIT, (1080, 1350))

    assert result == b"fake-image-bytes"

    args, kwargs = mock_post.call_args
    assert args[0].endswith(TEXT_ONLY_MODEL_ENDPOINT)
    sent_prompt = kwargs["json"]["prompt"]
    assert "no text" in sent_prompt.lower()
    assert "no words" in sent_prompt.lower()
    # Width/height sent to BFL must be valid (multiple of 32, in
    # [256, 1440]) even though the caller asked for 1080x1350.
    assert kwargs["json"]["width"] % 32 == 0
    assert kwargs["json"]["height"] % 32 == 0


def test_generate_ai_background_sends_strengthened_no_logo_instruction():
    """Same end-to-end shape as the no-text test above, but for the
    strengthened no-logo instruction added to fix the reference-image
    path reproducing a logo-like emblem from the uploaded reference."""
    post_response, poll_response, download_response = _mocked_job()

    with (
        patch("src.poster_background.requests.post", return_value=post_response) as mock_post,
        patch(
            "src.poster_background.requests.get", side_effect=[poll_response, download_response]
        ),
    ):
        generate_ai_background(SAMPLE_STYLE_KIT, (1080, 1350))

    sent_prompt = mock_post.call_args.kwargs["json"]["prompt"].lower()
    for phrase in ("zero logos", "emblems", "badges", "brand marks"):
        assert phrase in sent_prompt


def test_generate_ai_background_without_reference_image_uses_text_only_endpoint():
    """No reference_image_bytes given (theme-only) -> the original
    flux-pro-1.1 text-only path, unchanged."""
    post_response, poll_response, download_response = _mocked_job()

    with (
        patch("src.poster_background.requests.post", return_value=post_response) as mock_post,
        patch(
            "src.poster_background.requests.get", side_effect=[poll_response, download_response]
        ),
    ):
        generate_ai_background(SAMPLE_STYLE_KIT, (1080, 1350), reference_image_bytes=None)

    args, kwargs = mock_post.call_args
    assert args[0].endswith(TEXT_ONLY_MODEL_ENDPOINT)
    assert "input_image" not in kwargs["json"]


def test_generate_ai_background_with_reference_image_uses_image_edit_endpoint():
    """A reference image given -> Flux 2's image-editing endpoint, with
    the reference riding along as base64 input_image, and the exact
    requested width/height sent unrounded (flux-2-pro-preview doesn't
    share flux-pro-1.1's 32-multiple constraint)."""
    post_response, poll_response, download_response = _mocked_job()
    reference_bytes = b"raw-reference-image-bytes"

    with (
        patch("src.poster_background.requests.post", return_value=post_response) as mock_post,
        patch(
            "src.poster_background.requests.get", side_effect=[poll_response, download_response]
        ),
    ):
        result = generate_ai_background(
            SAMPLE_STYLE_KIT, (1080, 1350), reference_image_bytes=reference_bytes
        )

    assert result == b"fake-image-bytes"
    args, kwargs = mock_post.call_args
    assert args[0].endswith(IMAGE_EDIT_MODEL_ENDPOINT)
    payload = kwargs["json"]
    assert "input_image" in payload
    assert payload["width"] == 1080
    assert payload["height"] == 1350
    sent_prompt = payload["prompt"]
    assert "no text" in sent_prompt.lower()


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


def test_generate_ai_background_regenerates_once_when_content_check_flags_a_logo(monkeypatch):
    """The first generated image is flagged as containing a logo; a
    second, fresh Flux generation is attempted and accepted once the
    check reports it clean — confirms the retry path actually pays for
    a new generation rather than just re-checking the same bytes."""
    post_response, poll_response, download_response = _mocked_job(sample_bytes=b"unclean-then-clean")

    check_results = iter([(False, "circular emblem resembling a logo"), (True, None)])
    monkeypatch.setattr(
        "src.poster_background._check_background_is_clean", lambda image_bytes: next(check_results)
    )

    with (
        patch(
            "src.poster_background.requests.post",
            side_effect=[post_response, post_response],
        ) as mock_post,
        patch(
            "src.poster_background.requests.get",
            side_effect=[poll_response, download_response, poll_response, download_response],
        ),
    ):
        result = generate_ai_background(SAMPLE_STYLE_KIT, (1080, 1350))

    assert result == b"unclean-then-clean"
    assert mock_post.call_count == 2


def test_generate_ai_background_raises_after_exhausting_content_check_retries(monkeypatch):
    """Every attempt is flagged as unclean — generate_ai_background must
    give up after MAX_GENERATION_ATTEMPTS rather than returning a
    known-unclean image; the caller (poster_render.py) already treats
    any exception here as "fall back to the procedural background"."""
    post_response, poll_response, download_response = _mocked_job()

    monkeypatch.setattr(
        "src.poster_background._check_background_is_clean",
        lambda image_bytes: (False, "logo-like emblem"),
    )

    with (
        patch(
            "src.poster_background.requests.post",
            side_effect=[post_response] * MAX_GENERATION_ATTEMPTS,
        ) as mock_post,
        patch(
            "src.poster_background.requests.get",
            side_effect=[poll_response, download_response] * MAX_GENERATION_ATTEMPTS,
        ),
        pytest.raises(RuntimeError, match="logo-like emblem"),
    ):
        generate_ai_background(SAMPLE_STYLE_KIT, (1080, 1350))

    assert mock_post.call_count == MAX_GENERATION_ATTEMPTS


def test_parse_content_check_response_reads_clean_and_unclean_verdicts():
    assert _parse_content_check_response('{"clean": true}') == (True, None)
    assert _parse_content_check_response(
        '{"clean": false, "reason": "circular badge graphic"}'
    ) == (False, "circular badge graphic")


def test_parse_content_check_response_strips_a_markdown_fence():
    raw = '```json\n{"clean": false, "reason": "logo"}\n```'
    assert _parse_content_check_response(raw) == (False, "logo")


def test_parse_content_check_response_treats_unparseable_output_as_clean():
    """Best-effort: a malformed reply shouldn't block background
    generation over the safety-check's own failure — see
    _check_background_is_clean's docstring."""
    assert _parse_content_check_response("not json at all") == (True, None)


def test_generate_accent_element_returns_a_transparent_png():
    """generate_accent_element chroma-keys the solid-white background it
    asked BFL for into real alpha transparency — verify the returned
    bytes are a PNG with a non-opaque pixel where the source was white."""
    import io

    from PIL import Image

    # A tiny solid-white "generated" image, standing in for what BFL
    # would return for an isolated accent graphic on a white background.
    source = Image.new("RGB", (32, 32), (255, 255, 255))
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")

    post_response, poll_response, download_response = _mocked_job(sample_bytes=buffer.getvalue())

    with (
        patch("src.poster_background.requests.post", return_value=post_response) as mock_post,
        patch(
            "src.poster_background.requests.get", side_effect=[poll_response, download_response]
        ),
    ):
        result = generate_accent_element(SAMPLE_STYLE_KIT, "badge")

    args, _ = mock_post.call_args
    assert args[0].endswith(TEXT_ONLY_MODEL_ENDPOINT)

    output_image = Image.open(io.BytesIO(result))
    assert output_image.mode == "RGBA"
    # A pure-white source pixel must have been keyed down toward
    # transparent, not left fully opaque.
    assert output_image.getpixel((0, 0))[3] < 255
