"""generate_ai_background: optional AI-generated poster background via
the Black Forest Labs (BFL) Flux API.

Raw HTTP, not an SDK — BFL doesn't ship an official Python client, and
the surface here is small enough (one POST to start a job, one GET to
poll it, one GET to download the result) that raw requests calls are
simpler than adopting a dependency for it.

This module never falls back to anything itself — it either returns
real image bytes or raises. Falling back to the procedural background
is poster_render.py's job (see its use_ai_background parameter), so
this stays a plain "do the one thing or fail clearly" function.
"""

import logging
import os
import time

import requests
from dotenv import load_dotenv

from src.models import StyleOutput

load_dotenv()

logger = logging.getLogger(__name__)

BFL_API_BASE = "https://api.bfl.ai"
MODEL_ENDPOINT = "v1/flux-pro-1.1"
REQUEST_TIMEOUT_SECONDS = 15
DOWNLOAD_TIMEOUT_SECONDS = 30

# Flux endpoints require width/height to be multiples of 32, in
# [256, 1440] — a poster's actual canvas size (e.g. the default
# 1080x1350) usually isn't. Requested dimensions are rounded to the
# nearest valid value; the caller resizes the result back to the
# poster's real size, so the small rounding mismatch is never visible.
BFL_MIN_DIMENSION = 256
BFL_MAX_DIMENSION = 1440
BFL_DIMENSION_STEP = 32

RETRY_DELAYS = (1, 2, 4)  # seconds — for the initial "start the job" request only
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

POLL_INTERVAL_SECONDS = 0.75
POLL_TIMEOUT_SECONDS = 30
TERMINAL_FAILURE_STATUSES = {"Error", "Failed", "Content Moderated", "Request Moderated"}


def generate_ai_background(style_kit: StyleOutput, size: tuple[int, int]) -> bytes:
    """Generate an AI background image matching `style_kit`'s aesthetic.

    Returns raw image bytes (JPEG). Raises on any failure — see the
    module docstring for why this doesn't fall back to anything itself.

    Args:
        style_kit: Drives the generation prompt (vibe, font_mood,
            colors) — nothing about the poster's actual text (headline,
            event details) goes into the prompt.
        size: (width, height) in pixels the poster will ultimately be.
            Rounded to a valid Flux request size internally; resize the
            returned bytes to this exact size yourself if you need it.
    """
    api_key = _get_api_key()
    prompt = _build_prompt(style_kit)
    width, height = size
    request_width = _round_to_bfl_dimension(width)
    request_height = _round_to_bfl_dimension(height)

    request_id, polling_url = _submit_with_retry(prompt, request_width, request_height, api_key)
    sample_url = _poll_until_ready(request_id, polling_url, api_key)
    return _download(sample_url)


def _build_prompt(style_kit: StyleOutput) -> str:
    """Build a Flux prompt from the style kit's aesthetic fields only,
    with an explicit, repeated no-text instruction.

    Image models routinely garble any text they're asked to render, and
    every real piece of text on the poster (headline, subtext) is
    already drawn precisely by Pillow in poster_render.py — so any text
    the model added here would only ever be noise to paint over, never
    a wanted element.
    """
    colors = ", ".join(style_kit.colors)
    return (
        f"Abstract textural background. Vibe: {style_kit.vibe}. "
        f"Typography mood to complement: {style_kit.font_mood}. "
        f"Color palette: {colors}. "
        "Pure visual texture only — soft gradients, organic shapes, light, "
        "or subtle patterns. "
        "IMPORTANT: absolutely no text, no words, no letters, no numbers, "
        "no typography, no writing, no characters, no logos, no watermarks "
        "anywhere in the image — a completely text-free background."
    )


def _round_to_bfl_dimension(value: int) -> int:
    """Round `value` to the nearest multiple of 32 within BFL's
    [256, 1440] accepted range."""
    rounded = round(value / BFL_DIMENSION_STEP) * BFL_DIMENSION_STEP
    return max(BFL_MIN_DIMENSION, min(BFL_MAX_DIMENSION, rounded))


def _submit_with_retry(prompt: str, width: int, height: int, api_key: str) -> tuple[str, str]:
    """POST the generation request, retrying transient failures with
    exponential backoff. This is the one-shot "start the job" call only
    — the polling loop below has its own, separate timeout instead of
    retry/backoff, since polling is expected to take multiple requests
    by design.
    """
    delays = (0,) + RETRY_DELAYS
    last_error: Exception | None = None
    for attempt, delay in enumerate(delays):
        if delay:
            logger.warning(
                "BFL background request: retrying after %s (attempt %d/%d), waiting %ss",
                type(last_error).__name__,
                attempt,
                len(delays) - 1,
                delay,
            )
            time.sleep(delay)

        try:
            response = requests.post(
                f"{BFL_API_BASE}/{MODEL_ENDPOINT}",
                headers={
                    "accept": "application/json",
                    "x-key": api_key,
                    "Content-Type": "application/json",
                },
                json={"prompt": prompt, "width": width, "height": height},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            # Network-level failure (timeout, connection reset) — always
            # worth retrying, same as a 5xx.
            last_error = error
            continue

        if response.status_code == 200:
            data = response.json()
            return data["id"], data["polling_url"]

        if response.status_code not in RETRYABLE_STATUS_CODES:
            # A permanent failure (bad key, bad prompt, ...) fails the
            # same way on every attempt — retrying wastes 7+ seconds for
            # no chance of a different outcome, so raise immediately.
            logger.error(
                "BFL background request failed (non-retryable): %s %s",
                response.status_code,
                response.text[:200],
            )
            response.raise_for_status()

        last_error = RuntimeError(
            f"BFL request failed: {response.status_code} {response.text[:200]}"
        )

    assert last_error is not None
    logger.error("BFL background request failed after all retries: %s", last_error)
    raise last_error


def _poll_until_ready(request_id: str, polling_url: str, api_key: str) -> str:
    """Poll `polling_url` until the job's status is "Ready", returning
    the generated image's temporary download URL.

    Raises:
        TimeoutError: If the job isn't ready within POLL_TIMEOUT_SECONDS.
        RuntimeError: If BFL reports the job failed or was moderated.
    """
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = requests.get(
            polling_url,
            headers={"accept": "application/json", "x-key": api_key},
            params={"id": request_id},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        status = data.get("status")

        if status == "Ready":
            return str(data["result"]["sample"])
        if status in TERMINAL_FAILURE_STATUSES:
            raise RuntimeError(
                f"BFL background generation failed: {status} ({data.get('details')})"
            )

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"BFL background generation timed out after {POLL_TIMEOUT_SECONDS}s "
        f"(request id: {request_id})"
    )


def _download(sample_url: str) -> bytes:
    """Download the generated image from its temporary signed URL."""
    response = requests.get(sample_url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def _get_api_key() -> str:
    """Read BFL_API_KEY from the environment, failing with a clear message."""
    api_key = os.environ.get("BFL_API_KEY")
    if not api_key:
        raise OSError("BFL_API_KEY is not set. Copy .env.example to .env and add your key.")
    return api_key
