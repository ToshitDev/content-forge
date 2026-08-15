"""AI-generated poster assets via the Black Forest Labs (BFL) Flux API:
generate_ai_background() for the poster background, generate_accent_element()
for small decorative graphics composited on top.

Raw HTTP, not an SDK — BFL doesn't ship an official Python client, and
the surface here (POST to start a job, GET to poll it, GET to download
the result) is small enough that raw requests calls are simpler than
adopting a dependency for it.

Neither function falls back to anything itself — each either returns
real image bytes or raises. Falling back to the procedural background,
or skipping accents entirely, is poster_render.py's job (see its
use_ai_background / use_accent_elements parameters), so this module
stays a plain "do the one thing or fail clearly" set of functions.

CONTENT SAFETY CHECK (background only, not accents): generate_ai_background
verifies each generated image via Claude vision (_check_background_is_clean)
before accepting it, retrying generation up to MAX_GENERATION_ATTEMPTS
times if the check flags text/logo-like content — see that function's
docstring. This is a second, independent layer on top of the no-text/
no-logo prompt instructions below, not a replacement for them. KNOWN
LIMITATION: this relies entirely on Claude's judgment call from a
single still image; it does not run any pixel-level heuristic (e.g.
detecting a small isolated high-contrast badge-shaped region), which
would be needed to catch a subtler leak that reads as "clean" to a
vision model but still looks logo-like to a human. Revisit with a
real heuristic if that turns out to be a real gap in practice.
"""

import base64
import io
import json
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import anthropic
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

from src.models import StyleOutput

load_dotenv()

logger = logging.getLogger(__name__)

BFL_API_BASE = "https://api.bfl.ai"

# flux-pro-1.1: text-only generation (the original background path, and
# the model used for accent elements — small isolated graphics don't
# benefit from image-editing conditioning the way a background does).
TEXT_ONLY_MODEL_ENDPOINT = "v1/flux-pro-1.1"

# flux-2-pro-preview: Flux 2's image-editing endpoint. Takes an
# `input_image` alongside the prompt and conditions generation on it —
# verified live against the real API before this was written: the
# response echoes back a Flux-rewritten prompt that concretely described
# our test reference image's actual color bands, confirming it really
# looks at the pixels rather than silently ignoring input_image.
IMAGE_EDIT_MODEL_ENDPOINT = "v1/flux-2-pro-preview"

REQUEST_TIMEOUT_SECONDS = 15
DOWNLOAD_TIMEOUT_SECONDS = 30

# flux-pro-1.1 requires width/height to be multiples of 32, in
# [256, 1440] — a poster's actual canvas size (e.g. the default
# 1080x1350) usually isn't. Requested dimensions are rounded to the
# nearest valid value; the caller resizes the result back to the
# poster's real size, so the small rounding mismatch is never visible.
# flux-2-pro-preview does NOT share this constraint — verified live, it
# accepts arbitrary width/height directly (though it may still adjust
# the output slightly server-side, e.g. 1080x1350 came back as
# 1072x1344) — so this rounding is only applied on the text-only path.
BFL_MIN_DIMENSION = 256
BFL_MAX_DIMENSION = 1440
BFL_DIMENSION_STEP = 32

RETRY_DELAYS = (1, 2, 4)  # seconds — for the initial "start the job" request only
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

POLL_INTERVAL_SECONDS = 0.75
POLL_TIMEOUT_SECONDS = 30
TERMINAL_FAILURE_STATUSES = {"Error", "Failed", "Content Moderated", "Request Moderated"}

ACCENT_ELEMENT_SIZE = (512, 512)
# The chroma-key trick in _apply_chroma_key_transparency needs the
# generated element on a known, solid, high-contrast background to key
# out. Pure white works well in practice (verified live) and rarely
# appears as pure white *within* a generated icon/badge itself.
CHROMA_KEY_PROMPT_SUFFIX = "solid pure white background, isolated, product photography style"
# Soft ramp, not a hard cutoff: fully transparent within this distance
# of pure white, fully opaque past it, linearly blended in between. A
# hard cutoff leaves a visible white fringe at anti-aliased edges.
CHROMA_KEY_LOW_THRESHOLD = 8
CHROMA_KEY_HIGH_THRESHOLD = 40


def generate_ai_background(
    style_kit: StyleOutput, size: tuple[int, int], reference_image_bytes: bytes | None = None
) -> bytes:
    """Generate an AI background image matching `style_kit`'s aesthetic.

    Two distinct paths, chosen by whether a reference image is
    available — see _generate_from_reference_image and
    _generate_from_text_only for the mechanics of each:

    - reference_image_bytes given: real image-based style transfer via
      Flux 2's editing endpoint. The reference rides along as
      input_image, so generation conditions on its actual composition,
      texture, and color relationships — not just a paraphrased text
      description of them.
    - reference_image_bytes is None (theme-only, no image uploaded):
      the original pure-text-prompt path via flux-pro-1.1, unchanged
      from before image-based style transfer existed.

    Returns raw image bytes. Raises on any failure — see the module
    docstring for why this doesn't fall back to anything itself. That
    includes exhausting every content-safety-check retry (see
    _check_background_is_clean) without ever getting back a clean image.

    Args:
        style_kit: Drives the generation prompt (vibe, font_mood,
            colors) on both paths — nothing about the poster's actual
            text (headline, event details) goes into the prompt.
        size: (width, height) in pixels the poster will ultimately be.
            Adjusted to a valid Flux request size internally; resize
            the returned bytes to this exact size yourself if you need
            it (poster_render.py always does).
        reference_image_bytes: Raw bytes of the user's uploaded
            reference image (the same one the Style Agent analyzed), if
            any. None routes to the text-only path.
    """
    api_key = _get_api_key()
    width, height = size

    def _attempt() -> bytes:
        if reference_image_bytes is not None:
            return _generate_from_reference_image(
                style_kit, reference_image_bytes, width, height, api_key
            )
        return _generate_from_text_only(style_kit, width, height, api_key)

    return _generate_with_content_safety_check(_attempt)


# One retry beyond the first attempt: each attempt is a real, billed Flux
# generation, so this bounds the extra cost a persistent leak can cause
# rather than retrying indefinitely.
MAX_GENERATION_ATTEMPTS = 2


def _generate_with_content_safety_check(generate_once: Callable[[], bytes]) -> bytes:
    """Call `generate_once` (a zero-arg thunk that runs one full Flux
    generation and returns image bytes), verifying each result via
    Claude vision before accepting it. Regenerates from scratch — same
    prompt, a fresh Flux call — up to MAX_GENERATION_ATTEMPTS times if
    the check flags text/logo-like content, since Flux isn't seeded and
    a repeat call is a genuinely different image, not a no-op.

    Raises RuntimeError if every attempt fails the check — the caller
    (poster_render.py) already treats any exception from
    generate_ai_background as "fall back to the procedural background",
    so this deliberately never returns a known-unclean image.
    """
    last_reason: str | None = None
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        image_bytes = generate_once()
        is_clean, reason = _check_background_is_clean(image_bytes)
        if is_clean:
            return image_bytes
        last_reason = reason
        logger.warning(
            "AI background failed content safety check (attempt %d/%d): %s",
            attempt,
            MAX_GENERATION_ATTEMPTS,
            reason,
        )
    raise RuntimeError(
        f"AI background repeatedly contained text/logo-like content after "
        f"{MAX_GENERATION_ATTEMPTS} attempt(s): {last_reason}"
    )


ANTHROPIC_CONTENT_CHECK_MODEL = "claude-haiku-4-5-20251001"
CONTENT_CHECK_MAX_TOKENS = 200
_CONTENT_CHECK_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "background_content_check.txt"
)


def _check_background_is_clean(image_bytes: bytes) -> tuple[bool, str | None]:
    """Ask Claude vision whether `image_bytes` contains readable text,
    a logo/emblem/badge, or a watermark — the second, independent layer
    of defense on top of the prompt instructions (_NO_TEXT_INSTRUCTION /
    _NO_LOGO_INSTRUCTION), for exactly the case those don't fully stop:
    live testing showed the image-editing path can still reproduce a
    logo-like emblem straight out of a reference image even with the
    no-logo instruction in the prompt.

    Best-effort, not a hard gate on the whole feature: if ANTHROPIC_API_KEY
    isn't set, the API call fails, or the reply doesn't parse as the
    expected JSON, this logs a warning and returns (True, None) — treats
    the image as clean — rather than raising. The prompt-level defense
    above is the primary safeguard; this check augments it but its own
    failure shouldn't block background generation entirely.

    Returns (is_clean, reason) — reason is a short human-readable
    description of what was flagged, or None when clean / when the
    check couldn't run.
    """
    try:
        api_key = _get_anthropic_api_key()
    except OSError as error:
        logger.warning("Skipping AI background content safety check: %s", error)
        return True, None

    media_type = _detect_media_type(image_bytes)
    image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    prompt = _CONTENT_CHECK_PROMPT_PATH.read_text()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        # The SDK's MessageParam typing enumerates every specific content-block
        # TypedDict; our plain dicts are validated by the API at request time
        # instead, so the cast just tells mypy what we already know at
        # runtime (same pattern as StyleAgent._call_api).
        response = client.messages.create(
            model=ANTHROPIC_CONTENT_CHECK_MODEL,
            max_tokens=CONTENT_CHECK_MAX_TOKENS,
            messages=cast(
                Any,
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            ),
        )
    except Exception as error:  # noqa: BLE001 - safety-net check must never crash generation
        logger.warning("AI background content safety check failed, treating as clean: %s", error)
        return True, None

    raw = "".join(block.text for block in response.content if block.type == "text")
    return _parse_content_check_response(raw)


def _parse_content_check_response(raw: str) -> tuple[bool, str | None]:
    """Parse the {"clean": bool, "reason": str} JSON expected from
    _check_background_is_clean's prompt. Unparseable output (the model
    added stray prose, or something else went wrong) is treated as
    "clean" rather than raised — same best-effort spirit as the caller.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        is_clean = bool(data.get("clean", True))
        return is_clean, (data.get("reason") if not is_clean else None)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Could not parse AI background content check response: %r", raw[:200])
        return True, None


def _detect_media_type(image_bytes: bytes) -> str:
    """Sniff the actual image format so the Anthropic vision request
    declares the right media_type — BFL's response format isn't fixed
    to PNG (only accent elements explicitly request output_format=png),
    so guessing wrong here would send a mismatched media_type/bytes pair.
    """
    try:
        image_format = Image.open(io.BytesIO(image_bytes)).format
    except Exception:  # noqa: BLE001 - fall through to a sane default
        image_format = None
    return "image/jpeg" if image_format == "JPEG" else "image/png"


def _generate_from_text_only(style_kit: StyleOutput, width: int, height: int, api_key: str) -> bytes:
    """The original background path: flux-pro-1.1, pure text prompt, no
    reference image. Requires width/height rounded to a multiple of 32.
    """
    prompt = _build_background_prompt(style_kit)
    payload = {
        "prompt": prompt,
        "width": _round_to_bfl_dimension(width),
        "height": _round_to_bfl_dimension(height),
    }
    request_id, polling_url = _submit_with_retry(TEXT_ONLY_MODEL_ENDPOINT, payload, api_key)
    sample_url = _poll_until_ready(request_id, polling_url, api_key)
    return _download(sample_url)


def _generate_from_reference_image(
    style_kit: StyleOutput,
    reference_image_bytes: bytes,
    width: int,
    height: int,
    api_key: str,
) -> bytes:
    """Real style transfer: the reference image rides along as
    input_image on Flux 2's editing endpoint, so generation conditions
    on its actual pixels rather than a text paraphrase of them.
    """
    prompt = _build_reference_prompt(style_kit)
    image_b64 = base64.b64encode(reference_image_bytes).decode("ascii")
    payload = {
        "prompt": prompt,
        "input_image": image_b64,
        "width": width,
        "height": height,
    }
    request_id, polling_url = _submit_with_retry(IMAGE_EDIT_MODEL_ENDPOINT, payload, api_key)
    sample_url = _poll_until_ready(request_id, polling_url, api_key)
    return _download(sample_url)


def _build_background_prompt(style_kit: StyleOutput) -> str:
    """Build a text-only Flux prompt from the style kit's aesthetic
    fields, with an explicit, repeated no-text instruction.

    Image models routinely garble any text they're asked to render, and
    every real piece of text on the poster (headline, subtext) is
    already drawn precisely by Pillow in poster_render.py — so any text
    the model added here would only ever be noise to paint over, never
    a wanted element. Same reasoning applies to every prompt-builder in
    this module.
    """
    colors = ", ".join(style_kit.colors)
    return (
        f"Abstract textural background. Vibe: {style_kit.vibe}. "
        f"Typography mood to complement: {style_kit.font_mood}. "
        f"Color palette: {colors}. "
        "Pure visual texture only — soft gradients, organic shapes, light, "
        "or subtle patterns. "
        + _NO_TEXT_INSTRUCTION
        + " "
        + _NO_LOGO_INSTRUCTION
    )


def _build_reference_prompt(style_kit: StyleOutput) -> str:
    """Prompt for the image-conditioned path — shorter than
    _build_background_prompt() since Flux 2 already sees the reference
    image's actual colors/composition directly; this just nudges tone
    and repeats the no-text/no-logo instructions, rather than
    re-describing the palette in words the model can already see.

    The no-logo instruction matters most on THIS path: conditioning on
    a real reference image (e.g. an event flyer with its own circular
    logo emblem) makes the model prone to reproducing that logo as an
    actual graphic element, not just describing its style in words —
    confirmed by live testing, not a hypothetical.
    """
    return (
        "Reimagine this reference image as an abstract textural poster "
        "background — preserve its color relationships, composition, and "
        "texture, but make it purely abstract with no recognizable "
        f"subject. Vibe: {style_kit.vibe}. " + _NO_TEXT_INSTRUCTION
        + " "
        + _NO_LOGO_INSTRUCTION
    )


def _build_accent_prompt(style_kit: StyleOutput, element_type: str) -> str:
    """Prompt for one small decorative accent graphic."""
    colors = ", ".join(style_kit.colors)
    return (
        f"A small decorative {element_type}, {style_kit.vibe} style, "
        f"color palette: {colors}. Simple, clean, isolated graphic — "
        f"{CHROMA_KEY_PROMPT_SUFFIX}. " + _NO_TEXT_INSTRUCTION
    )


_NO_TEXT_INSTRUCTION = (
    "IMPORTANT: absolutely no text, no words, no letters, no numbers, no "
    "typography, no writing, no characters, no logos, no watermarks "
    "anywhere in the image — a completely text-free result."
)

# A second, more explicit pass at the same "no logos" idea already
# named above — live testing showed the plain mention in
# _NO_TEXT_INSTRUCTION wasn't enough to stop the image-editing path
# (_build_reference_prompt) from reproducing a logo-like emblem out of
# a reference image, so this spells out the specific shapes that tend
# to read as a logo. Not applied to _build_accent_prompt: accent
# elements (badge/icon/etc.) are deliberately small self-contained
# graphics, so a blanket "no logo-shaped graphics" instruction would
# fight the feature's own purpose there.
_NO_LOGO_INSTRUCTION = (
    "The background must contain ZERO logos, emblems, badges, or brand "
    "marks of any kind — purely abstract texture, color, and composition "
    "only, no circular or shield-shaped graphic elements resembling a logo."
)


def generate_accent_element(style_kit: StyleOutput, element_type: str) -> bytes:
    """Generate a small decorative accent graphic with a transparent
    background, ready to composite onto a poster.

    `element_type` is a short description like "badge", "icon", "corner
    flourish", or "geometric accent shape" — poster_render.py decides
    which type(s) to request (see its accent-selection helper) based on
    the style kit's vibe; this function just generates one of whatever
    type it's given.

    Flux has no native alpha-channel/transparent-background output — it
    generates a solid image regardless of what the prompt asks for
    (verified live: even output_format="png" with "transparent
    background" in the prompt came back as plain RGB on a white
    background). This works around that by explicitly requesting a
    solid white background, then chroma-keying white pixels to
    transparent afterward — see _apply_chroma_key_transparency, whose
    output was visually verified against a real generated asset before
    this was written.

    Returns RGBA PNG bytes. Raises on any failure — same "do the one
    thing or fail clearly" contract as generate_ai_background; the
    caller is responsible for skipping accents gracefully if this
    raises, exactly like it falls back for the background.
    """
    api_key = _get_api_key()
    prompt = _build_accent_prompt(style_kit, element_type)
    width, height = ACCENT_ELEMENT_SIZE
    payload = {
        "prompt": prompt,
        "width": _round_to_bfl_dimension(width),
        "height": _round_to_bfl_dimension(height),
        "output_format": "png",
    }
    request_id, polling_url = _submit_with_retry(TEXT_ONLY_MODEL_ENDPOINT, payload, api_key)
    sample_url = _poll_until_ready(request_id, polling_url, api_key)
    opaque_bytes = _download(sample_url)
    return _apply_chroma_key_transparency(opaque_bytes)


def _apply_chroma_key_transparency(image_bytes: bytes) -> bytes:
    """Turn a solid-white background into real alpha transparency.

    Computes each pixel's Euclidean distance from pure white across all
    three RGB channels (numpy, vectorized — a pure-Python per-pixel loop
    would work but is needlessly slow for a 512x512 image) and maps that
    distance through the soft ramp described by CHROMA_KEY_LOW_THRESHOLD
    / CHROMA_KEY_HIGH_THRESHOLD to get each pixel's alpha.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    pixels = np.array(image)
    rgb = pixels[:, :, :3].astype(int)
    distance_from_white = np.sqrt(((rgb - 255) ** 2).sum(axis=2))
    alpha = (
        np.clip(
            (distance_from_white - CHROMA_KEY_LOW_THRESHOLD)
            / (CHROMA_KEY_HIGH_THRESHOLD - CHROMA_KEY_LOW_THRESHOLD),
            0,
            1,
        )
        * 255
    )
    pixels[:, :, 3] = alpha.astype("uint8")

    output = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(output, format="PNG")
    return output.getvalue()


def _round_to_bfl_dimension(value: int) -> int:
    """Round `value` to the nearest multiple of 32 within BFL's
    [256, 1440] accepted range (flux-pro-1.1's constraint)."""
    rounded = round(value / BFL_DIMENSION_STEP) * BFL_DIMENSION_STEP
    return max(BFL_MIN_DIMENSION, min(BFL_MAX_DIMENSION, rounded))


def _submit_with_retry(endpoint: str, payload: dict, api_key: str) -> tuple[str, str]:
    """POST a generation request to `endpoint`, retrying transient
    failures with exponential backoff. This is the one-shot "start the
    job" call only — the polling loop below has its own, separate
    timeout instead of retry/backoff, since polling is expected to take
    multiple requests by design. Shared by every generation path in this
    module (background, both variants, and accent elements) — the
    retry/backoff mechanics are identical regardless of which Flux model
    or payload shape is being submitted.
    """
    delays = (0,) + RETRY_DELAYS
    last_error: Exception | None = None
    for attempt, delay in enumerate(delays):
        if delay:
            logger.warning(
                "BFL request to %s: retrying after %s (attempt %d/%d), waiting %ss",
                endpoint,
                type(last_error).__name__,
                attempt,
                len(delays) - 1,
                delay,
            )
            time.sleep(delay)

        try:
            response = requests.post(
                f"{BFL_API_BASE}/{endpoint}",
                headers={
                    "accept": "application/json",
                    "x-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
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
                "BFL request to %s failed (non-retryable): %s %s",
                endpoint,
                response.status_code,
                response.text[:200],
            )
            response.raise_for_status()

        last_error = RuntimeError(
            f"BFL request to {endpoint} failed: {response.status_code} {response.text[:200]}"
        )

    assert last_error is not None
    logger.error("BFL request to %s failed after all retries: %s", endpoint, last_error)
    raise last_error


def _poll_until_ready(request_id: str, polling_url: str, api_key: str) -> str:
    """Poll `polling_url` until the job's status is "Ready", returning
    the generated image's temporary download URL. Model-agnostic — used
    by every generation path in this module, since BFL's async job
    lifecycle (queued/pending -> Ready/Error/...) is the same shape
    regardless of which endpoint started the job.

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


def _get_anthropic_api_key() -> str:
    """Read ANTHROPIC_API_KEY from the environment, failing with a clear
    message — same pattern as _get_api_key(), but for the Claude vision
    content safety check rather than BFL image generation itself."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise OSError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    return api_key
