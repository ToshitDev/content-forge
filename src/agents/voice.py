"""VoiceAgent: turns a script into narration audio via ElevenLabs.

Not a BaseAgent subclass, and not async — see the class docstring for
why. It lives in src/agents/ anyway because it's the same kind of thing
conceptually (one class wrapping one external API for one pipeline
concern), even though the mechanics underneath differ.
"""

import logging
import os
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T")

AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "audio"

# ElevenLabs' "Rachel" voice — a stock voice present on every account,
# used when the user hasn't cloned their own.
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

RETRY_DELAYS = (1, 2, 4)  # seconds
# Only retry failures that are plausibly transient. A bad API key or a
# bad voice_id (400/401/404) will fail identically on every attempt, so
# retrying those just adds latency for no chance of a different outcome.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Rough, order-of-magnitude estimate only (ElevenLabs bills by
# subscription-tier "credits", not a flat per-character rate) — good
# enough for a ballpark cost figure in the run history, not a real
# billing calculation.
ESTIMATED_COST_PER_CHARACTER = 0.00018

# Matches bracketed production markers in a Script Agent script — e.g.
# "[HOOK - 0-3 sec]", "[ON-SCREEN TEXT: ...]", "[PAUSE 1 sec]", "[CUT]",
# "[VALUE]", "[CTA]" — that are meant for whoever's filming, not for a
# text-to-speech voice to read aloud.
BRACKETED_MARKER_PATTERN = re.compile(r"\[[^\[\]]*\]")


class VoiceAgent:
    """Generates narration audio from a script, via the ElevenLabs API.

    Synchronous, unlike the five Anthropic-backed agents: those are
    async so independent pipeline runs can have their API calls in
    flight concurrently (see BaseAgent's module docstring), but voice
    generation is a one-off action a user triggers from a button after
    the pipeline has already finished — there's nothing else for it to
    run alongside, so there's no concurrency to enable.

    Two ways to pick a voice: pass a cloned voice's id (from
    clone_voice()) to generate(), or omit it to use DEFAULT_VOICE_ID.
    """

    def __init__(self) -> None:
        self.client = ElevenLabs(api_key=_get_api_key())

    def clone_voice(self, sample_bytes: bytes, sample_filename: str, name: str) -> str:
        """Upload a short voice sample and return the new cloned voice's id."""
        voice = self._call_with_retry(
            lambda: self.client.voices.ivc.create(
                name=name,
                files=[(sample_filename, sample_bytes)],
                # Must be an actual dict, never omitted. The installed SDK
                # (elevenlabs 2.63.0) serializes an omitted `labels` as the
                # JSON string "null" instead of leaving the field out of
                # the request — jsonable_encoder(OMIT) returns None, and
                # that None still gets json.dumps()'d and sent, bypassing
                # the SDK's own omit-empty-fields logic. The API then
                # rejects "null" with 400 invalid_labels ("Labels must be
                # a serialized dictionary object"), since it parses to
                # None, not a dict. A real dict — even {} — serializes to
                # "{}" and is accepted.
                labels={"use_case": "content_forge_voiceover"},
            )
        )
        return voice.voice_id

    def generate(self, text: str, voice_id: str = DEFAULT_VOICE_ID) -> Path:
        """Synthesize `text` as speech in `voice_id` and save it under audio/.

        Returns the path to the saved MP3 file.
        """
        spoken_text = clean_script_for_voice(text)
        audio_chunks = self._call_with_retry(
            lambda: self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=spoken_text,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
        )
        AUDIO_DIR.mkdir(exist_ok=True)
        path = AUDIO_DIR / f"{uuid.uuid4().hex}.mp3"
        path.write_bytes(b"".join(audio_chunks))
        return path

    def _call_with_retry(self, call: Callable[[], T]) -> T:
        """Retry an ElevenLabs call with exponential backoff.

        Simpler than BaseAgent._call_with_retry in two ways: this isn't
        async (see the class docstring), and there's no shared rate
        limiter — ElevenLabs calls are one-off user actions, not part of
        a pipeline whose calls need pacing against each other. Also,
        unlike BaseAgent's fixed set of retryable exception types, every
        ElevenLabs failure raises the same ApiError class, so the
        retryable/non-retryable split has to be made on its status_code
        instead.
        """
        delays = (0,) + RETRY_DELAYS
        last_error: ApiError | None = None
        for attempt, delay in enumerate(delays):
            if delay:
                logger.warning(
                    "VoiceAgent: retrying after HTTP %s (attempt %d/%d), waiting %ss",
                    last_error.status_code if last_error else "?",
                    attempt,
                    len(delays) - 1,
                    delay,
                )
                time.sleep(delay)
            try:
                return call()
            except ApiError as error:
                if error.status_code not in RETRYABLE_STATUS_CODES:
                    logger.error("VoiceAgent call failed (non-retryable): %s", error)
                    raise
                last_error = error
        assert last_error is not None
        logger.error("VoiceAgent failed after all retries: %s", last_error)
        raise last_error


def clean_script_for_voice(script_text: str) -> str:
    """Strip bracketed production markers, leaving only spoken narration.

    A Script Agent script is a full production script, not a transcript
    — markers like "[HOOK - 0-3 sec]", "[ON-SCREEN TEXT: ...]",
    "[PAUSE 1 sec]", "[CUT]", "[VALUE]", "[CTA]" are stage directions
    mixed in among the words actually meant to be said. Read verbatim,
    ElevenLabs speaks the markers too; this removes them and cleans up
    the blank lines/whitespace left behind.
    """
    without_markers = BRACKETED_MARKER_PATTERN.sub("", script_text)
    # Markers on their own line leave a blank line behind; markers inline
    # leave doubled-up spaces. Collapse both.
    collapsed = re.sub(r"[ \t]+", " ", without_markers)
    collapsed = re.sub(r"\n[ \t]*\n+", "\n\n", collapsed)
    return collapsed.strip()


def estimate_cost(text: str) -> float:
    """Very rough dollar estimate for narrating `text`, for the run history.

    Not a real billing figure — see ESTIMATED_COST_PER_CHARACTER.
    """
    return round(len(text) * ESTIMATED_COST_PER_CHARACTER, 4)


def _get_api_key() -> str:
    """Read ELEVENLABS_API_KEY from the environment, failing with a clear message."""
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise OSError(
            "ELEVENLABS_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return api_key
