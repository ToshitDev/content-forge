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
from elevenlabs import VoiceSettings
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

# Matches any [bracketed] span — both the Script Agent's production
# markers ("[HOOK - 0-3 sec]", "[ON-SCREEN TEXT: ...]", "[PAUSE 1 sec]",
# "[CUT]", "[VALUE]", "[CTA]", "[PROBLEM]") and ElevenLabs v3 emotional
# audio tags ("[excited]", "[whispers]", ...). Capturing group so the
# content can be checked against EMOTIONAL_TAG_WHITELIST below.
BRACKETED_CONTENT_PATTERN = re.compile(r"\[([^\[\]]*)\]")

# Explicit whitelist, not a heuristic: clean_script_for_voice() keeps a
# bracketed span only if its content (case-insensitive) exactly matches
# one of these. Everything else — HOOK/PROBLEM/VALUE/CTA section labels,
# "[PAUSE 1 sec]", "[ON-SCREEN TEXT: ...]", "[CUT]" — is a production
# marker and gets stripped. Guessing at "looks like a short lowercase
# word" would also match things like a stray "[ok]" in dialogue; an
# explicit list doesn't.
EMOTIONAL_TAG_WHITELIST = frozenset(
    {
        "excited",
        "whispers",
        "sighs",
        "laughs",
        "curious",
        "serious",
        "pause",
        "rushed",
        "drawn out",
    }
)

# ElevenLabs' own default settings read as flat and robotic for spoken
# narration — stability around 0.5 leaves little room for natural pitch
# and pace variation. Lower stability and a nonzero style push toward
# more expressive, human-sounding delivery; only eleven_multilingual_v2
# and other non-Flash/Turbo models actually use "style" (Flash/Turbo
# mostly ignore it), which is exactly the model this agent uses.
DEFAULT_STABILITY = 0.35
DEFAULT_SIMILARITY_BOOST = 0.75
DEFAULT_STYLE = 0.25

V2_MODEL_ID = "eleven_multilingual_v2"
# Inline audio tags like [excited] only work on v3 — v2 just reads them
# as literal text ("bracket excited bracket"). v3 is picked automatically
# in generate() when the cleaned text still contains a tag.
V3_MODEL_ID = "eleven_v3"

# ElevenLabs' docs describe three named stability "modes" for v3 —
# Creative, Natural, Robust — but the API's voice_settings.stability
# field only ever accepts a float: passing the string "Creative" raises
# a pydantic ValidationError against the installed SDK (verified
# directly, not assumed). This is that mode's numeric equivalent — the
# low end of the 0-1 range. It matters specifically for v3: at high
# stability v3 tends to smooth inline tags away to stay consistent,
# so a request with real audio tags needs this to make them audible.
V3_TAG_STABILITY = 0.0


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

    def __init__(
        self,
        stability: float = DEFAULT_STABILITY,
        similarity_boost: float = DEFAULT_SIMILARITY_BOOST,
        style: float = DEFAULT_STYLE,
    ) -> None:
        """Create a VoiceAgent with the given delivery settings.

        Args:
            stability: 0-1. Lower means more expressive/varied delivery;
                higher means flatter but more consistent. ElevenLabs'
                own default (~0.5) reads as monotone for narration.
            similarity_boost: 0-1. How closely to match the reference
                voice's timbre.
            style: 0-1. Exaggerates the voice's inherent style for more
                expressive delivery. Only has an effect on models that
                support it — eleven_multilingual_v2 (used below) does;
                the Flash/Turbo models mostly ignore it. 0 disables it.
        """
        self.client = ElevenLabs(api_key=_get_api_key())
        self.voice_settings = VoiceSettings(
            stability=stability, similarity_boost=similarity_boost, style=style
        )

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

        # Anything still bracketed after cleaning is, by construction, a
        # whitelisted emotional tag (clean_script_for_voice strips every
        # other bracketed span) — so this alone tells us whether v3 is
        # needed for this particular script.
        if BRACKETED_CONTENT_PATTERN.search(spoken_text):
            model_id = V3_MODEL_ID
            voice_settings = VoiceSettings(
                stability=V3_TAG_STABILITY,
                similarity_boost=self.voice_settings.similarity_boost,
                style=self.voice_settings.style,
            )
        else:
            model_id = V2_MODEL_ID
            voice_settings = self.voice_settings

        audio_chunks = self._call_with_retry(
            lambda: self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=spoken_text,
                model_id=model_id,
                output_format="mp3_44100_128",
                voice_settings=voice_settings,
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
    """Strip bracketed production markers, but keep emotional audio tags.

    A Script Agent script is a full production script, not a plain
    transcript — markers like "[HOOK - 0-3 sec]", "[ON-SCREEN TEXT:
    ...]", "[PAUSE 1 sec]", "[CUT]", "[VALUE]", "[CTA]", "[PROBLEM]" are
    stage directions for whoever's filming/editing, not words to say.
    Read verbatim, ElevenLabs speaks the markers too, so those are
    removed. But the script may also contain ElevenLabs v3 emotional
    audio tags like "[excited]" or "[whispers]", which are meant to
    reach ElevenLabs intact — those are kept (see EMOTIONAL_TAG_WHITELIST
    for exactly which ones). Also collapses the blank lines/whitespace a
    removed marker leaves behind.
    """

    def keep_if_emotional_tag(match: re.Match[str]) -> str:
        content = match.group(1).strip().lower()
        return match.group(0) if content in EMOTIONAL_TAG_WHITELIST else ""

    without_production_markers = BRACKETED_CONTENT_PATTERN.sub(keep_if_emotional_tag, script_text)
    # Markers on their own line leave a blank line behind; markers inline
    # leave doubled-up spaces. Collapse both.
    collapsed = re.sub(r"[ \t]+", " ", without_production_markers)
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
