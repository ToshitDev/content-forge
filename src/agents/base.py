"""Shared base class for all pipeline agents.

Loads a prompt template, fills in its placeholders from a dict of inputs,
and calls the Anthropic API with retry logic for transient failures.
"""

import os
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Placeholders look like [NICHE], [BRAND_VOICE] — all caps, optional underscores.
PLACEHOLDER_PATTERN = re.compile(r"\[([A-Z][A-Z_]*)\]")

# Errors worth retrying: the request itself was fine, the API was just busy.
RETRYABLE_ERRORS = (anthropic.RateLimitError, anthropic.InternalServerError)
RETRY_DELAYS = (1, 2, 4)  # seconds, one entry per retry attempt


class BaseAgent:
    """Base class for a single-prompt pipeline agent.

    Subclasses just need to pick a `prompt_name` matching a file in
    src/prompts/ (e.g. "hook" for src/prompts/hook.txt).
    """

    def __init__(
        self,
        prompt_name: str,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1500,
    ):
        """Set up the agent and fail fast if the API key isn't configured.

        Args:
            prompt_name: Base filename (no extension) of the prompt template
                in src/prompts/.
            model: Anthropic model ID to call.
            max_tokens: Max tokens to generate per call.
        """
        self.prompt_name = prompt_name
        self.model = model
        self.max_tokens = max_tokens
        # max_retries=0: our own _call_with_retry is the only retry logic,
        # so the SDK's built-in retries don't quietly stack on top of ours.
        self.client = anthropic.Anthropic(api_key=_get_api_key(), max_retries=0)

    def run(self, inputs: dict) -> str:
        """Fill the prompt template with `inputs` and return Claude's reply.

        Args:
            inputs: Values to slot into the template. Keys are lowercase
                versions of the template's placeholder names — e.g. the
                template placeholder [NICHE] is filled by inputs["niche"].

        Returns:
            The raw text of Claude's response.

        Raises:
            ValueError: If any placeholder in the template has no matching
                key in `inputs` and is left unfilled.
        """
        template = self._load_template()
        prompt = self._fill_template(template, inputs)
        return self._call_with_retry(prompt)

    def _load_template(self) -> str:
        """Read the raw prompt template for this agent."""
        path = PROMPTS_DIR / f"{self.prompt_name}.txt"
        return path.read_text()

    def _fill_template(self, template: str, inputs: dict) -> str:
        """Replace [PLACEHOLDER] tokens with values from `inputs`.

        A placeholder [FOO] is replaced by inputs["foo"] (lowercased key).
        Any placeholder with no matching key is left as-is, then reported
        as an error once every substitution has been attempted.
        """

        def substitute(match: re.Match) -> str:
            key = match.group(1).lower()
            return str(inputs[key]) if key in inputs else match.group(0)

        filled = PLACEHOLDER_PATTERN.sub(substitute, template)

        remaining = list(dict.fromkeys(PLACEHOLDER_PATTERN.findall(filled)))
        if remaining:
            missing = ", ".join(f"[{name}]" for name in remaining)
            example_key = remaining[0].lower()
            raise ValueError(
                f"Missing input(s) for '{self.prompt_name}' prompt: {missing}. "
                f"Add lowercase keys to `inputs`, e.g. inputs={{'{example_key}': ...}}."
            )
        return filled

    def _call_with_retry(self, prompt: str) -> str:
        """Call the Anthropic API, retrying on rate-limit/overload errors.

        Makes one initial attempt, then up to 3 retries with exponential
        backoff (1s, 2s, 4s), before giving up and re-raising the last error.
        """
        delays = (0,) + RETRY_DELAYS  # no delay before the first attempt
        last_error: Exception | None = None
        for delay in delays:
            if delay:
                time.sleep(delay)
            try:
                return self._call_api(prompt)
            except RETRYABLE_ERRORS as error:
                last_error = error
        raise last_error

    def _call_api(self, prompt: str) -> str:
        """Send one request to the Anthropic API and return the text reply."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


def _get_api_key() -> str:
    """Read ANTHROPIC_API_KEY from the environment, failing with a clear message."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return api_key
