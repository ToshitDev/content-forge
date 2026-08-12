"""StyleAgent: extracts a lightweight style kit from a reference image."""

import asyncio
import base64
import hashlib
import logging
from typing import Any, cast

from src import cache
from src.agents.base import RATE_LIMITER, RETRY_DELAYS, RETRYABLE_ERRORS, BaseAgent

logger = logging.getLogger(__name__)


class StyleAgent(BaseAgent):
    """Runs the Style Agent prompt (src/prompts/style.txt).

    The first agent that needs vision input rather than pure text, so it
    overrides run()/run_parsed() to accept an optional reference image
    alongside the usual template inputs. Template loading/filling, JSON
    parsing, and retry/backoff are otherwise inherited from BaseAgent;
    _call_with_retry/_call_api are re-declared only to widen their
    "prompt is plain text" type to also allow image+text content.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 4000,
        use_cache: bool = True,
    ) -> None:
        """Create a StyleAgent bound to the "style" prompt template."""
        super().__init__(
            prompt_name="style", model=model, max_tokens=max_tokens, use_cache=use_cache
        )

    async def run(
        self,
        inputs: dict[str, Any],
        image_bytes: bytes | None = None,
        media_type: str = "image/png",
    ) -> str:
        """Fill the prompt template and return Claude's reply.

        Same contract as BaseAgent.run(), plus an optional reference
        image. With image_bytes, the filled template text rides alongside
        the image in a vision request; without it, this is text-only,
        using just inputs["theme"].

        Args:
            inputs: Template values — {"theme": "..."}. Pass "" for theme
                if the user didn't provide one.
            image_bytes: Raw bytes of a reference image to extract style
                cues from. None falls back to text-only.
            media_type: MIME type of image_bytes (e.g. "image/png").

        Returns:
            The raw text of Claude's response.
        """
        template = self._load_template()
        prompt = self._fill_template(template, inputs)

        if image_bytes is None:
            content: str | list[dict[str, Any]] = prompt
            cache_input = prompt
        else:
            image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
            content = [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                },
                {"type": "text", "text": prompt},
            ]
            # Hash the raw bytes rather than folding the full base64 payload
            # into the cache key — same uniqueness, much less to hash.
            cache_input = prompt + hashlib.sha256(image_bytes).hexdigest()

        key = cache.cache_key(cache_input)

        if self.use_cache:
            cached = cache.get(key)
            if cached is not None:
                logger.info("[cache hit] %s", self.prompt_name)
                return cached
            logger.warning("[cache miss] %s — calling the API", self.prompt_name)

        logger.info("Calling %s agent", self.prompt_name)
        result = await self._call_with_retry(content)
        logger.info("%s agent call succeeded", self.prompt_name)

        if self.use_cache:
            cache.set(key, result)

        return result

    async def run_parsed(
        self,
        inputs: dict[str, Any],
        image_bytes: bytes | None = None,
        media_type: str = "image/png",
    ) -> dict[str, Any]:
        """Run the agent and parse its reply as JSON in one step."""
        raw = await self.run(inputs, image_bytes=image_bytes, media_type=media_type)
        return self.parse_json(raw)

    async def _call_with_retry(self, content: str | list[dict[str, Any]]) -> str:
        """Same retry/backoff loop as BaseAgent._call_with_retry, widened to
        also accept a list of content blocks (image + text).
        """
        delays = (0,) + RETRY_DELAYS
        last_error: Exception | None = None
        for attempt, delay in enumerate(delays):
            if delay:
                logger.warning(
                    "%s agent: retrying after %s (attempt %d/%d), waiting %ss",
                    self.prompt_name,
                    type(last_error).__name__,
                    attempt,
                    len(delays) - 1,
                    delay,
                )
                await asyncio.sleep(delay)
            try:
                return await self._call_api(content)
            except RETRYABLE_ERRORS as error:
                last_error = error
        assert last_error is not None
        logger.error("%s agent failed after all retries: %s", self.prompt_name, last_error)
        raise last_error

    async def _call_api(self, content: str | list[dict[str, Any]]) -> str:
        """Same as BaseAgent._call_api, widened to accept image+text content."""
        await RATE_LIMITER.acquire()
        # The SDK's MessageParam typing enumerates every specific content-block
        # TypedDict; our plain dicts are validated by the API at request time
        # instead, so the cast just tells mypy what we already know at runtime.
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=cast(Any, [{"role": "user", "content": content}]),
        )
        if response.stop_reason == "max_tokens":
            logger.error(
                "%s agent response truncated (max_tokens=%d)", self.prompt_name, self.max_tokens
            )
            raise ValueError(
                f"Response from '{self.prompt_name}' was truncated (hit "
                f"max_tokens={self.max_tokens}). Increase max_tokens or shorten the prompt."
            )
        return "".join(block.text for block in response.content if block.type == "text")
