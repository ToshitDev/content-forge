"""PosterAgent: plans a poster's headline, subtext, colors, and layout."""

from src.agents.base import BaseAgent


class PosterAgent(BaseAgent):
    """Runs the Poster Agent prompt (src/prompts/poster.txt)."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 4000,
        use_cache: bool = True,
    ) -> None:
        """Create a PosterAgent bound to the "poster" prompt template."""
        super().__init__(
            prompt_name="poster", model=model, max_tokens=max_tokens, use_cache=use_cache
        )
