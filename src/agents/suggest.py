"""SuggestAgent: generates illustrative example audience comments."""

from src.agents.base import BaseAgent


class SuggestAgent(BaseAgent):
    """Runs the Suggest Agent prompt (src/prompts/suggest.txt)."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 4000,
        use_cache: bool = True,
    ) -> None:
        """Create a SuggestAgent bound to the "suggest" prompt template."""
        super().__init__(
            prompt_name="suggest", model=model, max_tokens=max_tokens, use_cache=use_cache
        )
