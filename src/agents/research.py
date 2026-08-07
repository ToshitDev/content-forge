"""ResearchAgent: turns raw audience material into pain points and content ideas."""

from src.agents.base import BaseAgent


class ResearchAgent(BaseAgent):
    """Runs the Research Agent prompt (src/prompts/research.txt)."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 4000,
        use_cache: bool = True,
    ) -> None:
        """Create a ResearchAgent bound to the "research" prompt template."""
        super().__init__(
            prompt_name="research", model=model, max_tokens=max_tokens, use_cache=use_cache
        )
