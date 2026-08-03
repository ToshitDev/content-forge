"""GrowthAgent: scores a finished post package and gives a publish/rework call."""

from src.agents.base import BaseAgent


class GrowthAgent(BaseAgent):
    """Runs the Growth Agent prompt (src/prompts/growth.txt)."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1500):
        """Create a GrowthAgent bound to the "growth" prompt template."""
        super().__init__(prompt_name="growth", model=model, max_tokens=max_tokens)
