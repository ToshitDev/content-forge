"""VisualAgent: turns a finished script into a shot/slide plan."""

from src.agents.base import BaseAgent


class VisualAgent(BaseAgent):
    """Runs the Visual Agent prompt (src/prompts/visual.txt)."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 4000) -> None:
        """Create a VisualAgent bound to the "visual" prompt template."""
        super().__init__(prompt_name="visual", model=model, max_tokens=max_tokens)
