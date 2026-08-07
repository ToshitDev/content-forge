"""ScriptAgent: turns a winning hook into a full platform-ready script."""

from src.agents.base import BaseAgent


class ScriptAgent(BaseAgent):
    """Runs the Script Agent prompt (src/prompts/script.txt)."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 4000,
        use_cache: bool = True,
    ) -> None:
        """Create a ScriptAgent bound to the "script" prompt template."""
        super().__init__(
            prompt_name="script", model=model, max_tokens=max_tokens, use_cache=use_cache
        )
