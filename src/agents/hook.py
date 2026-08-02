"""HookAgent: generates and ranks hook lines for a chosen content idea."""

from src.agents.base import BaseAgent


class HookAgent(BaseAgent):
    """Runs the Hook Agent prompt (src/prompts/hook.txt)."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1500):
        """Create a HookAgent bound to the "hook" prompt template."""
        super().__init__(prompt_name="hook", model=model, max_tokens=max_tokens)


if __name__ == "__main__":
    sample_inputs = {
        "niche": "student productivity",
        "audience": "college students who procrastinate",
        "platform": "Instagram",
        "brand_voice": "casual, direct, slightly funny",
        "chosen_idea": "why your study timetable fails by day 3",
    }

    agent = HookAgent()
    result = agent.run(sample_inputs)
    print(result)
