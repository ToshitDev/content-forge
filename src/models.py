"""Typed representations of each agent's JSON output.

Every dataclass has a from_dict() classmethod that builds an instance
from a parsed JSON dict (the output of BaseAgent.run_parsed()) and
raises a clear error if a required key is missing.
"""

from dataclasses import dataclass


def _require(d: dict, key: str, context: str):
    """Fetch d[key], raising a clear ValueError if it's missing."""
    if key not in d:
        raise ValueError(f"{context}: missing required key '{key}' in {d!r}")
    return d[key]


@dataclass
class Idea:
    """One content idea tied to a specific audience pain point."""

    idea: str
    pain_point: str
    unique_angle: str

    @classmethod
    def from_dict(cls, d: dict) -> "Idea":
        return cls(
            idea=_require(d, "idea", "Idea"),
            pain_point=_require(d, "pain_point", "Idea"),
            unique_angle=_require(d, "unique_angle", "Idea"),
        )


@dataclass
class TopPick:
    """A recommended idea with the reason it was chosen."""

    idea: str
    reason: str

    @classmethod
    def from_dict(cls, d: dict) -> "TopPick":
        return cls(
            idea=_require(d, "idea", "TopPick"),
            reason=_require(d, "reason", "TopPick"),
        )


@dataclass
class ResearchOutput:
    """Parsed output of the Research Agent."""

    pain_points: list[str]
    ideas: list[Idea]
    top_picks: list[TopPick]
    missing_info: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchOutput":
        return cls(
            pain_points=_require(d, "pain_points", "ResearchOutput"),
            ideas=[Idea.from_dict(i) for i in _require(d, "ideas", "ResearchOutput")],
            top_picks=[
                TopPick.from_dict(p) for p in _require(d, "top_picks", "ResearchOutput")
            ],
            missing_info=_require(d, "missing_info", "ResearchOutput"),
        )


@dataclass
class HookOption:
    """One candidate hook line and its type label."""

    text: str
    type: str

    @classmethod
    def from_dict(cls, d: dict) -> "HookOption":
        return cls(
            text=_require(d, "text", "HookOption"),
            type=_require(d, "type", "HookOption"),
        )


@dataclass
class HookChoice:
    """A selected hook (winner or runner-up) with reasoning."""

    text: str
    reasoning: str

    @classmethod
    def from_dict(cls, d: dict) -> "HookChoice":
        return cls(
            text=_require(d, "text", "HookChoice"),
            reasoning=_require(d, "reasoning", "HookChoice"),
        )


@dataclass
class HookOutput:
    """Parsed output of the Hook Agent."""

    hooks: list[HookOption]
    winner: HookChoice
    runner_up: HookChoice

    @classmethod
    def from_dict(cls, d: dict) -> "HookOutput":
        return cls(
            hooks=[HookOption.from_dict(h) for h in _require(d, "hooks", "HookOutput")],
            winner=HookChoice.from_dict(_require(d, "winner", "HookOutput")),
            runner_up=HookChoice.from_dict(_require(d, "runner_up", "HookOutput")),
        )


@dataclass
class ScriptOutput:
    """Parsed output of the Script Agent."""

    script: str
    word_count: int
    viewer_takeaway: str

    @classmethod
    def from_dict(cls, d: dict) -> "ScriptOutput":
        return cls(
            script=_require(d, "script", "ScriptOutput"),
            word_count=_require(d, "word_count", "ScriptOutput"),
            viewer_takeaway=_require(d, "viewer_takeaway", "ScriptOutput"),
        )


@dataclass
class FramePlanItem:
    """One slide/frame in the visual plan."""

    position: str
    on_screen_text: str
    visual_direction: str

    @classmethod
    def from_dict(cls, d: dict) -> "FramePlanItem":
        return cls(
            position=_require(d, "position", "FramePlanItem"),
            on_screen_text=_require(d, "on_screen_text", "FramePlanItem"),
            visual_direction=_require(d, "visual_direction", "FramePlanItem"),
        )


@dataclass
class VisualOutput:
    """Parsed output of the Visual Agent."""

    cover_options: list[str]
    frame_plan: list[FramePlanItem]
    assets_needed: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "VisualOutput":
        return cls(
            cover_options=_require(d, "cover_options", "VisualOutput"),
            frame_plan=[
                FramePlanItem.from_dict(f) for f in _require(d, "frame_plan", "VisualOutput")
            ],
            assets_needed=_require(d, "assets_needed", "VisualOutput"),
        )


@dataclass
class Scores:
    """The Growth Agent's six 1-10 quality scores."""

    clarity: int
    retention: int
    save_potential: int
    shareability: int
    audience_fit: int
    cta_strength: int

    @classmethod
    def from_dict(cls, d: dict) -> "Scores":
        return cls(
            clarity=_require(d, "clarity", "Scores"),
            retention=_require(d, "retention", "Scores"),
            save_potential=_require(d, "save_potential", "Scores"),
            shareability=_require(d, "shareability", "Scores"),
            audience_fit=_require(d, "audience_fit", "Scores"),
            cta_strength=_require(d, "cta_strength", "Scores"),
        )


@dataclass
class Weakness:
    """One flagged weakness with a concrete fix."""

    issue: str
    fix: str

    @classmethod
    def from_dict(cls, d: dict) -> "Weakness":
        return cls(
            issue=_require(d, "issue", "Weakness"),
            fix=_require(d, "fix", "Weakness"),
        )


@dataclass
class GrowthReview:
    """Parsed output of the Growth Agent."""

    scores: Scores
    justifications: dict
    weaknesses: list[Weakness]
    captions: list[str]
    final_call: str
    final_call_reason: str

    @classmethod
    def from_dict(cls, d: dict) -> "GrowthReview":
        return cls(
            scores=Scores.from_dict(_require(d, "scores", "GrowthReview")),
            justifications=_require(d, "justifications", "GrowthReview"),
            weaknesses=[
                Weakness.from_dict(w) for w in _require(d, "weaknesses", "GrowthReview")
            ],
            captions=_require(d, "captions", "GrowthReview"),
            final_call=_require(d, "final_call", "GrowthReview"),
            final_call_reason=_require(d, "final_call_reason", "GrowthReview"),
        )
