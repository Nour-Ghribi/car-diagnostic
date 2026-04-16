from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from backend.schemas import GoalName, IntentName, SignalName, StrictBaseModel, ToolName


LanguageHint = Literal["fr", "en", "mixed", "unknown"]
AmbiguityLevel = Literal["low", "medium", "high"]
ResolverScope = Literal["specific", "broad", "ambiguous"]


class RewriterOutput(StrictBaseModel):
    rewritten_prompt: str = Field(min_length=1)
    language: LanguageHint
    ambiguity_level: AmbiguityLevel
    preserved_meaning: bool = True
    needs_user_clarification: bool = False
    clarification_question: str | None = None


class RetrievalCandidate(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    public_intent: IntentName
    goal: GoalName
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanStep(StrictBaseModel):
    tool: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class ResolverDecision(StrictBaseModel):
    selected_public_intent: IntentName
    selected_goal: GoalName
    scope: ResolverScope
    confidence: float = Field(ge=0.0, le=1.0)
    needs_user_clarification: bool = False
    clarification_question: str | None = None
    reasoning_summary: str = Field(min_length=1)
    execution_plan: list[PlanStep] = Field(default_factory=list)
