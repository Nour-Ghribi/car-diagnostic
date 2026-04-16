from __future__ import annotations

from dataclasses import dataclass, field

from backend.schemas import GoalName, IntentName, MissingData
from nlp.schemas import PlanStep, RetrievalCandidate


@dataclass
class AgentState:
    """Execution state for one hybrid-agent request."""

    original_prompt: str
    rewritten_prompt: str = ""
    retrieval_candidates: tuple[RetrievalCandidate, ...] = ()
    selected_public_intent: IntentName = "UNKNOWN"
    selected_goal: GoalName = "UNKNOWN"
    validated_execution_plan: list[PlanStep] = field(default_factory=list)
    executed_steps: list[PlanStep] = field(default_factory=list)
    final_status: str = "created"
    clarification_question: str | None = None
    confidence: float = 0.0
    missing_data: list[MissingData] = field(default_factory=list)
