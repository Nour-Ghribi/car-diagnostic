from __future__ import annotations

from agent.settings import AgentSettings, get_settings
from nlp.schemas import ResolverDecision
from validation.safety import ensure_allowed_tools, ensure_max_plan_length, ensure_no_duplicate_steps, ensure_valid_arguments
from validation.schema_validator import validate_model


def validate_resolver_decision(payload: ResolverDecision | dict, settings: AgentSettings | None = None) -> ResolverDecision:
    """Validate a resolver decision and its execution plan."""
    active_settings = settings or get_settings()
    decision = payload if isinstance(payload, ResolverDecision) else validate_model(ResolverDecision, payload)
    if (
        decision.selected_goal != "UNKNOWN"
        and decision.selected_public_intent != "UNKNOWN"
        and not decision.needs_user_clarification
        and not decision.execution_plan
    ):
        raise ValueError("Accepted resolver decisions must contain a non-empty execution plan.")
    ensure_allowed_tools(decision.execution_plan)
    ensure_valid_arguments(decision.execution_plan)
    ensure_no_duplicate_steps(decision.execution_plan)
    ensure_max_plan_length(decision.execution_plan, max_steps=active_settings.max_plan_steps)
    return decision
