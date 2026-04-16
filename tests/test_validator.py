from __future__ import annotations

import pytest

from nlp.schemas import PlanStep, ResolverDecision
from validation.plan_validator import validate_resolver_decision


def test_validator_accepts_valid_resolver_decision() -> None:
    decision = ResolverDecision(
        selected_public_intent="READ_DTC",
        selected_goal="READ_DTC",
        scope="specific",
        confidence=0.8,
        reasoning_summary="Read DTCs first.",
        execution_plan=[PlanStep(tool="get_dtcs", arguments={"include_pending": True})],
    )
    validated = validate_resolver_decision(decision)
    assert validated.selected_goal == "READ_DTC"


def test_validator_rejects_duplicate_plan_steps() -> None:
    decision = ResolverDecision(
        selected_public_intent="READ_DTC",
        selected_goal="READ_DTC",
        scope="specific",
        confidence=0.8,
        reasoning_summary="Duplicate steps should be rejected.",
        execution_plan=[
            PlanStep(tool="get_dtcs", arguments={"include_pending": True}),
            PlanStep(tool="get_dtcs", arguments={"include_pending": True}),
        ],
    )
    with pytest.raises(ValueError):
        validate_resolver_decision(decision)


def test_validator_rejects_accepted_decision_with_empty_plan() -> None:
    decision = ResolverDecision(
        selected_public_intent="CHECK_ENGINE_HEALTH",
        selected_goal="VEHICLE_HEALTH_CHECK",
        scope="broad",
        confidence=0.8,
        needs_user_clarification=False,
        clarification_question=None,
        reasoning_summary="Accepted broad health goal.",
        execution_plan=[],
    )
    with pytest.raises(ValueError):
        validate_resolver_decision(decision)
