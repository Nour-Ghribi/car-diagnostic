from __future__ import annotations

from datetime import datetime, timezone

from agent.confidence import compute_final_diagnostic_confidence
from agent.response_composer import compose_agent_response
from agent.state import AgentState
from agent.tools import ALLOWED_SIGNALS
from agent.settings import get_settings
from backend.schemas import AgentRequest, AgentResponse, Intent, IntentParameters, MissingData, SignalName
from nlp.llm_resolver import resolve_and_plan
from nlp.llm_rewriter import rewrite_prompt
from nlp.retriever import retrieve_candidates
from tools.executor import ExecutionAccumulator, execute_plan
from validation.plan_validator import validate_resolver_decision


def parse_intent(prompt: str) -> Intent:
    """
    Backward-compatible deterministic parser kept as a last-resort safety net.

    It is no longer the main planner; the hybrid pipeline now uses a bounded
    resolver to generate the plan.
    """
    text = prompt.strip().lower()

    if any(token in text for token in ("cylinder", "cylindre")):
        cylinder_index = _extract_first_int(text) or 1
        return Intent(
            name="CHECK_CYLINDER",
            confidence=0.9,
            parameters=IntentParameters(cylinder_index=cylinder_index, detail="medium"),
        )

    if any(token in text for token in ("dtc", "fault code", "code défaut", "codes défauts", "codes moteur")):
        return Intent(
            name="READ_DTC",
            confidence=0.92,
            parameters=IntentParameters(include_pending=True, include_permanent=False),
        )

    if any(token in text for token in ("check engine", "voyant moteur", "mil")):
        return Intent(
            name="EXPLAIN_WARNING_LIGHT",
            confidence=0.9,
            parameters=IntentParameters(warning_type="check_engine"),
        )

    if any(token in text for token in ("vin", "vehicle context", "contexte véhicule", "contexte vehicule", "supported pids", "capacités obd", "capacites obd")):
        return Intent(
            name="GET_VEHICLE_CONTEXT",
            confidence=0.88,
            parameters=IntentParameters(include_calibration=True, refresh_capabilities=False),
        )

    if any(token in text for token in ("rpm", "engine load", "coolant", "temperature", "throttle", "o2", "speed", "voltage")):
        signal = _detect_signal(text)
        return Intent(
            name="CHECK_SIGNAL_STATUS",
            confidence=0.84,
            parameters=IntentParameters(signal=signal, max_age_ms=2000),
        )

    if any(
        token in text
        for token in (
            "engine health",
            "diagnostic global",
            "bilan moteur",
            "moteur sain",
            "moteur en bon état",
            "moteur en bon etat",
            "état du moteur",
            "etat du moteur",
        )
    ):
        return Intent(
            name="CHECK_ENGINE_HEALTH",
            confidence=0.86,
            parameters=IntentParameters(detail="medium", goal="VEHICLE_HEALTH_CHECK", scope="broad"),
        )

    return Intent(name="UNKNOWN", confidence=0.4, parameters=IntentParameters())


def diagnose(request: AgentRequest) -> AgentResponse:
    """
    Run the hybrid agent pipeline:
    rewriter -> retrieval -> resolver/planner -> validation -> execution -> response composition.
    """
    settings = get_settings()
    state = AgentState(original_prompt=request.user_prompt)

    try:
        rewritten = rewrite_prompt(request.user_prompt)
        state.rewritten_prompt = rewritten.rewritten_prompt
    except Exception:
        rewritten = None
        state.rewritten_prompt = request.user_prompt

    try:
        candidates = retrieve_candidates(state.rewritten_prompt or request.user_prompt, settings)
    except Exception:
        candidates = ()
    state.retrieval_candidates = candidates

    try:
        decision = resolve_and_plan(
            original_prompt=request.user_prompt,
            rewritten=rewritten or rewrite_prompt(request.user_prompt),
            candidates=candidates,
            settings=settings,
        )
        validated = validate_resolver_decision(decision, settings)
    except Exception:
        return _safe_clarification_response(request)

    intent = _decision_to_intent(validated)
    state.selected_public_intent = intent.name
    state.selected_goal = intent.parameters.goal or "UNKNOWN"
    state.validated_execution_plan = list(validated.execution_plan)

    if validated.needs_user_clarification or not validated.execution_plan:
        state.final_status = "clarification"
        execution = ExecutionAccumulator()
        confidence = validated.confidence
        return compose_agent_response(
            request_id=request.request_id,
            vehicle_id=request.vehicle_id,
            intent=intent,
            execution=execution,
            confidence=confidence,
            clarification_question=validated.clarification_question or intent.parameters.clarification_question,
        )

    try:
        execution = execute_plan(
            vehicle_id=request.vehicle_id,
            request_id=request.request_id,
            plan=list(validated.execution_plan),
        )
        state.executed_steps = list(validated.execution_plan)
        state.missing_data = execution.missing_data
        state.final_status = "completed"
        data_confidence = _extract_data_confidence(execution)
        confidence = compute_final_diagnostic_confidence(validated.confidence, data_confidence)
        state.confidence = confidence
        return compose_agent_response(
            request_id=request.request_id,
            vehicle_id=request.vehicle_id,
            intent=intent,
            execution=execution,
            confidence=confidence,
        )
    except Exception:
        return _safe_failure_response(request, intent)


def _decision_to_intent(decision) -> Intent:
    resolution_policy = (
        "CLARIFICATION_NEEDED"
        if decision.needs_user_clarification
        else "UNKNOWN"
        if decision.selected_public_intent == "UNKNOWN" or decision.selected_goal == "UNKNOWN"
        else "ACCEPT_BROAD_GOAL"
        if decision.scope == "broad"
        else "ACCEPT"
    )
    parameters = IntentParameters(
        goal=decision.selected_goal,
        scope="broad" if decision.scope == "broad" else "specific",
        resolution_policy=resolution_policy,  # type: ignore[arg-type]
        clarification_question=decision.clarification_question,
        signal=_extract_signal_from_plan(decision.execution_plan),
        include_pending=_extract_include_pending(decision.execution_plan),
        include_calibration=True,
    )
    return Intent(
        name=decision.selected_public_intent,
        confidence=round(decision.confidence, 2),
        parameters=parameters,
    )


def _extract_signal_from_plan(plan) -> SignalName | None:
    for step in plan:
        signals = step.arguments.get("signals")
        if isinstance(signals, list) and len(signals) == 1 and signals[0] in ALLOWED_SIGNALS:
            return signals[0]
    return None


def _extract_include_pending(plan) -> bool:
    for step in plan:
        if step.tool == "get_dtcs":
            return bool(step.arguments.get("include_pending", True))
    return True


def _extract_data_confidence(execution: ExecutionAccumulator) -> float:
    metrics = execution.aggregated.get("metrics")
    if metrics and "confidence" in metrics:
        return float(metrics["confidence"])
    return 0.0


def _safe_clarification_response(request: AgentRequest) -> AgentResponse:
    intent = Intent(
        name="UNKNOWN",
        confidence=0.2,
        parameters=IntentParameters(
            goal="UNKNOWN",
            scope="specific",
            resolution_policy="CLARIFICATION_NEEDED",
            clarification_question="Pouvez-vous préciser le problème automobile que vous voulez diagnostiquer ?",
        ),
    )
    return compose_agent_response(
        request_id=request.request_id,
        vehicle_id=request.vehicle_id,
        intent=intent,
        execution=ExecutionAccumulator(),
        confidence=intent.confidence,
        clarification_question=intent.parameters.clarification_question,
    )


def _safe_failure_response(request: AgentRequest, intent: Intent) -> AgentResponse:
    return AgentResponse(
        request_id=request.request_id,
        ts=datetime.now(timezone.utc),
        vehicle_id=request.vehicle_id,
        intent=intent,
        diagnosis="Diagnostic execution degraded due to an internal execution error.",
        confidence=0.0,
        evidence=[],
        signals_used=[],
        actions_taken=[],
        missing_data=[MissingData(key="orchestrator", reason="not_collected", impact="diagnosis_limited")],
        recommendations=["Retry the request after verifying resolver output and tool availability."],
    )


def _detect_signal(text: str) -> SignalName:
    mapping: dict[str, SignalName] = {
        "rpm": "rpm",
        "engine load": "engine_load",
        "charge moteur": "engine_load",
        "coolant": "coolant_temp",
        "temperature": "coolant_temp",
        "température": "coolant_temp",
        "throttle": "throttle_pos",
        "stft": "stft_b1",
        "ltft": "ltft_b1",
        "o2": "o2_b1s1",
        "speed": "vehicle_speed",
        "vitesse": "vehicle_speed",
        "voltage": "module_voltage",
        "tension": "module_voltage",
    }
    for token, signal in mapping.items():
        if token in text:
            return signal
    return "rpm"


def _extract_first_int(text: str) -> int | None:
    digits = ""
    for char in text:
        if char.isdigit():
            digits += char
        elif digits:
            break
    return int(digits) if digits else None
