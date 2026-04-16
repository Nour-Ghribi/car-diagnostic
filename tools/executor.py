from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.schemas import MissingData, ToolName
from nlp.schemas import PlanStep
from tools.registry import get_tool_spec


@dataclass(frozen=True)
class ToolExecutionResult:
    step: PlanStep
    raw_response: dict[str, Any]


@dataclass
class ExecutionAccumulator:
    actions_taken: list[ToolName] = field(default_factory=list)
    observations: list[ToolExecutionResult] = field(default_factory=list)
    missing_data: list[MissingData] = field(default_factory=list)
    aggregated: dict[str, Any] = field(
        default_factory=lambda: {
            "signals": {},
            "dtc": None,
            "mode06": None,
            "vehicle_context": None,
            "capabilities": None,
            "metrics": None,
        }
    )


def execute_plan(*, vehicle_id: str, request_id: str, plan: list[PlanStep]) -> ExecutionAccumulator:
    """Execute a validated plan step by step and accumulate normalized observations."""
    accumulator = ExecutionAccumulator()
    for step in plan:
        spec = get_tool_spec(step.tool)
        arguments = _resolve_runtime_arguments(step, accumulator)
        if step.tool == "score_confidence":
            response = spec.execute(request_id=request_id, **arguments)
        else:
            response = spec.execute(vehicle_id, request_id=request_id, **arguments)
        accumulator.actions_taken.append(step.tool)
        accumulator.observations.append(ToolExecutionResult(step=step, raw_response=response))
        _merge_tool_response(accumulator, response)
    return accumulator


def _resolve_runtime_arguments(step: PlanStep, accumulator: ExecutionAccumulator) -> dict[str, Any]:
    if step.tool != "score_confidence":
        return dict(step.arguments)

    present_count = len(accumulator.aggregated["signals"]) + _count_dtc_items(accumulator.aggregated.get("dtc"))
    coherent = _is_signal_set_coherent(accumulator.aggregated["signals"])
    missing_data = [{"key": item.key, "reason": item.reason} for item in accumulator.missing_data]
    return {"present_count": present_count, "missing_data": missing_data, "coherent": coherent}


def _merge_tool_response(accumulator: ExecutionAccumulator, response: dict[str, Any]) -> None:
    if response["status"] == "error":
        accumulator.missing_data.append(
            MissingData(key=response["tool_name"], reason="not_collected", impact="diagnosis_limited")
        )
        return

    data = response.get("data") or {}
    if data.get("signals"):
        accumulator.aggregated["signals"].update(data["signals"])
    if data.get("dtc") is not None:
        accumulator.aggregated["dtc"] = data["dtc"]
    if data.get("mode06") is not None:
        accumulator.aggregated["mode06"] = data["mode06"]
    if data.get("vehicle_context") is not None:
        accumulator.aggregated["vehicle_context"] = data["vehicle_context"]
    if data.get("capabilities") is not None:
        accumulator.aggregated["capabilities"] = data["capabilities"]
    if data.get("metrics") is not None:
        accumulator.aggregated["metrics"] = data["metrics"]

    for item in response.get("missing_data", []):
        reason = item["reason"]
        impact = "confidence_reduced" if reason in {"unsupported", "no_data", "timeout", "stale"} else "diagnosis_limited"
        normalized = MissingData(key=item["key"], reason=reason, impact=impact)
        if not any(existing.key == normalized.key and existing.reason == normalized.reason for existing in accumulator.missing_data):
            accumulator.missing_data.append(normalized)


def _count_dtc_items(dtc: dict[str, Any] | None) -> int:
    if not dtc:
        return 0
    return len(dtc.get("stored", [])) + len(dtc.get("pending", []))


def _is_signal_set_coherent(signals: dict[str, Any]) -> bool:
    rpm = _signal_numeric_value(signals, "rpm")
    coolant = _signal_numeric_value(signals, "coolant_temp")
    throttle = _signal_numeric_value(signals, "throttle_pos")
    engine_load = _signal_numeric_value(signals, "engine_load")
    if rpm is not None and rpm < 0:
        return False
    if coolant is not None and coolant > 140:
        return False
    if throttle is not None and not 0 <= throttle <= 100:
        return False
    if engine_load is not None and not 0 <= engine_load <= 100:
        return False
    if rpm is not None and throttle is not None and rpm < 500 and throttle > 60:
        return False
    return True


def _signal_numeric_value(signals: dict[str, Any], key: str) -> float | None:
    payload = signals.get(key)
    if not payload:
        return None
    value = payload.get("value")
    if isinstance(value, (int, float)):
        return float(value)
    return None
