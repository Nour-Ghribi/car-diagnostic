from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.schemas import (
    AgentRequest,
    AgentResponse,
    Evidence,
    Intent,
    IntentParameters,
    MissingData,
    SignalName,
    ToolName,
)
from agent.confidence import compute_final_diagnostic_confidence
from agent.intent_parser_llm import parse_intent_hybrid
from agent.intent_parser_v3 import parse_intent_v3
from agent.tools import (
    ALLOWED_SIGNALS,
    get_dtcs,
    get_latest_signals,
    get_vehicle_context,
    request_fresh_signals,
    request_mode06,
    score_confidence,
)


def parse_intent(prompt: str) -> Intent:
    """
    Parse a user prompt into a normalized Intent using simple rule-based logic.

    This is a temporary mock parser. It is intentionally deterministic and does
    not use any LLM.
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

    if any(
        token in text
        for token in ("vin", "vehicle context", "contexte véhicule", "contexte vehicule", "supported pids", "capacités obd", "capacites obd")
    ):
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


def get_required_signals(intent: Intent) -> list[SignalName]:
    """Return the required signal list for an intent."""
    goal = intent.parameters.goal
    if goal == "VEHICLE_HEALTH_CHECK":
        return ["rpm", "engine_load", "coolant_temp", "vehicle_speed", "module_voltage"]
    if goal == "BATTERY_CHECK":
        return ["module_voltage"]
    if goal == "ENGINE_TEMPERATURE_CHECK":
        return ["coolant_temp"]
    if goal == "PERFORMANCE_ISSUE_CHECK":
        return ["rpm", "engine_load", "stft_b1", "ltft_b1", "o2_b1s1"]
    if goal == "STARTING_PROBLEM_CHECK":
        return ["module_voltage", "rpm"]
    if goal == "FUEL_CONSUMPTION_CHECK":
        return ["stft_b1", "ltft_b1", "o2_b1s1", "vehicle_speed"]
    if intent.name == "CHECK_CYLINDER":
        return ["rpm", "engine_load", "stft_b1", "ltft_b1", "o2_b1s1"]
    if intent.name == "CHECK_ENGINE_HEALTH":
        return ["rpm", "engine_load", "coolant_temp"]
    if intent.name == "CHECK_SIGNAL_STATUS" and intent.parameters.signal is not None:
        return [intent.parameters.signal]
    if intent.name == "EXPLAIN_WARNING_LIGHT":
        return ["rpm", "engine_load", "coolant_temp"]
    return []


def build_diagnosis(intent: Intent, aggregated: dict[str, Any]) -> str:
    """Build a readable diagnosis string from the collected tool data."""
    signals = aggregated.get("signals", {})
    dtc = aggregated.get("dtc")
    mode06 = aggregated.get("mode06")
    vehicle_context = aggregated.get("vehicle_context")
    resolution_policy = intent.parameters.resolution_policy

    if resolution_policy == "CLARIFICATION_NEEDED" and intent.parameters.clarification_question:
        return intent.parameters.clarification_question

    if intent.parameters.goal == "VEHICLE_HEALTH_CHECK":
        return _diagnose_vehicle_health(signals, dtc, vehicle_context)

    if intent.name == "READ_DTC":
        return _diagnose_dtc_readout(dtc)

    if intent.name == "CHECK_CYLINDER":
        return _diagnose_cylinder(intent, signals, dtc, mode06)

    if intent.name == "CHECK_ENGINE_HEALTH":
        return _diagnose_engine_health(signals, dtc)

    if intent.name == "CHECK_SIGNAL_STATUS":
        return _diagnose_signal_status(intent, signals)

    if intent.name == "EXPLAIN_WARNING_LIGHT":
        return _diagnose_warning_light(signals, dtc)

    if intent.name == "GET_VEHICLE_CONTEXT":
        if vehicle_context:
            vin = vehicle_context.get("vin") or "unknown"
            ecu_name = vehicle_context.get("ecu_name") or "unknown ECU"
            calibration_id = vehicle_context.get("calibration_id") or "unknown calibration"
            return f"Vehicle context retrieved successfully for VIN {vin}, ECU {ecu_name}, calibration {calibration_id}."
        return "Vehicle context could not be fully retrieved."

    return "Request is outside the supported V2 diagnostic scope."


def diagnose(request: AgentRequest) -> AgentResponse:
    """
    Execute the full mock orchestration flow for an AgentRequest.

    The orchestrator:
    - parses the prompt into an intent,
    - determines required signals,
    - reads cache first,
    - falls back to on-demand reads if needed,
    - fetches DTC/context data when relevant,
    - computes confidence,
    - returns a structured AgentResponse.
    """
    try:
        intent = parse_intent_v3(request.user_prompt)
    except Exception:
        intent = parse_intent_hybrid(request.user_prompt)
    if intent.parameters.resolution_policy == "CLARIFICATION_NEEDED":
        diagnosis = intent.parameters.clarification_question or "Please clarify the vehicle issue you want checked."
        return AgentResponse(
            request_id=request.request_id,
            ts=datetime.now(timezone.utc),
            vehicle_id=request.vehicle_id,
            intent=intent,
            diagnosis=diagnosis,
            confidence=round(intent.confidence, 2),
            evidence=[],
            signals_used=[],
            actions_taken=[],
            missing_data=[],
            recommendations=["Clarify whether you want a broad health check or a specific problem diagnosis."],
        )
    required_signals = get_required_signals(intent)

    actions_taken: list[ToolName] = []
    evidence: list[Evidence] = []
    missing_data: list[MissingData] = []
    aggregated: dict[str, Any] = {
        "signals": {},
        "dtc": None,
        "mode06": None,
        "vehicle_context": None,
        "capabilities": None,
        "metrics": None,
    }

    try:
        if intent.parameters.goal == "VEHICLE_HEALTH_CHECK":
            context_response = get_vehicle_context(request.vehicle_id, request_id=request.request_id)
            actions_taken.append("get_vehicle_context")
            _merge_tool_response(aggregated, context_response, missing_data)

        if intent.name in {"READ_DTC", "CHECK_CYLINDER", "CHECK_ENGINE_HEALTH", "EXPLAIN_WARNING_LIGHT"}:
            dtc_response = get_dtcs(
                request.vehicle_id,
                include_pending=intent.parameters.include_pending,
                request_id=request.request_id,
            )
            actions_taken.append("get_dtcs")
            _merge_tool_response(aggregated, dtc_response, missing_data)

        if intent.name == "GET_VEHICLE_CONTEXT":
            context_response = get_vehicle_context(request.vehicle_id, request_id=request.request_id)
            actions_taken.append("get_vehicle_context")
            _merge_tool_response(aggregated, context_response, missing_data)

        if required_signals:
            cache_response = get_latest_signals(
                request.vehicle_id,
                required_signals,
                request_id=request.request_id,
            )
            actions_taken.append("get_latest_signals")
            _merge_tool_response(aggregated, cache_response, missing_data)

            missing_signal_names = _extract_missing_signals(missing_data)
            allow_on_demand = request.constraints.allow_on_demand if request.constraints else True
            if missing_signal_names and allow_on_demand:
                fresh_response = request_fresh_signals(
                    request.vehicle_id,
                    missing_signal_names,
                    request_id=request.request_id,
                )
                actions_taken.append("request_fresh_signals")
                _merge_tool_response(aggregated, fresh_response, missing_data, replace_signal_missing=True)

        if intent.name == "CHECK_CYLINDER":
            mode06_response = request_mode06(request.vehicle_id, request_id=request.request_id)
            actions_taken.append("request_mode06")
            _merge_tool_response(aggregated, mode06_response, missing_data)

        coherent = _is_signal_set_coherent(aggregated["signals"])
        confidence_response = score_confidence(
            present_count=len(aggregated["signals"]) + _count_dtc_items(aggregated.get("dtc")),
            missing_data=[{"key": item.key, "reason": item.reason} for item in missing_data],
            coherent=coherent,
            request_id=request.request_id,
        )
        actions_taken.append("score_confidence")
        _merge_tool_response(aggregated, confidence_response, missing_data)

        evidence = _build_evidence(aggregated)
        diagnosis = build_diagnosis(intent, aggregated)
        recommendations = _build_recommendations(intent, missing_data)
        data_confidence = _extract_confidence(aggregated)
        confidence = round(compute_final_diagnostic_confidence(intent.confidence, data_confidence), 2)

        return AgentResponse(
            request_id=request.request_id,
            ts=datetime.now(timezone.utc),
            vehicle_id=request.vehicle_id,
            intent=intent,
            diagnosis=diagnosis,
            confidence=confidence,
            evidence=evidence,
            signals_used=list(aggregated["signals"].keys()),
            actions_taken=actions_taken,
            missing_data=missing_data,
            recommendations=recommendations,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        fallback_intent = Intent(name="UNKNOWN", confidence=0.0, parameters=IntentParameters())
        return AgentResponse(
            request_id=request.request_id,
            ts=datetime.now(timezone.utc),
            vehicle_id=request.vehicle_id,
            intent=fallback_intent,
            diagnosis=f"Diagnostic fallback due to internal tool orchestration error: {exc}",
            confidence=0.0,
            evidence=[],
            signals_used=[],
            actions_taken=actions_taken,
            missing_data=[
                MissingData(
                    key="orchestrator",
                    reason="not_collected",
                    impact="diagnosis_limited",
                )
            ],
            recommendations=["Retry the request after verifying tool outputs."],
        )


def _merge_tool_response(
    aggregated: dict[str, Any],
    response: dict[str, Any],
    missing_data: list[MissingData],
    *,
    replace_signal_missing: bool = False,
) -> None:
    """Merge a tool response into the aggregated state without crashing."""
    if response["status"] == "error":
        missing_data.append(
            MissingData(
                key=response["tool_name"],
                reason="not_collected",
                impact="diagnosis_limited",
            )
        )
        return

    data = response.get("data") or {}
    if data.get("signals"):
        aggregated["signals"].update(data["signals"])
    if data.get("dtc") is not None:
        aggregated["dtc"] = data["dtc"]
    if data.get("mode06") is not None:
        aggregated["mode06"] = data["mode06"]
    if data.get("vehicle_context") is not None:
        aggregated["vehicle_context"] = data["vehicle_context"]
    if data.get("capabilities") is not None:
        aggregated["capabilities"] = data["capabilities"]
    if data.get("metrics") is not None:
        aggregated["metrics"] = data["metrics"]

    if replace_signal_missing and data.get("signals"):
        recovered_keys = set(data["signals"].keys())
        missing_data[:] = [
            item
            for item in missing_data
            if not (item.key in recovered_keys and item.reason in {"not_collected", "stale", "timeout"})
        ]

    for item in response.get("missing_data", []):
        reason = item["reason"]
        impact = "confidence_reduced" if reason in {"unsupported", "no_data", "timeout", "stale"} else "diagnosis_limited"
        normalized = MissingData(key=item["key"], reason=reason, impact=impact)
        if not any(existing.key == normalized.key and existing.reason == normalized.reason for existing in missing_data):
            missing_data.append(normalized)


def _build_evidence(aggregated: dict[str, Any]) -> list[Evidence]:
    """Convert aggregated tool data into structured evidence."""
    items: list[Evidence] = []
    for key, payload in aggregated.get("signals", {}).items():
        if "value" in payload:
            items.append(
                Evidence(
                    key=key,
                    label=key.replace("_", " ").title(),
                    value=payload["value"],
                    unit=payload.get("unit"),
                    observed_ts=_to_datetime(payload["observed_ts"]),
                    source=payload["source"],
                )
            )

    dtc = aggregated.get("dtc")
    if dtc is not None:
        items.append(
            Evidence(
                key="dtc.stored",
                label="Stored DTC",
                value=dtc.get("stored", []),
                unit=None,
                observed_ts=datetime.now(timezone.utc),
                source="db",
            )
        )

    if aggregated.get("mode06") is not None:
        items.append(
            Evidence(
                key="mode06",
                label="Mode 06",
                value=aggregated["mode06"],
                unit=None,
                observed_ts=_to_datetime(aggregated["mode06"]["observed_ts"]),
                source=aggregated["mode06"]["source"],
            )
        )

    return items


def _build_recommendations(intent: Intent, missing_data: list[MissingData]) -> list[str]:
    """Create lightweight recommendations from intent and missing data."""
    recommendations: list[str] = []
    if intent.parameters.resolution_policy == "CLARIFICATION_NEEDED":
        recommendations.append("Clarify the main concern so the agent can choose the right diagnostic plan.")
        return recommendations

    if intent.parameters.goal == "VEHICLE_HEALTH_CHECK":
        recommendations.append("Review the broad vehicle-health snapshot with live data, DTC status, and ECU context together.")
    if intent.name == "CHECK_CYLINDER":
        recommendations.append("Inspect ignition, injector behavior, and air-fuel balance for the requested cylinder.")
    elif intent.name == "READ_DTC":
        recommendations.append("Review stored and pending DTCs before clearing anything or escalating the diagnosis.")
    elif intent.name == "CHECK_ENGINE_HEALTH":
        recommendations.append("Re-run the engine health snapshot with fresh live data and compare against DTC status.")
    elif intent.name == "GET_VEHICLE_CONTEXT":
        recommendations.append("Use VIN and ECU metadata to refine downstream diagnostics.")
    elif intent.name == "CHECK_SIGNAL_STATUS":
        recommendations.append("Compare the requested signal with neighboring operating conditions before drawing conclusions.")
    elif intent.name == "EXPLAIN_WARNING_LIGHT":
        recommendations.append("Correlate the warning light with DTCs and live signals before deciding on mechanical action.")

    if any(item.reason in {"no_data", "unsupported", "timeout"} for item in missing_data):
        recommendations.append("Consider a follow-up request if additional data becomes available.")
    if not recommendations:
        recommendations.append("Use a supported V2 request such as READ_DTC or CHECK_SIGNAL_STATUS.")
    return recommendations


def _extract_missing_signals(missing_data: list[MissingData]) -> list[SignalName]:
    """Return signal-like missing keys that are eligible for on-demand refresh."""
    eligible: list[SignalName] = []
    for item in missing_data:
        if item.key in ALLOWED_SIGNALS and item.reason in {"not_collected", "stale", "timeout"}:
            eligible.append(item.key)
    return eligible


def _extract_confidence(aggregated: dict[str, Any]) -> float:
    """Read the computed confidence from aggregated metrics."""
    metrics = aggregated.get("metrics")
    if metrics and "confidence" in metrics:
        return float(metrics["confidence"])
    return 0.0


def _count_dtc_items(dtc: dict[str, Any] | None) -> int:
    """Count DTC entries for confidence scoring."""
    if not dtc:
        return 0
    return len(dtc.get("stored", [])) + len(dtc.get("pending", []))


def _is_signal_set_coherent(signals: dict[str, Any]) -> bool:
    """Apply a very small coherence heuristic suitable for mock orchestration."""
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
    """Extract a numeric signal value when possible."""
    payload = signals.get(key)
    if not payload:
        return None
    value = payload.get("value")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _detect_signal(text: str) -> SignalName:
    """Map a prompt fragment to a supported signal."""
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
    """Extract the first integer found in a string."""
    digits = ""
    for char in text:
        if char.isdigit():
            digits += char
        elif digits:
            break
    return int(digits) if digits else None


def _to_datetime(value: str) -> datetime:
    """Convert an ISO-like string to datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _diagnose_dtc_readout(dtc: dict[str, Any] | None) -> str:
    """Summarize DTC state in a more useful diagnostic sentence."""
    if not dtc:
        return "No DTC information is currently available."
    stored = dtc.get("stored", [])
    pending = dtc.get("pending", [])
    permanent = dtc.get("permanent", [])
    parts: list[str] = []
    if stored:
        parts.append(f"stored faults detected: {stored}")
    if pending:
        parts.append(f"pending faults detected: {pending}")
    if permanent:
        parts.append(f"permanent faults detected: {permanent}")
    if not parts:
        return "No stored, pending, or permanent DTC detected."
    return " ; ".join(parts) + "."


def _diagnose_cylinder(
    intent: Intent,
    signals: dict[str, Any],
    dtc: dict[str, Any] | None,
    mode06: dict[str, Any] | None,
) -> str:
    """Produce a cylinder-oriented diagnosis from DTCs and live signals."""
    cyl = intent.parameters.cylinder_index or "requested"
    stored = (dtc or {}).get("stored", [])
    pending = (dtc or {}).get("pending", [])
    stft = _signal_numeric_value(signals, "stft_b1")
    ltft = _signal_numeric_value(signals, "ltft_b1")
    o2 = _signal_numeric_value(signals, "o2_b1s1")
    rpm = _signal_numeric_value(signals, "rpm")
    load = _signal_numeric_value(signals, "engine_load")

    reasons: list[str] = []
    severity = "moderate"

    misfire_codes = [code for code in stored + pending if code.startswith("P03")]
    if misfire_codes:
        reasons.append(f"misfire-related DTCs detected ({misfire_codes})")
        severity = "high"

    if stft is not None and stft > 12:
        reasons.append(f"short-term fuel trim is strongly positive ({stft:.1f}%) suggesting a lean correction")
    elif stft is not None and stft < -12:
        reasons.append(f"short-term fuel trim is strongly negative ({stft:.1f}%) suggesting a rich correction")

    if ltft is not None and abs(ltft) > 10:
        reasons.append(f"long-term fuel trim is significantly shifted ({ltft:.1f}%)")

    if o2 is not None and o2 < 0.2:
        reasons.append(f"upstream O2 voltage is low ({o2:.2f} V), consistent with a lean trend")
    elif o2 is not None and o2 > 0.8:
        reasons.append(f"upstream O2 voltage is high ({o2:.2f} V), consistent with a rich trend")

    if mode06:
        reasons.append("optional Mode 06 data is available and can reinforce the cylinder suspicion")
    else:
        reasons.append("Mode 06 is unavailable or not contributing additional evidence")

    if rpm is not None and load is not None:
        reasons.append(f"snapshot captured around {rpm:.0f} RPM and {load:.0f}% engine load")

    if not reasons:
        return f"No clear cylinder-specific anomaly was detected for cylinder {cyl} with the currently available data."

    return (
        f"Cylinder {cyl} analysis suggests a {severity} suspicion of abnormal combustion behavior: "
        + "; ".join(reasons)
        + "."
    )


def _diagnose_engine_health(signals: dict[str, Any], dtc: dict[str, Any] | None) -> str:
    """Produce a more nuanced engine health summary."""
    stored = (dtc or {}).get("stored", [])
    pending = (dtc or {}).get("pending", [])
    rpm = _signal_numeric_value(signals, "rpm")
    load = _signal_numeric_value(signals, "engine_load")
    coolant = _signal_numeric_value(signals, "coolant_temp")

    observations: list[str] = []
    risk_level = "normal"

    if stored:
        observations.append(f"stored DTCs are present ({stored})")
        risk_level = "elevated"
    if pending:
        observations.append(f"pending DTCs are present ({pending})")
        if risk_level == "normal":
            risk_level = "watch"

    if coolant is not None:
        if coolant >= 110:
            observations.append(f"coolant temperature is high ({coolant:.0f} C)")
            risk_level = "elevated"
        elif 80 <= coolant <= 105:
            observations.append(f"coolant temperature is within a plausible warm operating range ({coolant:.0f} C)")
        else:
            observations.append(f"coolant temperature is outside a typical fully-warm range ({coolant:.0f} C)")

    if rpm is not None:
        observations.append(f"engine speed is {rpm:.0f} RPM")
    if load is not None:
        observations.append(f"engine load is {load:.0f}%")

    if risk_level == "normal" and not stored and not pending and coolant is not None and coolant < 110:
        return "Engine appears to be operating within normal parameters based on current live data and the absence of active DTCs."

    if not observations:
        return "Engine health cannot be assessed reliably because the available data is insufficient."

    return f"Engine health assessment is {risk_level}: " + "; ".join(observations) + "."


def _diagnose_vehicle_health(
    signals: dict[str, Any],
    dtc: dict[str, Any] | None,
    vehicle_context: dict[str, Any] | None,
) -> str:
    """Produce a broad vehicle-health summary for valid high-level health requests."""
    engine_summary = _diagnose_engine_health(signals, dtc)
    context_bits: list[str] = []
    if vehicle_context:
        vin = vehicle_context.get("vin")
        ecu = vehicle_context.get("ecu_name")
        if vin:
            context_bits.append(f"VIN {vin}")
        if ecu:
            context_bits.append(f"ECU {ecu}")
    if context_bits:
        return "Vehicle health snapshot: " + engine_summary + " Context: " + ", ".join(context_bits) + "."
    return "Vehicle health snapshot: " + engine_summary


def _diagnose_signal_status(intent: Intent, signals: dict[str, Any]) -> str:
    """Describe a single requested signal in a more useful way."""
    signal = intent.parameters.signal
    if not signal:
        return "No signal was specified for signal status evaluation."
    payload = signals.get(signal)
    if not payload:
        return f"Signal {signal} is unavailable, stale, or unsupported in the current context."

    value = payload.get("value")
    unit = payload.get("unit") or ""
    observed_ts = payload.get("observed_ts")
    source = payload.get("source")
    plausibility = _signal_plausibility_note(signal, value)
    return f"Signal {signal} is available at {value} {unit}".strip() + f", observed at {observed_ts} from {source}; {plausibility}."


def _diagnose_warning_light(signals: dict[str, Any], dtc: dict[str, Any] | None) -> str:
    """Explain the warning light using DTCs and minimal live context."""
    stored = (dtc or {}).get("stored", [])
    pending = (dtc or {}).get("pending", [])
    if stored:
        return f"Check Engine light is most likely explained by stored DTCs {stored}, which indicate an active or confirmed fault condition."
    if pending:
        return f"Check Engine investigation found pending DTCs {pending}; the issue may be intermittent or not yet fully confirmed."

    rpm = _signal_numeric_value(signals, "rpm")
    coolant = _signal_numeric_value(signals, "coolant_temp")
    context = []
    if rpm is not None:
        context.append(f"RPM={rpm:.0f}")
    if coolant is not None:
        context.append(f"coolant={coolant:.0f} C")
    if context:
        return "No DTC currently explains the warning light; available live context is " + ", ".join(context) + "."
    return "The warning light cannot be explained reliably because neither DTCs nor sufficient live context are available."


def _signal_plausibility_note(signal: str, value: Any) -> str:
    """Return a short plausibility note for a signal value."""
    if not isinstance(value, (int, float)):
        return "value requires contextual interpretation"
    numeric = float(value)
    if signal == "rpm":
        return "value looks plausible for an engine speed measurement" if 0 <= numeric <= 8000 else "value looks unusual for engine speed"
    if signal == "coolant_temp":
        return "value looks plausible for coolant temperature" if -40 <= numeric <= 140 else "value looks unusual for coolant temperature"
    if signal in {"engine_load", "throttle_pos", "stft_b1", "ltft_b1"}:
        return "value should be interpreted alongside other engine conditions"
    if signal == "o2_b1s1":
        return "value should be interpreted as part of mixture control behavior"
    if signal in {"vehicle_speed", "module_voltage"}:
        return "value looks plausible within normal operating boundaries"
    return "value requires contextual interpretation"
