from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.schemas import AgentResponse, Evidence, Intent, MissingData
from tools.executor import ExecutionAccumulator


def compose_agent_response(
    *,
    request_id: str,
    vehicle_id: str,
    intent: Intent,
    execution: ExecutionAccumulator,
    confidence: float,
    clarification_question: str | None = None,
) -> AgentResponse:
    """Compose a grounded user-facing response from executed observations."""
    diagnosis = _compose_diagnosis(intent, execution.aggregated, clarification_question)
    evidence = _build_evidence(execution.aggregated)
    recommendations = _build_recommendations(intent, execution.missing_data, clarification_question)
    return AgentResponse(
        request_id=request_id,
        ts=datetime.now(timezone.utc),
        vehicle_id=vehicle_id,
        intent=intent,
        diagnosis=diagnosis,
        confidence=round(confidence, 2),
        evidence=evidence,
        signals_used=list(execution.aggregated["signals"].keys()),
        actions_taken=execution.actions_taken,
        missing_data=execution.missing_data,
        recommendations=recommendations,
    )


def _compose_diagnosis(intent: Intent, aggregated: dict[str, Any], clarification_question: str | None) -> str:
    if clarification_question:
        return clarification_question
    signals = aggregated.get("signals", {})
    dtc = aggregated.get("dtc")
    mode06 = aggregated.get("mode06")
    vehicle_context = aggregated.get("vehicle_context")

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
    return "No grounded diagnostic result could be produced from the executed observations."


def _build_evidence(aggregated: dict[str, Any]) -> list[Evidence]:
    items: list[Evidence] = []
    for key, payload in aggregated.get("signals", {}).items():
        if "value" in payload:
            items.append(
                Evidence(
                    key=key,
                    label=key.replace("_", " ").title(),
                    value=payload["value"],
                    unit=payload.get("unit"),
                    observed_ts=datetime.fromisoformat(payload["observed_ts"].replace("Z", "+00:00")),
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


def _build_recommendations(intent: Intent, missing_data: list[MissingData], clarification_question: str | None) -> list[str]:
    if clarification_question:
        return ["Clarify the request so the resolver can generate a more grounded diagnostic plan."]
    recommendations: list[str] = []
    if intent.parameters.goal == "VEHICLE_HEALTH_CHECK":
        recommendations.append("Review the broad vehicle-health snapshot with live data, DTC status, and ECU context together.")
    elif intent.name == "CHECK_CYLINDER":
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
        recommendations.append("Use a supported automotive diagnostic request.")
    return recommendations


def _diagnose_dtc_readout(dtc: dict[str, Any] | None) -> str:
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


def _diagnose_cylinder(intent: Intent, signals: dict[str, Any], dtc: dict[str, Any] | None, mode06: dict[str, Any] | None) -> str:
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
    return f"Cylinder {cyl} analysis suggests a {severity} suspicion of abnormal combustion behavior: " + "; ".join(reasons) + "."


def _diagnose_engine_health(signals: dict[str, Any], dtc: dict[str, Any] | None) -> str:
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


def _diagnose_vehicle_health(signals: dict[str, Any], dtc: dict[str, Any] | None, vehicle_context: dict[str, Any] | None) -> str:
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


def _signal_numeric_value(signals: dict[str, Any], key: str) -> float | None:
    payload = signals.get(key)
    if not payload:
        return None
    value = payload.get("value")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
