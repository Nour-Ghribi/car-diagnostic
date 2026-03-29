from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal


SignalName = Literal[
    "rpm",
    "engine_load",
    "coolant_temp",
    "throttle_pos",
    "stft_b1",
    "ltft_b1",
    "o2_b1s1",
    "vehicle_speed",
    "module_voltage",
]

ToolName = Literal[
    "get_capabilities",
    "get_vehicle_context",
    "get_latest_signals",
    "get_signal_history",
    "get_dtcs",
    "request_fresh_signals",
    "request_mode06",
    "score_confidence",
]

ToolStatus = Literal["success", "partial", "error"]
MissingReason = Literal["unsupported", "not_collected", "stale", "timeout", "no_data"]

ALLOWED_SIGNALS: tuple[SignalName, ...] = (
    "rpm",
    "engine_load",
    "coolant_temp",
    "throttle_pos",
    "stft_b1",
    "ltft_b1",
    "o2_b1s1",
    "vehicle_speed",
    "module_voltage",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MOCK_DIR = BASE_DIR / "fake_data" / "tool_responses"


def _load_mock(filename: str) -> dict[str, Any]:
    with (MOCK_DIR / filename).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_vehicle_profiles() -> dict[str, Any]:
    return _load_mock("vehicle_profiles.json")


def _get_vehicle_profile(vehicle_id: str) -> dict[str, Any]:
    profiles = _load_vehicle_profiles()
    vehicles = profiles.get("vehicles", {})
    return deepcopy(vehicles.get(vehicle_id) or vehicles.get("veh_001", {}))


def _empty_data() -> dict[str, Any]:
    return {
        "signals": {},
        "dtc": None,
        "mode06": None,
        "vehicle_context": None,
        "capabilities": None,
        "metrics": None,
    }


def _build_response(
    request_id: str,
    tool_name: ToolName,
    status: ToolStatus,
    *,
    data: dict[str, Any] | None = None,
    missing_data: list[dict[str, str]] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "tool_name": tool_name,
        "status": status,
        "data": data if data is not None else _empty_data(),
        "missing_data": missing_data or [],
        "error_message": error_message,
    }


def _filter_signals(
    signals: dict[str, Any],
    requested_signals: list[SignalName],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    selected: dict[str, Any] = {}
    missing: list[dict[str, str]] = []
    for signal in requested_signals:
        if signal in signals:
            selected[signal] = signals[signal]
        else:
            missing.append({"key": signal, "reason": "not_collected"})
    return selected, missing


def _validate_signals(signals: list[str]) -> list[SignalName]:
    invalid = [signal for signal in signals if signal not in ALLOWED_SIGNALS]
    if invalid:
        raise ValueError(f"Unsupported signals requested: {', '.join(invalid)}")
    return signals  # type: ignore[return-value]


def get_capabilities(vehicle_id: str, request_id: str = "mock_get_capabilities") -> dict[str, Any]:
    """
    Return vehicle diagnostic capabilities from mock data.

    Input:
    - vehicle_id: logical vehicle identifier.
    - request_id: correlation identifier for the tool call.

    Output:
    - ToolResponse containing vehicle capabilities in a dedicated `capabilities` block
      and mode06 support in the `mode06` block.

    Possible errors:
    - returns status 'error' if vehicle_id is empty or mock loading fails.
    """
    if not vehicle_id:
        return _build_response(
            request_id,
            "get_capabilities",
            "error",
            error_message="vehicle_id is required",
        )

    profile = _get_vehicle_profile(vehicle_id)
    payload = deepcopy(_load_mock("get_capabilities.json"))
    if profile.get("capabilities") is not None:
        payload["data"]["capabilities"] = profile["capabilities"]
        payload["data"]["mode06"] = {"supported": profile["capabilities"].get("mode06_supported", False)}
    return _build_response(
        request_id,
        "get_capabilities",
        "success",
        data=payload["data"],
        missing_data=payload.get("missing_data", []),
    )


def get_vehicle_context(vehicle_id: str, request_id: str = "mock_get_vehicle_context") -> dict[str, Any]:
    """
    Return vehicle identity and ECU context from mock data.

    Input:
    - vehicle_id: logical vehicle identifier.
    - request_id: correlation identifier for the tool call.

    Output:
    - ToolResponse containing VIN, calibration and ECU context in a dedicated
      `vehicle_context` block.

    Possible errors:
    - returns status 'error' if vehicle_id is empty.
    """
    if not vehicle_id:
        return _build_response(
            request_id,
            "get_vehicle_context",
            "error",
            error_message="vehicle_id is required",
        )

    profile = _get_vehicle_profile(vehicle_id)
    payload = deepcopy(_load_mock("get_vehicle_context.json"))
    if profile.get("vehicle_context") is not None:
        payload["data"]["vehicle_context"] = profile["vehicle_context"]
    return _build_response(
        request_id,
        "get_vehicle_context",
        "success",
        data=payload["data"],
        missing_data=payload.get("missing_data", []),
    )


def get_latest_signals(
    vehicle_id: str,
    signals: list[SignalName],
    request_id: str = "mock_get_latest_signals",
) -> dict[str, Any]:
    """
    Return the latest cached signals without triggering any hardware read.

    Input:
    - vehicle_id: logical vehicle identifier.
    - signals: allowed signal names to retrieve from cache.
    - request_id: correlation identifier for the tool call.

    Output:
    - ToolResponse containing only cached values for requested signals.

    Possible errors:
    - returns status 'error' if vehicle_id is empty.
    - raises ValueError if an unsupported signal is requested.
    """
    if not vehicle_id:
        return _build_response(
            request_id,
            "get_latest_signals",
            "error",
            error_message="vehicle_id is required",
        )

    validated = _validate_signals(signals)
    profile = _get_vehicle_profile(vehicle_id)
    payload = deepcopy(_load_mock("get_latest_signals.json"))
    if profile.get("latest_signals") is not None:
        payload["data"]["signals"] = profile["latest_signals"]
    selected, missing = _filter_signals(payload["data"]["signals"], validated)
    status: ToolStatus = "success" if not missing else "partial"
    return _build_response(
        request_id,
        "get_latest_signals",
        status,
        data={**_empty_data(), "signals": selected},
        missing_data=missing,
    )


def get_signal_history(
    vehicle_id: str,
    signal: SignalName,
    request_id: str = "mock_get_signal_history",
) -> dict[str, Any]:
    """
    Return a simulated history for one signal.

    Input:
    - vehicle_id: logical vehicle identifier.
    - signal: one allowed signal name.
    - request_id: correlation identifier for the tool call.

    Output:
    - ToolResponse whose `data.signals` contains a dedicated `history` list for the
      requested signal.
    - The returned history is fully simulated from mock data and does not come from
      a real time-series backend.

    Possible errors:
    - returns status 'error' if vehicle_id is empty.
    """
    if not vehicle_id:
        return _build_response(
            request_id,
            "get_signal_history",
            "error",
            error_message="vehicle_id is required",
        )

    if signal not in ALLOWED_SIGNALS:
        raise ValueError(f"Unsupported signal requested: {signal}")

    profile = _get_vehicle_profile(vehicle_id)
    latest = deepcopy(_load_mock("get_latest_signals.json"))
    if profile.get("latest_signals") is not None:
        latest["data"]["signals"] = profile["latest_signals"]
    if signal not in latest["data"]["signals"]:
        return _build_response(
            request_id,
            "get_signal_history",
            "partial",
            data=_empty_data(),
            missing_data=[{"key": signal, "reason": "not_collected"}],
        )

    current = latest["data"]["signals"][signal]
    history_points = [
        {
            "value": current["value"],
            "unit": current["unit"],
            "observed_ts": "2026-03-26T15:09:30Z",
            "source": current["source"],
        },
        {
            "value": current["value"],
            "unit": current["unit"],
            "observed_ts": "2026-03-26T15:09:45Z",
            "source": current["source"],
        },
        current,
    ]
    return _build_response(
        request_id,
        "get_signal_history",
        "success",
        data={
            **_empty_data(),
            "signals": {
                signal: {
                    "history": history_points,
                    "simulated": True,
                }
            },
        },
    )


def get_dtcs(
    vehicle_id: str,
    include_pending: bool = True,
    request_id: str = "mock_get_dtcs",
) -> dict[str, Any]:
    """
    Return stored and pending DTCs from mock data.

    Input:
    - vehicle_id: logical vehicle identifier.
    - include_pending: whether pending DTCs should be included.
    - request_id: correlation identifier for the tool call.

    Output:
    - ToolResponse containing structured DTC bundles.

    Possible errors:
    - returns status 'error' if vehicle_id is empty.
    """
    if not vehicle_id:
        return _build_response(
            request_id,
            "get_dtcs",
            "error",
            error_message="vehicle_id is required",
        )

    profile = _get_vehicle_profile(vehicle_id)
    payload = deepcopy(_load_mock("get_dtcs.json"))
    if profile.get("dtc") is not None:
        payload["data"]["dtc"] = profile["dtc"]
    if not include_pending:
        payload["data"]["dtc"]["pending"] = []
    return _build_response(
        request_id,
        "get_dtcs",
        "success",
        data=payload["data"],
        missing_data=payload.get("missing_data", []),
    )


def request_fresh_signals(
    vehicle_id: str,
    signals: list[SignalName],
    request_id: str = "mock_request_fresh_signals",
) -> dict[str, Any]:
    """
    Simulate an on-demand backend request that would ask the STM32 for fresh signals.

    Input:
    - vehicle_id: logical vehicle identifier.
    - signals: allowed signal names to refresh.
    - request_id: correlation identifier for the tool call.

    Output:
    - ToolResponse containing refreshed signals.

    Possible errors:
    - returns status 'error' if vehicle_id is empty.
    - raises ValueError if an unsupported signal is requested.
    """
    if not vehicle_id:
        return _build_response(
            request_id,
            "request_fresh_signals",
            "error",
            error_message="vehicle_id is required",
        )

    validated = _validate_signals(signals)
    profile = _get_vehicle_profile(vehicle_id)
    payload = deepcopy(_load_mock("request_fresh_signals.json"))
    if profile.get("fresh_signals") is not None:
        payload["data"]["signals"] = profile["fresh_signals"]
    selected, missing = _filter_signals(payload["data"]["signals"], validated)
    status: ToolStatus = "success" if not missing else "partial"
    return _build_response(
        request_id,
        "request_fresh_signals",
        status,
        data={**_empty_data(), "signals": selected},
        missing_data=missing,
    )


def request_mode06(
    vehicle_id: str,
    request_id: str = "mock_request_mode06",
    *,
    nodata: bool = False,
) -> dict[str, Any]:
    """
    Simulate a Mode 06 request executed through the backend toward the embedded device.

    Input:
    - vehicle_id: logical vehicle identifier.
    - request_id: correlation identifier for the tool call.
    - nodata: when True, simulate an ECU returning NO DATA.

    Output:
    - ToolResponse containing mode06 data or a partial response with missing_data.

    Possible errors:
    - returns status 'error' if vehicle_id is empty.
    """
    if not vehicle_id:
        return _build_response(
            request_id,
            "request_mode06",
            "error",
            error_message="vehicle_id is required",
        )

    profile = _get_vehicle_profile(vehicle_id)
    if not nodata and profile.get("mode06_response") == "no_data":
        nodata = True

    filename = "request_mode06_nodata.json" if nodata else "request_mode06_success.json"
    payload = deepcopy(_load_mock(filename))
    if not nodata and profile.get("mode06") is not None:
        payload["data"]["mode06"] = profile["mode06"]
    return _build_response(
        request_id,
        "request_mode06",
        payload["status"],
        data=payload["data"],
        missing_data=payload.get("missing_data", []),
        error_message=payload.get("error_message"),
    )


def score_confidence(
    *,
    present_count: int,
    missing_data: list[dict[str, str]] | None = None,
    coherent: bool = True,
    request_id: str = "mock_score_confidence",
) -> dict[str, Any]:
    """
    Compute a mock confidence score from data availability and signal consistency.

    Input:
    - present_count: number of useful data points available.
    - missing_data: normalized missing data list.
    - coherent: whether signals appear mutually coherent.
    - request_id: correlation identifier for the tool call.

    Output:
    - ToolResponse containing a confidence score in a dedicated `metrics` block.

    Possible errors:
    - returns status 'error' if present_count is negative.
    """
    if present_count < 0:
        return _build_response(
            request_id,
            "score_confidence",
            "error",
            error_message="present_count must be >= 0",
        )

    missing_data = missing_data or []
    score = min(1.0, 0.35 + present_count * 0.08)
    score -= min(0.45, len(missing_data) * 0.08)
    if not coherent:
        score -= 0.15
    score = max(0.0, round(score, 2))
    status: ToolStatus = "success" if not missing_data else "partial"
    return _build_response(
        request_id,
        "score_confidence",
        status,
        data={
            **_empty_data(),
            "metrics": {"confidence": score, "coherent": coherent},
        },
        missing_data=missing_data,
    )
