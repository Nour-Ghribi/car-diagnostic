from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent import orchestrator
from backend.schemas import AgentRequest
from tools import registry as tool_registry


def _build_request(prompt: str, request_id: str = "req_test") -> AgentRequest:
    return AgentRequest(
        request_id=request_id,
        ts=datetime.now(timezone.utc),
        vehicle_id="veh_001",
        user_prompt=prompt,
    )


def test_diagnose_check_cylinder() -> None:
    response = orchestrator.diagnose(_build_request("check cylinder 2", "req_cyl"))
    assert response.request_id == "req_cyl"
    assert response.intent.name == "CHECK_CYLINDER"
    assert response.diagnosis
    assert 0.0 <= response.confidence <= 1.0
    assert "get_dtcs" in response.actions_taken
    assert "get_latest_signals" in response.actions_taken
    assert "request_mode06" in response.actions_taken
    assert "rpm" in response.signals_used
    assert isinstance(response.missing_data, list)


def test_diagnose_read_dtc() -> None:
    response = orchestrator.diagnose(_build_request("read dtc", "req_dtc"))
    assert response.intent.name == "READ_DTC"
    assert response.request_id == "req_dtc"
    assert response.diagnosis
    assert 0.0 <= response.confidence <= 1.0
    assert "get_dtcs" in response.actions_taken


def test_diagnose_check_signal_status() -> None:
    response = orchestrator.diagnose(_build_request("show rpm", "req_rpm"))
    assert response.intent.name == "CHECK_SIGNAL_STATUS"
    assert response.diagnosis
    assert "rpm" in response.signals_used
    assert "get_latest_signals" in response.actions_taken


def test_diagnose_get_vehicle_context() -> None:
    response = orchestrator.diagnose(_build_request("get vin", "req_vin"))
    assert response.intent.name == "GET_VEHICLE_CONTEXT"
    assert response.request_id == "req_vin"
    assert response.diagnosis
    assert "get_vehicle_context" in response.actions_taken


def test_diagnose_unknown() -> None:
    response = orchestrator.diagnose(_build_request("repair my whole car please", "req_unknown"))
    assert response.intent.name == "UNKNOWN"
    assert response.diagnosis
    assert 0.0 <= response.confidence <= 1.0
    assert isinstance(response.missing_data, list)


def test_diagnose_uses_request_fresh_signals_when_cache_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"fresh": False}

    def fake_get_latest_signals(vehicle_id: str, signals: list[str], request_id: str) -> dict:
        return {
            "request_id": request_id,
            "tool_name": "get_latest_signals",
            "status": "partial",
            "data": {
                "signals": {"rpm": {"value": 800, "unit": "RPM", "observed_ts": "2026-03-26T15:10:00Z", "source": "push_cache"}},
                "dtc": None,
                "mode06": None,
                "vehicle_context": None,
                "capabilities": None,
                "metrics": None,
            },
            "missing_data": [{"key": "engine_load", "reason": "not_collected"}],
            "error_message": None,
        }

    def fake_request_fresh_signals(vehicle_id: str, signals: list[str], request_id: str) -> dict:
        called["fresh"] = True
        return {
            "request_id": request_id,
            "tool_name": "request_fresh_signals",
            "status": "success",
            "data": {
                "signals": {
                    "engine_load": {
                        "value": 20,
                        "unit": "%",
                        "observed_ts": "2026-03-26T15:10:01Z",
                        "source": "on_demand",
                    }
                },
                "dtc": None,
                "mode06": None,
                "vehicle_context": None,
                "capabilities": None,
                "metrics": None,
            },
            "missing_data": [],
            "error_message": None,
        }

    monkeypatch.setattr(tool_registry.legacy_tools, "get_latest_signals", fake_get_latest_signals)
    monkeypatch.setattr(tool_registry.legacy_tools, "request_fresh_signals", fake_request_fresh_signals)

    response = orchestrator.diagnose(_build_request("show rpm", "req_refresh"))
    assert called["fresh"] is True
    assert "request_fresh_signals" in response.actions_taken


def test_diagnose_continues_when_mode06_returns_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request_mode06(vehicle_id: str, request_id: str) -> dict:
        return {
            "request_id": request_id,
            "tool_name": "request_mode06",
            "status": "partial",
            "data": {
                "signals": {},
                "dtc": None,
                "mode06": None,
                "vehicle_context": None,
                "capabilities": None,
                "metrics": None,
            },
            "missing_data": [{"key": "mode06", "reason": "no_data"}],
            "error_message": None,
        }

    monkeypatch.setattr(tool_registry.legacy_tools, "request_mode06", fake_request_mode06)
    response = orchestrator.diagnose(_build_request("check cylinder 2", "req_mode06"))
    assert response.intent.name == "CHECK_CYLINDER"
    assert response.diagnosis
    assert any(item.key == "mode06" for item in response.missing_data)


def test_diagnose_does_not_crash_when_tool_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_get_dtcs(vehicle_id: str, include_pending: bool, request_id: str) -> dict:
        return {
            "request_id": request_id,
            "tool_name": "get_dtcs",
            "status": "error",
            "data": {
                "signals": {},
                "dtc": None,
                "mode06": None,
                "vehicle_context": None,
                "capabilities": None,
                "metrics": None,
            },
            "missing_data": [],
            "error_message": "forced failure",
        }

    monkeypatch.setattr(tool_registry.legacy_tools, "get_dtcs", failing_get_dtcs)
    response = orchestrator.diagnose(_build_request("read dtc", "req_err"))
    assert response.request_id == "req_err"
    assert response.diagnosis
    assert isinstance(response.missing_data, list)


def test_diagnose_returns_safe_clarification_when_pipeline_cannot_ground_request() -> None:
    response = orchestrator.diagnose(_build_request("it feels unclear and weird", "req_parse_fallback"))
    assert response.intent.parameters.resolution_policy in {"CLARIFICATION_NEEDED", "UNKNOWN"}


def test_diagnose_broad_vehicle_health_builds_default_plan() -> None:
    response = orchestrator.diagnose(_build_request("je veux connaître le health de ma voiture", "req_broad"))
    assert response.intent.name == "CHECK_ENGINE_HEALTH"
    assert response.intent.parameters.goal == "VEHICLE_HEALTH_CHECK"
    assert response.intent.parameters.scope == "broad"
    assert "get_vehicle_context" in response.actions_taken
    assert "get_dtcs" in response.actions_taken
    assert "get_latest_signals" in response.actions_taken


def test_diagnose_clarification_needed_returns_question() -> None:
    response = orchestrator.diagnose(_build_request("it's weird", "req_clarify"))
    assert response.intent.parameters.resolution_policy == "CLARIFICATION_NEEDED"
    assert "clarify" in response.diagnosis.lower() or "voulez" in response.diagnosis.lower() or "probleme" in response.diagnosis.lower()
