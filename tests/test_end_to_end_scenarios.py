from __future__ import annotations

from datetime import datetime, timezone

from agent.orchestrator import diagnose
from backend.schemas import AgentRequest


def _request(prompt: str, request_id: str) -> AgentRequest:
    return AgentRequest(
        request_id=request_id,
        ts=datetime.now(timezone.utc),
        vehicle_id="veh_001",
        user_prompt=prompt,
    )


def _request_for_vehicle(prompt: str, request_id: str, vehicle_id: str) -> AgentRequest:
    return AgentRequest(
        request_id=request_id,
        ts=datetime.now(timezone.utc),
        vehicle_id=vehicle_id,
        user_prompt=prompt,
    )


def _assert_structured_response(response) -> None:
    assert response.request_id
    assert response.intent.name
    assert response.diagnosis
    assert 0.0 <= response.confidence <= 1.0
    assert isinstance(response.evidence, list)
    assert isinstance(response.missing_data, list)
    assert isinstance(response.actions_taken, list)


def test_scenario_check_cylinder_2() -> None:
    response = diagnose(_request("check cylinder 2", "scenario_1"))
    _assert_structured_response(response)
    assert response.intent.name == "CHECK_CYLINDER"


def test_scenario_read_dtc() -> None:
    response = diagnose(_request("read dtc", "scenario_2"))
    _assert_structured_response(response)
    assert response.intent.name == "READ_DTC"


def test_scenario_show_rpm() -> None:
    response = diagnose(_request("show rpm", "scenario_3"))
    _assert_structured_response(response)
    assert response.intent.name == "CHECK_SIGNAL_STATUS"
    assert "rpm" in response.signals_used


def test_scenario_get_vin() -> None:
    response = diagnose(_request("get vin", "scenario_4"))
    _assert_structured_response(response)
    assert response.intent.name == "GET_VEHICLE_CONTEXT"


def test_scenario_unsupported_prompt() -> None:
    response = diagnose(_request("please fix everything automatically", "scenario_5"))
    _assert_structured_response(response)
    assert response.intent.name == "UNKNOWN"


def test_scenario_broad_vehicle_health_prompt() -> None:
    response = diagnose(_request("check my car health", "scenario_broad"))
    _assert_structured_response(response)
    assert response.intent.name == "CHECK_ENGINE_HEALTH"
    assert response.intent.parameters.goal == "VEHICLE_HEALTH_CHECK"
    assert "get_vehicle_context" in response.actions_taken


def test_vehicle_profile_healthy_engine_veh_002() -> None:
    response = diagnose(_request_for_vehicle("je veux connaître l'état du moteur", "scenario_veh_002", "veh_002"))
    _assert_structured_response(response)
    assert response.intent.name == "CHECK_ENGINE_HEALTH"
    assert "normal parameters" in response.diagnosis.lower()


def test_vehicle_profile_overheat_risk_veh_004() -> None:
    response = diagnose(_request_for_vehicle("je veux connaître l'état du moteur", "scenario_veh_004", "veh_004"))
    _assert_structured_response(response)
    assert response.intent.name == "CHECK_ENGINE_HEALTH"
    assert "elevated" in response.diagnosis.lower() or "high" in response.diagnosis.lower()
    assert "coolant" in response.diagnosis.lower()


def test_vehicle_profile_sparse_cache_veh_005_triggers_on_demand() -> None:
    response = diagnose(_request_for_vehicle("check cylinder 2", "scenario_veh_005", "veh_005"))
    _assert_structured_response(response)
    assert response.intent.name == "CHECK_CYLINDER"
    assert "request_fresh_signals" in response.actions_taken
    assert "engine_load" in response.signals_used


def test_vehicle_profile_lean_condition_veh_003() -> None:
    response = diagnose(_request_for_vehicle("check cylinder 2", "scenario_veh_003", "veh_003"))
    _assert_structured_response(response)
    assert response.intent.name == "CHECK_CYLINDER"
    assert "lean" in response.diagnosis.lower() or "fuel trim" in response.diagnosis.lower()
