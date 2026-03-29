from __future__ import annotations

from agent import tools


def _assert_tool_response_shape(response: dict) -> None:
    assert response["status"] in {"success", "partial", "error"}
    assert "data" in response
    assert "missing_data" in response
    assert "error_message" in response


def test_get_capabilities_returns_valid_tool_response() -> None:
    response = tools.get_capabilities("veh_001")
    _assert_tool_response_shape(response)
    assert response["status"] == "success"
    assert "capabilities" in response["data"]
    assert response["data"]["capabilities"]["mode06_supported"] in {True, False}


def test_get_vehicle_context_returns_valid_tool_response() -> None:
    response = tools.get_vehicle_context("veh_001")
    _assert_tool_response_shape(response)
    assert response["status"] == "success"
    assert response["data"]["vehicle_context"]["vin"]


def test_get_latest_signals_returns_requested_signals() -> None:
    requested = ["rpm", "engine_load"]
    response = tools.get_latest_signals("veh_001", requested)
    _assert_tool_response_shape(response)
    assert response["status"] == "success"
    assert set(response["data"]["signals"].keys()) == set(requested)


def test_request_fresh_signals_returns_coherent_structure() -> None:
    requested = ["rpm", "coolant_temp"]
    response = tools.request_fresh_signals("veh_001", requested)
    _assert_tool_response_shape(response)
    assert response["status"] == "success"
    assert set(response["data"]["signals"].keys()) == set(requested)
    assert response["data"]["dtc"] is None


def test_get_dtcs_returns_stored_and_pending() -> None:
    response = tools.get_dtcs("veh_001", include_pending=True)
    _assert_tool_response_shape(response)
    assert response["status"] == "success"
    assert "stored" in response["data"]["dtc"]
    assert "pending" in response["data"]["dtc"]


def test_request_mode06_handles_success_and_no_data() -> None:
    success_response = tools.request_mode06("veh_001", nodata=False)
    nodata_response = tools.request_mode06("veh_001", nodata=True)

    _assert_tool_response_shape(success_response)
    _assert_tool_response_shape(nodata_response)

    assert success_response["status"] == "success"
    assert success_response["data"]["mode06"] is not None

    assert nodata_response["status"] == "partial"
    assert nodata_response["data"]["mode06"] is None
    assert nodata_response["missing_data"][0]["reason"] == "no_data"


def test_score_confidence_returns_score_between_zero_and_one() -> None:
    response = tools.score_confidence(present_count=4, missing_data=[], coherent=True)
    _assert_tool_response_shape(response)
    assert response["status"] == "success"
    confidence = response["data"]["metrics"]["confidence"]
    assert 0.0 <= confidence <= 1.0


def test_tool_responses_have_contract_keys() -> None:
    responses = [
        tools.get_capabilities("veh_001"),
        tools.get_vehicle_context("veh_001"),
        tools.get_latest_signals("veh_001", ["rpm"]),
        tools.request_fresh_signals("veh_001", ["rpm"]),
        tools.get_dtcs("veh_001"),
        tools.request_mode06("veh_001"),
    ]
    for response in responses:
        _assert_tool_response_shape(response)
