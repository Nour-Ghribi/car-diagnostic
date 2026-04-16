from __future__ import annotations

from data.mock_runtime_scenarios import get_mock_runtime_scenario, list_mock_runtime_scenarios


def test_mock_runtime_scenarios_expose_expected_metadata() -> None:
    scenarios = list_mock_runtime_scenarios()
    assert len(scenarios) >= 7

    vehicle_ids = {scenario.vehicle_id for scenario in scenarios}
    assert {"veh_001", "veh_006", "veh_007"}.issubset(vehicle_ids)


def test_electrical_low_voltage_scenario_is_available() -> None:
    scenario = get_mock_runtime_scenario("veh_006")
    assert scenario.scenario_family == "electrical_low_voltage"
    assert "battery" in scenario.profile_tags
    assert scenario.latest_signals["module_voltage"]["value"] < 12.0
    assert "P0562" in scenario.dtc["stored"]


def test_rich_running_scenario_is_available() -> None:
    scenario = get_mock_runtime_scenario("veh_007")
    assert scenario.scenario_family == "rich_running"
    assert scenario.latest_signals["stft_b1"]["value"] < 0
    assert scenario.latest_signals["ltft_b1"]["value"] < 0
    assert "P0172" in scenario.dtc["stored"]
