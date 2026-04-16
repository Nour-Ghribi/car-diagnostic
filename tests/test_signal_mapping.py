from __future__ import annotations

from data.signal_mapping import signal_key_to_category, signal_key_to_signal_name, signal_key_to_tool_name


def test_signal_mapping_covers_common_csv_keys() -> None:
    assert signal_key_to_signal_name("engine_rpm") == "rpm"
    assert signal_key_to_signal_name("coolant_temp_sensor_2_c") == "coolant_temp"
    assert signal_key_to_signal_name("absolute_engine_load_pct") == "engine_load"


def test_signal_mapping_exposes_categories_and_tools() -> None:
    assert signal_key_to_category("maf_g_s") == "airflow_throttle"
    assert signal_key_to_category("fuel_pressure_kpa") == "fuel_control"
    assert signal_key_to_tool_name("control_module_voltage_v") == "get_latest_signals"

