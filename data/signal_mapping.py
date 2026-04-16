from __future__ import annotations

from backend.schemas import SignalName, ToolName


SignalCategoryName = str


_SIGNAL_KEY_TO_SIGNAL_NAME: dict[str, SignalName] = {
    "engine_rpm": "rpm",
    "engine_load": "engine_load",
    "absolute_engine_load_pct": "engine_load",
    "coolant_temp_c": "coolant_temp",
    "coolant_temp_sensor_2_c": "coolant_temp",
    "throttle_position_pct": "throttle_pos",
    "relative_throttle_position_pct": "throttle_pos",
    "absolute_throttle_position_b_pct": "throttle_pos",
    "short_term_fuel_trim_bank1": "stft_b1",
    "long_term_fuel_trim_bank1": "ltft_b1",
    "o2_sensor_b1s1": "o2_b1s1",
    "vehicle_speed": "vehicle_speed",
    "control_module_voltage_v": "module_voltage",
}


_SIGNAL_KEY_TO_TOOL_NAME: dict[str, ToolName] = {
    "engine_rpm": "get_latest_signals",
    "engine_load": "get_latest_signals",
    "absolute_engine_load_pct": "get_latest_signals",
    "vehicle_speed": "get_latest_signals",
    "throttle_position_pct": "get_latest_signals",
    "relative_throttle_position_pct": "get_latest_signals",
    "absolute_throttle_position_b_pct": "get_latest_signals",
    "relative_accelerator_pedal_pct": "get_latest_signals",
    "maf_g_s": "get_latest_signals",
    "intake_manifold_pressure_kpa": "get_latest_signals",
    "timing_advance_deg": "get_latest_signals",
    "coolant_temp_c": "get_latest_signals",
    "coolant_temp_sensor_2_c": "get_latest_signals",
    "engine_oil_temp_c": "get_latest_signals",
    "ambient_air_temp_c": "get_latest_signals",
    "intake_air_temp_c": "get_latest_signals",
    "runtime_since_engine_start_s": "get_latest_signals",
    "fuel_system_status": "get_latest_signals",
    "fuel_pressure_kpa": "get_latest_signals",
    "fuel_rail_pressure_relative_kpa": "get_latest_signals",
    "fuel_rail_pressure_direct_kpa": "get_latest_signals",
    "short_term_fuel_trim_bank1": "get_latest_signals",
    "long_term_fuel_trim_bank1": "get_latest_signals",
    "short_term_fuel_trim_bank2": "get_latest_signals",
    "long_term_fuel_trim_bank2": "get_latest_signals",
    "o2_sensor_b1s1": "get_latest_signals",
    "o2_sensor_b1s2": "get_latest_signals",
    "commanded_air_fuel_ratio_lambda": "get_latest_signals",
    "barometric_pressure_kpa": "get_latest_signals",
    "control_module_voltage_v": "get_latest_signals",
    "commanded_throttle_actuator_pct": "get_latest_signals",
}


_SIGNAL_KEY_TO_CATEGORY: dict[str, SignalCategoryName] = {
    "engine_rpm": "engine_core",
    "engine_load": "engine_core",
    "absolute_engine_load_pct": "engine_core",
    "vehicle_speed": "drivability_core",
    "throttle_position_pct": "airflow_throttle",
    "relative_throttle_position_pct": "airflow_throttle",
    "absolute_throttle_position_b_pct": "airflow_throttle",
    "relative_accelerator_pedal_pct": "airflow_throttle",
    "maf_g_s": "airflow_throttle",
    "intake_manifold_pressure_kpa": "airflow_throttle",
    "timing_advance_deg": "engine_core",
    "coolant_temp_c": "cooling_core",
    "coolant_temp_sensor_2_c": "cooling_core",
    "engine_oil_temp_c": "cooling_core",
    "ambient_air_temp_c": "cooling_core",
    "intake_air_temp_c": "cooling_core",
    "runtime_since_engine_start_s": "runtime_context",
    "fuel_system_status": "fuel_control",
    "fuel_pressure_kpa": "fuel_control",
    "fuel_rail_pressure_relative_kpa": "fuel_control",
    "fuel_rail_pressure_direct_kpa": "fuel_control",
    "short_term_fuel_trim_bank1": "fuel_control",
    "long_term_fuel_trim_bank1": "fuel_control",
    "short_term_fuel_trim_bank2": "fuel_control",
    "long_term_fuel_trim_bank2": "fuel_control",
    "o2_sensor_b1s1": "fuel_control",
    "o2_sensor_b1s2": "fuel_control",
    "commanded_air_fuel_ratio_lambda": "fuel_control",
    "barometric_pressure_kpa": "airflow_throttle",
    "control_module_voltage_v": "electrical_core",
    "commanded_throttle_actuator_pct": "airflow_throttle",
}


def signal_key_to_signal_name(key: str) -> SignalName | None:
    """Map one CSV signal key to the internal canonical signal name."""

    return _SIGNAL_KEY_TO_SIGNAL_NAME.get(key.strip().lower())


def signal_key_to_tool_name(key: str) -> ToolName | None:
    """Map one CSV signal key to the most relevant stable runtime tool."""

    return _SIGNAL_KEY_TO_TOOL_NAME.get(key.strip().lower())


def signal_key_to_category(key: str) -> SignalCategoryName | None:
    """Map one CSV signal key to a higher-level diagnostic signal family."""

    return _SIGNAL_KEY_TO_CATEGORY.get(key.strip().lower())


def category_to_tool_name(category: SignalCategoryName) -> ToolName:
    """Return the stable runtime tool that currently owns one signal family."""

    if category in {"engine_core", "cooling_core", "airflow_throttle", "fuel_control", "electrical_core", "runtime_context", "drivability_core"}:
        return "get_latest_signals"
    return "get_latest_signals"
