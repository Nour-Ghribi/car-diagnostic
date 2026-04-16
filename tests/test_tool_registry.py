from __future__ import annotations

from agent.embedding_provider import clear_embedding_cache
from agent.settings import clear_settings_cache
from data.diagnostic_profiles import clear_diagnostic_profile_cache
from nlp.schemas import RetrievalCandidate
from tools.registry import get_tool_spec


def test_tool_registry_enriches_description_from_csv(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "profiles.csv"
    csv_path.write_text(
        "\n".join(
            [
                "id,code,name,description,defaultRequestedPidsJson,includeDtcsByDefault,createdAt,updatedAt",
                '1,cooling_overheating__car_overheats__under_load,Overheating,System: Cooling system. Overheating.,"[{""key"": ""coolant_temp_c"", ""pid"": ""05"", ""mode"": ""01"", ""priority"": 1}]",True,2026-04-13,2026-04-13',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DIAGNOSTIC_PROFILE_CSV_PATH", str(csv_path))
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "mock")
    clear_settings_cache()
    clear_embedding_cache()
    clear_diagnostic_profile_cache()

    spec = get_tool_spec("get_latest_signals")

    assert "Common profile domains:" in spec.description
    assert "Signal families:" in spec.description
    assert "cooling overheating" in spec.profile_text


def test_tool_default_step_uses_supporting_profiles_when_required_signals_are_missing() -> None:
    candidate = RetrievalCandidate(
        candidate_id="ENGINE_TEMPERATURE_CHECK",
        public_intent="CHECK_SIGNAL_STATUS",
        goal="ENGINE_TEMPERATURE_CHECK",
        score=0.8,
        metadata={
            "required_signals": [],
            "default_parameters": {},
            "supporting_profiles": [
                {
                    "requested_pids": [
                        {"key": "coolant_temp_c", "pid": "05", "mode": "01", "priority": 1},
                        {"key": "engine_rpm", "pid": "0C", "mode": "01", "priority": 2},
                    ]
                }
            ],
        },
    )

    step = get_tool_spec("get_latest_signals").default_step_for_candidate(candidate)

    assert step is not None
    assert "coolant_temp" in step.arguments["signals"]
    assert "rpm" in step.arguments["signals"]


def test_tool_default_step_falls_back_to_goal_summary_signal_keys() -> None:
    candidate = RetrievalCandidate(
        candidate_id="BATTERY_CHECK",
        public_intent="CHECK_SIGNAL_STATUS",
        goal="BATTERY_CHECK",
        score=0.8,
        metadata={
            "required_signals": [],
            "default_parameters": {},
            "supporting_profiles": [],
            "goal_profile_summary": {
                "requested_signal_keys": ["control_module_voltage_v", "engine_rpm"],
            },
        },
    )

    step = get_tool_spec("get_latest_signals").default_step_for_candidate(candidate)

    assert step is not None
    assert "module_voltage" in step.arguments["signals"]
    assert "rpm" in step.arguments["signals"]


def test_tool_default_step_uses_public_fallback_for_broad_vehicle_health() -> None:
    candidate = RetrievalCandidate(
        candidate_id="VEHICLE_HEALTH_CHECK",
        public_intent="CHECK_ENGINE_HEALTH",
        goal="VEHICLE_HEALTH_CHECK",
        score=0.8,
        metadata={
            "required_signals": [],
            "default_parameters": {},
            "supporting_profiles": [],
            "goal_profile_summary": None,
        },
    )

    step = get_tool_spec("get_latest_signals").default_step_for_candidate(candidate)

    assert step is not None
    assert step.arguments["signals"] == ["rpm", "engine_load", "coolant_temp"]
