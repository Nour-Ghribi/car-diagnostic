from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
MOCK_DIR = BASE_DIR / "fake_data" / "tool_responses"


@dataclass(frozen=True)
class MockRuntimeScenario:
    vehicle_id: str
    scenario_family: str
    description: str
    diagnostic_theme: str
    expected_findings: tuple[str, ...]
    profile_tags: tuple[str, ...]
    capabilities: dict[str, Any]
    vehicle_context: dict[str, Any]
    latest_signals: dict[str, Any]
    fresh_signals: dict[str, Any]
    dtc: dict[str, Any]
    mode06_response: str
    mode06: dict[str, Any] | None

    def to_runtime_payload(self) -> dict[str, Any]:
        return {
            "scenario_family": self.scenario_family,
            "description": self.description,
            "diagnostic_theme": self.diagnostic_theme,
            "expected_findings": list(self.expected_findings),
            "profile_tags": list(self.profile_tags),
            "capabilities": deepcopy(self.capabilities),
            "vehicle_context": deepcopy(self.vehicle_context),
            "latest_signals": deepcopy(self.latest_signals),
            "fresh_signals": deepcopy(self.fresh_signals),
            "dtc": deepcopy(self.dtc),
            "mode06_response": self.mode06_response,
            "mode06": deepcopy(self.mode06),
        }


def load_mock_runtime_scenarios() -> dict[str, MockRuntimeScenario]:
    payload = _load_vehicle_profile_payload()
    vehicles = payload.get("vehicles", {})
    scenarios: dict[str, MockRuntimeScenario] = {}
    for vehicle_id, raw in vehicles.items():
        if not isinstance(raw, dict):
            continue
        scenarios[vehicle_id] = MockRuntimeScenario(
            vehicle_id=vehicle_id,
            scenario_family=str(raw.get("scenario_family", "baseline")),
            description=str(raw.get("description", "")).strip(),
            diagnostic_theme=str(raw.get("diagnostic_theme", "general_diagnostics")).strip(),
            expected_findings=tuple(_normalize_string_list(raw.get("expected_findings", []))),
            profile_tags=tuple(_normalize_string_list(raw.get("profile_tags", []))),
            capabilities=deepcopy(raw.get("capabilities", {})),
            vehicle_context=deepcopy(raw.get("vehicle_context", {})),
            latest_signals=deepcopy(raw.get("latest_signals", {})),
            fresh_signals=deepcopy(raw.get("fresh_signals", {})),
            dtc=deepcopy(raw.get("dtc", {})),
            mode06_response=str(raw.get("mode06_response", "no_data")),
            mode06=deepcopy(raw.get("mode06")),
        )
    return scenarios


def get_mock_runtime_scenario(vehicle_id: str) -> MockRuntimeScenario:
    scenarios = load_mock_runtime_scenarios()
    return scenarios.get(vehicle_id) or scenarios["veh_001"]


def list_mock_runtime_scenarios() -> tuple[MockRuntimeScenario, ...]:
    scenarios = load_mock_runtime_scenarios()
    return tuple(scenarios[key] for key in sorted(scenarios))


def _load_vehicle_profile_payload() -> dict[str, Any]:
    with (MOCK_DIR / "vehicle_profiles.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for item in values:
        value = str(item).strip()
        if value:
            normalized.append(value)
    return normalized
