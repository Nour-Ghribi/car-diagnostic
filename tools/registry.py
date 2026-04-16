from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field

from agent import tools as legacy_tools
from backend.schemas import SignalName, ToolName
from data.diagnostic_profiles import get_diagnostic_profiles
from data.signal_mapping import signal_key_to_category, signal_key_to_signal_name, signal_key_to_tool_name
from nlp.schemas import PlanStep, RetrievalCandidate


class EmptyArgs(BaseModel):
    pass


class SignalArgs(BaseModel):
    signals: list[SignalName] = Field(default_factory=list)


class DTCArgs(BaseModel):
    include_pending: bool = True


@dataclass(frozen=True)
class ToolSpec:
    name: ToolName
    description: str
    arg_model: type[BaseModel]
    execute: Callable[..., dict[str, Any]]
    profile_text: str

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments_schema": self.arg_model.model_json_schema(),
        }

    def default_step_for_candidate(self, candidate: RetrievalCandidate) -> PlanStep | None:
        metadata = candidate.metadata
        required_signals = list(metadata.get("required_signals", []))
        default_parameters = dict(metadata.get("default_parameters", {}))
        if not required_signals:
            required_signals = _signals_from_supporting_profiles(candidate, tool_name=self.name)

        if self.name in {"get_vehicle_context", "request_mode06", "score_confidence"}:
            return PlanStep(tool=self.name, arguments={})
        if self.name == "get_dtcs":
            return PlanStep(tool=self.name, arguments={"include_pending": bool(default_parameters.get("include_pending", True))})
        if self.name in {"get_latest_signals", "request_fresh_signals"}:
            if not required_signals and default_parameters.get("signal"):
                required_signals = [default_parameters["signal"]]
            if not required_signals:
                return None
            return PlanStep(tool=self.name, arguments={"signals": required_signals})
        return None


def get_tool_registry() -> tuple[ToolSpec, ...]:
    enrichments = _tool_enrichments()
    return (
        ToolSpec(
            name="get_vehicle_context",
            description=_merge_tool_description(
                "Retrieve VIN, ECU metadata, calibration id, and overall vehicle context.",
                enrichments.get("get_vehicle_context", {}),
            ),
            arg_model=EmptyArgs,
            execute=legacy_tools.get_vehicle_context,
            profile_text=_merge_tool_profile_text(
                "vehicle context vin ecu calibration metadata identity overall diagnostic context",
                enrichments.get("get_vehicle_context", {}),
            ),
        ),
        ToolSpec(
            name="get_dtcs",
            description=_merge_tool_description(
                "Read stored, pending, and permanent diagnostic trouble codes from the ECU.",
                enrichments.get("get_dtcs", {}),
            ),
            arg_model=DTCArgs,
            execute=legacy_tools.get_dtcs,
            profile_text=_merge_tool_profile_text(
                "read dtc diagnostic trouble codes stored pending permanent faults warning light engine issue",
                enrichments.get("get_dtcs", {}),
            ),
        ),
        ToolSpec(
            name="get_latest_signals",
            description=_merge_tool_description(
                "Read latest cached live signals such as rpm, engine load, coolant temperature, oxygen sensor, speed, and module voltage.",
                enrichments.get("get_latest_signals", {}),
            ),
            arg_model=SignalArgs,
            execute=legacy_tools.get_latest_signals,
            profile_text=_merge_tool_profile_text(
                "latest cached live signals rpm engine load coolant temperature oxygen sensor speed module voltage telemetry",
                enrichments.get("get_latest_signals", {}),
            ),
        ),
        ToolSpec(
            name="request_fresh_signals",
            description=_merge_tool_description(
                "Request fresh on-demand live signals when cached values are missing, stale, or insufficient.",
                enrichments.get("request_fresh_signals", {}),
            ),
            arg_model=SignalArgs,
            execute=legacy_tools.request_fresh_signals,
            profile_text=_merge_tool_profile_text(
                "fresh on-demand live signals fallback stale missing telemetry current snapshot",
                enrichments.get("request_fresh_signals", {}),
            ),
        ),
        ToolSpec(
            name="request_mode06",
            description=_merge_tool_description(
                "Request mode 06 monitor data for deeper combustion and monitor analysis when relevant.",
                enrichments.get("request_mode06", {}),
            ),
            arg_model=EmptyArgs,
            execute=legacy_tools.request_mode06,
            profile_text=_merge_tool_profile_text(
                "mode06 monitor data combustion misfire cylinder advanced diagnostic",
                enrichments.get("request_mode06", {}),
            ),
        ),
        ToolSpec(
            name="score_confidence",
            description=_merge_tool_description(
                "Compute final diagnostic confidence from observation completeness, missing data, and signal coherence.",
                enrichments.get("score_confidence", {}),
            ),
            arg_model=EmptyArgs,
            execute=legacy_tools.score_confidence,
            profile_text=_merge_tool_profile_text(
                "confidence scoring completeness missing data coherence final diagnostic certainty",
                enrichments.get("score_confidence", {}),
            ),
        ),
    )


def get_tool_spec(tool_name: ToolName) -> ToolSpec:
    for spec in get_tool_registry():
        if spec.name == tool_name:
            return spec
    raise KeyError(f"Unknown tool: {tool_name}")


def _signals_from_supporting_profiles(candidate: RetrievalCandidate, *, tool_name: ToolName) -> list[SignalName]:
    profiles = candidate.metadata.get("supporting_profiles", [])
    if not profiles:
        profiles = candidate.metadata.get("supporting_profiles_global", [])
    ordered_signals = _collect_signals_from_profiles(profiles, tool_name=tool_name)
    if ordered_signals:
        return ordered_signals

    ordered_signals = _collect_signals_from_goal_summary(candidate, tool_name=tool_name)
    if ordered_signals:
        return ordered_signals

    return _fallback_signals_for_goal(candidate.goal, tool_name=tool_name)


def _collect_signals_from_profiles(profiles: list[object], *, tool_name: ToolName) -> list[SignalName]:
    ordered_signals: list[SignalName] = []
    seen: set[SignalName] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        for item in profile.get("requested_pids", []):
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", ""))
            mapped_tool = signal_key_to_tool_name(key)
            signal_name = signal_key_to_signal_name(key)
            if mapped_tool != tool_name or signal_name is None or signal_name in seen:
                continue
            seen.add(signal_name)
            ordered_signals.append(signal_name)
            if len(ordered_signals) >= 5:
                return ordered_signals
    return ordered_signals


def _collect_signals_from_goal_summary(candidate: RetrievalCandidate, *, tool_name: ToolName) -> list[SignalName]:
    summary = candidate.metadata.get("goal_profile_summary")
    if not isinstance(summary, dict):
        return []
    ordered_signals: list[SignalName] = []
    seen: set[SignalName] = set()
    for key in summary.get("requested_signal_keys", []):
        mapped_tool = signal_key_to_tool_name(str(key))
        signal_name = signal_key_to_signal_name(str(key))
        if mapped_tool != tool_name or signal_name is None or signal_name in seen:
            continue
        seen.add(signal_name)
        ordered_signals.append(signal_name)
        if len(ordered_signals) >= 5:
            break
    return ordered_signals


def _fallback_signals_for_goal(goal_name: str, tool_name: ToolName) -> list[SignalName]:
    if tool_name not in {"get_latest_signals", "request_fresh_signals"}:
        return []
    if goal_name == "VEHICLE_HEALTH_CHECK":
        return ["rpm", "engine_load", "coolant_temp"]
    if goal_name == "CYLINDER_CHECK":
        return ["rpm", "engine_load", "stft_b1", "ltft_b1", "o2_b1s1"]
    if goal_name == "VEHICLE_CONTEXT_LOOKUP":
        return []
    return []


def _tool_enrichments() -> dict[ToolName, dict[str, object]]:
    profiles = get_diagnostic_profiles()
    enrichments: dict[ToolName, dict[str, object]] = {
        "get_vehicle_context": {"domains": [], "symptoms": [], "signals": [], "categories": []},
        "get_dtcs": {"domains": [], "symptoms": [], "signals": [], "categories": []},
        "get_latest_signals": {"domains": [], "symptoms": [], "signals": [], "categories": []},
        "request_fresh_signals": {"domains": [], "symptoms": [], "signals": [], "categories": []},
        "request_mode06": {"domains": [], "symptoms": [], "signals": [], "categories": []},
        "score_confidence": {"domains": [], "symptoms": [], "signals": [], "categories": []},
    }
    if not profiles:
        return enrichments

    for profile in profiles:
        if profile.include_dtcs:
            _append_unique(enrichments["get_dtcs"]["domains"], profile.domain)
            _append_unique(enrichments["get_dtcs"]["symptoms"], profile.symptom)
            _append_unique(enrichments["get_dtcs"]["categories"], "dtc_context")
        if profile.include_dtcs or profile.requested_pids:
            _append_unique(enrichments["score_confidence"]["domains"], profile.domain)
            _append_unique(enrichments["score_confidence"]["categories"], "confidence_context")
        if _looks_like_context_profile(profile):
            _append_unique(enrichments["get_vehicle_context"]["domains"], profile.domain)
            _append_unique(enrichments["get_vehicle_context"]["symptoms"], profile.symptom)
            _append_unique(enrichments["get_vehicle_context"]["categories"], "vehicle_context")
        if _looks_like_mode06_profile(profile):
            _append_unique(enrichments["request_mode06"]["domains"], profile.domain)
            _append_unique(enrichments["request_mode06"]["symptoms"], profile.symptom)
            _append_unique(enrichments["request_mode06"]["categories"], "combustion_core")

        for pid in profile.requested_pids:
            signal_name = signal_key_to_signal_name(pid.key)
            mapped_tool = signal_key_to_tool_name(pid.key)
            category = signal_key_to_category(pid.key)
            if signal_name is None or mapped_tool is None:
                continue
            for target_tool in ("get_latest_signals", "request_fresh_signals"):
                _append_unique(enrichments[target_tool]["signals"], signal_name)
                _append_unique(enrichments[target_tool]["domains"], profile.domain)
                _append_unique(enrichments[target_tool]["symptoms"], profile.symptom)
                if category:
                    _append_unique(enrichments[target_tool]["categories"], category)
            if mapped_tool == "get_latest_signals" and category:
                _append_unique(enrichments["get_latest_signals"]["categories"], category)

    return enrichments


def _merge_tool_description(base: str, enrichment: dict[str, object]) -> str:
    domains = list(enrichment.get("domains", []))[:4]
    categories = list(enrichment.get("categories", []))[:4]
    if not domains and not categories:
        return base
    suffix_parts: list[str] = []
    if domains:
        suffix_parts.append("Common profile domains: " + ", ".join(domains))
    if categories:
        suffix_parts.append("Signal families: " + ", ".join(categories))
    return base + " " + ". ".join(suffix_parts) + "."


def _merge_tool_profile_text(base: str, enrichment: dict[str, object]) -> str:
    domains = " ".join(str(item) for item in list(enrichment.get("domains", []))[:6])
    symptoms = " ".join(str(item) for item in list(enrichment.get("symptoms", []))[:6])
    signals = " ".join(str(item) for item in list(enrichment.get("signals", []))[:10])
    categories = " ".join(str(item) for item in list(enrichment.get("categories", []))[:8])
    extra = " ".join(part for part in (domains, symptoms, signals, categories) if part)
    return base if not extra else base + " " + extra


def _append_unique(target: object, value: str) -> None:
    if not isinstance(target, list):
        return
    normalized = value.strip()
    if not normalized or normalized in target:
        return
    target.append(normalized)


def _looks_like_context_profile(profile: object) -> bool:
    text = " ".join(
        [
            str(getattr(profile, "domain", "")),
            str(getattr(profile, "symptom", "")),
            str(getattr(profile, "context", "")),
            str(getattr(profile, "name", "")),
        ]
    ).lower()
    return any(token in text for token in ("overall", "global", "context", "healthy", "health"))


def _looks_like_mode06_profile(profile: object) -> bool:
    text = " ".join(
        [
            str(getattr(profile, "domain", "")),
            str(getattr(profile, "symptom", "")),
            str(getattr(profile, "description", "")),
            str(getattr(profile, "name", "")),
        ]
    ).lower()
    return any(token in text for token in ("misfire", "combustion", "ignition", "cylinder"))
