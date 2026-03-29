from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


IntentName = Literal[
    "READ_DTC",
    "CHECK_CYLINDER",
    "CHECK_ENGINE_HEALTH",
    "CHECK_SIGNAL_STATUS",
    "EXPLAIN_WARNING_LIGHT",
    "GET_VEHICLE_CONTEXT",
    "UNKNOWN",
]

GoalName = Literal[
    "VEHICLE_HEALTH_CHECK",
    "READ_DTC",
    "CYLINDER_CHECK",
    "SIGNAL_STATUS_CHECK",
    "WARNING_LIGHT_CHECK",
    "BATTERY_CHECK",
    "ENGINE_TEMPERATURE_CHECK",
    "PERFORMANCE_ISSUE_CHECK",
    "STARTING_PROBLEM_CHECK",
    "FUEL_CONSUMPTION_CHECK",
    "VEHICLE_CONTEXT_LOOKUP",
    "UNKNOWN",
]

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

DetailLevel = Literal["low", "medium", "high"]
ScopeLevel = Literal["broad", "specific"]
ResolutionPolicy = Literal["ACCEPT", "ACCEPT_BROAD_GOAL", "CLARIFICATION_NEEDED", "UNKNOWN"]
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
DataSource = Literal["push_cache", "db", "on_demand"]
MissingReason = Literal["unsupported", "not_collected", "stale", "timeout", "no_data"]
MissingImpact = Literal["confidence_reduced", "diagnosis_limited", "followup_needed"]
WarningType = Literal["check_engine"]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SignalValue(StrictBaseModel):
    value: float | str | bool | None
    unit: str | None = None
    observed_ts: datetime
    source: DataSource


class DTCBundle(StrictBaseModel):
    stored: list[str] = Field(default_factory=list)
    pending: list[str] = Field(default_factory=list)
    permanent: list[str] = Field(default_factory=list)


class RequestConstraints(StrictBaseModel):
    allow_on_demand: bool = True
    max_latency_ms: int | None = Field(default=6000, ge=0)
    max_tool_calls: int | None = Field(default=8, ge=1)


class PrefetchData(StrictBaseModel):
    signals: dict[SignalName, SignalValue] = Field(default_factory=dict)
    dtc: DTCBundle | None = None


class IntentParameters(StrictBaseModel):
    detail: DetailLevel | None = None
    goal: GoalName | None = None
    scope: ScopeLevel | None = None
    resolution_policy: ResolutionPolicy | None = None
    cylinder_index: int | None = Field(default=None, ge=1)
    bank: int | None = Field(default=None, ge=1)
    include_pending: bool = True
    include_permanent: bool = False
    signal: SignalName | None = None
    max_age_ms: int | None = Field(default=None, ge=0)
    warning_type: WarningType | None = None
    include_calibration: bool = True
    refresh_capabilities: bool = False
    clarification_question: str | None = None


class Intent(StrictBaseModel):
    name: IntentName
    confidence: float = Field(ge=0.0, le=1.0)
    parameters: IntentParameters = Field(default_factory=IntentParameters)


class AgentRequest(StrictBaseModel):
    request_id: str = Field(min_length=1)
    ts: datetime
    vehicle_id: str = Field(min_length=1)
    session_id: str | None = None
    user_prompt: str = Field(min_length=1)
    locale: str = "fr"
    constraints: RequestConstraints | None = None
    prefetch: PrefetchData | None = None


class ToolOptions(StrictBaseModel):
    max_age_ms: int | None = Field(default=None, ge=0)
    mode06_optional: bool = False
    include_pending: bool = True
    detail: DetailLevel | None = None
    cylinder_index: int | None = Field(default=None, ge=1)


class ToolRequest(StrictBaseModel):
    request_id: str = Field(min_length=1)
    tool_name: ToolName
    vehicle_id: str = Field(min_length=1)
    signal_keys: list[SignalName] = Field(default_factory=list)
    options: ToolOptions = Field(default_factory=ToolOptions)


class MissingData(StrictBaseModel):
    key: str = Field(min_length=1)
    reason: MissingReason
    impact: MissingImpact | None = None


class VehicleContext(StrictBaseModel):
    vin: str | None = None
    ecu_name: str | None = None
    calibration_id: str | None = None
    observed_ts: datetime | None = None
    source: DataSource | None = None


class Capabilities(StrictBaseModel):
    supported_signals: list[SignalName] = Field(default_factory=list)
    mode06_supported: bool = False


class Metrics(StrictBaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    coherent: bool


class ToolResponseData(StrictBaseModel):
    signals: dict[str, Any] = Field(default_factory=dict)
    dtc: DTCBundle | None = None
    mode06: dict[str, Any] | None = None
    vehicle_context: VehicleContext | None = None
    capabilities: Capabilities | None = None
    metrics: Metrics | None = None


class ToolResponse(StrictBaseModel):
    request_id: str = Field(min_length=1)
    tool_name: ToolName
    status: ToolStatus
    data: ToolResponseData
    missing_data: list[MissingData] = Field(default_factory=list)
    error_message: str | None = None


class Evidence(StrictBaseModel):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: Any
    unit: str | None = None
    observed_ts: datetime
    source: DataSource


class AgentResponse(StrictBaseModel):
    request_id: str = Field(min_length=1)
    ts: datetime
    vehicle_id: str = Field(min_length=1)
    intent: Intent
    diagnosis: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    signals_used: list[SignalName] = Field(default_factory=list)
    actions_taken: list[ToolName] = Field(default_factory=list)
    missing_data: list[MissingData] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


IntentParameters.model_rebuild()
Intent.model_rebuild()
AgentRequest.model_rebuild()
ToolOptions.model_rebuild()
ToolRequest.model_rebuild()
MissingData.model_rebuild()
VehicleContext.model_rebuild()
Capabilities.model_rebuild()
Metrics.model_rebuild()
ToolResponseData.model_rebuild()
ToolResponse.model_rebuild()
Evidence.model_rebuild()
AgentResponse.model_rebuild()
