from __future__ import annotations

from datetime import datetime, timezone

from agent.orchestrator import diagnose
from backend.schemas import AgentRequest


def test_hybrid_agent_pipeline_executes_noisy_broad_prompt_end_to_end() -> None:
    request = AgentRequest(
        request_id="e2e_hybrid_broad",
        ts=datetime.now(timezone.utc),
        vehicle_id="veh_002",
        user_prompt="je vx conaitre l etat de ma voiture",
    )
    response = diagnose(request)
    assert response.intent.name == "CHECK_ENGINE_HEALTH"
    assert response.intent.parameters.goal == "VEHICLE_HEALTH_CHECK"
    assert "get_dtcs" in response.actions_taken
    assert "get_latest_signals" in response.actions_taken
    assert response.diagnosis
