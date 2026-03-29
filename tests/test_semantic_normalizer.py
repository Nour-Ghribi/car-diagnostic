from __future__ import annotations

from agent.semantic_normalizer import normalize_prompt


def test_normalize_prompt_detects_broad_vehicle_health() -> None:
    normalized = normalize_prompt("je veux connaître le health de ma voiture")
    assert normalized.automotive_context is True
    assert normalized.broad_request is True
    assert "vehicle" in normalized.detected_keywords
    assert "health" in normalized.detected_keywords


def test_normalize_prompt_handles_misspelling_noise() -> None:
    normalized = normalize_prompt("je vx conaitre l etat de ma voiture")
    assert normalized.automotive_context is True
    assert normalized.broad_request is True


def test_normalize_prompt_flags_unsupported_repair_request() -> None:
    normalized = normalize_prompt("please repair my whole car automatically")
    assert normalized.unsupported_action is True
