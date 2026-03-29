from __future__ import annotations

import pytest

from agent.intent_parser_llm import (
    IntentParserError,
    parse_intent_hybrid,
    parse_intent_with_llm,
    validate_llm_intent_payload,
)


def test_parse_intent_with_llm_check_cylinder_2() -> None:
    intent = parse_intent_with_llm("check cylinder 2")
    assert intent.name == "CHECK_CYLINDER"
    assert intent.parameters.cylinder_index == 2
    assert 0.0 <= intent.confidence <= 1.0


def test_parse_intent_with_llm_read_dtc() -> None:
    intent = parse_intent_with_llm("read dtc")
    assert intent.name == "READ_DTC"
    assert intent.parameters.include_pending is True


def test_parse_intent_with_llm_show_rpm() -> None:
    intent = parse_intent_with_llm("show rpm")
    assert intent.name == "CHECK_SIGNAL_STATUS"
    assert intent.parameters.signal == "rpm"


def test_parse_intent_with_llm_vin() -> None:
    intent = parse_intent_with_llm("vin")
    assert intent.name == "GET_VEHICLE_CONTEXT"


def test_parse_intent_with_llm_ambiguous_prompt_returns_unknown_or_valid_fallback() -> None:
    intent = parse_intent_hybrid("please do something unclear for the car")
    assert intent.name in {
        "UNKNOWN",
        "READ_DTC",
        "CHECK_CYLINDER",
        "CHECK_ENGINE_HEALTH",
        "CHECK_SIGNAL_STATUS",
        "EXPLAIN_WARNING_LIGHT",
        "GET_VEHICLE_CONTEXT",
    }


def test_validate_llm_intent_payload_rejects_unsupported_signal() -> None:
    payload = {
        "name": "CHECK_SIGNAL_STATUS",
        "confidence": 0.9,
        "parameters": {"signal": "maf"},
    }
    with pytest.raises(IntentParserError):
        validate_llm_intent_payload(payload)


def test_validate_llm_intent_payload_rejects_unsupported_intent() -> None:
    payload = {"name": "REPAIR_CAR", "confidence": 0.9, "parameters": {}}
    with pytest.raises(IntentParserError):
        validate_llm_intent_payload(payload)


def test_parse_intent_hybrid_falls_back_when_llm_payload_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent import intent_parser_llm

    monkeypatch.setattr(
        intent_parser_llm,
        "mock_llm_response",
        lambda prompt: {"name": "CHECK_SIGNAL_STATUS", "confidence": 0.9, "parameters": {"signal": "maf"}},
    )
    intent = parse_intent_hybrid("show rpm")
    assert intent.name == "CHECK_SIGNAL_STATUS"
    assert intent.parameters.signal == "rpm"
