from __future__ import annotations

import pytest

from agent.ambiguity import detect_ambiguity
from agent.confidence import compute_intent_confidence
from agent.embedding_provider import MockEmbeddingProvider, embed_intent_cards, embed_prompt
from agent.intent_index import get_intent_cards
from agent.intent_parser_v3 import parse_intent_v3, parse_intent_v3_detailed
from agent.reranker import validate_reranker_payload
from agent.retriever import retrieve_top_k_intents
from backend.schemas import Intent, IntentParameters


def test_parse_intent_v3_engine_health() -> None:
    intent = parse_intent_v3("je veux connaître l'état du moteur")
    assert intent.name == "CHECK_ENGINE_HEALTH"
    assert intent.parameters.goal == "VEHICLE_HEALTH_CHECK"
    assert 0.0 <= intent.confidence <= 1.0


def test_parse_intent_v3_broad_vehicle_health_is_accepted() -> None:
    result = parse_intent_v3_detailed("je veux connaître le health de ma voiture")
    assert result.intent.name == "CHECK_ENGINE_HEALTH"
    assert result.intent.parameters.goal == "VEHICLE_HEALTH_CHECK"
    assert result.intent.parameters.scope == "broad"
    assert result.intent.parameters.resolution_policy == "ACCEPT_BROAD_GOAL"


def test_parse_intent_v3_misspelled_vehicle_health_is_accepted() -> None:
    intent = parse_intent_v3("je vx conaitre l etat de ma voiture")
    assert intent.name == "CHECK_ENGINE_HEALTH"
    assert intent.parameters.goal == "VEHICLE_HEALTH_CHECK"
    assert intent.parameters.scope == "broad"


def test_parse_intent_v3_mixed_engine_health_phrase_is_accepted() -> None:
    intent = parse_intent_v3("is the moteur is good")
    assert intent.name == "CHECK_ENGINE_HEALTH"
    assert intent.parameters.goal == "VEHICLE_HEALTH_CHECK"


def test_parse_intent_v3_read_dtc() -> None:
    intent = parse_intent_v3("read dtc")
    assert intent.name == "READ_DTC"
    assert intent.parameters.goal == "READ_DTC"
    assert intent.parameters.include_pending is True


def test_parse_intent_v3_show_rpm() -> None:
    intent = parse_intent_v3("show rpm")
    assert intent.name == "CHECK_SIGNAL_STATUS"
    assert intent.parameters.goal == "SIGNAL_STATUS_CHECK"
    assert intent.parameters.signal == "rpm"


def test_parse_intent_v3_get_vin() -> None:
    intent = parse_intent_v3("get vin")
    assert intent.name == "GET_VEHICLE_CONTEXT"
    assert intent.parameters.goal == "VEHICLE_CONTEXT_LOOKUP"


def test_parse_intent_v3_battery_prompt() -> None:
    intent = parse_intent_v3("why is my battery weak")
    assert intent.name == "CHECK_SIGNAL_STATUS"
    assert intent.parameters.goal == "BATTERY_CHECK"
    assert intent.parameters.signal == "module_voltage"


def test_parse_intent_v3_temperature_prompt() -> None:
    intent = parse_intent_v3("engine temperature too high")
    assert intent.name == "CHECK_SIGNAL_STATUS"
    assert intent.parameters.goal == "ENGINE_TEMPERATURE_CHECK"
    assert intent.parameters.signal == "coolant_temp"


def test_parse_intent_v3_generic_symptom_prefers_clarification() -> None:
    intent = parse_intent_v3("it's weird")
    assert intent.parameters.resolution_policy == "CLARIFICATION_NEEDED"
    assert intent.parameters.clarification_question


def test_parse_intent_v3_generic_doubt_stays_generic() -> None:
    intent = parse_intent_v3("j ai un doute")
    assert intent.name == "UNKNOWN"
    assert intent.parameters.resolution_policy == "CLARIFICATION_NEEDED"


def test_parse_intent_v3_unsupported_prompt_returns_unknown() -> None:
    intent = parse_intent_v3("please repair everything automatically")
    assert intent.name == "UNKNOWN"
    assert intent.parameters.resolution_policy == "UNKNOWN"


def test_top_k_retrieval_output_shape() -> None:
    provider = MockEmbeddingProvider()
    cards = get_intent_cards()
    indexed = embed_intent_cards(provider, cards)
    prompt_vector = embed_prompt(provider, "show rpm")
    candidates = retrieve_top_k_intents(prompt_vector, indexed, top_k=4)
    assert len(candidates) == 4
    assert candidates[0].rank == 1
    assert hasattr(candidates[0], "goal_name")
    assert hasattr(candidates[0], "intent_name")
    assert hasattr(candidates[0], "similarity_score")


def test_reranker_validation_accepts_valid_payload() -> None:
    payload = {
        "goal": "SIGNAL_STATUS_CHECK",
        "intent": {"name": "CHECK_SIGNAL_STATUS", "scope": "specific", "parameters": {"signal": "rpm", "max_age_ms": 2000}},
        "rerank_score": 0.82,
        "rationale_short": "Prompt asks for one signal value.",
        "clarification_question": None,
    }
    result = validate_reranker_payload(payload, allowed_candidate_names=["SIGNAL_STATUS_CHECK", "VEHICLE_HEALTH_CHECK"])
    assert result.goal_name == "SIGNAL_STATUS_CHECK"
    assert result.intent_name == "CHECK_SIGNAL_STATUS"
    assert result.parameters.signal == "rpm"


def test_ambiguity_detection_accepts_broad_vehicle_prompt() -> None:
    report = detect_ambiguity(
        prompt="diagnostic voiture",
        top_similarity=0.63,
        second_similarity=0.58,
        rerank_score=0.61,
        selected_goal="VEHICLE_HEALTH_CHECK",
        selected_scope="broad",
    )
    assert report.policy in {"ACCEPT_BROAD_GOAL", "CLARIFICATION_NEEDED"}


def test_intent_confidence_is_clipped_to_valid_range() -> None:
    report = compute_intent_confidence(
        calibrated_probability_proxy=0.95,
        rerank_score=0.90,
        retrieval_similarity=0.88,
        top1_score=0.88,
        top2_score=0.68,
        ambiguity_penalty=0.05,
        automotive_context=True,
        broad_request=False,
    )
    assert 0.0 <= report.score <= 1.0


def test_parse_intent_v3_falls_back_if_pipeline_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent import intent_parser_v3

    monkeypatch.setattr(intent_parser_v3, "parse_intent_v3_detailed", lambda prompt: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(
        intent_parser_v3,
        "_fallback_parse",
        lambda prompt: Intent(name="READ_DTC", confidence=0.8, parameters=IntentParameters(include_pending=True)),
    )
    intent = intent_parser_v3.parse_intent_v3("read dtc")
    assert intent.name == "READ_DTC"
