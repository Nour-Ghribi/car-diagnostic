from __future__ import annotations

from dataclasses import fields

from agent.intent_index import get_intent_card, render_intent_card


def test_intent_card_is_now_a_thin_public_contract() -> None:
    field_names = {field.name for field in fields(get_intent_card("VEHICLE_HEALTH_CHECK"))}
    assert field_names == {
        "goal_name",
        "intent_name",
        "default_scope",
        "description",
        "expected_parameters",
        "clarification_question",
        "default_parameters",
    }


def test_render_intent_card_only_renders_contract_fields() -> None:
    card = get_intent_card("SIGNAL_STATUS_CHECK")
    rendered = render_intent_card(card)

    assert "goal: SIGNAL_STATUS_CHECK" in rendered
    assert "intent: CHECK_SIGNAL_STATUS" in rendered
    assert "expected_parameters:" in rendered
    assert "default_parameters:" in rendered
    assert "examples_fr:" not in rendered
    assert "examples_en:" not in rendered
    assert "anti_examples:" not in rendered
    assert "required_signals:" not in rendered
    assert "semantic_hints:" not in rendered


def test_legacy_properties_remain_available_but_empty() -> None:
    card = get_intent_card("READ_DTC")
    assert card.examples_fr == ()
    assert card.examples_en == ()
    assert card.anti_examples == ()
    assert card.required_signals == ()
    assert card.semantic_hints == ()
