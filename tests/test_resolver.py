from __future__ import annotations

from nlp import llm_resolver
from nlp.prompt_builder import build_resolver_messages
from nlp.llm_resolver import resolve_and_plan
from nlp.llm_rewriter import rewrite_prompt
from nlp.retriever import retrieve_candidates


def test_resolver_returns_structured_plan_for_broad_health() -> None:
    prompt = "check my car health"
    rewritten = rewrite_prompt(prompt)
    candidates = retrieve_candidates(rewritten.rewritten_prompt)
    decision = resolve_and_plan(original_prompt=prompt, rewritten=rewritten, candidates=candidates)
    assert decision.selected_public_intent == "CHECK_ENGINE_HEALTH"
    assert decision.selected_goal == "VEHICLE_HEALTH_CHECK"
    assert decision.execution_plan
    assert any(step.tool == "get_dtcs" for step in decision.execution_plan)
    assert any(step.tool == "get_latest_signals" for step in decision.execution_plan)


def test_resolver_falls_back_cleanly_when_openrouter_has_no_key(monkeypatch) -> None:
    monkeypatch.setenv("INTENT_RESOLVER_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    prompt = "check my car health"
    rewritten = rewrite_prompt(prompt)
    candidates = retrieve_candidates(rewritten.rewritten_prompt)
    decision = resolve_and_plan(original_prompt=prompt, rewritten=rewritten, candidates=candidates)
    assert decision.selected_goal == "VEHICLE_HEALTH_CHECK"
    assert decision.execution_plan


def test_resolver_falls_back_when_remote_accepts_goal_without_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_resolver,
        "_call_openrouter_resolver",
        lambda **kwargs: {
            "selected_public_intent": "CHECK_ENGINE_HEALTH",
            "selected_goal": "VEHICLE_HEALTH_CHECK",
            "scope": "broad",
            "confidence": 0.75,
            "needs_user_clarification": False,
            "clarification_question": None,
            "reasoning_summary": "Accepted broad health request.",
            "execution_plan": [],
        },
    )
    monkeypatch.setattr(llm_resolver, "_call_gemini_resolver", lambda **kwargs: None)
    monkeypatch.setattr(llm_resolver, "_call_openai_resolver", lambda **kwargs: None)

    prompt = "check my car health"
    rewritten = rewrite_prompt(prompt)
    candidates = retrieve_candidates(rewritten.rewritten_prompt)
    decision = resolve_and_plan(original_prompt=prompt, rewritten=rewritten, candidates=candidates)
    assert decision.selected_goal == "VEHICLE_HEALTH_CHECK"
    assert decision.execution_plan


def test_resolver_messages_prefer_foundational_observations_for_broad_health() -> None:
    messages = build_resolver_messages(
        original_prompt="check my car health",
        rewritten_prompt="check my car health",
        candidates=retrieve_candidates("check my car health"),
        tools=(),
        vehicle_context=None,
    )
    system = messages[0]["content"]
    assert "vehicle context, DTCs, and current live signals" in system
    assert "before specialized checks like Mode 06" in system


def test_resolver_messages_include_supporting_profiles_payload() -> None:
    messages = build_resolver_messages(
        original_prompt="check my car health",
        rewritten_prompt="check my car health",
        candidates=retrieve_candidates("check my car health"),
        tools=(),
        vehicle_context=None,
    )
    payload = messages[1]["content"]
    assert "supporting_profiles_global" in payload
    assert "goal_profile_summaries" in payload


def test_local_resolver_uses_sober_plan_for_read_dtc() -> None:
    prompt = "read dtc"
    rewritten = rewrite_prompt(prompt)
    candidates = retrieve_candidates(rewritten.rewritten_prompt)
    decision = resolve_and_plan(original_prompt=prompt, rewritten=rewritten, candidates=candidates)

    assert decision.selected_goal == "READ_DTC"
    assert [step.tool for step in decision.execution_plan] == ["get_dtcs", "score_confidence"]
