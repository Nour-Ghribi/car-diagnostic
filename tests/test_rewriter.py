from __future__ import annotations

from nlp.prompt_builder import build_rewriter_messages
from nlp.llm_rewriter import rewrite_prompt


def test_rewriter_cleans_noisy_broad_prompt() -> None:
    rewritten = rewrite_prompt("je vx conaitre l etat de ma voiture")
    assert rewritten.preserved_meaning is True
    assert rewritten.rewritten_prompt
    assert rewritten.language in {"fr", "mixed"}


def test_rewriter_marks_generic_non_automotive_prompt_as_clarification() -> None:
    rewritten = rewrite_prompt("it's weird")
    assert rewritten.needs_user_clarification is True
    assert rewritten.clarification_question


def test_rewriter_falls_back_cleanly_when_openrouter_has_no_key(monkeypatch) -> None:
    monkeypatch.setenv("INTENT_REWRITER_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    rewritten = rewrite_prompt("check my car health")
    assert rewritten.rewritten_prompt
    assert rewritten.preserved_meaning is True


def test_rewriter_messages_forbid_creative_rephrasing() -> None:
    messages = build_rewriter_messages("battery issue maybe")
    system = messages[0]["content"]
    assert "Do not turn the prompt into a new question" in system
    assert "Prefer minimal edits" in system


def test_rewriter_preserves_specific_vin_prompt() -> None:
    rewritten = rewrite_prompt("get vin")
    assert rewritten.rewritten_prompt == "get vin"


def test_rewriter_preserves_specific_rpm_prompt() -> None:
    rewritten = rewrite_prompt("show rpm")
    assert rewritten.rewritten_prompt == "show rpm"
