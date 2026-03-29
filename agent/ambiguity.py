from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.semantic_normalizer import NormalizedPrompt


DecisionPolicy = Literal["ACCEPT", "ACCEPT_BROAD_GOAL", "CLARIFICATION_NEEDED", "UNKNOWN"]


@dataclass(frozen=True)
class AmbiguityReport:
    """Structured ambiguity report for the hybrid goal-intent pipeline."""

    is_ambiguous: bool
    level: str
    policy: DecisionPolicy
    reasons: tuple[str, ...]
    top_similarity: float
    second_similarity: float
    margin: float
    rerank_score: float
    prompt_token_count: int
    automotive_context: bool
    broad_request: bool
    generic_prompt: bool
    underspecified: bool
    clarification_question: str | None
    penalty: float


def detect_ambiguity(
    *,
    normalized_prompt: NormalizedPrompt | None = None,
    prompt: str | None = None,
    top_similarity: float,
    second_similarity: float,
    rerank_score: float,
    selected_goal: str = "UNKNOWN",
    selected_scope: str = "specific",
    clarification_question: str | None = None,
) -> AmbiguityReport:
    """
    Detect ambiguity while preserving broad but valid automotive requests.

    Broad vehicle-health wording is treated as a valid broad goal, not as a
    failure, as long as the prompt still carries automotive meaning.
    """
    resolved_prompt = normalized_prompt or normalize_for_compat(prompt)
    tokens = [token for token in resolved_prompt.normalized_text.split(" ") if token]
    margin = max(0.0, top_similarity - second_similarity)
    generic_prompt = resolved_prompt.broad_request and selected_scope == "broad"
    underspecified = len(tokens) <= 2 and not resolved_prompt.automotive_context

    reasons: list[str] = []
    if top_similarity < 0.42:
        reasons.append("low_top_similarity")
    if margin < 0.05:
        reasons.append("low_retrieval_margin")
    if rerank_score < 0.48:
        reasons.append("low_rerank_score")
    if underspecified:
        reasons.append("underspecified_prompt")
    if resolved_prompt.spelling_noise:
        reasons.append("spelling_noise_detected")
    if resolved_prompt.unsupported_action:
        reasons.append("unsupported_action_request")

    if resolved_prompt.unsupported_action:
        policy = "UNKNOWN"
        level = "high"
        penalty = 0.45
    elif resolved_prompt.broad_request and resolved_prompt.automotive_context:
        policy: DecisionPolicy = "ACCEPT_BROAD_GOAL"
        level = "medium" if reasons else "low"
        penalty = 0.12 if reasons else 0.04
    elif selected_scope == "specific" and top_similarity >= 0.55 and rerank_score >= 0.62:
        policy = "ACCEPT"
        level = "low"
        penalty = 0.04
    elif top_similarity >= 0.60 and rerank_score >= 0.60 and margin >= 0.06:
        policy = "ACCEPT"
        level = "low"
        penalty = 0.04
    elif resolved_prompt.automotive_context and (top_similarity >= 0.32 or rerank_score >= 0.34):
        policy = "CLARIFICATION_NEEDED"
        level = "medium"
        penalty = 0.22
        if "low_top_similarity" not in reasons:
            reasons.append("clarification_preferred")
    elif "doubt" in resolved_prompt.detected_keywords and (top_similarity >= 0.24 or rerank_score >= 0.24):
        policy = "CLARIFICATION_NEEDED"
        level = "medium"
        penalty = 0.26
        reasons.append("generic_symptom_without_context")
    else:
        policy = "UNKNOWN"
        level = "high"
        penalty = 0.42
        if "non_automotive_or_too_weak" not in reasons:
            reasons.append("non_automotive_or_too_weak")

    return AmbiguityReport(
        is_ambiguous=policy in {"CLARIFICATION_NEEDED", "UNKNOWN"} or bool(reasons),
        level=level,
        policy=policy,
        reasons=tuple(dict.fromkeys(reasons)),
        top_similarity=top_similarity,
        second_similarity=second_similarity,
        margin=margin,
        rerank_score=rerank_score,
        prompt_token_count=len(tokens),
        automotive_context=resolved_prompt.automotive_context,
        broad_request=resolved_prompt.broad_request,
        generic_prompt=generic_prompt,
        underspecified=underspecified,
        clarification_question=clarification_question,
        penalty=penalty,
    )


def should_return_unknown(
    report: AmbiguityReport,
    *,
    intent_confidence: float,
    selected_intent: str,
    unknown_threshold: float,
) -> bool:
    """Return True when the pipeline should finalize as UNKNOWN."""
    if selected_intent == "UNKNOWN":
        return True
    if report.policy == "UNKNOWN":
        return True
    if report.policy == "CLARIFICATION_NEEDED" and intent_confidence < max(unknown_threshold, 0.30):
        return True
    return False


def normalize_for_compat(prompt: str | None) -> NormalizedPrompt:
    """Compatibility helper so legacy callers can still pass a raw prompt."""
    return NormalizedPrompt(
        original_text=prompt or "",
        normalized_text=(prompt or "").strip().lower(),
        language_hint="unknown",
        detected_keywords=(),
        automotive_context=any(token in (prompt or "").lower() for token in ("car", "voiture", "engine", "moteur", "rpm", "dtc", "vin")),
        broad_request=any(token in (prompt or "").lower() for token in ("health", "etat", "state", "diagnostic", "bilan")),
        spelling_noise=False,
        unsupported_action=any(token in (prompt or "").lower() for token in ("repair", "fix", "repare")),
    )
