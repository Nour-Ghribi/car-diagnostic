from __future__ import annotations

from dataclasses import dataclass

from backend.schemas import Intent, IntentParameters
from agent.ambiguity import AmbiguityReport, detect_ambiguity, should_return_unknown
from agent.confidence import IntentConfidenceReport, compute_intent_confidence
from agent.embedding_provider import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    embed_intent_cards,
    embed_prompt,
    get_embedding_provider,
)
from agent.intent_index import IntentCard, get_intent_cards
from agent.reranker import (
    RerankResult,
    llm_fallback_disambiguate,
    mock_rerank_response,
    rerank_candidates,
    validate_goal_fallback_payload,
    validate_reranker_payload,
)
from agent.retriever import RetrievedIntentCandidate, retrieve_top_k_intents
from agent.semantic_normalizer import NormalizedPrompt, normalize_prompt
from agent.settings import AgentSettings, get_settings


@dataclass(frozen=True)
class IntentParseArtifacts:
    """Detailed output of the hybrid goal-intent pipeline."""

    intent: Intent
    normalized_prompt: NormalizedPrompt
    retrieved_candidates: tuple[RetrievedIntentCandidate, ...]
    rerank_result: RerankResult
    ambiguity_report: AmbiguityReport
    confidence_report: IntentConfidenceReport
    llm_fallback_used: bool


def parse_intent_v3(prompt: str) -> Intent:
    """
    Parse a prompt with semantic normalization + retrieval + rerank + ambiguity.

    Safety behavior:
    - deterministic normalization first
    - strict candidate set from supported goal cards
    - broad automotive prompts can map to a broad valid goal
    - LLM fallback is optional and only used on low-confidence cases
    - safe fallback to the previous parser stack on provider/runtime failures
    """
    try:
        return parse_intent_v3_detailed(prompt).intent
    except Exception:
        try:
            return _parse_intent_v3_with_provider(prompt, provider=MockEmbeddingProvider(), allow_llm_fallback=False).intent
        except Exception:
            return _fallback_parse(prompt)


def parse_intent_v3_detailed(prompt: str) -> IntentParseArtifacts:
    """Run the full hybrid pipeline and return internal scoring artifacts."""
    return _parse_intent_v3_with_provider(prompt)


def _parse_intent_v3_with_provider(
    prompt: str,
    *,
    provider: EmbeddingProvider | None = None,
    allow_llm_fallback: bool | None = None,
) -> IntentParseArtifacts:
    """Run the full parser, optionally forcing a provider for safe local fallback."""
    settings = get_settings()
    normalized_prompt = normalize_prompt(prompt)
    embedding_provider = provider or _get_embedding_provider(settings)
    cards = get_intent_cards()
    indexed_cards = embed_intent_cards(embedding_provider, cards)
    prompt_vector = embed_prompt(embedding_provider, normalized_prompt.embedding_text)
    retrieved = retrieve_top_k_intents(prompt_vector, indexed_cards, top_k=settings.retrieval_top_k)
    reranked = rerank_candidates(prompt, retrieved)
    llm_fallback_used = False

    top1_score = retrieved[0].similarity_score if retrieved else 0.0
    top2_score = retrieved[1].similarity_score if len(retrieved) > 1 else 0.0
    selected_similarity = _selected_similarity(retrieved, reranked.goal_name)
    probability_proxy = _compute_probability_proxy(selected_similarity, reranked.rerank_score)
    ambiguity_report = detect_ambiguity(
        normalized_prompt=normalized_prompt,
        top_similarity=selected_similarity,
        second_similarity=top2_score if reranked.goal_name == retrieved[0].goal_name else top1_score,
        rerank_score=reranked.rerank_score,
        selected_goal=reranked.goal_name,
        selected_scope=reranked.scope,
        clarification_question=reranked.clarification_question,
    )
    confidence_report = compute_intent_confidence(
        calibrated_probability_proxy=probability_proxy,
        rerank_score=reranked.rerank_score,
        retrieval_similarity=selected_similarity,
        top1_score=top1_score,
        top2_score=top2_score,
        ambiguity_penalty=ambiguity_report.penalty,
        automotive_context=normalized_prompt.automotive_context,
        broad_request=normalized_prompt.broad_request,
    )

    effective_llm_fallback = settings.enable_llm_fallback if allow_llm_fallback is None else allow_llm_fallback
    if _should_try_llm_fallback(settings, ambiguity_report, confidence_report, allow_llm_fallback=effective_llm_fallback):
        maybe_result = _resolve_with_llm_fallback(prompt, retrieved, settings)
        if maybe_result is not None:
            reranked = maybe_result
            llm_fallback_used = True
            selected_similarity = _selected_similarity(retrieved, reranked.goal_name)
            probability_proxy = _compute_probability_proxy(selected_similarity, reranked.rerank_score)
            ambiguity_report = detect_ambiguity(
                normalized_prompt=normalized_prompt,
                top_similarity=selected_similarity,
                second_similarity=top2_score,
                rerank_score=reranked.rerank_score,
                selected_goal=reranked.goal_name,
                selected_scope=reranked.scope,
                clarification_question=reranked.clarification_question,
            )
            confidence_report = compute_intent_confidence(
                calibrated_probability_proxy=probability_proxy,
                rerank_score=reranked.rerank_score,
                retrieval_similarity=selected_similarity,
                top1_score=top1_score,
                top2_score=top2_score,
                ambiguity_penalty=ambiguity_report.penalty,
                automotive_context=normalized_prompt.automotive_context,
                broad_request=normalized_prompt.broad_request,
            )

    final_intent = _build_final_intent(
        reranked=reranked,
        ambiguity_report=ambiguity_report,
        confidence_report=confidence_report,
        settings=settings,
    )

    return IntentParseArtifacts(
        intent=final_intent,
        normalized_prompt=normalized_prompt,
        retrieved_candidates=tuple(retrieved),
        rerank_result=reranked,
        ambiguity_report=ambiguity_report,
        confidence_report=confidence_report,
        llm_fallback_used=llm_fallback_used,
    )


def _get_embedding_provider(settings: AgentSettings | None = None) -> EmbeddingProvider:
    """Return the configured embedding provider."""
    return get_embedding_provider(settings)


def _selected_similarity(
    retrieved: list[RetrievedIntentCandidate],
    selected_goal: str,
) -> float:
    for candidate in retrieved:
        if candidate.goal_name == selected_goal:
            return candidate.similarity_score
    return retrieved[0].similarity_score if retrieved else 0.0


def _compute_probability_proxy(similarity: float, rerank_score: float) -> float:
    """Blend retrieval and rerank quality into a probability-like proxy."""
    return max(0.0, min(1.0, 0.55 * similarity + 0.45 * rerank_score))


def _should_try_llm_fallback(
    settings: AgentSettings,
    ambiguity_report: AmbiguityReport,
    confidence_report: IntentConfidenceReport,
    *,
    allow_llm_fallback: bool,
) -> bool:
    if not allow_llm_fallback:
        return False
    if confidence_report.score > settings.llm_fallback_threshold and ambiguity_report.policy == "ACCEPT_BROAD_GOAL":
        return False
    return (
        confidence_report.score < settings.llm_fallback_threshold
        or ambiguity_report.policy == "CLARIFICATION_NEEDED"
    )


def _resolve_with_llm_fallback(
    prompt: str,
    retrieved: list[RetrievedIntentCandidate],
    settings: AgentSettings,
) -> RerankResult | None:
    payload = llm_fallback_disambiguate(prompt, retrieved, settings)
    if payload is None:
        return None

    validated = validate_goal_fallback_payload(
        payload,
        allowed_candidate_goals=[candidate.goal_name for candidate in retrieved],
    )
    candidate = next((item for item in retrieved if item.goal_name == validated["goal"]), None)
    if candidate is None:
        return None

    seed_payload = mock_rerank_response(prompt, [candidate])
    seed_payload["goal"] = validated["goal"]
    seed_payload["intent"]["scope"] = validated["scope"]
    seed_payload["rerank_score"] = max(float(seed_payload["rerank_score"]), float(validated["confidence"]))
    seed_payload["rationale_short"] = validated["reason"]
    seed_payload["clarification_question"] = validated["clarification_question"]
    return validate_reranker_payload(seed_payload, allowed_candidate_names=[candidate.goal_name])


def _build_final_intent(
    *,
    reranked: RerankResult,
    ambiguity_report: AmbiguityReport,
    confidence_report: IntentConfidenceReport,
    settings: AgentSettings,
) -> Intent:
    if should_return_unknown(
        ambiguity_report,
        intent_confidence=confidence_report.score,
        selected_intent=reranked.intent_name,
        unknown_threshold=settings.unknown_threshold,
    ):
        resolution_policy = ambiguity_report.policy if ambiguity_report.policy in {"CLARIFICATION_NEEDED", "UNKNOWN"} else "UNKNOWN"
        parameters = IntentParameters(
            goal=reranked.goal_name,
            scope=reranked.scope,  # type: ignore[arg-type]
            clarification_question=reranked.clarification_question or ambiguity_report.clarification_question,
            resolution_policy=resolution_policy,  # type: ignore[arg-type]
        )
        return Intent(name="UNKNOWN", confidence=round(confidence_report.score, 2), parameters=parameters)

    parameters = reranked.parameters.model_copy(
        update={
            "goal": reranked.goal_name,
            "scope": reranked.scope,
            "clarification_question": reranked.clarification_question,
            "resolution_policy": ambiguity_report.policy,
        }
    )
    return Intent(
        name=reranked.intent_name,
        confidence=round(confidence_report.score, 2),
        parameters=parameters,
    )


def _fallback_parse(prompt: str) -> Intent:
    """Delegate to the previous parser stack as a final safety net."""
    from agent.intent_parser_llm import parse_intent_hybrid

    return parse_intent_hybrid(prompt)
