from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntentConfidenceReport:
    """Structured confidence output for the intent understanding pipeline."""

    score: float
    top1_score: float
    top2_score: float
    score_gap: float
    decision_reason: str


def compute_intent_confidence(
    *,
    calibrated_probability_proxy: float,
    rerank_score: float,
    retrieval_similarity: float,
    top1_score: float,
    top2_score: float,
    ambiguity_penalty: float,
    automotive_context: bool,
    broad_request: bool,
) -> IntentConfidenceReport:
    """
    Compute a practical confidence report clipped to [0, 1].

    Inputs:
    - calibrated_probability_proxy: proxy for the selected goal probability after retrieval/rerank.
    - rerank_score: reranker score in [0, 1].
    - retrieval_similarity: semantic retrieval similarity for the selected goal in [0, 1].
    - top1_score: top retrieved similarity.
    - top2_score: second-best retrieved similarity.
    - ambiguity_penalty: ambiguity-derived penalty in [0, 1].
    - automotive_context: whether the prompt still clearly refers to the automotive domain.
    - broad_request: whether the prompt expresses a broad health/overall diagnostic need.
    """
    score_gap = max(0.0, top1_score - top2_score)
    raw = (
        0.32 * calibrated_probability_proxy
        + 0.28 * rerank_score
        + 0.24 * retrieval_similarity
        + 0.16 * score_gap
        - 0.24 * ambiguity_penalty
    )

    if automotive_context:
        raw += 0.05
    if broad_request and automotive_context:
        raw += 0.06

    score = max(0.0, min(1.0, raw))
    if score >= 0.72:
        reason = "high_confidence_semantic_match"
    elif broad_request and automotive_context and score >= 0.50:
        reason = "broad_automotive_request_accepted"
    elif automotive_context and score >= 0.36:
        reason = "automotive_but_needs_clarification"
    else:
        reason = "insufficient_semantic_evidence"

    return IntentConfidenceReport(
        score=score,
        top1_score=top1_score,
        top2_score=top2_score,
        score_gap=score_gap,
        decision_reason=reason,
    )


def compute_final_diagnostic_confidence(
    intent_confidence: float,
    data_confidence: float,
) -> float:
    """
    Combine intent confidence and data confidence into one final diagnostic score.

    Multiplication remains intentionally conservative: if either the intent
    understanding or the supporting data quality is weak, the final score drops.
    """
    return max(0.0, min(1.0, intent_confidence * data_confidence))
