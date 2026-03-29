from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from agent.embedding_provider import EmbeddedIntentCard
from agent.intent_index import IntentCard


@dataclass(frozen=True)
class RetrievedIntentCandidate:
    """Structured semantic retrieval output for one goal candidate."""

    intent_name: str
    similarity_score: float
    rank: int
    card: IntentCard
    goal_name: str = ""

    def __post_init__(self) -> None:
        if not self.goal_name:
            object.__setattr__(self, "goal_name", self.card.goal_name)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensionality.")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def retrieve_top_k_intents(
    prompt_embedding: Sequence[float],
    indexed_cards: Sequence[EmbeddedIntentCard],
    *,
    top_k: int = 4,
) -> list[RetrievedIntentCandidate]:
    """
    Retrieve the top-k goal candidates using cosine similarity.

    Returns structured candidates containing:
    - goal_name
    - intent_name
    - similarity_score
    - rank
    """
    scored = [
        (embedded.card, max(0.0, min(1.0, (cosine_similarity(prompt_embedding, embedded.vector) + 1.0) / 2.0)))
        for embedded in indexed_cards
    ]
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
    return [
        RetrievedIntentCandidate(
            goal_name=card.goal_name,
            intent_name=card.intent_name,
            similarity_score=round(score, 6),
            rank=index,
            card=card,
        )
        for index, (card, score) in enumerate(ranked, start=1)
    ]
