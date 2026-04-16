from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent.embedding_provider import EmbeddingProvider
from agent.intent_index import IntentCard
from data.diagnostic_profiles import NormalizedDiagnosticProfile, map_profiles_to_intent_cards


@dataclass(frozen=True)
class GoalCatalogEntry:
    """CSV-derived business summary attached to one stable project goal."""

    goal_name: str
    public_intent: str
    profile_count: int
    domains: tuple[str, ...]
    symptoms: tuple[str, ...]
    contexts: tuple[str, ...]
    requested_signal_keys: tuple[str, ...]
    include_dtcs_ratio: float

    def to_prompt_payload(self) -> dict[str, object]:
        return {
            "goal_name": self.goal_name,
            "public_intent": self.public_intent,
            "profile_count": self.profile_count,
            "domains": list(self.domains),
            "symptoms": list(self.symptoms),
            "contexts": list(self.contexts),
            "requested_signal_keys": list(self.requested_signal_keys),
            "include_dtcs_ratio": self.include_dtcs_ratio,
        }


def build_goal_catalog(
    *,
    profiles: Sequence[NormalizedDiagnosticProfile],
    cards: Sequence[IntentCard],
    provider: EmbeddingProvider,
) -> dict[str, GoalCatalogEntry]:
    """
    Build a compact goal-level business catalog from the CSV-derived profiles.

    This keeps goals stable while moving the knowledge content toward the CSV.
    """

    if not profiles or not cards:
        return {}
    mapping = map_profiles_to_intent_cards(profiles=profiles, cards=cards, provider=provider)
    grouped: dict[str, list[NormalizedDiagnosticProfile]] = {}
    for profile in profiles:
        card = mapping.get(profile.profile_code)
        if card is None:
            continue
        grouped.setdefault(card.goal_name, []).append(profile)

    entries: dict[str, GoalCatalogEntry] = {}
    cards_by_goal = {card.goal_name: card for card in cards}
    for goal_name, matched_profiles in grouped.items():
        card = cards_by_goal.get(goal_name)
        if card is None:
            continue
        include_dtcs_ratio = round(
            sum(1 for profile in matched_profiles if profile.include_dtcs) / max(1, len(matched_profiles)),
            3,
        )
        entries[goal_name] = GoalCatalogEntry(
            goal_name=goal_name,
            public_intent=card.intent_name,
            profile_count=len(matched_profiles),
            domains=tuple(_top_values([profile.domain for profile in matched_profiles], limit=6)),
            symptoms=tuple(_top_values([profile.symptom for profile in matched_profiles], limit=8)),
            contexts=tuple(_top_values([profile.context for profile in matched_profiles if profile.context], limit=6)),
            requested_signal_keys=tuple(
                _top_values(
                    [pid.key for profile in matched_profiles for pid in profile.requested_pids],
                    limit=10,
                )
            ),
            include_dtcs_ratio=include_dtcs_ratio,
        )
    return entries


def _top_values(values: Sequence[str] | object, *, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for raw_value in values:
        value = str(raw_value).strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [value for value, _ in ranked[:limit]]
