from __future__ import annotations

import re

from agent.embedding_provider import embed_intent_cards, embed_prompt, get_embedding_provider
from agent.intent_index import get_intent_cards
from agent.retriever import RetrievedIntentCandidate, retrieve_top_k_intents
from agent.settings import AgentSettings, get_settings
from data.diagnostic_profiles import (
    get_diagnostic_profiles,
    map_profiles_to_intent_cards,
    retrieve_top_k_profiles,
)
from data.goal_catalog import build_goal_catalog
from nlp.schemas import RetrievalCandidate


def retrieve_candidates(prompt: str, settings: AgentSettings | None = None) -> tuple[RetrievalCandidate, ...]:
    """Return top-k semantic candidates for a prompt."""
    active_settings = settings or get_settings()
    provider = get_embedding_provider(active_settings)
    cards = get_intent_cards()
    indexed_cards = embed_intent_cards(provider, cards)
    prompt_vector = embed_prompt(provider, prompt)
    candidates = retrieve_top_k_intents(prompt_vector, indexed_cards, top_k=active_settings.retrieval_top_k)
    profiles = get_diagnostic_profiles()
    supporting_profiles_by_goal: dict[str, list[dict[str, object]]] = {}
    global_supporting_profiles: list[dict[str, object]] = []
    goal_catalog_by_name: dict[str, dict[str, object]] = {}

    if profiles:
        goal_catalog = build_goal_catalog(profiles=profiles, cards=cards, provider=provider)
        goal_catalog_by_name = {goal_name: entry.to_prompt_payload() for goal_name, entry in goal_catalog.items()}
        top_profiles = retrieve_top_k_profiles(
            prompt=prompt,
            profiles=profiles,
            provider=provider,
            top_k=max(active_settings.retrieval_top_k, 4),
        )
        profile_card_mapping = map_profiles_to_intent_cards(profiles=profiles, cards=cards, provider=provider)
        for retrieved_profile in top_profiles:
            profile_payload = {
                "profile_code": retrieved_profile.profile.profile_code,
                "name": retrieved_profile.profile.name,
                "domain": retrieved_profile.profile.domain,
                "symptom": retrieved_profile.profile.symptom,
                "context": retrieved_profile.profile.context,
                "include_dtcs": retrieved_profile.profile.include_dtcs,
                "requested_pids": [
                    {"key": pid.key, "pid": pid.pid, "mode": pid.mode, "priority": pid.priority}
                    for pid in retrieved_profile.profile.requested_pids[:8]
                ],
                "score": retrieved_profile.similarity_score,
            }
            global_supporting_profiles.append(profile_payload)
            mapped_card = profile_card_mapping.get(retrieved_profile.profile.profile_code)
            if mapped_card is None:
                continue
            goal_name = mapped_card.goal_name
            supporting_profiles_by_goal.setdefault(goal_name, []).append(profile_payload)

    enriched = [
        _to_schema(candidate, supporting_profiles_by_goal, global_supporting_profiles, goal_catalog_by_name)
        for candidate in candidates
    ]
    _apply_public_contract_boosts(enriched, prompt)
    ranked = sorted(enriched, key=lambda candidate: candidate.score, reverse=True)[: active_settings.retrieval_top_k]
    return tuple(ranked)


def _to_schema(
    candidate: RetrievedIntentCandidate,
    supporting_profiles_by_goal: dict[str, list[dict[str, object]]],
    global_supporting_profiles: list[dict[str, object]],
    goal_catalog_by_name: dict[str, dict[str, object]],
) -> RetrievalCandidate:
    card = candidate.card
    profile_support = supporting_profiles_by_goal.get(card.goal_name, [])
    return RetrievalCandidate(
        candidate_id=card.goal_name,
        public_intent=card.intent_name,
        goal=card.goal_name,
        score=candidate.similarity_score,
        metadata={
            "description": card.description,
            "expected_parameters": list(card.expected_parameters),
            "default_scope": card.default_scope,
            "clarification_question": card.clarification_question,
            "default_parameters": dict(card.default_parameters),
            # Backward-compatible empty placeholders: semantic business knowledge
            # must now come from the normalized CSV profile layer.
            "required_signals": [],
            "semantic_hints": [],
            "supporting_profiles": profile_support,
            "supporting_profiles_global": global_supporting_profiles[:5],
            "goal_profile_summary": goal_catalog_by_name.get(card.goal_name),
        },
    )


def _apply_public_contract_boosts(candidates: list[RetrievalCandidate], prompt: str) -> None:
    tokens = set(re.split(r"[^a-z0-9_]+", prompt.lower()))
    if not candidates or not tokens:
        return

    for candidate in candidates:
        boost = 0.0
        if "vin" in tokens and candidate.goal == "VEHICLE_CONTEXT_LOOKUP":
            boost += 0.12
        if {"dtc", "dtcs", "fault", "code", "codes"}.intersection(tokens) and candidate.goal == "READ_DTC":
            boost += 0.08
        if "rpm" in tokens and candidate.goal == "SIGNAL_STATUS_CHECK":
            boost += 0.08
        if {"coolant", "temperature", "temp"}.intersection(tokens) and candidate.goal == "ENGINE_TEMPERATURE_CHECK":
            boost += 0.08
        if {"battery", "voltage"}.intersection(tokens) and candidate.goal == "BATTERY_CHECK":
            boost += 0.08
        if boost:
            candidate.score = round(min(1.0, candidate.score + boost), 6)
