from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from agent.embedding_provider import EmbeddingProvider
from agent.intent_index import IntentCard, render_intent_card
from agent.retriever import cosine_similarity


DEFAULT_DIAGNOSTIC_PROFILE_CSV_PATH = Path.home() / "Downloads" / "DiagnosticProfile (1).csv"
_PROFILE_EMBEDDING_CACHE: dict[tuple[str, tuple[str, ...]], list[tuple[float, ...]]] = {}
_PROFILE_CARD_MAPPING_CACHE: dict[tuple[str, tuple[str, ...], tuple[str, ...]], dict[str, IntentCard]] = {}


@dataclass(frozen=True)
class RequestedPid:
    """One requested PID entry extracted from the raw diagnostic profile CSV."""

    key: str
    pid: str
    mode: str
    priority: int


@dataclass(frozen=True)
class NormalizedDiagnosticProfile:
    """
    Canonical normalized representation of one diagnostic profile.

    This is the central business-facing shape derived from the raw CSV.
    """

    id: str
    profile_code: str
    name: str
    description: str
    domain: str
    symptom: str
    context: str
    include_dtcs: bool
    requested_pids: tuple[RequestedPid, ...]

    @property
    def signal_keys(self) -> tuple[str, ...]:
        return tuple(pid.key for pid in self.requested_pids)


@dataclass(frozen=True)
class RetrievedDiagnosticProfile:
    """Semantic retrieval result for a normalized diagnostic profile."""

    profile: NormalizedDiagnosticProfile
    similarity_score: float
    rank: int


def get_diagnostic_profile_csv_path() -> Path | None:
    """
    Return the configured diagnostic profile CSV path.

    Resolution order:
    1. `DIAGNOSTIC_PROFILE_CSV_PATH`
    2. user Downloads fallback if present
    """

    configured = os.getenv("DIAGNOSTIC_PROFILE_CSV_PATH", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.exists() else None
    return DEFAULT_DIAGNOSTIC_PROFILE_CSV_PATH if DEFAULT_DIAGNOSTIC_PROFILE_CSV_PATH.exists() else None


def get_diagnostic_profiles(csv_path: str | Path | None = None) -> tuple[NormalizedDiagnosticProfile, ...]:
    """Load and normalize diagnostic profiles from CSV."""

    path = Path(csv_path).expanduser() if csv_path is not None else get_diagnostic_profile_csv_path()
    if path is None or not path.exists():
        return ()
    return _load_diagnostic_profiles(path.resolve())


def retrieve_top_k_profiles(
    *,
    prompt: str,
    profiles: Sequence[NormalizedDiagnosticProfile],
    provider: EmbeddingProvider,
    top_k: int = 6,
) -> list[RetrievedDiagnosticProfile]:
    """Retrieve the top-k diagnostic profiles for a prompt."""

    if not profiles:
        return []
    profile_texts = tuple(render_diagnostic_profile(profile) for profile in profiles)
    cache_key = (provider.cache_identity(), profile_texts)
    vectors = _PROFILE_EMBEDDING_CACHE.get(cache_key)
    if vectors is None:
        vectors = provider.embed_texts(profile_texts)
        _PROFILE_EMBEDDING_CACHE[cache_key] = vectors
    prompt_vector = provider.embed_texts([prompt])[0]
    scored = [
        (profile, max(0.0, min(1.0, (cosine_similarity(prompt_vector, vector) + 1.0) / 2.0)))
        for profile, vector in zip(profiles, vectors, strict=True)
    ]
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
    return [
        RetrievedDiagnosticProfile(profile=profile, similarity_score=round(score, 6), rank=index)
        for index, (profile, score) in enumerate(ranked, start=1)
    ]


def map_profiles_to_intent_cards(
    *,
    profiles: Sequence[NormalizedDiagnosticProfile],
    cards: Sequence[IntentCard],
    provider: EmbeddingProvider,
    minimum_similarity: float = 0.58,
) -> dict[str, IntentCard]:
    """
    Map each normalized diagnostic profile to the nearest existing intent card.

    This keeps the migration progressive: profiles become the richer business layer,
    while current public intents and goals remain stable.
    """

    if not profiles or not cards:
        return {}
    profile_texts = tuple(render_diagnostic_profile(profile) for profile in profiles)
    card_texts = tuple(render_intent_card(card) for card in cards)
    cache_key = (provider.cache_identity(), profile_texts, card_texts)
    cached = _PROFILE_CARD_MAPPING_CACHE.get(cache_key)
    if cached is not None:
        return cached

    profile_vectors = _PROFILE_EMBEDDING_CACHE.get((provider.cache_identity(), profile_texts))
    if profile_vectors is None:
        profile_vectors = provider.embed_texts(profile_texts)
        _PROFILE_EMBEDDING_CACHE[(provider.cache_identity(), profile_texts)] = profile_vectors
    card_vectors = provider.embed_texts(card_texts)

    mapping: dict[str, IntentCard] = {}
    card_vector_pairs = list(zip(cards, card_vectors, strict=True))
    for profile, profile_vector in zip(profiles, profile_vectors, strict=True):
        best_card, best_vector = max(card_vector_pairs, key=lambda item: cosine_similarity(profile_vector, item[1]))
        similarity = max(0.0, min(1.0, (cosine_similarity(profile_vector, best_vector) + 1.0) / 2.0))
        if similarity < minimum_similarity:
            continue
        if not _profile_matches_public_contract(profile, best_card):
            continue
        mapping[profile.profile_code] = best_card
    _PROFILE_CARD_MAPPING_CACHE[cache_key] = mapping
    return mapping


def render_diagnostic_profile(profile: NormalizedDiagnosticProfile) -> str:
    """Render one normalized profile into one embedding-friendly text block."""

    pid_section = " | ".join(
        f"{pid.key} mode {pid.mode} pid {pid.pid} priority {pid.priority}" for pid in profile.requested_pids
    )
    sections = [
        f"profile_code: {profile.profile_code}",
        f"name: {profile.name}",
        f"domain: {profile.domain}",
        f"symptom: {profile.symptom}",
        f"context: {profile.context}",
        f"description: {profile.description}",
        f"include_dtcs: {profile.include_dtcs}",
        f"requested_pids: {pid_section}",
    ]
    return "\n".join(sections)


def clear_diagnostic_profile_cache() -> None:
    """Clear all in-process CSV/profile caches, mainly for tests."""

    _load_diagnostic_profiles.cache_clear()
    _PROFILE_EMBEDDING_CACHE.clear()
    _PROFILE_CARD_MAPPING_CACHE.clear()


@lru_cache(maxsize=4)
def _load_diagnostic_profiles(path: Path) -> tuple[NormalizedDiagnosticProfile, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(_normalize_profile_row(row) for row in reader)


def _normalize_profile_row(row: dict[str, Any]) -> NormalizedDiagnosticProfile:
    code = str(row.get("code", "")).strip()
    name = str(row.get("name", "")).strip()
    description = str(row.get("description", "")).strip()
    requested_pid_payload = row.get("defaultRequestedPidsJson", "[]")
    include_dtcs = str(row.get("includeDtcsByDefault", "")).strip().lower() == "true"
    code_parts = [part.strip().replace("_", " ") for part in code.split("__")]
    domain = code_parts[0] if code_parts else "unknown"
    symptom = code_parts[1] if len(code_parts) > 1 else name
    context = code_parts[2] if len(code_parts) > 2 else ""
    return NormalizedDiagnosticProfile(
        id=str(row.get("id", "")).strip(),
        profile_code=code,
        name=name,
        description=description,
        domain=domain,
        symptom=symptom,
        context=context,
        include_dtcs=include_dtcs,
        requested_pids=_parse_requested_pids(requested_pid_payload),
    )


def _parse_requested_pids(payload: Any) -> tuple[RequestedPid, ...]:
    try:
        raw_items = json.loads(payload or "[]")
    except Exception:
        raw_items = []
    items: list[RequestedPid] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        pid = str(item.get("pid", "")).strip()
        mode = str(item.get("mode", "")).strip()
        if not key or not pid or not mode:
            continue
        try:
            priority = int(item.get("priority", 0))
        except Exception:
            priority = 0
        items.append(RequestedPid(key=key, pid=pid, mode=mode, priority=priority))
    return tuple(sorted(items, key=lambda item: item.priority))


def _profile_matches_public_contract(profile: NormalizedDiagnosticProfile, card: IntentCard) -> bool:
    searchable = " ".join(
        [
            profile.profile_code,
            profile.name,
            profile.description,
            profile.domain,
            profile.symptom,
            profile.context,
        ]
    ).lower()

    if card.goal_name == "READ_DTC":
        if any(phrase in searchable for phrase in ("without dtc", "no dtc", "without fault code", "no fault code")):
            return False
        code_match = re.search(r"\b(dtc|fault codes?|stored codes?|pending codes?|permanent codes?|codes?)\b", searchable)
        read_match = re.search(r"\b(read|show|list|scan|retrieve)\b", searchable)
        return bool(code_match and read_match)
    if card.goal_name == "VEHICLE_CONTEXT_LOOKUP":
        return any(token in searchable for token in ("vin", "ecu", "calibration", "context", "capability"))
    return True
