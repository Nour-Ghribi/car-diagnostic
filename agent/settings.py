from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal


EmbeddingProviderName = Literal["mock", "sbert", "openai"]
RerankerProviderName = Literal["mock", "openai"]


@dataclass(frozen=True)
class AgentSettings:
    """Environment-driven settings for the hybrid semantic intent pipeline."""

    embedding_provider: EmbeddingProviderName = "sbert"
    reranker_provider: RerankerProviderName = "mock"
    sbert_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    reranker_model: str = "gpt-5.1"
    debug_logging: bool = False
    retrieval_top_k: int = 6
    high_confidence_threshold: float = 0.72
    broad_goal_accept_threshold: float = 0.50
    llm_fallback_threshold: float = 0.46
    unknown_threshold: float = 0.22
    enable_llm_fallback: bool = True
    max_plan_steps: int = 5


def get_settings() -> AgentSettings:
    """Return cached settings loaded from environment variables."""
    return _load_settings()


def clear_settings_cache() -> None:
    """Clear the cached settings so tests can reload environment changes."""
    _load_settings.cache_clear()


@lru_cache(maxsize=1)
def _load_settings() -> AgentSettings:
    return AgentSettings(
        embedding_provider=_read_embedding_provider("INTENT_EMBEDDING_PROVIDER", default="sbert"),
        reranker_provider=_read_reranker_provider("INTENT_RERANKER_PROVIDER", default="mock"),
        sbert_model=os.getenv(
            "SBERT_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ).strip()
        or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
        or "text-embedding-3-small",
        reranker_model=os.getenv("OPENAI_RERANKER_MODEL", "gpt-5.1").strip() or "gpt-5.1",
        debug_logging=_read_bool("AGENT_DEBUG", default=False),
        retrieval_top_k=_read_int("INTENT_RETRIEVAL_TOP_K", default=6, minimum=2),
        high_confidence_threshold=_read_float("INTENT_HIGH_CONFIDENCE_THRESHOLD", default=0.72),
        broad_goal_accept_threshold=_read_float("INTENT_BROAD_ACCEPT_THRESHOLD", default=0.50),
        llm_fallback_threshold=_read_float("INTENT_LLM_FALLBACK_THRESHOLD", default=0.46),
        unknown_threshold=_read_float("INTENT_UNKNOWN_THRESHOLD", default=0.22),
        enable_llm_fallback=_read_bool("INTENT_ENABLE_LLM_FALLBACK", default=True),
        max_plan_steps=_read_int("AGENT_MAX_PLAN_STEPS", default=5, minimum=1),
    )


def _read_embedding_provider(env_name: str, *, default: EmbeddingProviderName) -> EmbeddingProviderName:
    value = os.getenv(env_name, default).strip().lower()
    if value in {"mock", "sbert", "openai"}:
        return value  # type: ignore[return-value]
    return default


def _read_reranker_provider(env_name: str, *, default: RerankerProviderName) -> RerankerProviderName:
    value = os.getenv(env_name, default).strip().lower()
    if value in {"mock", "openai"}:
        return value  # type: ignore[return-value]
    return default


def _read_bool(env_name: str, *, default: bool) -> bool:
    value = os.getenv(env_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_int(env_name: str, *, default: int, minimum: int) -> int:
    value = os.getenv(env_name)
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return parsed if parsed >= minimum else default


def _read_float(env_name: str, *, default: float) -> float:
    value = os.getenv(env_name)
    if value is None:
        return default
    try:
        parsed = float(value.strip())
    except ValueError:
        return default
    return max(0.0, min(1.0, parsed))
