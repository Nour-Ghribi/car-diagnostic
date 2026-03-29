from __future__ import annotations

from dataclasses import dataclass

from agent.embedding_provider import MockEmbeddingProvider, get_embedding_provider
from agent.intent_index import get_intent_card
from agent.intent_parser_v3 import parse_intent_v3
from agent.retriever import RetrievedIntentCandidate
from agent.reranker import rerank_candidates
from agent.settings import clear_settings_cache


def _build_candidates() -> list[RetrievedIntentCandidate]:
    return [
        RetrievedIntentCandidate(
            intent_name="CHECK_SIGNAL_STATUS",
            similarity_score=0.82,
            rank=1,
            card=get_intent_card("CHECK_SIGNAL_STATUS"),
        ),
        RetrievedIntentCandidate(
            intent_name="CHECK_ENGINE_HEALTH",
            similarity_score=0.64,
            rank=2,
            card=get_intent_card("CHECK_ENGINE_HEALTH"),
        ),
        RetrievedIntentCandidate(
            intent_name="UNKNOWN",
            similarity_score=0.50,
            rank=3,
            card=get_intent_card("UNKNOWN"),
        ),
    ]


def test_embedding_provider_openai_missing_key_falls_back_to_mock(monkeypatch) -> None:
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    clear_settings_cache()

    provider = get_embedding_provider()
    assert isinstance(provider, MockEmbeddingProvider)

    clear_settings_cache()


def test_embedding_provider_sbert_missing_dependency_falls_back_to_mock(monkeypatch) -> None:
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "sbert")
    clear_settings_cache()

    from agent import embedding_provider

    monkeypatch.setattr(
        embedding_provider,
        "_load_sbert_model",
        lambda model_name: (_ for _ in ()).throw(RuntimeError("missing sentence-transformers")),
    )
    provider = get_embedding_provider()
    assert isinstance(provider, MockEmbeddingProvider)

    clear_settings_cache()


def test_reranker_openai_missing_key_falls_back_to_mock(monkeypatch) -> None:
    monkeypatch.setenv("INTENT_RERANKER_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    clear_settings_cache()

    result = rerank_candidates("show rpm", _build_candidates())
    assert result.intent_name == "CHECK_SIGNAL_STATUS"
    assert result.parameters.signal == "rpm"

    clear_settings_cache()


def test_pipeline_survives_real_provider_runtime_exception(monkeypatch) -> None:
    monkeypatch.setenv("INTENT_RERANKER_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    clear_settings_cache()

    from agent import reranker

    monkeypatch.setattr(reranker, "call_openai_reranker", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider failed")))
    intent = parse_intent_v3("show rpm")
    assert intent.name in {"CHECK_SIGNAL_STATUS", "UNKNOWN"}

    clear_settings_cache()


def test_pipeline_survives_embedding_provider_exception(monkeypatch) -> None:
    @dataclass
    class BrokenProvider:
        def embed_texts(self, texts):
            raise RuntimeError("embedding failure")

        def cache_identity(self) -> str:
            return "broken"

    from agent import intent_parser_v3

    monkeypatch.setattr(intent_parser_v3, "_get_embedding_provider", lambda settings=None: BrokenProvider())
    intent = parse_intent_v3("read dtc")
    assert intent.name == "READ_DTC"


def test_broad_prompt_survives_primary_provider_failure(monkeypatch) -> None:
    @dataclass
    class BrokenProvider:
        def embed_texts(self, texts):
            raise RuntimeError("embedding failure")

        def cache_identity(self) -> str:
            return "broken"

    from agent import intent_parser_v3

    monkeypatch.setattr(intent_parser_v3, "_get_embedding_provider", lambda settings=None: BrokenProvider())
    intent = parse_intent_v3("je veux connaître le health de ma voiture")
    assert intent.name == "CHECK_ENGINE_HEALTH"
    assert intent.parameters.goal == "VEHICLE_HEALTH_CHECK"
