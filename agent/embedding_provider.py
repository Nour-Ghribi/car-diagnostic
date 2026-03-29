from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from agent.intent_index import IntentCard, render_intent_card
from agent.settings import AgentSettings, get_settings

STOPWORDS = {
    "a",
    "an",
    "and",
    "de",
    "des",
    "du",
    "do",
    "does",
    "est",
    "et",
    "for",
    "get",
    "i",
    "is",
    "je",
    "la",
    "le",
    "les",
    "l",
    "ma",
    "me",
    "moi",
    "mon",
    "my",
    "of",
    "please",
    "show",
    "the",
    "to",
    "un",
    "une",
    "veux",
    "voir",
    "what",
}
logger = logging.getLogger(__name__)
_INTENT_CARD_EMBEDDING_CACHE: dict[tuple[str, tuple[str, ...]], list["EmbeddedIntentCard"]] = {}
_SBERT_MODEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class EmbeddedIntentCard:
    """Goal card plus its embedding vector."""

    card: IntentCard
    vector: tuple[float, ...]


class EmbeddingProvider(Protocol):
    """Pluggable embedding provider interface."""

    def embed_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed a list of texts into normalized vectors."""

    def cache_identity(self) -> str:
        """Return a stable identifier used for process-local embedding cache keys."""


class MockEmbeddingProvider:
    """
    Deterministic local embedding provider based on hashed lexical-semantic features.

    This mock stays offline and gives stable vectors for tests. It is not meant to
    match a production semantic model, but it supports semantic retrieval without
    naive single-keyword routing.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self._embed_one(text) for text in texts]

    def cache_identity(self) -> str:
        return f"mock:{self.dimensions}"

    def _embed_one(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self.dimensions
        normalized = _normalize_text(text)
        features = list(_token_features(normalized))
        if not features:
            return tuple(values)

        for feature in features:
            digest = hashlib.md5(feature.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            values[index] += sign

        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            return tuple(values)
        return tuple(value / norm for value in values)


class SBERTEmbeddingProvider:
    """
    SentenceTransformers-based embedding provider.

    Recommended default model:
    `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

    Why this model:
    - multilingual sentence embeddings suitable for FR/EN and mixed prompts
    - compact enough for practical production usage
    - widely used for semantic similarity and retrieval workloads
    """

    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name

    def cache_identity(self) -> str:
        return f"sbert:{self.model_name}"

    def embed_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        model = _load_sbert_model(self.model_name)
        with _suppress_sbert_noise():
            vectors = model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return [tuple(float(value) for value in vector) for vector in vectors]


class OpenAIEmbeddingProvider:
    """
    Optional OpenAI embedding provider.

    Intended target:
    - model: text-embedding-3-small
    - env var: OPENAI_API_KEY
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env

    def is_configured(self) -> bool:
        return bool(os.getenv(self.api_key_env))

    def cache_identity(self) -> str:
        return f"openai:{self.model}"

    def embed_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set.")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is not installed.") from exc

        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=self.model, input=list(texts))
        return [tuple(item.embedding) for item in response.data]


def embed_intent_cards(
    provider: EmbeddingProvider,
    cards: Sequence[IntentCard],
) -> list[EmbeddedIntentCard]:
    """Embed goal cards with the given provider and cache them in-process."""
    card_texts = [render_intent_card(card) for card in cards]
    cache_key = (provider.cache_identity(), tuple(card_texts))
    cached = _INTENT_CARD_EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached
    vectors = provider.embed_texts(card_texts)
    embedded = [EmbeddedIntentCard(card=card, vector=vector) for card, vector in zip(cards, vectors, strict=True)]
    _INTENT_CARD_EMBEDDING_CACHE[cache_key] = embedded
    return embedded


def embed_prompt(provider: EmbeddingProvider, prompt: str) -> tuple[float, ...]:
    """Embed one prompt with the given provider."""
    return provider.embed_texts([prompt])[0]


def get_embedding_provider(settings: AgentSettings | None = None) -> EmbeddingProvider:
    """
    Return the configured embedding provider with safe fallback to mock.

    Behavior:
    - default prefers SBERT for better multilingual semantic similarity
    - if SBERT or OpenAI is not usable, the function falls back to mock embeddings
    - offline tests keep running without network access by default
    """
    resolved_settings = settings or get_settings()
    if resolved_settings.embedding_provider == "mock":
        return MockEmbeddingProvider()

    if resolved_settings.embedding_provider == "sbert":
        try:
            _load_sbert_model(resolved_settings.sbert_model)
            return SBERTEmbeddingProvider(model_name=resolved_settings.sbert_model)
        except Exception as exc:
            logger.warning("SBERT embedding provider unavailable, falling back to mock embeddings: %s", exc)
            return MockEmbeddingProvider()

    provider = OpenAIEmbeddingProvider(model=resolved_settings.openai_embedding_model)
    if not provider.is_configured():
        logger.warning("OpenAI embedding provider requested but OPENAI_API_KEY is missing. Falling back to mock embeddings.")
        return MockEmbeddingProvider()
    try:
        import openai  # type: ignore # noqa: F401
    except Exception:
        logger.warning("OpenAI embedding provider requested but the openai package is unavailable. Falling back to mock embeddings.")
        return MockEmbeddingProvider()
    return provider


def get_embedding_provider_name(provider: EmbeddingProvider) -> str:
    """Return a human-readable provider label for CLI/demo output."""
    if isinstance(provider, SBERTEmbeddingProvider):
        return f"sbert:{provider.model_name}"
    if isinstance(provider, OpenAIEmbeddingProvider):
        return f"openai:{provider.model}"
    if isinstance(provider, MockEmbeddingProvider):
        return f"mock:{provider.dimensions}"
    return provider.__class__.__name__.lower()


def clear_embedding_cache() -> None:
    """Clear the in-process card embedding cache, mainly for tests."""
    _INTENT_CARD_EMBEDDING_CACHE.clear()
    _SBERT_MODEL_CACHE.clear()


def _load_sbert_model(model_name: str) -> Any:
    if model_name in _SBERT_MODEL_CACHE:
        return _SBERT_MODEL_CACHE[model_name]

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("sentence-transformers is not installed.") from exc

    with _suppress_sbert_noise():
        model = SentenceTransformer(model_name)
    _SBERT_MODEL_CACHE[model_name] = model
    return model


@contextlib.contextmanager
def _suppress_sbert_noise() -> Any:
    """
    Temporarily silence noisy HF/transformers console output.

    This keeps real exceptions intact while avoiding progress bars and verbose
    model load reports during local CLI usage.
    """
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    loggers = [
        logging.getLogger("huggingface_hub"),
        logging.getLogger("huggingface_hub.file_download"),
        logging.getLogger("sentence_transformers"),
        logging.getLogger("transformers"),
    ]
    previous_levels = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.ERROR)
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            yield
    finally:
        for logger, level in zip(loggers, previous_levels, strict=True):
            logger.setLevel(level)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return ascii_text.lower().strip()


def _token_features(text: str) -> list[str]:
    tokens = [token for token in re.split(r"[^a-z0-9_]+", text) if token and token not in STOPWORDS]
    features: list[str] = []
    for token in tokens:
        features.append(f"tok:{token}")
        if len(token) > 3:
            features.extend(f"pref:{token[:size]}" for size in (3, 4) if len(token) >= size)

    condensed = text.replace(" ", "")
    for index in range(max(0, len(condensed) - 2)):
        gram = condensed[index : index + 3]
        if len(gram) == 3:
            features.append(f"tri:{gram}")
    return features
