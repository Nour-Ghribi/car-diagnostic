from __future__ import annotations

import json
import logging
import os
from typing import Any

from agent.semantic_normalizer import normalize_prompt
from nlp.prompt_builder import build_rewriter_messages
from nlp.schemas import RewriterOutput

LOGGER = logging.getLogger(__name__)


def rewrite_prompt(prompt: str) -> RewriterOutput:
    """
    Rewrite a prompt conservatively for semantic retrieval.

    Safe behavior:
    - if the provider fails, fall back to a local deterministic rewrite
    - preserve the original intent and ambiguity
    """
    try:
        payload = _call_gemini_rewriter(prompt)
        if payload is not None:
            return RewriterOutput.model_validate(payload)
        payload = _call_openrouter_rewriter(prompt)
        if payload is not None:
            return RewriterOutput.model_validate(payload)
        payload = _call_openai_rewriter(prompt)
        if payload is not None:
            return RewriterOutput.model_validate(payload)
    except Exception as exc:  # pragma: no cover - optional remote provider
        LOGGER.warning("LLM rewriter failed, using local rewrite: %s", exc)
    return _mock_rewriter(prompt)


def _mock_rewriter(prompt: str) -> RewriterOutput:
    normalized = normalize_prompt(prompt)
    preserved_literal = _preserve_specific_diagnostic_prompt(prompt, normalized.normalized_text)
    if normalized.unsupported_action:
        return RewriterOutput(
            rewritten_prompt="unsupported automotive request",
            language=normalized.language_hint,  # type: ignore[arg-type]
            ambiguity_level="high",
            preserved_meaning=True,
            needs_user_clarification=False,
            clarification_question=None,
        )
    if not normalized.automotive_context:
        return RewriterOutput(
            rewritten_prompt=preserved_literal or normalized.normalized_text or prompt.strip(),
            language=normalized.language_hint,  # type: ignore[arg-type]
            ambiguity_level="high",
            preserved_meaning=True,
            needs_user_clarification=True,
            clarification_question="Pouvez-vous préciser ce que vous voulez savoir sur le véhicule ?",
        )
    ambiguity = "high" if normalized.spelling_noise or not normalized.automotive_context else "medium" if normalized.broad_request else "low"
    needs_clarification = ambiguity == "high" and normalized.automotive_context
    return RewriterOutput(
        rewritten_prompt=preserved_literal or normalized.normalized_text or prompt.strip(),
        language=normalized.language_hint,  # type: ignore[arg-type]
        ambiguity_level=ambiguity,  # type: ignore[arg-type]
        preserved_meaning=True,
        needs_user_clarification=needs_clarification,
        clarification_question=(
            "Pouvez-vous préciser si vous voulez un bilan global ou un problème précis ?"
            if needs_clarification
            else None
        ),
    )


def _preserve_specific_diagnostic_prompt(prompt: str, normalized_text: str) -> str | None:
    cleaned_prompt = " ".join(prompt.strip().lower().split())
    if not cleaned_prompt:
        return None

    protected_keywords = {
        "vin",
        "dtc",
        "dtcs",
        "rpm",
        "ecu",
        "calibration",
        "coolant",
        "temperature",
        "voltage",
        "battery",
    }
    tokens = set(cleaned_prompt.replace("?", " ").replace("!", " ").split())
    if len(tokens) <= 4 and tokens.intersection(protected_keywords):
        return cleaned_prompt
    if "vin" in tokens and "context" in normalized_text.split() and "vin" not in normalized_text.split():
        return cleaned_prompt
    return None


def _call_openai_rewriter(prompt: str) -> dict[str, Any] | None:
    if os.getenv("INTENT_REWRITER_PROVIDER", "mock").strip().lower() != "openai":
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("openai package is not installed") from exc

    client = OpenAI(api_key=api_key)
    messages = build_rewriter_messages(prompt)
    response = client.responses.create(
        model=os.getenv("OPENAI_REWRITER_MODEL", "gpt-5.1-mini"),
        input=messages,
        text={
            "format": {
                "type": "json_schema",
                "name": "rewriter_output",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "rewritten_prompt",
                        "language",
                        "ambiguity_level",
                        "preserved_meaning",
                        "needs_user_clarification",
                        "clarification_question",
                    ],
                    "properties": {
                        "rewritten_prompt": {"type": "string"},
                        "language": {"type": "string", "enum": ["fr", "en", "mixed", "unknown"]},
                        "ambiguity_level": {"type": "string", "enum": ["low", "medium", "high"]},
                        "preserved_meaning": {"type": "boolean"},
                        "needs_user_clarification": {"type": "boolean"},
                        "clarification_question": {"type": ["string", "null"]},
                    },
                },
            }
        },
    )
    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("OpenAI rewriter returned an empty response.")
    return json.loads(output_text)


def _call_openrouter_rewriter(prompt: str) -> dict[str, Any] | None:
    if os.getenv("INTENT_REWRITER_PROVIDER", "mock").strip().lower() != "openrouter":
        return None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("openai package is not installed") from exc

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    messages = build_rewriter_messages(prompt)
    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_REWRITER_MODEL", "google/gemini-2.5-flash"),
        messages=messages,
        max_tokens=int(os.getenv("OPENROUTER_REWRITER_MAX_TOKENS", "220")),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "rewriter_output",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "rewritten_prompt",
                        "language",
                        "ambiguity_level",
                        "preserved_meaning",
                        "needs_user_clarification",
                        "clarification_question",
                    ],
                    "properties": {
                        "rewritten_prompt": {"type": "string"},
                        "language": {"type": "string", "enum": ["fr", "en", "mixed", "unknown"]},
                        "ambiguity_level": {"type": "string", "enum": ["low", "medium", "high"]},
                        "preserved_meaning": {"type": "boolean"},
                        "needs_user_clarification": {"type": "boolean"},
                        "clarification_question": {"type": ["string", "null"]},
                    },
                },
            },
        },
    )
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise RuntimeError("OpenRouter rewriter returned an empty response.")
    return json.loads(content)


def _call_gemini_rewriter(prompt: str) -> dict[str, Any] | None:
    if os.getenv("INTENT_REWRITER_PROVIDER", "mock").strip().lower() != "gemini":
        return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("google-genai package is not installed") from exc

    client = genai.Client(api_key=api_key)
    messages = build_rewriter_messages(prompt)
    flattened = "\n\n".join(message["content"] for message in messages)
    response = client.models.generate_content(
        model=os.getenv("GEMINI_REWRITER_MODEL", "gemini-2.0-flash-lite"),
        contents=flattened,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "rewritten_prompt",
                    "language",
                    "ambiguity_level",
                    "preserved_meaning",
                    "needs_user_clarification",
                    "clarification_question",
                ],
                "properties": {
                    "rewritten_prompt": {"type": "string"},
                    "language": {"type": "string", "enum": ["fr", "en", "mixed", "unknown"]},
                    "ambiguity_level": {"type": "string", "enum": ["low", "medium", "high"]},
                    "preserved_meaning": {"type": "boolean"},
                    "needs_user_clarification": {"type": "boolean"},
                    "clarification_question": {"type": ["string", "null"]},
                },
            }
        },
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini rewriter returned an empty response.")
    return json.loads(text)
