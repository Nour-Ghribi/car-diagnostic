from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Sequence

from backend.schemas import GoalName, IntentName, IntentParameters
from agent.retriever import RetrievedIntentCandidate
from agent.semantic_normalizer import normalize_prompt
from agent.settings import AgentSettings, get_settings


DEFAULT_RERANK_MODEL = "gpt-5.1"
ALLOWED_INTENTS: tuple[IntentName, ...] = (
    "READ_DTC",
    "CHECK_CYLINDER",
    "CHECK_ENGINE_HEALTH",
    "CHECK_SIGNAL_STATUS",
    "EXPLAIN_WARNING_LIGHT",
    "GET_VEHICLE_CONTEXT",
    "UNKNOWN",
)
ALLOWED_GOALS: tuple[GoalName, ...] = (
    "VEHICLE_HEALTH_CHECK",
    "READ_DTC",
    "CYLINDER_CHECK",
    "SIGNAL_STATUS_CHECK",
    "WARNING_LIGHT_CHECK",
    "BATTERY_CHECK",
    "ENGINE_TEMPERATURE_CHECK",
    "PERFORMANCE_ISSUE_CHECK",
    "STARTING_PROBLEM_CHECK",
    "FUEL_CONSUMPTION_CHECK",
    "VEHICLE_CONTEXT_LOOKUP",
    "UNKNOWN",
)
ALLOWED_SIGNALS = {
    "rpm",
    "engine_load",
    "coolant_temp",
    "throttle_pos",
    "stft_b1",
    "ltft_b1",
    "o2_b1s1",
    "vehicle_speed",
    "module_voltage",
}
logger = logging.getLogger(__name__)


class RerankerError(ValueError):
    """Raised when a reranker payload is invalid or unsafe."""


@dataclass(frozen=True)
class RerankResult:
    """Validated goal-level reranker output."""

    goal_name: GoalName
    intent_name: IntentName
    scope: str
    parameters: IntentParameters
    rerank_score: float
    rationale_short: str
    clarification_question: str | None


def build_reranker_messages(
    prompt: str,
    candidate_intents: Sequence[RetrievedIntentCandidate],
) -> list[dict[str, str]]:
    """Build structured messages for optional OpenAI fallback disambiguation."""
    candidate_payload = [
        {
            "goal": candidate.goal_name,
            "intent": candidate.intent_name,
            "scope": candidate.card.default_scope,
            "description": candidate.card.description,
            "examples_fr": list(candidate.card.examples_fr[:4]),
            "examples_en": list(candidate.card.examples_en[:4]),
            "retrieval_similarity": candidate.similarity_score,
            "clarification_question": candidate.card.clarification_question,
        }
        for candidate in candidate_intents
    ]
    system_prompt = (
        "You are a strict automotive goal disambiguator. "
        "Choose only one goal from the provided candidates. "
        "Do not invent tools, OBD commands, signals, or new goals. "
        "Return JSON only."
    )
    user_prompt = json.dumps(
        {
            "user_prompt": prompt,
            "candidate_goals": candidate_payload,
            "required_output": {
                "goal": "one of the provided candidate goal names",
                "scope": "broad or specific",
                "confidence": "float between 0 and 1",
                "reason": "brief explanation without chain-of-thought",
                "clarification_question": "string or null",
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def mock_rerank_response(
    prompt: str,
    candidate_intents: Sequence[RetrievedIntentCandidate],
) -> dict[str, Any]:
    """
    Deterministic local reranker for goal candidates.

    This reranker combines retrieval similarity, keyword overlap, and semantic
    hints from the normalized prompt and the candidate cards.
    """
    normalized = normalize_prompt(prompt)
    prompt_tokens = set(normalized.normalized_text.split())

    best_candidate: RetrievedIntentCandidate | None = None
    best_score = -1.0
    for candidate in candidate_intents:
        card_tokens = set(_candidate_terms(candidate))
        overlap = len(prompt_tokens & card_tokens) / max(1, len(prompt_tokens))
        hint_overlap = len(set(normalized.detected_keywords) & set(candidate.card.semantic_hints)) / max(
            1, len(candidate.card.semantic_hints) or 1
        )
        broad_alignment = 0.14 if normalized.broad_request and candidate.card.default_scope == "broad" else 0.0
        automotive_alignment = 0.06 if normalized.automotive_context else -0.06
        generic_symptom_alignment = 0.18 if (
            "doubt" in normalized.detected_keywords
            and not normalized.automotive_context
            and candidate.goal_name == "UNKNOWN"
        ) else 0.0
        generic_symptom_penalty = -0.10 if (
            "doubt" in normalized.detected_keywords
            and not normalized.automotive_context
            and candidate.goal_name != "UNKNOWN"
        ) else 0.0
        composite = (
            0.58 * candidate.similarity_score
            + 0.22 * overlap
            + 0.14 * hint_overlap
            + broad_alignment
            + automotive_alignment
            + generic_symptom_alignment
            + generic_symptom_penalty
        )
        if composite > best_score:
            best_score = composite
            best_candidate = candidate

    if best_candidate is None:
        raise RerankerError("No candidate goals were provided to the reranker.")

    parameters = _extract_parameters(prompt, best_candidate)
    rationale = _build_rationale(normalized.detected_keywords, best_candidate)
    return {
        "goal": best_candidate.goal_name,
        "intent": {
            "name": best_candidate.intent_name,
            "scope": best_candidate.card.default_scope,
            "parameters": parameters,
        },
        "rerank_score": round(max(0.0, min(1.0, best_score)), 4),
        "rationale_short": rationale,
        "clarification_question": best_candidate.card.clarification_question,
    }


def validate_reranker_payload(
    payload: dict[str, Any],
    *,
    allowed_candidate_names: Sequence[str] | None = None,
) -> RerankResult:
    """Validate reranker payload structure, safety, and parameter normalization."""
    if not isinstance(payload, dict):
        raise RerankerError("Reranker payload must be an object.")

    goal_name = payload.get("goal")
    if goal_name is None:
        legacy_intent = payload.get("intent", {})
        if isinstance(legacy_intent, dict):
            goal_name = _infer_goal_from_intent_name(legacy_intent.get("name"))
    if goal_name not in ALLOWED_GOALS:
        raise RerankerError(f"Unsupported reranked goal: {goal_name}")
    intent = payload.get("intent")
    if not isinstance(intent, dict):
        raise RerankerError("Reranker payload must include an intent object.")

    intent_name = intent.get("name")
    if intent_name not in ALLOWED_INTENTS:
        raise RerankerError(f"Unsupported reranked intent: {intent_name}")
    if allowed_candidate_names is not None:
        allowed = set(allowed_candidate_names)
        if goal_name not in allowed and intent_name not in allowed:
            raise RerankerError("Reranker selected a goal outside the candidate set.")

    scope = str(intent.get("scope", "specific")).strip().lower()
    if scope not in {"broad", "specific"}:
        raise RerankerError("scope must be broad or specific.")

    rerank_score = payload.get("rerank_score")
    if not isinstance(rerank_score, (int, float)):
        raise RerankerError("rerank_score must be numeric.")
    rerank_score = float(rerank_score)
    if not 0.0 <= rerank_score <= 1.0:
        raise RerankerError("rerank_score must be between 0 and 1.")

    rationale_short = str(payload.get("rationale_short", "")).strip()
    if not rationale_short:
        raise RerankerError("rationale_short is required.")
    if len(rationale_short) > 180:
        raise RerankerError("rationale_short must stay brief.")
    if _looks_like_obd_command(rationale_short):
        raise RerankerError("rationale_short must not contain raw OBD commands.")

    clarification_question = payload.get("clarification_question")
    if clarification_question is not None:
        clarification_question = str(clarification_question).strip() or None

    parameters = _normalize_parameters(intent_name, goal_name, scope, intent.get("parameters") or {})
    return RerankResult(
        goal_name=goal_name,
        intent_name=intent_name,
        scope=scope,
        parameters=IntentParameters(**parameters),
        rerank_score=round(rerank_score, 4),
        rationale_short=rationale_short,
        clarification_question=clarification_question,
    )


def rerank_candidates(
    prompt: str,
    candidate_intents: Sequence[RetrievedIntentCandidate],
) -> RerankResult:
    """Rerank goal candidates with the deterministic local strategy."""
    if not candidate_intents:
        raise RerankerError("candidate_intents cannot be empty.")

    payload = mock_rerank_response(prompt, candidate_intents)
    return validate_reranker_payload(
        payload,
        allowed_candidate_names=[candidate.goal_name for candidate in candidate_intents],
    )


def get_reranker_target_model() -> str:
    """Return the configured future reranker target model name."""
    return os.getenv("OPENAI_RERANKER_MODEL", DEFAULT_RERANK_MODEL)


def get_reranker_provider_name(settings: AgentSettings | None = None) -> str:
    """Return a human-readable label describing the configured reranker path."""
    resolved_settings = settings or get_settings()
    if resolved_settings.reranker_provider == "openai" and os.getenv("OPENAI_API_KEY"):
        try:
            import openai  # type: ignore # noqa: F401
        except Exception:
            return "mock(fallback)"
        return f"openai:{resolved_settings.reranker_model}"
    return "mock"


def llm_fallback_disambiguate(
    prompt: str,
    candidate_intents: Sequence[RetrievedIntentCandidate],
    settings: AgentSettings | None = None,
) -> dict[str, Any] | None:
    """
    Optionally ask an LLM to disambiguate among candidate goals.

    This is only used by the parser on low-confidence cases. If the provider is
    not configured or fails, the caller can continue with deterministic logic.
    """
    resolved_settings = settings or get_settings()
    if resolved_settings.reranker_provider != "openai":
        return None
    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        return call_openai_goal_fallback(prompt, candidate_intents, model=resolved_settings.reranker_model)
    except Exception as exc:
        logger.warning("OpenAI goal fallback failed: %s", exc)
        return None


def call_openai_reranker(
    prompt: str,
    candidate_intents: Sequence[RetrievedIntentCandidate],
    *,
    model: str,
) -> dict[str, Any]:
    """Backward-compatible alias used by older tests."""
    return call_openai_goal_fallback(prompt, candidate_intents, model=model)


def validate_goal_fallback_payload(
    payload: dict[str, Any],
    *,
    allowed_candidate_goals: Sequence[str],
) -> dict[str, Any]:
    """Validate the structured JSON returned by the optional LLM fallback."""
    if not isinstance(payload, dict):
        raise RerankerError("Fallback payload must be an object.")

    goal = payload.get("goal")
    if goal not in set(allowed_candidate_goals):
        raise RerankerError("Fallback goal must belong to the retrieved candidates.")

    scope = str(payload.get("scope", "specific")).strip().lower()
    if scope not in {"broad", "specific"}:
        raise RerankerError("Fallback scope must be broad or specific.")

    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)):
        raise RerankerError("Fallback confidence must be numeric.")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise RerankerError("Fallback confidence must be between 0 and 1.")

    reason = str(payload.get("reason", "")).strip()
    if not reason or len(reason) > 180:
        raise RerankerError("Fallback reason must be a brief non-empty string.")
    if _looks_like_obd_command(reason):
        raise RerankerError("Fallback reason must not contain raw OBD commands.")

    clarification_question = payload.get("clarification_question")
    if clarification_question is not None:
        clarification_question = str(clarification_question).strip() or None

    return {
        "goal": goal,
        "scope": scope,
        "confidence": confidence,
        "reason": reason,
        "clarification_question": clarification_question,
    }


def call_openai_goal_fallback(
    prompt: str,
    candidate_intents: Sequence[RetrievedIntentCandidate],
    *,
    model: str,
) -> dict[str, Any]:
    """Call the OpenAI Responses API for structured goal disambiguation."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("openai package is not installed.") from exc

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=build_reranker_messages(prompt, candidate_intents),
        text={
            "format": {
                "type": "json_schema",
                "name": "goal_disambiguation_output",
                "schema": _build_goal_fallback_schema([candidate.goal_name for candidate in candidate_intents]),
                "strict": True,
            }
        },
    )
    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("OpenAI goal fallback returned an empty response.")
    payload = json.loads(output_text)
    if not isinstance(payload, dict):
        raise RuntimeError("OpenAI goal fallback response must decode to an object.")
    return payload


def _candidate_terms(candidate: RetrievedIntentCandidate) -> tuple[str, ...]:
    card = candidate.card
    joined = " ".join(
        (
            card.description,
            " ".join(card.examples_fr),
            " ".join(card.examples_en),
            " ".join(card.semantic_hints),
        )
    ).lower()
    return tuple(token for token in re.split(r"[^a-z0-9_]+", joined) if token)


def _extract_parameters(prompt: str, candidate: RetrievedIntentCandidate) -> dict[str, Any]:
    normalized = normalize_prompt(prompt).normalized_text
    card = candidate.card
    parameters: dict[str, Any] = {
        "goal": candidate.goal_name,
        "scope": card.default_scope,
    }
    parameters.update(card.default_parameters)

    if candidate.goal_name == "CYLINDER_CHECK":
        match = re.search(r"\b(\d+)\b", normalized)
        parameters["cylinder_index"] = int(match.group(1)) if match else 1
    elif candidate.goal_name == "SIGNAL_STATUS_CHECK":
        parameters["signal"] = _detect_signal_from_prompt(normalized) or "rpm"
        parameters["max_age_ms"] = 2000
    elif candidate.goal_name == "VEHICLE_CONTEXT_LOOKUP":
        parameters["refresh_capabilities"] = any(token in normalized for token in ("capabilities", "supported", "pid"))

    return parameters


def _normalize_parameters(
    intent_name: IntentName,
    goal_name: GoalName,
    scope: str,
    raw_parameters: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw_parameters, dict):
        raise RerankerError("intent.parameters must be an object.")

    normalized: dict[str, Any] = {
        "goal": goal_name,
        "scope": scope,
    }
    if "detail" in raw_parameters and raw_parameters["detail"] is not None:
        detail = str(raw_parameters["detail"]).strip().lower()
        if detail not in {"low", "medium", "high"}:
            raise RerankerError(f"Unsupported detail: {detail}")
        normalized["detail"] = detail

    if "cylinder_index" in raw_parameters and raw_parameters["cylinder_index"] is not None:
        cylinder_index = int(raw_parameters["cylinder_index"])
        if cylinder_index < 1:
            raise RerankerError("cylinder_index must be >= 1.")
        normalized["cylinder_index"] = cylinder_index

    if "include_pending" in raw_parameters:
        normalized["include_pending"] = bool(raw_parameters["include_pending"])

    if "include_permanent" in raw_parameters:
        normalized["include_permanent"] = bool(raw_parameters["include_permanent"])

    if "signal" in raw_parameters and raw_parameters["signal"] is not None:
        signal = str(raw_parameters["signal"]).strip().lower()
        if signal not in ALLOWED_SIGNALS:
            raise RerankerError(f"Unsupported signal: {signal}")
        normalized["signal"] = signal

    if "max_age_ms" in raw_parameters and raw_parameters["max_age_ms"] is not None:
        max_age_ms = int(raw_parameters["max_age_ms"])
        if max_age_ms < 0:
            raise RerankerError("max_age_ms must be >= 0.")
        normalized["max_age_ms"] = max_age_ms

    if "warning_type" in raw_parameters and raw_parameters["warning_type"] is not None:
        warning_type = str(raw_parameters["warning_type"]).strip().lower()
        if warning_type != "check_engine":
            raise RerankerError("warning_type must be check_engine in V2.")
        normalized["warning_type"] = warning_type

    if "include_calibration" in raw_parameters:
        normalized["include_calibration"] = bool(raw_parameters["include_calibration"])

    if "refresh_capabilities" in raw_parameters:
        normalized["refresh_capabilities"] = bool(raw_parameters["refresh_capabilities"])

    if "clarification_question" in raw_parameters and raw_parameters["clarification_question"] is not None:
        normalized["clarification_question"] = str(raw_parameters["clarification_question"]).strip()

    if intent_name == "CHECK_CYLINDER" and "cylinder_index" not in normalized:
        raise RerankerError("CHECK_CYLINDER requires cylinder_index.")
    if intent_name == "CHECK_SIGNAL_STATUS" and "signal" not in normalized:
        raise RerankerError("CHECK_SIGNAL_STATUS requires signal.")

    unknown_keys = set(raw_parameters) - {
        "detail",
        "goal",
        "scope",
        "cylinder_index",
        "include_pending",
        "include_permanent",
        "signal",
        "max_age_ms",
        "warning_type",
        "include_calibration",
        "refresh_capabilities",
        "clarification_question",
    }
    if unknown_keys:
        raise RerankerError(f"Unknown parameters returned by reranker: {sorted(unknown_keys)}")
    return normalized


def _build_rationale(detected_keywords: Sequence[str], candidate: RetrievedIntentCandidate) -> str:
    if candidate.goal_name == "VEHICLE_HEALTH_CHECK":
        return "Prompt asks for a broad health or overall vehicle diagnostic."
    if candidate.goal_name == "BATTERY_CHECK":
        return "Prompt focuses on battery or module voltage symptoms."
    if candidate.goal_name == "ENGINE_TEMPERATURE_CHECK":
        return "Prompt focuses on coolant or engine temperature."
    if candidate.goal_name == "WARNING_LIGHT_CHECK":
        return "Prompt explicitly mentions a warning light."
    if candidate.goal_name == "READ_DTC":
        return "Prompt asks for diagnostic fault codes."
    if candidate.goal_name == "SIGNAL_STATUS_CHECK":
        return "Prompt asks for one live signal or sensor value."
    if candidate.goal_name == "CYLINDER_CHECK":
        return "Prompt targets a specific cylinder-related issue."
    if "vehicle" in detected_keywords and "health" in detected_keywords:
        return "Prompt remains automotive and close to a global health request."
    return "Prompt most closely matches the selected automotive goal."


def _looks_like_obd_command(text: str) -> bool:
    normalized = text.lower().strip()
    return bool(re.search(r"\b0[0-9a-f]{3}\b", normalized)) or "elm327" in normalized


def _build_goal_fallback_schema(candidate_goals: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["goal", "scope", "confidence", "reason", "clarification_question"],
        "properties": {
            "goal": {"type": "string", "enum": list(candidate_goals)},
            "scope": {"type": "string", "enum": ["broad", "specific"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "maxLength": 180},
            "clarification_question": {"type": ["string", "null"], "maxLength": 220},
        },
        "additionalProperties": False,
    }


def _infer_goal_from_intent_name(intent_name: Any) -> GoalName:
    mapping: dict[str, GoalName] = {
        "READ_DTC": "READ_DTC",
        "CHECK_CYLINDER": "CYLINDER_CHECK",
        "CHECK_ENGINE_HEALTH": "VEHICLE_HEALTH_CHECK",
        "CHECK_SIGNAL_STATUS": "SIGNAL_STATUS_CHECK",
        "EXPLAIN_WARNING_LIGHT": "WARNING_LIGHT_CHECK",
        "GET_VEHICLE_CONTEXT": "VEHICLE_CONTEXT_LOOKUP",
        "UNKNOWN": "UNKNOWN",
    }
    return mapping.get(str(intent_name), "UNKNOWN")


def _detect_signal_from_prompt(normalized_prompt: str) -> str | None:
    if any(token in normalized_prompt for token in ("rpm", "regime")):
        return "rpm"
    if any(token in normalized_prompt for token in ("coolant", "temperature", "temp")):
        return "coolant_temp"
    if any(token in normalized_prompt for token in ("load", "charge")):
        return "engine_load"
    if any(token in normalized_prompt for token in ("voltage", "tension", "battery", "batterie")):
        return "module_voltage"
    if any(token in normalized_prompt for token in ("o2", "oxygen")):
        return "o2_b1s1"
    if "stft" in normalized_prompt:
        return "stft_b1"
    if "ltft" in normalized_prompt:
        return "ltft_b1"
    if any(token in normalized_prompt for token in ("speed", "vitesse")):
        return "vehicle_speed"
    return None
