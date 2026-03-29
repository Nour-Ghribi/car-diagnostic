from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path
from typing import Any

from backend.schemas import Intent, IntentName, IntentParameters, SignalName


ALLOWED_INTENTS: tuple[IntentName, ...] = (
    "READ_DTC",
    "CHECK_CYLINDER",
    "CHECK_ENGINE_HEALTH",
    "CHECK_SIGNAL_STATUS",
    "EXPLAIN_WARNING_LIGHT",
    "GET_VEHICLE_CONTEXT",
    "UNKNOWN",
)

ALLOWED_SIGNALS: tuple[SignalName, ...] = (
    "rpm",
    "engine_load",
    "coolant_temp",
    "throttle_pos",
    "stft_b1",
    "ltft_b1",
    "o2_b1s1",
    "vehicle_speed",
    "module_voltage",
)

ALLOWED_DETAILS = {"low", "medium", "high"}
ALLOWED_AMBIGUITY = {"low", "medium", "high"}
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class IntentParserError(ValueError):
    """Raised when the LLM payload cannot be validated as a strict intent."""


def _read_prompt_template(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _normalize_text(text: str) -> str:
    """Lowercase and strip accents to make prompt handling more robust."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower().strip()


def build_intent_parser_messages(prompt: str) -> list[dict[str, str]]:
    """
    Build structured messages for an LLM intent parser call.

    The output is provider-agnostic and intentionally compact so it can later
    be adapted to structured output or function calling without changing the
    orchestrator contract.
    """
    system_prompt = _read_prompt_template("intent_parser_system.txt")
    user_template = _read_prompt_template("intent_parser_user_template.txt")
    user_prompt = user_template.replace("{{user_prompt}}", prompt)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def is_gemini_configured() -> bool:
    """Return True when a Gemini API key is available in the environment."""
    return bool(os.getenv("GEMINI_API_KEY"))


def get_gemini_model_name() -> str:
    """Return the Gemini model name configured for intent parsing."""
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def _build_gemini_intent_schema() -> dict[str, Any]:
    """Return a JSON schema for richer intent parsing."""
    return {
        "type": "object",
        "required": [
            "selected_intent",
            "intent_confidence",
            "alternative_intents",
            "parameters",
            "ambiguity",
            "needs_clarification",
        ],
        "properties": {
            "selected_intent": {"type": "string", "enum": list(ALLOWED_INTENTS)},
            "intent_confidence": {"type": "number"},
            "alternative_intents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "score"],
                    "properties": {
                        "name": {"type": "string", "enum": list(ALLOWED_INTENTS)},
                        "score": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
            },
            "parameters": {
                "type": "object",
                "properties": {
                    "detail": {"type": "string", "enum": sorted(ALLOWED_DETAILS)},
                    "cylinder_index": {"type": "integer", "minimum": 1},
                    "bank": {"type": "integer", "minimum": 1},
                    "include_pending": {"type": "boolean"},
                    "include_permanent": {"type": "boolean"},
                    "signal": {"type": "string", "enum": list(ALLOWED_SIGNALS)},
                    "max_age_ms": {"type": "integer", "minimum": 0},
                    "warning_type": {"type": "string", "enum": ["check_engine"]},
                    "include_calibration": {"type": "boolean"},
                    "refresh_capabilities": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            "ambiguity": {"type": "string", "enum": sorted(ALLOWED_AMBIGUITY)},
            "needs_clarification": {"type": "boolean"},
        },
        "additionalProperties": False,
    }


def call_gemini_intent_parser(prompt: str) -> dict[str, Any]:
    """
    Call Gemini with structured output enabled and return a JSON payload.

    This function is optional at runtime:
    - if GEMINI_API_KEY is missing, it raises IntentParserError
    - if the SDK is not installed, it raises IntentParserError
    - if Gemini returns invalid JSON, validation will fail upstream
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise IntentParserError("GEMINI_API_KEY is not set.")

    try:
        from google import genai  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise IntentParserError("google-genai SDK is not installed.") from exc

    messages = build_intent_parser_messages(prompt)
    flattened_prompt = "\n\n".join(msg["content"] for msg in messages)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=get_gemini_model_name(),
        contents=flattened_prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": _build_gemini_intent_schema(),
        },
    )

    text = getattr(response, "text", None)
    if not text:
        raise IntentParserError("Gemini returned an empty response.")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise IntentParserError("Gemini did not return valid JSON.") from exc
    if not isinstance(payload, dict):
        raise IntentParserError("Gemini response must decode to an object.")
    return payload


def mock_llm_response(prompt: str) -> dict[str, Any]:
    """
    Return a mocked structured payload for representative prompts.

    The mock is richer than before:
    - selected_intent
    - intent_confidence
    - alternative_intents
    - ambiguity
    - needs_clarification
    """
    text = _normalize_text(prompt)

    if "check cylinder 2" in text or "cylinder 2" in text or "cylindre 2" in text:
        return {
            "selected_intent": "CHECK_CYLINDER",
            "intent_confidence": 0.93,
            "alternative_intents": [
                {"name": "CHECK_CYLINDER", "score": 0.93},
                {"name": "CHECK_ENGINE_HEALTH", "score": 0.05},
                {"name": "UNKNOWN", "score": 0.02},
            ],
            "parameters": {"cylinder_index": 2, "detail": "medium"},
            "ambiguity": "low",
            "needs_clarification": False,
        }

    if "read dtc" in text or "fault code" in text or "codes defaut" in text or "read fault codes" in text:
        return {
            "selected_intent": "READ_DTC",
            "intent_confidence": 0.95,
            "alternative_intents": [
                {"name": "READ_DTC", "score": 0.95},
                {"name": "EXPLAIN_WARNING_LIGHT", "score": 0.04},
                {"name": "UNKNOWN", "score": 0.01},
            ],
            "parameters": {"include_pending": True},
            "ambiguity": "low",
            "needs_clarification": False,
        }

    if "show rpm" in text or text == "rpm" or "montre les rpm" in text:
        return {
            "selected_intent": "CHECK_SIGNAL_STATUS",
            "intent_confidence": 0.91,
            "alternative_intents": [
                {"name": "CHECK_SIGNAL_STATUS", "score": 0.91},
                {"name": "CHECK_ENGINE_HEALTH", "score": 0.06},
                {"name": "UNKNOWN", "score": 0.03},
            ],
            "parameters": {"signal": "rpm", "max_age_ms": 2000},
            "ambiguity": "low",
            "needs_clarification": False,
        }

    if "check engine" in text or "voyant moteur" in text:
        return {
            "selected_intent": "EXPLAIN_WARNING_LIGHT",
            "intent_confidence": 0.89,
            "alternative_intents": [
                {"name": "EXPLAIN_WARNING_LIGHT", "score": 0.89},
                {"name": "READ_DTC", "score": 0.08},
                {"name": "UNKNOWN", "score": 0.03},
            ],
            "parameters": {"warning_type": "check_engine"},
            "ambiguity": "low",
            "needs_clarification": False,
        }

    if "vin" in text or "vehicle context" in text or "contexte vehicule" in text:
        return {
            "selected_intent": "GET_VEHICLE_CONTEXT",
            "intent_confidence": 0.86,
            "alternative_intents": [
                {"name": "GET_VEHICLE_CONTEXT", "score": 0.86},
                {"name": "UNKNOWN", "score": 0.09},
                {"name": "CHECK_ENGINE_HEALTH", "score": 0.05},
            ],
            "parameters": {"include_calibration": True, "refresh_capabilities": False},
            "ambiguity": "low",
            "needs_clarification": False,
        }

    if any(
        token in text
        for token in (
            "engine health",
            "etat du moteur",
            "bilan moteur",
            "moteur en bon etat",
            "diagnostic moteur",
        )
    ):
        return {
            "selected_intent": "CHECK_ENGINE_HEALTH",
            "intent_confidence": 0.9,
            "alternative_intents": [
                {"name": "CHECK_ENGINE_HEALTH", "score": 0.9},
                {"name": "READ_DTC", "score": 0.07},
                {"name": "UNKNOWN", "score": 0.03},
            ],
            "parameters": {"detail": "medium"},
            "ambiguity": "low",
            "needs_clarification": False,
        }

    if any(token in text for token in ("etat de voiture", "etat voiture", "state of the car", "etat general")):
        return {
            "selected_intent": "CHECK_ENGINE_HEALTH",
            "intent_confidence": 0.58,
            "alternative_intents": [
                {"name": "CHECK_ENGINE_HEALTH", "score": 0.58},
                {"name": "GET_VEHICLE_CONTEXT", "score": 0.27},
                {"name": "UNKNOWN", "score": 0.15},
            ],
            "parameters": {"detail": "medium"},
            "ambiguity": "medium",
            "needs_clarification": False,
        }

    return {
        "selected_intent": "UNKNOWN",
        "intent_confidence": 0.45,
        "alternative_intents": [
            {"name": "UNKNOWN", "score": 0.45},
            {"name": "CHECK_ENGINE_HEALTH", "score": 0.30},
            {"name": "GET_VEHICLE_CONTEXT", "score": 0.25},
        ],
        "parameters": {},
        "ambiguity": "high",
        "needs_clarification": True,
    }


def validate_llm_intent_payload(payload: dict[str, Any]) -> Intent:
    """
    Validate and normalize a structured LLM payload into a strict Intent model.

    The validator accepts both:
    - the richer V3-like payload
    - the previous compact payload
    """
    if not isinstance(payload, dict):
        raise IntentParserError("LLM payload must be a dictionary.")

    normalized_payload = _normalize_payload_shape(payload)
    name = normalized_payload["selected_intent"]
    raw_confidence = normalized_payload["intent_confidence"]
    alternatives = normalized_payload["alternative_intents"]
    ambiguity = normalized_payload["ambiguity"]
    needs_clarification = normalized_payload["needs_clarification"]

    raw_parameters = normalized_payload.get("parameters", {})
    if raw_parameters is None:
        raw_parameters = {}
    if not isinstance(raw_parameters, dict):
        raise IntentParserError("Intent parameters must be an object.")

    normalized: dict[str, Any] = {}

    if "detail" in raw_parameters and raw_parameters["detail"] is not None:
        detail = str(raw_parameters["detail"]).strip().lower()
        if detail not in ALLOWED_DETAILS:
            raise IntentParserError(f"Unsupported detail value: {detail}")
        normalized["detail"] = detail

    if "cylinder_index" in raw_parameters and raw_parameters["cylinder_index"] is not None:
        try:
            cylinder_index = int(raw_parameters["cylinder_index"])
        except (TypeError, ValueError) as exc:
            raise IntentParserError("cylinder_index must be an integer.") from exc
        if cylinder_index < 1:
            raise IntentParserError("cylinder_index must be >= 1.")
        normalized["cylinder_index"] = cylinder_index

    if "bank" in raw_parameters and raw_parameters["bank"] is not None:
        try:
            bank = int(raw_parameters["bank"])
        except (TypeError, ValueError) as exc:
            raise IntentParserError("bank must be an integer.") from exc
        if bank < 1:
            raise IntentParserError("bank must be >= 1.")
        normalized["bank"] = bank

    if "include_pending" in raw_parameters:
        normalized["include_pending"] = bool(raw_parameters["include_pending"])

    if "include_permanent" in raw_parameters:
        normalized["include_permanent"] = bool(raw_parameters["include_permanent"])

    if "signal" in raw_parameters and raw_parameters["signal"] is not None:
        signal = str(raw_parameters["signal"]).strip().lower()
        if signal not in ALLOWED_SIGNALS:
            raise IntentParserError(f"Unsupported signal returned by LLM: {signal}")
        normalized["signal"] = signal

    if "max_age_ms" in raw_parameters and raw_parameters["max_age_ms"] is not None:
        try:
            max_age_ms = int(raw_parameters["max_age_ms"])
        except (TypeError, ValueError) as exc:
            raise IntentParserError("max_age_ms must be an integer.") from exc
        if max_age_ms < 0:
            raise IntentParserError("max_age_ms must be >= 0.")
        normalized["max_age_ms"] = max_age_ms

    if "warning_type" in raw_parameters and raw_parameters["warning_type"] is not None:
        warning_type = str(raw_parameters["warning_type"]).strip().lower()
        if warning_type != "check_engine":
            raise IntentParserError("warning_type must be check_engine in V2.")
        normalized["warning_type"] = warning_type

    if "include_calibration" in raw_parameters:
        normalized["include_calibration"] = bool(raw_parameters["include_calibration"])

    if "refresh_capabilities" in raw_parameters:
        normalized["refresh_capabilities"] = bool(raw_parameters["refresh_capabilities"])

    if name == "CHECK_CYLINDER" and "cylinder_index" not in normalized:
        raise IntentParserError("CHECK_CYLINDER requires cylinder_index.")
    if name == "CHECK_SIGNAL_STATUS" and "signal" not in normalized:
        raise IntentParserError("CHECK_SIGNAL_STATUS requires signal.")

    calibrated_confidence = _calibrate_intent_confidence(
        selected_intent=name,
        raw_confidence=raw_confidence,
        alternatives=alternatives,
        ambiguity=ambiguity,
        needs_clarification=needs_clarification,
        normalized_parameters=normalized,
    )

    if _should_abstain(name, calibrated_confidence, ambiguity, needs_clarification):
        return Intent(name="UNKNOWN", confidence=round(calibrated_confidence, 2), parameters=IntentParameters())

    return Intent(
        name=name,
        confidence=round(calibrated_confidence, 2),
        parameters=IntentParameters(**normalized),
    )


def parse_intent_with_llm(prompt: str) -> Intent:
    """
    Parse intent through a structured LLM flow with strict validation.

    If Gemini is configured, the real provider is used.
    Otherwise, the local mock is used.
    """
    if is_gemini_configured():
        payload = call_gemini_intent_parser(prompt)
    else:
        payload = mock_llm_response(prompt)
    return validate_llm_intent_payload(payload)


def parse_intent_hybrid(prompt: str) -> Intent:
    """
    Try the LLM parser first, then fall back to the deterministic parser.

    The fallback remains as a resilience mechanism, not as the primary
    understanding strategy.
    """
    try:
        return parse_intent_with_llm(prompt)
    except Exception:
        from agent.orchestrator import parse_intent  # local import to avoid circular dependency at import time

        return parse_intent(prompt)


def debug_dump_messages(prompt: str) -> str:
    """Return the built messages as formatted JSON for inspection."""
    return json.dumps(build_intent_parser_messages(prompt), ensure_ascii=False, indent=2)


def _normalize_payload_shape(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize old and new payload shapes into the richer structure."""
    if "selected_intent" in payload:
        name = payload.get("selected_intent")
        confidence = payload.get("intent_confidence")
        alternatives = payload.get("alternative_intents", [])
        ambiguity = payload.get("ambiguity", "low")
        needs_clarification = payload.get("needs_clarification", False)
    else:
        name = payload.get("name")
        confidence = payload.get("confidence")
        alternatives = [{"name": name, "score": confidence}] if name is not None and confidence is not None else []
        ambiguity = "low"
        needs_clarification = False

    if name not in ALLOWED_INTENTS:
        raise IntentParserError(f"Unsupported intent returned by LLM: {name}")

    if not isinstance(confidence, (int, float)):
        raise IntentParserError("Intent confidence must be numeric.")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise IntentParserError("Intent confidence must be between 0 and 1.")

    if not isinstance(alternatives, list):
        raise IntentParserError("alternative_intents must be a list.")

    normalized_alternatives: list[dict[str, float | str]] = []
    for item in alternatives:
        if not isinstance(item, dict):
            raise IntentParserError("Each alternative intent must be an object.")
        alt_name = item.get("name")
        alt_score = item.get("score")
        if alt_name not in ALLOWED_INTENTS:
            raise IntentParserError(f"Unsupported alternative intent: {alt_name}")
        if not isinstance(alt_score, (int, float)):
            raise IntentParserError("Alternative intent score must be numeric.")
        alt_score = float(alt_score)
        if not 0.0 <= alt_score <= 1.0:
            raise IntentParserError("Alternative intent score must be between 0 and 1.")
        normalized_alternatives.append({"name": alt_name, "score": alt_score})

    if not normalized_alternatives:
        normalized_alternatives.append({"name": name, "score": confidence})

    ambiguity = str(ambiguity).strip().lower()
    if ambiguity not in ALLOWED_AMBIGUITY:
        raise IntentParserError(f"Unsupported ambiguity value: {ambiguity}")

    return {
        "selected_intent": name,
        "intent_confidence": confidence,
        "alternative_intents": normalized_alternatives,
        "parameters": payload.get("parameters", {}),
        "ambiguity": ambiguity,
        "needs_clarification": bool(needs_clarification),
    }


def _calibrate_intent_confidence(
    *,
    selected_intent: str,
    raw_confidence: float,
    alternatives: list[dict[str, float | str]],
    ambiguity: str,
    needs_clarification: bool,
    normalized_parameters: dict[str, Any],
) -> float:
    """
    Calibrate intent confidence instead of trusting the model-reported score alone.

    The calibrated score blends:
    - raw model confidence
    - top-1 / top-2 separation
    - parameter completeness
    - ambiguity penalty
    """
    ranked = sorted((float(item["score"]) for item in alternatives), reverse=True)
    top1 = ranked[0] if ranked else raw_confidence
    top2 = ranked[1] if len(ranked) > 1 else 0.0
    margin = max(0.0, top1 - top2)
    completeness = _parameter_completeness_score(selected_intent, normalized_parameters)

    calibrated = (0.55 * raw_confidence) + (0.25 * margin) + (0.20 * completeness)

    if ambiguity == "medium":
        calibrated -= 0.10
    elif ambiguity == "high":
        calibrated -= 0.20

    if needs_clarification:
        calibrated -= 0.10

    return max(0.0, min(1.0, calibrated))


def _parameter_completeness_score(selected_intent: str, normalized_parameters: dict[str, Any]) -> float:
    """Return how complete the extracted parameters are for the selected intent."""
    if selected_intent == "CHECK_CYLINDER":
        return 1.0 if "cylinder_index" in normalized_parameters else 0.0
    if selected_intent == "CHECK_SIGNAL_STATUS":
        return 1.0 if "signal" in normalized_parameters else 0.0
    if selected_intent in {
        "READ_DTC",
        "CHECK_ENGINE_HEALTH",
        "EXPLAIN_WARNING_LIGHT",
        "GET_VEHICLE_CONTEXT",
        "UNKNOWN",
    }:
        return 1.0
    return 0.5


def _should_abstain(selected_intent: str, confidence: float, ambiguity: str, needs_clarification: bool) -> bool:
    """Decide whether the parser should abstain and return UNKNOWN."""
    if selected_intent == "UNKNOWN":
        return True
    if needs_clarification and confidence < 0.70:
        return True
    if ambiguity == "high" and confidence < 0.65:
        return True
    return False
