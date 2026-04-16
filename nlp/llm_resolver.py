from __future__ import annotations

import json
import logging
import os
from typing import Any, Sequence

from agent.embedding_provider import get_embedding_provider
from agent.intent_index import get_intent_card, render_intent_card
from agent.settings import AgentSettings, get_settings
from nlp.prompt_builder import build_resolver_messages
from nlp.schemas import PlanStep, RetrievalCandidate, RewriterOutput, ResolverDecision
from tools.registry import ToolSpec, get_tool_registry

LOGGER = logging.getLogger(__name__)


def resolve_and_plan(
    *,
    original_prompt: str,
    rewritten: RewriterOutput,
    candidates: Sequence[RetrievalCandidate],
    vehicle_context: dict[str, Any] | None = None,
    settings: AgentSettings | None = None,
) -> ResolverDecision:
    """Resolve the final goal and execution plan from bounded candidates."""
    active_settings = settings or get_settings()
    tools = tuple(get_tool_registry())
    try:
        payload = _call_gemini_resolver(
            original_prompt=original_prompt,
            rewritten=rewritten,
            candidates=candidates,
            tools=tools,
            vehicle_context=vehicle_context,
        )
        if payload is not None:
            decision = ResolverDecision.model_validate(payload)
            if _should_fallback_to_local(decision):
                raise RuntimeError("Remote resolver returned an accepted goal without an executable plan.")
            return decision
        payload = _call_openrouter_resolver(
            original_prompt=original_prompt,
            rewritten=rewritten,
            candidates=candidates,
            tools=tools,
            vehicle_context=vehicle_context,
        )
        if payload is not None:
            decision = ResolverDecision.model_validate(payload)
            if _should_fallback_to_local(decision):
                raise RuntimeError("Remote resolver returned an accepted goal without an executable plan.")
            return decision
        payload = _call_openai_resolver(
            original_prompt=original_prompt,
            rewritten=rewritten,
            candidates=candidates,
            tools=tools,
            vehicle_context=vehicle_context,
        )
        if payload is not None:
            decision = ResolverDecision.model_validate(payload)
            if _should_fallback_to_local(decision):
                raise RuntimeError("Remote resolver returned an accepted goal without an executable plan.")
            return decision
    except Exception as exc:  # pragma: no cover - remote optional
        LOGGER.warning("LLM resolver failed, using local resolver: %s", exc)
    return _mock_resolver(
        original_prompt=original_prompt,
        rewritten=rewritten,
        candidates=candidates,
        tools=tools,
        settings=active_settings,
    )


def _mock_resolver(
    *,
    original_prompt: str,
    rewritten: RewriterOutput,
    candidates: Sequence[RetrievalCandidate],
    tools: Sequence[ToolSpec],
    settings: AgentSettings,
) -> ResolverDecision:
    selected = candidates[0] if candidates else RetrievalCandidate(candidate_id="UNKNOWN", public_intent="UNKNOWN", goal="UNKNOWN", score=0.0)
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    gap = max(0.0, selected.score - second_score)
    confidence = max(0.0, min(1.0, 0.7 * selected.score + 0.3 * gap))

    if rewritten.needs_user_clarification or confidence < settings.unknown_threshold:
        return ResolverDecision(
            selected_public_intent="UNKNOWN",
            selected_goal="UNKNOWN",
            scope="ambiguous",
            confidence=round(confidence, 2),
            needs_user_clarification=True,
            clarification_question=rewritten.clarification_question or "Pouvez-vous préciser votre demande automobile ?",
            reasoning_summary="Prompt remains too ambiguous for a grounded plan.",
            execution_plan=[],
        )
    if selected.goal == "UNKNOWN" or selected.public_intent == "UNKNOWN":
        return ResolverDecision(
            selected_public_intent="UNKNOWN",
            selected_goal="UNKNOWN",
            scope="specific",
            confidence=round(confidence, 2),
            needs_user_clarification=False,
            clarification_question=get_intent_card("UNKNOWN").clarification_question,
            reasoning_summary="Retriever could not ground the request into a supported automotive goal.",
            execution_plan=[],
        )

    card = get_intent_card(selected.goal)
    scope = "broad" if card.default_scope == "broad" else "specific"
    plan = _semantic_plan_for_candidate(
        original_prompt=original_prompt,
        rewritten_prompt=rewritten.rewritten_prompt,
        candidate=selected,
        tools=tools,
        settings=settings,
    )
    needs_clarification = rewritten.ambiguity_level == "high" and confidence < settings.broad_goal_accept_threshold
    return ResolverDecision(
        selected_public_intent=selected.public_intent,
        selected_goal=selected.goal,
        scope="ambiguous" if needs_clarification else scope,  # type: ignore[arg-type]
        confidence=round(confidence, 2),
        needs_user_clarification=needs_clarification,
        clarification_question=card.clarification_question,
        reasoning_summary=f"Selected {selected.goal} from bounded semantic candidates.",
        execution_plan=[] if needs_clarification else plan,
    )


def _semantic_plan_for_candidate(
    *,
    original_prompt: str,
    rewritten_prompt: str,
    candidate: RetrievalCandidate,
    tools: Sequence[ToolSpec],
    settings: AgentSettings,
) -> list[PlanStep]:
    contract_first = _contract_first_plan(candidate)
    if contract_first is not None:
        return _normalize_plan_order(contract_first)

    provider = get_embedding_provider(settings)
    tool_vectors = provider.embed_texts([tool.profile_text for tool in tools])
    card = get_intent_card(candidate.goal)
    query_text = "\n".join(
        [
            original_prompt,
            rewritten_prompt,
            render_intent_card(card),
            _render_profile_support(candidate),
            "Need a grounded diagnostic execution plan and confidence assessment.",
        ]
    )
    query_vector = provider.embed_texts([query_text])[0]
    candidate_terms = set(_candidate_terms(candidate))
    scored_tools: list[tuple[float, ToolSpec]] = []
    for tool, vector in zip(tools, tool_vectors):
        score = _cosine_similarity(query_vector, vector)
        score += 0.03 * len(candidate_terms.intersection(set(tool.profile_text.split())))
        if tool.name == "get_latest_signals" and candidate.metadata.get("required_signals"):
            score += 0.16
        if tool.name == "request_fresh_signals" and candidate.metadata.get("required_signals"):
            score += 0.12
        if tool.name == "get_vehicle_context" and card.default_scope == "broad":
            score += 0.12
        if tool.name == "get_vehicle_context" and card.default_scope != "broad" and candidate.goal != "VEHICLE_CONTEXT_LOOKUP":
            score -= 0.08
        if tool.name == "get_dtcs" and {"dtc", "fault", "engine", "diagnostic"}.intersection(candidate_terms):
            score += 0.08
        if tool.name == "get_dtcs" and candidate.goal == "CYLINDER_CHECK":
            score += 0.18
        if tool.name == "request_mode06" and candidate.goal == "CYLINDER_CHECK":
            score += 0.14
        if tool.name == "request_mode06" and not {"cylinder", "misfire", "combustion"}.intersection(candidate_terms):
            score -= 0.25
        if tool.name == "score_confidence":
            score += 0.02
        scored_tools.append((score, tool))
    ranked_tools = [tool for _, tool in sorted(scored_tools, key=lambda item: item[0], reverse=True)]

    selected_steps = []
    for tool in ranked_tools:
        step = tool.default_step_for_candidate(candidate)
        if step is None:
            continue
        if any(existing.tool == step.tool and existing.arguments == step.arguments for existing in selected_steps):
            continue
        selected_steps.append(step)
        if len(selected_steps) >= settings.max_plan_steps:
            break
    return _normalize_plan_order(selected_steps)


def _contract_first_plan(candidate: RetrievalCandidate) -> list[PlanStep] | None:
    if candidate.goal == "READ_DTC":
        return [
            PlanStep(tool="get_dtcs", arguments={"include_pending": True}),
            PlanStep(tool="score_confidence", arguments={}),
        ]
    if candidate.goal == "VEHICLE_CONTEXT_LOOKUP":
        return [
            PlanStep(tool="get_vehicle_context", arguments={}),
            PlanStep(tool="score_confidence", arguments={}),
        ]
    return None


def _call_openai_resolver(
    *,
    original_prompt: str,
    rewritten: RewriterOutput,
    candidates: Sequence[RetrievalCandidate],
    tools: Sequence[ToolSpec],
    vehicle_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if os.getenv("INTENT_RESOLVER_PROVIDER", "mock").strip().lower() != "openai":
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is not installed") from exc

    client = OpenAI(api_key=api_key)
    messages = build_resolver_messages(
        original_prompt=original_prompt,
        rewritten_prompt=rewritten.rewritten_prompt,
        candidates=candidates,
        tools=tools,
        vehicle_context=vehicle_context,
    )
    response = client.responses.create(
        model=os.getenv("OPENAI_RESOLVER_MODEL", "gpt-5.1"),
        input=messages,
        text={"format": {"type": "json_schema", "name": "resolver_decision", "schema": _build_resolver_schema()}},
    )
    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("OpenAI resolver returned an empty response.")
    return json.loads(output_text)


def _call_gemini_resolver(
    *,
    original_prompt: str,
    rewritten: RewriterOutput,
    candidates: Sequence[RetrievalCandidate],
    tools: Sequence[ToolSpec],
    vehicle_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if os.getenv("INTENT_RESOLVER_PROVIDER", "mock").strip().lower() != "gemini":
        return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("google-genai package is not installed") from exc

    client = genai.Client(api_key=api_key)
    messages = build_resolver_messages(
        original_prompt=original_prompt,
        rewritten_prompt=rewritten.rewritten_prompt,
        candidates=candidates,
        tools=tools,
        vehicle_context=vehicle_context,
    )
    flattened = "\n\n".join(message["content"] for message in messages)
    response = client.models.generate_content(
        model=os.getenv("GEMINI_RESOLVER_MODEL", "gemini-2.0-flash"),
        contents=flattened,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": _build_resolver_schema(),
        },
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini resolver returned an empty response.")
    return json.loads(text)


def _call_openrouter_resolver(
    *,
    original_prompt: str,
    rewritten: RewriterOutput,
    candidates: Sequence[RetrievalCandidate],
    tools: Sequence[ToolSpec],
    vehicle_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if os.getenv("INTENT_RESOLVER_PROVIDER", "mock").strip().lower() != "openrouter":
        return None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is not installed") from exc

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    messages = build_resolver_messages(
        original_prompt=original_prompt,
        rewritten_prompt=rewritten.rewritten_prompt,
        candidates=candidates,
        tools=tools,
        vehicle_context=vehicle_context,
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_RESOLVER_MODEL", "google/gemini-2.5-flash"),
        messages=messages,
        max_tokens=int(os.getenv("OPENROUTER_RESOLVER_MAX_TOKENS", "700")),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "resolver_decision",
                "strict": True,
                "schema": _build_resolver_schema(),
            },
        },
    )
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise RuntimeError("OpenRouter resolver returned an empty response.")
    return json.loads(content)


def _build_resolver_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "selected_public_intent",
            "selected_goal",
            "scope",
            "confidence",
            "needs_user_clarification",
            "clarification_question",
            "reasoning_summary",
            "execution_plan",
        ],
        "properties": {
            "selected_public_intent": {"type": "string"},
            "selected_goal": {"type": "string"},
            "scope": {"type": "string", "enum": ["specific", "broad", "ambiguous"]},
            "confidence": {"type": "number"},
            "needs_user_clarification": {"type": "boolean"},
            "clarification_question": {"type": ["string", "null"]},
            "reasoning_summary": {"type": "string"},
            "execution_plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["tool", "arguments"],
                    "properties": {
                        "tool": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                },
            },
        },
    }


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _candidate_terms(candidate: RetrievalCandidate) -> tuple[str, ...]:
    metadata = candidate.metadata
    values = [
        candidate.goal,
        candidate.public_intent,
        metadata.get("description", ""),
        " ".join(metadata.get("semantic_hints", [])),
        " ".join(metadata.get("required_signals", [])),
        " ".join(str(item) for item in (metadata.get("goal_profile_summary") or {}).get("domains", [])),
        " ".join(str(item) for item in (metadata.get("goal_profile_summary") or {}).get("symptoms", [])),
        " ".join(str(item) for item in (metadata.get("goal_profile_summary") or {}).get("requested_signal_keys", [])),
        " ".join(
            str(profile.get("profile_code", ""))
            + " "
            + str(profile.get("name", ""))
            + " "
            + str(profile.get("domain", ""))
            + " "
            + str(profile.get("symptom", ""))
            + " "
            + " ".join(str(item.get("key", "")) for item in profile.get("requested_pids", []))
            for profile in metadata.get("supporting_profiles", [])
            if isinstance(profile, dict)
        ),
    ]
    tokens: list[str] = []
    for value in values:
        tokens.extend(str(value).replace("_", " ").lower().split())
    return tuple(tokens)


def _normalize_plan_order(steps: list[PlanStep]) -> list[PlanStep]:
    priority = {
        "get_vehicle_context": 10,
        "get_dtcs": 20,
        "get_latest_signals": 30,
        "request_fresh_signals": 40,
        "request_mode06": 50,
        "score_confidence": 90,
    }
    return sorted(steps, key=lambda step: priority.get(step.tool, 70))


def _render_profile_support(candidate: RetrievalCandidate) -> str:
    summary = candidate.metadata.get("goal_profile_summary")
    profiles = candidate.metadata.get("supporting_profiles", [])
    if not profiles:
        profiles = candidate.metadata.get("supporting_profiles_global", [])
    lines = []
    if isinstance(summary, dict):
        lines.extend(
            [
                "goal_profile_summary:",
                " - domains: " + ", ".join(str(item) for item in summary.get("domains", [])[:6]),
                " - symptoms: " + ", ".join(str(item) for item in summary.get("symptoms", [])[:6]),
                " - requested_signal_keys: " + ", ".join(str(item) for item in summary.get("requested_signal_keys", [])[:8]),
                " - include_dtcs_ratio: " + str(summary.get("include_dtcs_ratio", "")),
            ]
        )
    if not profiles:
        return "\n".join(lines) if lines else "supporting_profiles: none"
    lines.append("supporting_profiles:")
    for profile in profiles[:5]:
        if not isinstance(profile, dict):
            continue
        requested = ", ".join(
            str(item.get("key", "")) for item in profile.get("requested_pids", [])[:8] if isinstance(item, dict)
        )
        lines.append(
            " - "
            + str(profile.get("profile_code", ""))
            + " | "
            + str(profile.get("domain", ""))
            + " | "
            + str(profile.get("symptom", ""))
            + " | pids: "
            + requested
        )
    return "\n".join(lines)


def _should_fallback_to_local(decision: ResolverDecision) -> bool:
    if decision.selected_goal == "UNKNOWN" or decision.selected_public_intent == "UNKNOWN":
        return False
    if decision.needs_user_clarification:
        return decision.scope != "ambiguous"
    return not decision.execution_plan
