from __future__ import annotations

import json
from typing import Any, Sequence

from nlp.schemas import RetrievalCandidate
from tools.registry import ToolSpec


def build_rewriter_messages(prompt: str) -> list[dict[str, str]]:
    """Build constrained messages for prompt rewriting."""
    system = (
        "You are a constrained automotive prompt rewriter. "
        "Preserve meaning exactly. Clean only obvious noise such as spelling, spacing, punctuation, and light multilingual mixing. "
        "Do not invent issues, tools, diagnoses, plans, symptoms, or extra specificity. "
        "Do not turn the prompt into a new question unless the original prompt was already a question. "
        "Do not ask the user for more details inside rewritten_prompt. "
        "Prefer minimal edits. Keep the same scope, tone, and uncertainty level as the original user prompt. "
        "If the prompt is already understandable, keep it very close to the original. "
        "Return JSON only."
    )
    user = (
        "Rewrite the user prompt conservatively for semantic retrieval.\n"
        "Examples:\n"
        '- "je vx conaitre l etat de ma voiture" -> "je veux connaitre l etat de ma voiture"\n'
        '- "battery issue maybe" -> "battery issue maybe"\n'
        '- "it feels weird" -> "it feels weird"\n'
        '- "show rpm" -> "show rpm"\n'
        '- "get vin" -> "get vin"\n'
        '- "read dtc" -> "read dtc"\n'
        '- "check coolant temperature" -> "check coolant temperature"\n'
        "Never replace a specific diagnostic identifier with a broader word.\n"
        "For example: keep 'vin' as 'vin', keep 'dtc' as 'dtc', keep 'rpm' as 'rpm'.\n"
        f"prompt: {prompt}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_resolver_messages(
    *,
    original_prompt: str,
    rewritten_prompt: str,
    candidates: Sequence[RetrievalCandidate],
    tools: Sequence[ToolSpec],
    vehicle_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build structured resolver messages with bounded candidates and tools."""
    system = (
        "You are a bounded automotive resolver and planner. "
        "You must choose only from the provided candidates and allowed tools. "
        "Return strict JSON only. Do not emit raw OBD commands. "
        "If you accept a supported goal and do not need user clarification, you must return a non-empty execution_plan. "
        "Broad valid requests such as overall car health checks are acceptable and should normally produce a diagnostic plan, "
        "not a clarification request, unless the user request is truly too underspecified to proceed safely. "
        "Do not request clarification for a broad but valid overall car health request when the retrieval candidates clearly support a vehicle health check. "
        "Do request clarification for vague prompts with weak situational detail such as 'it feels weird' or similarly underspecified complaints. "
        "Prefer plans grounded in the allowed tools and candidate metadata, and avoid unnecessary specialized checks when simpler observations are sufficient. "
        "For broad health or engine status checks, prefer collecting foundational observations first, such as vehicle context, DTCs, and current live signals, "
        "before specialized checks like Mode 06 unless the candidate evidence strongly points to cylinder or misfire analysis. "
        "For single-signal requests, prefer direct signal collection over broad diagnostic plans."
    )
    payload = {
        "original_prompt": original_prompt,
        "rewritten_prompt": rewritten_prompt,
        "retrieval_candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "goal_profile_summaries": _collect_goal_profile_summaries(candidates),
        "supporting_profiles_global": _collect_global_supporting_profiles(candidates),
        "allowed_tools": [tool.to_prompt_payload() for tool in tools],
        "vehicle_context": vehicle_context or {},
    }
    user = json.dumps(payload, ensure_ascii=False, indent=2)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _collect_global_supporting_profiles(candidates: Sequence[RetrievalCandidate]) -> list[dict[str, Any]]:
    seen_codes: set[str] = set()
    collected: list[dict[str, Any]] = []
    for candidate in candidates:
        for profile in candidate.metadata.get("supporting_profiles_global", []):
            if not isinstance(profile, dict):
                continue
            code = str(profile.get("profile_code", "")).strip()
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            collected.append(profile)
            if len(collected) >= 8:
                return collected
    return collected


def _collect_goal_profile_summaries(candidates: Sequence[RetrievalCandidate]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen_goals: set[str] = set()
    for candidate in candidates:
        goal_name = candidate.goal
        if goal_name in seen_goals:
            continue
        seen_goals.add(goal_name)
        summary = candidate.metadata.get("goal_profile_summary")
        if isinstance(summary, dict):
            collected.append(summary)
    return collected
