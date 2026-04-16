from __future__ import annotations

from nlp.schemas import PlanStep
from tools.registry import get_tool_registry, get_tool_spec


def ensure_allowed_tools(plan: list[PlanStep]) -> None:
    allowed = {tool.name for tool in get_tool_registry()}
    for step in plan:
        if step.tool not in allowed:
            raise ValueError(f"Unknown or disallowed tool requested: {step.tool}")


def ensure_max_plan_length(plan: list[PlanStep], *, max_steps: int) -> None:
    if len(plan) > max_steps:
        raise ValueError(f"Execution plan exceeds max_steps={max_steps}")


def ensure_no_duplicate_steps(plan: list[PlanStep]) -> None:
    seen: set[tuple[str, tuple[tuple[str, object], ...]]] = set()
    for step in plan:
        key = (step.tool, tuple(sorted((name, _freeze(value)) for name, value in step.arguments.items())))
        if key in seen:
            raise ValueError(f"Duplicate plan step detected for tool {step.tool}")
        seen.add(key)


def ensure_valid_arguments(plan: list[PlanStep]) -> None:
    for step in plan:
        spec = get_tool_spec(step.tool)
        spec.arg_model.model_validate(step.arguments)


def _freeze(value: object) -> object:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    return value
