from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nlp.schemas import PlanStep


@dataclass(frozen=True)
class Observation:
    """Normalized observation captured after one tool step."""

    step: PlanStep
    payload: dict[str, Any]
