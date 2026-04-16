from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LocalStateStore:
    """Minimal in-memory local store for request-scoped debugging snapshots."""

    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def save(self, key: str, payload: dict[str, Any]) -> None:
        self.records[key] = payload

    def load(self, key: str) -> dict[str, Any] | None:
        return self.records.get(key)
