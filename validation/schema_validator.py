from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

TModel = TypeVar("TModel", bound=BaseModel)


def validate_model(model_type: type[TModel], payload: Any) -> TModel:
    """Validate arbitrary payload against a Pydantic model."""
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
