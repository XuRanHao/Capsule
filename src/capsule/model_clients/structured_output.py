"""Helpers for strict JSON Schema output from Responses-compatible APIs."""

from copy import deepcopy
from typing import Any

from pydantic import BaseModel


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a strict copy of a JSON Schema without mutating ``schema``.

    Strict structured output requires every object to reject undeclared fields
    and to list all declared properties as required.  Walking every mapping and
    sequence also applies those rules to objects stored below ``$defs``, array
    items, and composition keywords such as ``anyOf``.
    """

    strict_schema = deepcopy(schema)
    _make_objects_strict(strict_schema)
    return strict_schema


def responses_json_schema_format(
    output_type: type[BaseModel],
    *,
    name: str | None = None,
    strip_annotations: bool = False,
) -> dict[str, Any]:
    """Build the ``text.format`` payload for strict Responses JSON output."""

    schema = output_type.model_json_schema()
    if strip_annotations:
        _strip_annotations(schema)
    return {
        "type": "json_schema",
        "name": name or output_type.__name__,
        "strict": True,
        "schema": strict_json_schema(schema),
    }


def _make_objects_strict(node: Any) -> None:
    if isinstance(node, dict):
        for value in tuple(node.values()):
            _make_objects_strict(value)

        properties = node.get("properties")
        if node.get("type") == "object" or isinstance(properties, dict):
            node["additionalProperties"] = False
            node["required"] = list(properties) if isinstance(properties, dict) else []
    elif isinstance(node, list):
        for value in node:
            _make_objects_strict(value)


def _strip_annotations(node: Any) -> None:
    if isinstance(node, dict):
        for key in ("title", "description", "default", "examples"):
            node.pop(key, None)
        for value in node.values():
            _strip_annotations(value)
    elif isinstance(node, list):
        for value in node:
            _strip_annotations(value)
