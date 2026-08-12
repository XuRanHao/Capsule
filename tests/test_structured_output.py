from typing import Any

from pydantic import BaseModel, Field

from capsule.model_clients.structured_output import (
    responses_json_schema_format,
    strict_json_schema,
)


class _Evidence(BaseModel):
    text: str
    confidence: float | None = None


class _ResultItem(BaseModel):
    asset_id: str
    evidence: _Evidence


class _ResultBatch(BaseModel):
    items: list[_ResultItem]
    summary: str | None = None


class _NamedLikeSchemaAnnotations(BaseModel):
    description: str = Field(description="annotation to strip")
    title: str = Field(description="another annotation to strip")


def test_responses_json_schema_format_makes_root_and_defs_strict() -> None:
    result = responses_json_schema_format(_ResultBatch, name="result_batch")

    assert result["type"] == "json_schema"
    assert result["name"] == "result_batch"
    assert result["strict"] is True

    schema = result["schema"]
    assert isinstance(schema, dict)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["items", "summary"]

    definitions = schema["$defs"]
    assert definitions["_ResultItem"]["additionalProperties"] is False
    assert definitions["_ResultItem"]["required"] == ["asset_id", "evidence"]
    assert definitions["_Evidence"]["additionalProperties"] is False
    assert definitions["_Evidence"]["required"] == ["text", "confidence"]


def test_strict_json_schema_does_not_mutate_input_schema() -> None:
    source: dict[str, Any] = {
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            }
        },
    }
    expected = {
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            }
        },
    }

    result = strict_json_schema(source)

    assert source == expected
    assert result is not source
    assert result["additionalProperties"] is False
    assert result["required"] == ["nested"]
    nested = result["properties"]["nested"]
    assert nested["additionalProperties"] is False
    assert nested["required"] == ["value"]


def test_responses_json_schema_format_uses_model_name_by_default() -> None:
    result = responses_json_schema_format(_ResultBatch)

    assert result["name"] == "_ResultBatch"


def test_responses_json_schema_format_can_strip_prompt_annotations() -> None:
    result = responses_json_schema_format(
        _ResultBatch,
        strip_annotations=True,
    )

    rendered = str(result["schema"])
    assert "title" not in rendered
    assert "description" not in rendered
    assert "default" not in rendered
    assert result["schema"]["required"] == ["items", "summary"]


def test_strip_annotations_preserves_fields_named_description_and_title() -> None:
    result = responses_json_schema_format(
        _NamedLikeSchemaAnnotations,
        strip_annotations=True,
    )

    properties = result["schema"]["properties"]
    assert set(properties) == {"description", "title"}
    assert properties["description"] == {"type": "string"}
    assert properties["title"] == {"type": "string"}
    assert result["schema"]["required"] == ["description", "title"]
