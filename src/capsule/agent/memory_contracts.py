"""Strict contracts exchanged by memory extraction, retrieval, and persistence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MemoryScope = Literal["workspace", "global"]
MemoryMutationAction = Literal["create", "merge", "deactivate", "lower_confidence"]


class ConversationSummary(BaseModel):
    """Compact, derived short-term context for one conversational thread."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=8_000)
    topic: str | None = Field(default=None, max_length=128)
    covered_sequence: int = Field(ge=0)


class MemoryCandidate(BaseModel):
    """One structured memory proposed from a consolidated message range."""

    model_config = ConfigDict(extra="forbid")

    scope: MemoryScope
    kind: str = Field(min_length=1, max_length=64)
    memory_key: str = Field(min_length=1, max_length=255)
    value: dict[str, Any] = Field(default_factory=dict)
    display_text: str = Field(min_length=1, max_length=4_000)
    topics: list[str] = Field(default_factory=list, max_length=5)
    initial_confidence: float = Field(ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class MemoryMutation(BaseModel):
    """A resolver-approved, idempotent mutation of persistent memory."""

    model_config = ConfigDict(extra="forbid")

    action: MemoryMutationAction
    candidate: MemoryCandidate | None = None
    target_memory_id: str | None = Field(default=None, min_length=1, max_length=64)
    confidence_delta: float = Field(default=0.0, ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_target(self) -> MemoryMutation:
        if self.action == "create" and self.candidate is None:
            raise ValueError("create mutations require a candidate")
        if self.action != "create" and self.target_memory_id is None:
            raise ValueError("non-create mutations require target_memory_id")
        return self


class MemoryConsolidation(BaseModel):
    """Complete output of one memory-worker run for one message range."""

    model_config = ConfigDict(extra="forbid")

    summary: ConversationSummary
    mutations: list[MemoryMutation] = Field(default_factory=list, max_length=3)
