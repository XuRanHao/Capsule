"""Structured recall-query generation for one new Agent turn."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

logger = logging.getLogger(__name__)

RecallQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=30),
]


class MemoryRecallIntent(BaseModel):
    """Only the model-generated additions; raw user input is added by the graph."""

    model_config = ConfigDict(extra="forbid")

    extra_queries: list[RecallQuery] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def deduplicate_queries(self) -> MemoryRecallIntent:
        seen: set[str] = set()
        unique: list[str] = []
        for query in self.extra_queries:
            normalized = query.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(query)
        self.extra_queries = unique
        return self


class MemoryIntentRecognizer(Protocol):
    """Produce short supplemental queries from the current short-term context."""

    async def recognize(
        self,
        *,
        user_input: str,
        short_term_context: Mapping[str, object],
    ) -> list[str]: ...


class StructuredMemoryIntentModel(Protocol):
    """The strict JSON capability shared by the configured model client."""

    async def generate_structured(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        output_type: type[MemoryRecallIntent],
        schema_name: str,
        max_output_tokens: int,
        model: str | None = None,
    ) -> MemoryRecallIntent: ...


class RawInputMemoryIntentRecognizer:
    """Safe fallback: the graph still recalls with the original user input."""

    async def recognize(
        self,
        *,
        user_input: str,
        short_term_context: Mapping[str, object],
    ) -> list[str]:
        del user_input, short_term_context
        return []


class ModelMemoryIntentRecognizer:
    """Generate bounded recall rewrites without giving the model planning authority."""

    def __init__(
        self,
        *,
        model: StructuredMemoryIntentModel,
        model_name: str,
        max_output_tokens: int,
    ) -> None:
        if not model_name.strip():
            raise ValueError("memory intent model must not be blank")
        if max_output_tokens < 1:
            raise ValueError("memory intent max output tokens must be positive")
        self._model = model
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens

    async def recognize(
        self,
        *,
        user_input: str,
        short_term_context: Mapping[str, object],
    ) -> list[str]:
        payload = {
            "original_input": user_input,
            "short_term_context": dict(short_term_context),
        }
        try:
            intent = await self._model.generate_structured(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        ),
                    },
                ],
                output_type=MemoryRecallIntent,
                schema_name="capsule_agent_memory_recall_intent",
                max_output_tokens=self._max_output_tokens,
                model=self._model_name,
            )
        except Exception:
            logger.warning("agent memory intent recognition failed", exc_info=True)
            return []
        return intent.extra_queries


_SYSTEM_PROMPT = """你负责为长期记忆召回补充检索短句。仅输出 JSON。
原始输入会由系统固定参与检索；根据原始输入和短期会话上下文，额外输出至多两条，
每条不超过 30 字的短句。
短句应表达可能保存在跨会话记忆中的稳定规则、项目背景、已确认决定或用户偏好；不要回答问题，不要生成工具参数，
不要复述原始输入，也不要输出临时任务、工具结果、口令或敏感信息。没有合适补充时返回空数组。"""
