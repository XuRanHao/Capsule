import asyncio
import base64
import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field

from capsule.config import Settings
from capsule.db.repositories import AssetRepository, EmbeddingAsset, EmbeddingRepository
from capsule.enums import AssetType
from capsule.media.model_image import ModelImageCache
from capsule.schemas import AssetUnderstanding

logger = logging.getLogger(__name__)


class UnderstandingClient(Protocol):
    async def understand_asset(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> AssetUnderstanding: ...


class ArtifactReader(Protocol):
    async def download_uri(self, uri: str) -> bytes: ...


class UnderstandingRunResult(BaseModel):
    workspace_id: str
    requested_asset_count: int
    completed_count: int = 0
    failed_count: int = 0
    understanding_duration_ms: float = 0
    feature_ready_duration_ms: float = 0
    errors: list[dict[str, str]] = Field(default_factory=list)


class AssetUnderstandingService:
    """Create names, descriptions, and searchable feature fields concurrently."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedding_repository: EmbeddingRepository,
        asset_repository: AssetRepository,
        model_client: UnderstandingClient,
        artifact_reader: ArtifactReader | None = None,
        image_cache: ModelImageCache | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_repository = embedding_repository
        self._asset_repository = asset_repository
        self._model_client = model_client
        self._artifact_reader = artifact_reader
        self._image_cache = image_cache or ModelImageCache(
            target_bytes=settings.model_image_target_bytes,
            max_edge=settings.model_image_max_edge,
            max_entries=settings.model_image_cache_entries,
        )

    async def run(
        self,
        *,
        workspace_id: str,
        asset_ids: Sequence[str] | None = None,
    ) -> UnderstandingRunResult:
        assets = await self._embedding_repository.list_assets(
            workspace_id=workspace_id,
            asset_ids=asset_ids,
        )
        assets = [
            asset
            for asset in assets
            if not asset.asset_description or not asset.asset_features
        ]
        run_started = time.perf_counter()
        semaphore = asyncio.Semaphore(self._settings.understanding_concurrency)
        outcomes = await asyncio.gather(
            *(self._understand_one(asset, semaphore=semaphore) for asset in assets)
        )
        errors = [
            {"asset_id": asset_id, "error": error}
            for asset_id, error, _, _ in outcomes
            if error is not None
        ]
        elapsed_ms = (time.perf_counter() - run_started) * 1000 if assets else 0.0
        model_ms = sum(outcome[2] for outcome in outcomes)
        storage_ms = sum(outcome[3] for outcome in outcomes)
        measured_ms = model_ms + storage_ms
        understanding_ms = (
            elapsed_ms * model_ms / measured_ms if measured_ms else elapsed_ms
        )
        return UnderstandingRunResult(
            workspace_id=workspace_id,
            requested_asset_count=len(assets),
            completed_count=len(assets) - len(errors),
            failed_count=len(errors),
            understanding_duration_ms=understanding_ms,
            feature_ready_duration_ms=max(0.0, elapsed_ms - understanding_ms),
            errors=errors,
        )

    async def _understand_one(
        self,
        asset: EmbeddingAsset,
        *,
        semaphore: asyncio.Semaphore,
    ) -> tuple[str, str | None, float, float]:
        async with semaphore:
            model_ms = 0.0
            storage_ms = 0.0
            try:
                messages = await self._messages(asset)
                phase_started = time.perf_counter()
                try:
                    understanding = await self._model_client.understand_asset(messages)
                finally:
                    model_ms = (time.perf_counter() - phase_started) * 1000
                phase_started = time.perf_counter()
                try:
                    await self._asset_repository.store_understanding(
                        asset_id=asset.asset_id,
                        understanding=understanding,
                    )
                finally:
                    storage_ms = (time.perf_counter() - phase_started) * 1000
                return asset.asset_id, None, model_ms, storage_ms
            except Exception as exc:
                error = str(exc) or type(exc).__name__
                logger.exception("understanding failed for asset %s", asset.asset_id)
                return asset.asset_id, error[:2000], model_ms, storage_ms

    async def _messages(self, asset: EmbeddingAsset) -> list[dict[str, Any]]:
        system = {
            "role": "system",
            "content": (
                "你是多模态 Asset 特征提取器，只描述当前 Asset；上下文仅用于消歧。"
                "十个 Feature 彼此独立，不跨维度重复或推导。asset_name 不超过 20 字；"
                "asset_description 用 40 到 120 字客观描述可检索内容。每个 Feature 的 value "
                "只含该维度 0 到 5 个中文关键词，以分号连接；evidence 最多一条且不超过 40 字。"
                "维度边界：subject_content=主体与动作；scene_theme=场景题材；"
                "visual_style=表现技法；color_composition=色彩构图；"
                "mood_atmosphere=情绪氛围；character_state_or_psychology=人物可观察状态；"
                "asset_usage=用途；target_audience=受众；provenance=客观来源；"
                "rights_version_authorship=有证据的权利版本作者。无证据使用 null/unknown，"
                "不得虚构。只输出约定 JSON，不要 Markdown。"
            ),
        }
        metadata = _asset_context_payload(asset)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "请理解以下素材并输出约定 JSON。\n"
                    f"metadata={json.dumps(metadata, ensure_ascii=False)}"
                ),
            }
        ]
        if asset.asset_type in {
            AssetType.MARKDOWN_BLOCK.value,
            AssetType.TEXT_BLOCK.value,
        }:
            content.append(
                {
                    "type": "text",
                    "text": f"素材正文：\n{(asset.raw_content or '')[:60_000]}",
                }
            )
        elif asset.asset_type == AssetType.IMAGE.value:
            prepared_image = await self._image_cache.prepare(
                cache_key=asset.content_hash,
                mime_type=asset.source_mime_type,
                loader=lambda: _read_local_source(asset.source_storage_uri),
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_uri(
                            prepared_image.mime_type,
                            prepared_image.content,
                        ),
                    },
                }
            )
        elif asset.asset_type == AssetType.VIDEO_SEGMENT.value:
            keyframes = await self._video_keyframes(asset)
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_uri("image/jpeg", frame)},
                }
                for frame in keyframes
            )
            if not keyframes:
                content.append(
                    {
                        "type": "text",
                        "text": "视频关键帧暂不可读，请仅依据文件信息和关联上下文输出保守描述。",
                    }
                )
        return [system, {"role": "user", "content": content}]

    async def _video_keyframes(self, asset: EmbeddingAsset) -> list[bytes]:
        if self._artifact_reader is None:
            return []
        raw_keyframes = asset.file_info.get("keyframes")
        if not isinstance(raw_keyframes, list):
            return []
        frames: list[bytes] = []
        for item in raw_keyframes[:3]:
            if not isinstance(item, Mapping):
                continue
            uri = item.get("uri")
            if not isinstance(uri, str):
                continue
            try:
                frames.append(await self._artifact_reader.download_uri(uri))
            except Exception:
                logger.warning("could not load video keyframe %s", uri, exc_info=True)
        return frames


def _read_local_source(storage_uri: str) -> bytes:
    parsed = urlparse(storage_uri)
    if parsed.scheme != "file":
        raise ValueError("understanding currently requires a local source file")
    path = Path(unquote(parsed.path))
    if not path.is_file():
        raise ValueError(f"source file no longer exists: {path}")
    return path.read_bytes()


def _data_uri(mime_type: str, content: bytes) -> str:
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _asset_context_payload(asset: EmbeddingAsset) -> dict[str, Any]:
    source_contexts = [
        dict(context)
        for context in asset.source_contexts
        if isinstance(context, Mapping)
    ]
    associated_text = list(
        dict.fromkeys(
            text.strip()[:2_000]
            for context in source_contexts
            if context.get("relation_type") in {"caption", "preceding_text"}
            and isinstance((text := context.get("text")), str)
            and text.strip()
        )
    )[:4]
    heading_path = asset.source_locator.get("heading_path")
    if not isinstance(heading_path, list):
        heading_path = next(
            (
                context.get("heading_path")
                for context in source_contexts
                if isinstance(context.get("heading_path"), list)
            ),
            [],
        )
    normalized_heading_path = (
        [item for item in heading_path if isinstance(item, str)]
        if isinstance(heading_path, list)
        else []
    )
    document_title = next(
        (
            context.get("document_title")
            for context in source_contexts
            if isinstance(context.get("document_title"), str)
        ),
        None,
    )
    return {
        "asset": {
            "asset_type": asset.asset_type,
            "file_name": asset.file_name,
            "file_info": _compact_file_info(asset.file_info),
        },
        "context": {
            "source_path": asset.source_relative_path,
            "document_title": document_title,
            "heading_path": normalized_heading_path[:8],
            "associated_text": associated_text,
            "relations": [
                {
                    key: context[key]
                    for key in ("relation_type", "source_path", "paragraph_id")
                    if key in context
                }
                for context in source_contexts[:4]
            ],
            "file_tree_context": asset.file_tree_context[-12:],
        },
    }


def _compact_file_info(file_info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in file_info.items()
        if key
        in {
            "width",
            "height",
            "duration_seconds",
            "frame_count",
            "format",
            "mime_type",
        }
        and isinstance(value, (str, int, float, bool))
    }
