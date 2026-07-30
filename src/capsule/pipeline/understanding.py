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
from capsule.media.model_image import prepare_model_image
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
    ) -> None:
        self._settings = settings
        self._embedding_repository = embedding_repository
        self._asset_repository = asset_repository
        self._model_client = model_client
        self._artifact_reader = artifact_reader

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
                "你是多模态 Asset 特征提取器。请结合 Asset 本体与上下文信息理解当前 Asset。"
                "上下文用于补充名称、含义、主题、人物关系、场景背景和来源，但最终结果始终"
                "描述当前 Asset，而不是概括整篇文档。十个 Feature 是相互独立的特征空间："
                "每条信息只归入定义最准确的一个维度；每个维度只回答自己的问题，彼此不补充、"
                "不推导、不重复。"
                "只能输出合法 JSON，不要 Markdown。JSON 必须包含 "
                "asset_name、asset_description、features。features 必须完整包含 "
                "subject_content、scene_theme、visual_style、color_composition、"
                "mood_atmosphere、character_state_or_psychology、asset_usage、"
                "target_audience、provenance、rights_version_authorship。每个 Feature 都必须包含 "
                "value、status、confidence、evidence；status 只能是 observed、inferred、"
                "metadata、user_supplied、unknown、not_applicable，confidence 为 0 到 1；"
                "evidence 必须是 JSON 字符串数组，即使只有一条也必须写成 [\"证据\"]。"
                "每个 value 只输出该维度内部的 0 到 5 个关键词，使用中文分号连接；关键词应为"
                "2 到 8 个汉字的名词、形容词或短语，不写完整句子；无法确定或不适用时 value "
                "使用 null。各维度分别回答：subject_content 当前有什么；scene_theme 是什么"
                "题材或概念；visual_style 采用什么视觉表现；color_composition 如何组织颜色"
                "和画面；mood_atmosphere 形成什么情绪氛围；"
                "character_state_or_psychology 角色处于什么动作、姿态、表情或互动状态；"
                "asset_usage 可用于什么场景；target_audience 面向什么人群；provenance 有什么"
                "客观来源和载体；rights_version_authorship 有什么明确的权利、版本和作者信息。"
                "asset_name 简洁可辨识，asset_description 描述可检索的主体、场景、风格、"
                "色彩、构图和氛围。不得虚构无法从素材、目录和关联段落中得到的事实。"
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
            image_bytes = await asyncio.to_thread(_read_local_source, asset.source_storage_uri)
            prepared_image = await asyncio.to_thread(
                prepare_model_image,
                image_bytes,
                asset.source_mime_type,
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
            text.strip()
            for context in source_contexts
            if context.get("relation_type") in {"caption", "preceding_text"}
            and isinstance((text := context.get("text")), str)
            and text.strip()
        )
    )
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
            "asset_id": asset.asset_id,
            "asset_type": asset.asset_type,
            "file_name": asset.file_name,
            "file_info": asset.file_info,
        },
        "context": {
            "source_file_id": asset.source_file_id,
            "source_path": asset.source_relative_path,
            "document_title": document_title,
            "heading_path": heading_path,
            "associated_text": associated_text,
            "source_contexts": source_contexts,
            "file_tree_context": asset.file_tree_context,
        },
    }
