import asyncio
import base64
import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field

from capsule.config import Settings
from capsule.db.repositories import AssetRepository, EmbeddingAsset, EmbeddingRepository
from capsule.enums import AssetType, FeatureStatus
from capsule.media.model_image import ModelImageCache
from capsule.schemas import AssetUnderstanding

logger = logging.getLogger(__name__)

_DESCRIPTION_CONTEXT_RULES = (
    "asset_name 与 asset_description 必须以素材本身可见或可读内容为主体。"
    "文件名、相对路径、目录层级、文档标题、标题路径和关联段落只作为语义上下文："
    "其中与素材内容一致且有实际语义的信息必须自然融入描述，不得写成元数据说明；"
    "路径或文字与素材内容冲突时，以素材本身为准。忽略纯编号、序号、通用词组成的"
    "文件名。禁止在结果中机械复述文件名、扩展名、目录、路径、来源路径或"
    "“位于某文件夹”等措辞。以上限制不适用于 asset_usage.description；素材用途说明"
    "必须明确写出 metadata.context.source_path。"
)

_USAGE_PATH_HINTS = (
    ("海报", "海报制作"),
    ("宣传", "宣传推广"),
    ("广告", "广告投放"),
    ("预告", "预告宣传"),
    ("封面", "封面设计"),
    ("头像", "头像制作"),
    ("壁纸", "壁纸使用"),
    ("电商", "电商展示"),
    ("社交媒体", "社交媒体发布"),
    ("社媒", "社交媒体发布"),
    ("参考", "创作参考"),
    ("插画", "插画创作"),
)

_GENERIC_USAGE_PATH_PARTS = {
    "asset",
    "assets",
    "file",
    "files",
    "image",
    "images",
    "img",
    "素材",
    "文件",
    "图片",
    "图像",
    "png",
    "jpg",
    "jpeg",
    "webp",
    "gif",
}


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
        force: bool = False,
    ) -> UnderstandingRunResult:
        assets = await self._embedding_repository.list_assets(
            workspace_id=workspace_id,
            asset_ids=asset_ids,
        )
        if not force:
            assets = [
                asset for asset in assets if not asset.asset_description or not asset.asset_features
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
        understanding_ms = elapsed_ms * model_ms / measured_ms if measured_ms else elapsed_ms
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
                    _attach_asset_usage_path_context(understanding, asset)
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
                f"{_DESCRIPTION_CONTEXT_RULES}"
                "维度边界：subject_content=主体与动作；scene_theme=场景题材；"
                "visual_style=表现技法；color_composition=色彩构图；"
                "mood_atmosphere=情绪氛围；character_state_or_psychology=人物可观察状态；"
                "asset_usage=用途；target_audience=受众；provenance=客观来源；"
                "rights_version_authorship=有证据的权利版本作者。unknown 表示该维度适用但"
                "当前证据不足；not_applicable 表示当前 Asset 不存在该维度所需对象或该维度"
                "不适用。status 为 unknown 或 not_applicable 时 value 必须为 null。"
                "character_state_or_psychology 采用严格适用性判断：图像或视频中只有清晰可见"
                "的人物或拟人角色，文本中只有明确描述人物状态时才适用；纯场景、建筑、物体、"
                "树木、遗骸或非拟人怪物必须返回 null/not_applicable。不得根据文件名、IP 背景、"
                "场景叙事或画面情绪推断不存在的人物，也不得把物体或场景状态写成人物状态。"
                "asset_usage 必须优先使用 metadata.context.source_path 和 file_tree_context："
                "当目录名能表达海报、宣传、封面、广告、预告、参考等用途时，status 使用 "
                "metadata，value 只写规范化用途语义；description 必须自然说明完整相对路径"
                "及其对应用途，source_path 必须原样返回该相对路径。不得返回本地绝对路径。"
                "无证据不得虚构。只输出约定 JSON，不要 Markdown。"
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
            keyframes = await self._video_keyframe_data_uris(asset)
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": keyframe_data_uri},
                }
                for keyframe_data_uri in keyframes
            )
            if not keyframes:
                content.append(
                    {
                        "type": "text",
                        "text": "视频关键帧暂不可读，请仅依据文件信息和关联上下文输出保守描述。",
                    }
                )
        return [system, {"role": "user", "content": content}]

    async def _video_keyframe_data_uris(self, asset: EmbeddingAsset) -> list[str]:
        raw_keyframes = asset.file_info.get("keyframes")
        if not isinstance(raw_keyframes, list):
            return []
        data_uris: list[str] = []
        for item in raw_keyframes[:3]:
            if not isinstance(item, Mapping):
                continue
            uri = item.get("uri")
            if not isinstance(uri, str):
                continue
            parsed = urlparse(uri)
            if parsed.scheme == "data":
                data_uris.append(uri)
                continue
            if parsed.scheme != "s3" or self._artifact_reader is None:
                continue
            try:
                content = await self._artifact_reader.download_uri(uri)
                data_uris.append(_data_uri("image/jpeg", content))
            except Exception:
                logger.warning("could not load video keyframe %s", uri, exc_info=True)
        return data_uris


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
        dict(context) for context in asset.source_contexts if isinstance(context, Mapping)
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


def _attach_asset_usage_path_context(
    understanding: AssetUnderstanding,
    asset: EmbeddingAsset,
) -> None:
    """Make Asset usage path evidence deterministic instead of model-optional."""
    source_path = _normalized_relative_path(asset.source_relative_path)
    if source_path is None:
        return

    usage = understanding.features.asset_usage
    usage.source_path = source_path
    path_hint = _usage_hint_from_path(
        source_path=source_path,
        file_tree_context=asset.file_tree_context,
    )
    if path_hint is not None:
        if not usage.value:
            usage.value = path_hint
        if usage.status in {
            FeatureStatus.UNKNOWN,
            FeatureStatus.NOT_APPLICABLE,
            FeatureStatus.INFERRED,
        }:
            usage.status = FeatureStatus.METADATA
        usage.confidence = max(usage.confidence, 0.9)

    directory = _source_directory(source_path)
    if usage.value:
        normalized_usage = usage.value.replace("；", "、")
        directory_clause = f"，所属目录为「{directory}」" if directory else ""
        usage.description = (
            f"该素材对应相对文件路径「{source_path}」{directory_clause}，"
            f"路径语义与素材信息表明其用途为{normalized_usage}。"
        )
    else:
        usage.description = (
            f"该素材对应相对文件路径「{source_path}」，当前路径和素材内容尚未提供可确认的具体用途。"
        )
    usage.evidence = [f"相对文件路径：{source_path}"]


def _normalized_relative_path(value: str) -> str | None:
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        return None
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path.as_posix()


def _source_directory(source_path: str) -> str:
    directory = PurePosixPath(source_path).parent.as_posix()
    return "" if directory == "." else directory


def _usage_hint_from_path(
    *,
    source_path: str,
    file_tree_context: Sequence[str],
) -> str | None:
    directory = _source_directory(source_path)
    context_parts = [
        item.strip() for item in file_tree_context if isinstance(item, str) and item.strip()
    ]
    combined = "/".join([directory, *context_parts])
    for token, usage in _USAGE_PATH_HINTS:
        if token in combined:
            return usage

    meaningful_parts = [
        part
        for part in dict.fromkeys([*PurePosixPath(directory).parts, *context_parts])
        if part and part.lower() not in _GENERIC_USAGE_PATH_PARTS
    ]
    return "；".join(meaningful_parts[:3]) or None


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
