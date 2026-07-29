import asyncio
import base64
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field

from capsule.config import Settings
from capsule.db.repositories import AssetRepository, EmbeddingAsset, EmbeddingRepository
from capsule.enums import AssetType
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
        semaphore = asyncio.Semaphore(self._settings.understanding_concurrency)
        outcomes = await asyncio.gather(
            *(self._understand_one(asset, semaphore=semaphore) for asset in assets)
        )
        errors = [
            {"asset_id": asset_id, "error": error}
            for asset_id, error in outcomes
            if error is not None
        ]
        return UnderstandingRunResult(
            workspace_id=workspace_id,
            requested_asset_count=len(assets),
            completed_count=len(assets) - len(errors),
            failed_count=len(errors),
            errors=errors,
        )

    async def _understand_one(
        self,
        asset: EmbeddingAsset,
        *,
        semaphore: asyncio.Semaphore,
    ) -> tuple[str, str | None]:
        async with semaphore:
            try:
                messages = await self._messages(asset)
                understanding = await self._model_client.understand_asset(messages)
                await self._asset_repository.store_understanding(
                    asset_id=asset.asset_id,
                    understanding=understanding,
                )
                return asset.asset_id, None
            except Exception as exc:
                error = str(exc) or type(exc).__name__
                logger.exception("understanding failed for asset %s", asset.asset_id)
                return asset.asset_id, error[:2000]

    async def _messages(self, asset: EmbeddingAsset) -> list[dict[str, Any]]:
        system = {
            "role": "system",
            "content": (
                "你是多模态素材理解器。只能输出合法 JSON，不要 Markdown。JSON 必须包含 "
                "asset_name、asset_description、features。features 必须完整包含 "
                "subject_content、scene_theme、visual_style、color_composition、"
                "mood_atmosphere、character_state_or_psychology、asset_usage、"
                "target_audience、provenance、rights_version_authorship。每个 Feature 都必须包含 "
                "value、status、confidence、evidence；status 只能是 observed、inferred、"
                "metadata、user_supplied、unknown、not_applicable，confidence 为 0 到 1；"
                "evidence 必须是 JSON 字符串数组，即使只有一条也必须写成 [\"证据\"]。"
                "asset_name 简洁可辨识，asset_description 描述可检索的主体、场景、风格、"
                "色彩、构图和氛围。不得虚构无法从素材、目录和关联段落中得到的事实。"
            ),
        }
        metadata = {
            "asset_id": asset.asset_id,
            "asset_type": asset.asset_type,
            "file_name": asset.file_name,
            "file_tree_context": asset.file_tree_context,
            "source_contexts": asset.source_contexts,
            "file_info": asset.file_info,
        }
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
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_uri(asset.source_mime_type, image_bytes),
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
