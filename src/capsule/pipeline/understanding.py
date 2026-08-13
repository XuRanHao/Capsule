import asyncio
import base64
import json
import logging
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field

from capsule.config import Settings
from capsule.db.repositories import AssetRepository, EmbeddingAsset, EmbeddingRepository
from capsule.enums import AssetType
from capsule.features import FEATURE_DIMENSION_DISAMBIGUATION_PROMPT
from capsule.media.model_image import ModelImageCache
from capsule.media.video_frames import (
    FFmpegVideoFrameExtractor,
    VideoFrameExtractor,
    logical_video_frame_request,
)
from capsule.schemas import AssetUnderstanding
from capsule.video_output import is_logical_video_asset

logger = logging.getLogger(__name__)

_DESCRIPTION_CONTEXT_RULES = (
    "asset_name 与 asset_description 聚焦素材本身可见或可读的内容。"
    "描述直接呈现可观察或可阅读的事实。文件名、目录等元数据由独立路径处理，"
    "无需在内容描述中解释其来源。"
)

_SUBJECT_OUTPUT_RULES = (
    "subject_content 走内容提取路径，聚焦画面、正文或视频中最显著的具体人物、物体和场景对象。"
    "subject 写简短稳定的实体名称，description 写身份、类别、外观、动作或关系等区分信息。"
    "这里无需采用文件名或目录给主体命名；元数据实体会由独立路径提取，并在随后合并去重。"
    "salience 使用 0 到 1 的相对数值，体现各实体在当前素材中的重要程度。"
)

_GENERIC_SUBJECT_HINTS = frozenset(
    {
        "asset",
        "file",
        "image",
        "img",
        "photo",
        "picture",
        "screenshot",
        "untitled",
        "素材",
        "图片",
        "图像",
        "照片",
        "截图",
        "未命名",
        "场景",
        "参考",
        "参考图",
        "三视图",
        "四视图",
    }
)


class UnderstandingClient(Protocol):
    async def understand_asset(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        asset_id: str | None = None,
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
        video_frame_extractor: VideoFrameExtractor | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_repository = embedding_repository
        self._asset_repository = asset_repository
        self._model_client = model_client
        self._artifact_reader = artifact_reader
        self._image_cache = image_cache or ModelImageCache(
            target_bytes=settings.model_image_target_bytes,
            max_edge=settings.model_image_max_edge,
            fixed_size=settings.understanding_image_size,
            max_entries=settings.model_image_cache_entries,
        )
        self._video_frame_extractor = video_frame_extractor or FFmpegVideoFrameExtractor(
            concurrency=settings.ffmpeg_concurrency
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
                    understanding = await self._model_client.understand_asset(
                        messages,
                        asset_id=asset.asset_id,
                    )
                    # Drop transient video-frame data URIs before the database write.
                    del messages
                    _attach_ocr_confidence(understanding, asset)
                finally:
                    model_ms = (time.perf_counter() - phase_started) * 1000
                phase_started = time.perf_counter()
                try:
                    if asset.asset_type == AssetType.AUDIO_SEGMENT.value:
                        await self._asset_repository.store_understanding(
                            asset_id=asset.asset_id,
                            understanding=understanding,
                            raw_content=understanding.transcript or "",
                            file_info_updates={
                                "transcription": {
                                    "status": "completed",
                                    "model": self._settings.understanding_model,
                                }
                            },
                        )
                    else:
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
                "asset_name 不超过 20 字；asset_description 用 40 到 120 字客观描述可检索"
                "内容。三个 Feature 彼此独立，各自描述范围以 JSON Schema 中对应字段的说明"
                f"为准。{FEATURE_DIMENSION_DISAMBIGUATION_PROMPT}"
                "根据当前维度相关信息在素材中的占比、显著性和丰富程度自适应调整信息"
                "密度：信息丰富时保留更多有区分度的事实，信息有限时只输出少量可靠内容，不为"
                "填满数量而扩写。主体维度把不同实体拆成不同 item。事实来自素材或可靠上下文。"
                "使用本地 OCR 内容时，evidence 以“OCR：”开头，真实"
                "OCR 置信度由后端附加。"
                "处理音频时完整转写 transcript，evidence 引用听到的内容时以“转写：”开头。"
                f"{_SUBJECT_OUTPUT_RULES}"
                f"{_DESCRIPTION_CONTEXT_RULES}"
                "以有证据、可复用的表达为主。"
            ),
        }
        content_context = _content_context_payload(asset)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "请理解以下素材并输出约定 JSON。\n"
                    f"content_context={json.dumps(content_context, ensure_ascii=False)}"
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
            image_uri = _image_source_uri(asset)
            prepared_image = await self._image_cache.prepare(
                cache_key=asset.content_hash,
                mime_type=_image_mime_type(asset),
                loader=lambda: _read_local_source(image_uri),
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
            if asset.raw_content and asset.raw_content.strip():
                content.append(
                    {
                        "type": "text",
                        "text": f"本地 OCR 识别文字：\n{asset.raw_content[:8_000]}",
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
        elif asset.asset_type == AssetType.AUDIO_SEGMENT.value:
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": await asyncio.to_thread(_audio_segment_data_uri, asset),
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        "请直接听取当前音频片段。transcript 填写完整转写；若没有可辨识人声，"
                        "填写空字符串。其余字段同时描述可听见的讲话、音乐、环境声和事件。"
                    ),
                }
            )
        return [system, {"role": "user", "content": content}]

    async def _video_keyframe_data_uris(self, asset: EmbeddingAsset) -> list[str]:
        raw_keyframes = asset.file_info.get("keyframes")
        data_uris: list[str] = []
        if isinstance(raw_keyframes, list):
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
        if data_uris or not is_logical_video_asset(
            asset_type=asset.asset_type,
            file_info=asset.file_info,
        ):
            return data_uris
        frames = await self._video_frame_extractor.extract(
            logical_video_frame_request(
                source_uri=asset.source_storage_uri,
                file_info=asset.file_info,
                source_locator=asset.source_locator,
            )
        )
        data_uris.extend(_data_uri("image/jpeg", frame) for frame in frames)
        return data_uris


def _read_local_source(storage_uri: str) -> bytes:
    parsed = urlparse(storage_uri)
    if parsed.scheme != "file":
        raise ValueError("understanding currently requires a local source file")
    path = Path(unquote(parsed.path))
    if not path.is_file():
        raise ValueError(f"source file no longer exists: {path}")
    return path.read_bytes()


def _image_source_uri(asset: EmbeddingAsset) -> str:
    """Document image children point at a materialised image, not the DOCX/PDF."""

    return asset.derived_file_uri or asset.source_storage_uri


def _image_mime_type(asset: EmbeddingAsset) -> str:
    value = asset.file_info.get("mime_type")
    return value if isinstance(value, str) and value else asset.source_mime_type


def _data_uri(mime_type: str, content: bytes) -> str:
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _audio_segment_data_uri(asset: EmbeddingAsset) -> str:
    parsed = urlparse(asset.source_storage_uri)
    if parsed.scheme != "file":
        raise ValueError("audio understanding currently requires a local source file")
    source = Path(unquote(parsed.path))
    start_ms = int(asset.source_locator.get("start_ms", 0))
    end_ms = int(asset.source_locator.get("end_ms", 0))
    if not source.is_file() or end_ms <= start_ms:
        raise ValueError("audio segment source or time range is invalid")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("ffmpeg is required for audio understanding")
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-to",
            f"{end_ms / 1000:.3f}",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode or not completed.stdout:
        detail = completed.stderr.decode(errors="replace").strip()
        raise ValueError(detail or "audio extraction failed")
    return _data_uri("audio/wav", completed.stdout)


def _asset_context_payload(asset: EmbeddingAsset) -> dict[str, Any]:
    source_contexts = [
        dict(context) for context in asset.source_contexts if isinstance(context, Mapping)
    ]
    associated_text = list(
        dict.fromkeys(
            text.strip()[:2_000]
            for context in source_contexts
            if context.get("relation_type") in {"caption", "preceding_text", "ocr_text"}
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
    entity_hints = _entity_hints(
        asset=asset,
        document_title=document_title,
        heading_path=normalized_heading_path,
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
        "entity_hints": entity_hints,
    }


def _content_context_payload(asset: EmbeddingAsset) -> dict[str, Any]:
    """Return technical context only, keeping entity metadata on its own path."""

    return {
        "asset_type": asset.asset_type,
        "file_info": _compact_file_info(asset.file_info),
    }


def _entity_hints(
    *,
    asset: EmbeddingAsset,
    document_title: str | None,
    heading_path: Sequence[str],
) -> list[dict[str, str]]:
    """Return stable context candidates before asset-local naming hints."""

    raw_candidates: list[tuple[str, str, str]] = []
    if asset.source_relative_path:
        relative_path = PurePosixPath(asset.source_relative_path)
        raw_candidates.extend(
            ("directory", part, "collection") for part in reversed(relative_path.parts[:-1])
        )
    if document_title:
        raw_candidates.append(("document_title", document_title, "document"))
    raw_candidates.extend(("heading", item, "section") for item in reversed(heading_path))
    raw_candidates.extend(
        ("file_tree", item, "collection") for item in reversed(asset.file_tree_context[-12:])
    )
    if asset.file_name:
        raw_candidates.append(("file_name", PurePosixPath(asset.file_name).stem, "asset"))

    hints: list[dict[str, str]] = []
    seen: set[str] = set()
    for source, raw_value, scope in raw_candidates:
        value = _normalize_subject_hint(raw_value)
        if value is None or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        hints.append({"source": source, "value": value, "scope": scope})
        if len(hints) == 8:
            break
    return hints


def _normalize_subject_hint(value: str) -> str | None:
    normalized = value.strip().strip("/\\").strip()
    if not normalized or len(normalized) > 80:
        return None
    if PurePosixPath(normalized).suffix:
        normalized = PurePosixPath(normalized).stem.strip()
    if not normalized or normalized.casefold() in _GENERIC_SUBJECT_HINTS:
        return None
    if re.fullmatch(r"[\W_]*\d[\d\W_]*", normalized):
        return None
    if re.fullmatch(
        r"(?i)(?:img|image|photo|screenshot|jimeng)[-_ ]?\d[\w\-. ]*",
        normalized,
    ):
        return None
    return normalized


def _attach_ocr_confidence(
    understanding: AssetUnderstanding,
    asset: EmbeddingAsset,
) -> None:
    ocr = asset.file_info.get("ocr")
    if not isinstance(ocr, Mapping) or ocr.get("status") != "accepted":
        return
    raw_confidence = ocr.get("confidence")
    if raw_confidence is None or isinstance(raw_confidence, bool):
        return
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError, OverflowError):
        return
    if not 0.0 <= confidence <= 1.0:
        return
    for feature_name in type(understanding.features).model_fields:
        feature = getattr(understanding.features, feature_name)
        for item in feature.items:
            if any(evidence.lstrip().lower().startswith("ocr：") for evidence in item.evidence):
                item.ocr_confidence = confidence


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
