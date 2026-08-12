"""Scan reusable subject entities from one Capsule workspace without writing data."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from capsule.config import Settings
from capsule.db.models import Asset, Workspace
from capsule.db.session import Database


class SubjectMention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    subject: str = Field(description="简短、稳定、可跨资产复用的主体名称")
    description: str = Field(description="用于区分同名主体的客观主体描述")


class SubjectScanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subjects: list[SubjectMention]


def _subject_descriptions(features: Mapping[str, Any]) -> list[str]:
    subject = features.get("subject_content")
    if not isinstance(subject, Mapping):
        return []
    items = subject.get("items")
    if not isinstance(items, list):
        return []
    descriptions: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        description = item.get("description")
        if isinstance(description, str) and description.strip():
            descriptions.append(description.strip())
    return descriptions


_GENERIC_FILE_STEMS = {
    "image",
    "img",
    "picture",
    "photo",
    "screenshot",
    "whiteboard_exported_image",
}


def _metadata_hints(asset: Asset) -> list[str]:
    """Return semantic metadata hints while dropping obvious generated names."""

    candidates: list[str] = []
    stem = PurePath(asset.file_name).stem.strip()
    normalized_stem = re.sub(r"\s*\(\d+\)\s*$", "", stem).strip()
    if (
        normalized_stem
        and normalized_stem.lower() not in _GENERIC_FILE_STEMS
        and not re.fullmatch(r"[\d\W_]+", normalized_stem)
        and not re.fullmatch(r"\d{8}-\d{6}", normalized_stem)
    ):
        candidates.append(normalized_stem)

    if isinstance(asset.asset_name, str) and asset.asset_name.strip():
        candidates.append(asset.asset_name.strip())

    for item in asset.file_tree_context:
        if isinstance(item, str) and item.strip():
            candidates.append(item.strip())

    return list(dict.fromkeys(candidates))[:6]


async def _load_workspace_assets(
    database: Database,
    workspace_name: str,
    limit: int | None,
) -> tuple[str, list[dict[str, Any]]]:
    async with database.session() as session:
        workspace = (
            await session.execute(
                select(Workspace).where(Workspace.name == workspace_name)
            )
        ).scalar_one()
        statement = (
            select(Asset)
            .where(
                Asset.workspace_id == workspace.workspace_id,
                Asset.index_role.in_(("standalone", "parent")),
            )
            .order_by(Asset.created_at, Asset.asset_id)
        )
        if limit is not None:
            statement = statement.limit(limit)
        assets = list((await session.execute(statement)).scalars())

    rows = []
    for asset in assets:
        descriptions = _subject_descriptions(asset.asset_features)
        if not descriptions:
            continue
        rows.append(
            {
                "asset_id": asset.asset_id,
                "metadata_hints": _metadata_hints(asset),
                "subject_descriptions": descriptions,
            }
        )
    return workspace.workspace_id, rows


async def _scan_subjects(
    settings: Settings,
    assets: list[dict[str, Any]],
) -> SubjectScanResult:
    if settings.deepseek_api_key is None:
        raise RuntimeError("CAPSULE_DEEPSEEK_API_KEY is required")

    system_prompt = (
        "你是素材主体扫描器。输入包含 Asset 的 metadata_hints 和已有主体描述。"
        "对每个 Asset 输出 0 到 3 组主体名称和主体描述。"
        "主体是人物、角色、动物、物体、建筑或可被明确指认的地点；"
        "不要提取颜色、风格、动作、状态、构图、时间、天气、泛化背景和零碎部件。"
        "先检查 metadata_hints：如果其中存在具体角色名、作品名、道具名或产品名，"
        "并且与主体描述不冲突，优先用它命名主体。忽略纯编号、时间戳、通用文件名、"
        "导出工具名，以及单独出现的‘场景’‘参考’‘四视图’等处理性词语。"
        "metadata_hints 只能帮助命名主体，不能凭空添加主体。"
        "subject 必须简短稳定；description 只描述这个主体的身份、类别和有区分度的"
        "客观特征，不描述画面风格和构图，长度不超过 80 字。"
        "同一 Asset 内同义主体只保留一次，不要跨 Asset 猜测两个未命名角色身份相同。"
        "asset_id 必须原样复制。严格输出以下 json 对象："
        '{"subjects":[{"asset_id":"...","subject":"...",'
        '"description":"..."}]}。'
    )
    async with httpx.AsyncClient(
        base_url=settings.deepseek_base_url.rstrip("/"),
        trust_env=False,
        headers={
            "Authorization": f"Bearer {settings.deepseek_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        },
    ) as client:
        async def scan_batch(batch: list[dict[str, Any]]) -> list[SubjectMention]:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": settings.search_query_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": json.dumps({"assets": batch}, ensure_ascii=False),
                        },
                    ],
                    "thinking": {"type": "disabled"},
                    "max_tokens": 2048,
                    "response_format": {"type": "json_object"},
                },
                timeout=180,
            )
            if response.is_error:
                raise RuntimeError(
                    f"subject scan request failed ({response.status_code}): {response.text[:1000]}"
                )
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            return _decode_subjects(json.loads(content))

        batch_results = await asyncio.gather(
            *(scan_batch(assets[start : start + 10]) for start in range(0, len(assets), 10))
        )
    return SubjectScanResult(subjects=[item for batch in batch_results for item in batch])


def _decode_subjects(decoded: Mapping[str, Any]) -> list[SubjectMention]:
    if isinstance(decoded.get("subjects"), list):
        return SubjectScanResult.model_validate(decoded).subjects

    flattened: list[dict[str, Any]] = []
    for asset in decoded.get("assets", []):
        if not isinstance(asset, Mapping):
            continue
        asset_id = asset.get("asset_id")
        if not isinstance(asset_id, str):
            continue
        nested_subjects = asset.get("subjects")
        if not isinstance(nested_subjects, list):
            nested_subjects = [asset]
        for subject in nested_subjects:
            if not isinstance(subject, Mapping):
                continue
            subject_name = (
                subject.get("subject")
                or subject.get("entity_name")
                or subject.get("name")
            )
            description = subject.get("description") or subject.get("evidence") or subject_name
            if isinstance(subject_name, str) and subject_name.strip():
                flattened.append(
                    {
                        "asset_id": asset_id,
                        "subject": subject_name,
                        "description": description,
                    }
                )
    return SubjectScanResult(subjects=flattened).subjects


async def _run(args: argparse.Namespace) -> None:
    settings = Settings()
    database = Database(settings)
    try:
        workspace_id, assets = await _load_workspace_assets(
            database,
            args.workspace,
            args.limit,
        )
        result = await _scan_subjects(settings, assets)
    finally:
        await database.dispose()

    output = {
        "workspace_id": workspace_id,
        "workspace_name": args.workspace,
        "asset_count": len(assets),
        "subjects": [
            subject.model_dump()
            for subject in result.subjects
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="信息")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
