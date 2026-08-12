"""Compare concise prompts for metadata/content entity reconciliation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from capsule.config import Settings
from capsule.model_clients.doubao import DoubaoClient

PROMPTS = {
    "direct": (
        "合并元数据实体和内容实体。same_entity 表示元数据名称就是内容主体的身份；"
        "contains_content 表示元数据是容纳内容主体的场景或集合；其余为 related。"
        "结合组内多项内容整体判断并输出简短描述。"
    ),
    "positive": (
        "请判断两条实体路径怎样连接。人物或物体名称与多项内容描述一致时选择 same_entity；"
        "地点、空间或集合承载不同内容主体时选择 contains_content；其他关联选择 related。"
        "关注可跨素材复用的关系。"
    ),
    "examples": (
        "将 metadata_entity 与一组 content_entities 建立最贴切的关系。"
        "例如角色名‘封不觉’与多张同一人物图是 same_entity；场景名‘飞船内部’与其中的"
        "雕塑、宴会厅或人物是 contains_content。其他情况使用 related。输出简短关系描述。"
    ),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _groups(graph: dict[str, object]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for asset in graph["assets"]:  # type: ignore[index,union-attr]
        source_path = asset["source_path"]  # type: ignore[index]
        metadata_entity = source_path.split("/", 1)[0]
        primary = asset["primary_subject"]  # type: ignore[index]
        grouped.setdefault(metadata_entity, []).append(
            {
                "subject": primary["subject"],
                "description": primary["description"],
            }
        )
    return [
        {"metadata_entity": name, "content_entities": entities}
        for name, entities in grouped.items()
    ]


async def _main() -> None:
    args = _arguments()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    groups = _groups(graph)
    client = DoubaoClient(Settings())
    try:
        results = {}
        for name, prompt in PROMPTS.items():
            response = await client.resolve_metadata_content_entities(
                groups,
                guidance=prompt,
            )
            results[name] = response.model_dump(mode="json")
    finally:
        await client.close()
    args.output.write_text(
        json.dumps({"groups": groups, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(_main())
