"""Build a small Asset→primary Entity relationship graph from local images."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
from datetime import UTC, datetime
from pathlib import Path

from capsule.config import Settings
from capsule.db.repositories import EmbeddingAsset
from capsule.enums import AssetType
from capsule.model_clients.doubao import DoubaoClient
from capsule.pipeline.understanding import AssetUnderstandingService
from capsule.relation_graph import apply_asset_entity_relations, build_relation_graph
from capsule.schemas import AssetUnderstanding

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path, help="Directory containing the test images")
    parser.add_argument("--output", type=Path, required=True, help="Graph JSON output path")
    parser.add_argument(
        "--understandings",
        type=Path,
        help="Reuse an existing Asset understanding JSON instead of calling the model",
    )
    parser.add_argument(
        "--relations",
        type=Path,
        help="Reuse metadata/content relation decisions from an existing graph JSON",
    )
    return parser.parse_args()


def _sample_assets(source_root: Path) -> list[EmbeddingAsset]:
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
    )
    assets: list[EmbeddingAsset] = []
    for path in paths:
        relative_path = path.relative_to(source_root).as_posix()
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        assets.append(
            EmbeddingAsset(
                asset_id=f"sample_{hashlib.sha256(relative_path.encode()).hexdigest()[:12]}",
                workspace_id="thriller_park_graph_demo",
                project_id="thriller_park",
                source_file_id=f"source_{content_hash[:12]}",
                asset_type=AssetType.IMAGE.value,
                file_type=path.suffix.casefold(),
                content_hash=content_hash,
                embedding_revision=1,
                created_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
                raw_content=None,
                asset_description=None,
                asset_features={},
                derived_file_uri=None,
                source_storage_uri=path.resolve().as_uri(),
                source_mime_type=mime_type,
                file_name=path.name,
                source_relative_path=relative_path,
                file_tree_context=list(path.relative_to(source_root).parts[:-1]),
                file_info={"mime_type": mime_type},
            )
        )
    return assets


async def _understand_assets(
    assets: list[EmbeddingAsset],
) -> dict[str, AssetUnderstanding]:
    settings = Settings()
    client = DoubaoClient(settings)
    service = AssetUnderstandingService(
        settings=settings,
        embedding_repository=None,  # type: ignore[arg-type]
        asset_repository=None,  # type: ignore[arg-type]
        model_client=client,
    )

    async def understand(asset: EmbeddingAsset) -> tuple[str, AssetUnderstanding]:
        messages = await service._messages(asset)
        result = await client.understand_asset(messages, asset_id=asset.asset_id)
        return asset.asset_id, result

    try:
        results = await asyncio.gather(*(understand(asset) for asset in assets))
    finally:
        await client.close()
    return dict(results)


async def _resolve_metadata_content_relations(
    client: DoubaoClient,
    assets: list[EmbeddingAsset],
    understandings: dict[str, AssetUnderstanding],
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for asset in assets:
        parts = Path(asset.source_relative_path).parts[:-1]
        if not parts:
            continue
        metadata_entity = parts[0]
        primary = max(
            understandings[asset.asset_id].features.subject_content.items,
            key=lambda item: item.salience,
            default=None,
        )
        if primary is None:
            continue
        grouped.setdefault(metadata_entity, []).append(
            {
                "asset_id": asset.asset_id,
                "subject": primary.subject,
                "description": primary.description,
            }
        )
    resolution = await client.resolve_metadata_content_entities(
        [
            {"metadata_entity": name, "content_entities": entities}
            for name, entities in grouped.items()
            if len(entities) >= 2
        ]
    )
    return {
        "".join(
            character
            for character in item.metadata_entity.casefold()
            if character.isalnum()
        ): item.model_dump(mode="json")
        for item in resolution.decisions
    }


async def _generate_asset_entity_relations(
    client: DoubaoClient,
    graph: dict[str, object],
):
    assets_by_id = {
        item["asset_id"]: item for item in graph["assets"]  # type: ignore[index,union-attr]
    }
    entities_by_id = {
        item["entity_id"]: item for item in graph["entities"]  # type: ignore[index,union-attr]
    }
    candidates = []
    for edge in graph["edges"]:  # type: ignore[index,union-attr]
        source = assets_by_id.get(edge["source"])
        target = entities_by_id.get(edge["target"])
        if source is None or target is None:
            continue
        path_parts = Path(source["source_path"]).parts
        collection_name = path_parts[0] if len(path_parts) > 1 else None
        candidates.append(
            {
                "source_id": edge["source"],
                "target_id": edge["target"],
                "asset": {
                    "metadata": {
                        "source_path": source["source_path"],
                        "asset_name": source["asset_name"],
                        "entity_hints": (
                            [
                                {
                                    "value": collection_name,
                                    "scope": "collection",
                                }
                            ]
                            if collection_name
                            else []
                        ),
                    },
                    "content_description": source["asset_description"],
                    "content_subject": source["primary_subject"],
                },
                "entity": {
                    "name": target["name"],
                    "semantic": target["semantic"],
                },
            }
        )
    return await client.generate_asset_entity_relations(candidates)


async def _main() -> None:
    args = _arguments()
    source_root = args.source_root.resolve()
    assets = _sample_assets(source_root)
    if not assets:
        raise ValueError(f"no supported images found under {source_root}")
    understandings: dict[str, AssetUnderstanding] = {}
    if args.understandings:
        raw_understandings = json.loads(args.understandings.read_text(encoding="utf-8"))
        understandings.update({
            asset_id: AssetUnderstanding.model_validate(value)
            for asset_id, value in raw_understandings.items()
        })
    missing_assets = [asset for asset in assets if asset.asset_id not in understandings]
    if missing_assets:
        understandings.update(await _understand_assets(missing_assets))
    understanding_path = args.output.with_suffix(".understandings.json")
    understanding_path.parent.mkdir(parents=True, exist_ok=True)
    understanding_path.write_text(
        json.dumps(
            {
                asset_id: understanding.model_dump(mode="json")
                for asset_id, understanding in understandings.items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.relations:
        relation_graph = json.loads(args.relations.read_text(encoding="utf-8"))
        relations = relation_graph["metadata_content_relations"]
    else:
        settings = Settings()
        client = DoubaoClient(settings)
        try:
            relations = await _resolve_metadata_content_relations(
                client, assets, understandings
            )
        finally:
            await client.close()
    graph = build_relation_graph(
        assets,
        understandings,
        metadata_content_relations=relations,
    )
    client = DoubaoClient(Settings())
    try:
        edge_relations = await _generate_asset_entity_relations(client, graph)
    finally:
        await client.close()
    apply_asset_entity_relations(graph, edge_relations)
    graph["asset_entity_relation_decisions"] = edge_relations.model_dump(mode="json")
    graph["metadata_content_relations"] = relations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: graph[key] for key in ("asset_count", "entity_count", "edge_count")}))


if __name__ == "__main__":
    asyncio.run(_main())
