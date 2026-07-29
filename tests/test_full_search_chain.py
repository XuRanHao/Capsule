import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from capsule.enums import EmbeddingType
from capsule.search.fusion import FusionEngine
from capsule.search.models import (
    ChannelMatch,
    ChannelRecall,
    DimensionQuery,
    FusedHit,
    FusionMethod,
    ParsedQuery,
    QueryDimensionSource,
    QueryType,
    QueryVector,
    RerankBatch,
    RerankItem,
    SearchAssetRecord,
    SearchRequest,
    VectorSearchHit,
)
from capsule.search.query_parser import QueryParser
from capsule.search.rerank import SearchReranker
from capsule.search.result_builder import SearchResultBuilder


def _hit(asset_id: str, similarity: float) -> VectorSearchHit:
    return VectorSearchHit(
        embedding_id=f"embedding_{asset_id}",
        asset_id=asset_id,
        source_file_id=f"source_{asset_id}",
        asset_type="image",
        embedding_revision=1,
        similarity=similarity,
    )


def test_normalized_weighted_similarity_is_selectable() -> None:
    subject = ChannelRecall(
        query_vector=QueryVector(
            channel="subject_content",
            embedding_type=EmbeddingType.SUBJECT_CONTENT,
            vector=[1],
            weight=0.6,
        ),
        hits=(_hit("a", 0.7), _hit("b", 0.9)),
    )
    mood = ChannelRecall(
        query_vector=QueryVector(
            channel="mood_atmosphere",
            embedding_type=EmbeddingType.MOOD_ATMOSPHERE,
            vector=[1],
            weight=0.4,
        ),
        hits=(_hit("a", 0.95), _hit("b", 0.5)),
    )

    result = FusionEngine(candidate_cap=10).fuse(
        (subject, mood),
        FusionMethod.NORMALIZED_WEIGHTED_SIMILARITY,
    )

    assert [item.asset_id for item in result] == ["b", "a"]
    assert result[0].score == 0.6
    assert result[0].matched_channels[0].rrf_contribution == 0


class FakeUnderstandingClient:
    async def parse_search_query(
        self,
        request: SearchRequest,
        *,
        image_url: str | None,
    ) -> ParsedQuery:
        assert image_url == "https://example.com/query.jpg"
        return ParsedQuery(
            query_summary="图片中的人物、构图和黄昏氛围",
            dimension_queries=[
                DimensionQuery(
                    embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
                    query="参考图片",
                    weight=0.45,
                    source=QueryDimensionSource.IMAGE,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.SUBJECT_CONTENT,
                    query="人物",
                    weight=0.15,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.SCENE_THEME,
                    query="户外黄昏",
                    weight=0.10,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.VISUAL_STYLE,
                    query="动画电影",
                    weight=0.10,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.COLOR_COMPOSITION,
                    query="蓝紫色与暖金色",
                    weight=0.10,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.MOOD_ATMOSPHERE,
                    query="安静而有叙事感",
                    weight=0.10,
                ),
            ],
        )

    async def rerank_search_results(
        self,
        request: SearchRequest,
        *,
        image_url: str | None,
        candidates: Sequence[Mapping[str, object]],
    ) -> RerankBatch:
        return RerankBatch(
            items=[
                RerankItem(asset_id="b", relevance_score=0.95, reason="约束全部命中"),
                RerankItem(asset_id="a", relevance_score=0.30, reason="氛围偏差"),
            ]
        )


async def test_precision_image_parser_keeps_documented_six_routes() -> None:
    request = SearchRequest(
        workspace_id="workspace_demo",
        query_type=QueryType.IMAGE,
        query_image_url="https://example.com/query.jpg",
        precision_mode=True,
    )
    parsed, reasons = await QueryParser(FakeUnderstandingClient()).parse(
        request,
        image_url=request.query_image_url,
    )

    assert reasons == ()
    assert len(parsed.dimension_queries) == 6
    assert math.isclose(sum(item.weight for item in parsed.dimension_queries), 1)
    assert parsed.dimension_queries[0].weight == 0.45


async def test_seed_reranker_reorders_only_hydrated_candidates() -> None:
    request = SearchRequest(
        workspace_id="workspace_demo",
        query_type=QueryType.TEXT,
        query_text="黄昏",
        rerank="doubao_seed_2_lite",
    )
    assets = {asset_id: _asset(asset_id, source_id=f"source_{asset_id}") for asset_id in ("a", "b")}
    ranked, annotations, reasons = await SearchReranker(FakeUnderstandingClient()).rerank(
        request=request,
        image_url=None,
        ranked_hits=[
            FusedHit(asset_id="a", source_file_id="source_a", asset_type="image", score=1),
            FusedHit(asset_id="b", source_file_id="source_b", asset_type="image", score=0.5),
        ],
        assets=assets,
    )

    assert reasons == ()
    assert [item.asset_id for item in ranked] == ["b", "a"]
    assert annotations["b"] == (0.95, "约束全部命中")


def test_adjacent_video_segments_are_folded_before_same_source_limit() -> None:
    match = ChannelMatch(
        channel="native_multimodal",
        embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
        embedding_id="embedding_current",
        embedding_revision=1,
        rank=1,
        similarity=0.9,
        fusion_contribution=0.1,
    )
    assets = {
        "video_a": _asset(
            "video_a",
            source_id="video_source",
            asset_type="video_segment",
            locator={"start_ms": 0, "end_ms": 5_000},
        ),
        "video_b": _asset(
            "video_b",
            source_id="video_source",
            asset_type="video_segment",
            locator={"start_ms": 6_000, "end_ms": 10_000},
        ),
    }
    results = SearchResultBuilder(same_source_limit=3).build(
        ranked_hits=[
            FusedHit(
                asset_id="video_a",
                source_file_id="video_source",
                asset_type="video_segment",
                score=1,
                matched_channels=[match],
            ),
            FusedHit(
                asset_id="video_b",
                source_file_id="video_source",
                asset_type="video_segment",
                score=0.9,
                matched_channels=[match],
            ),
        ],
        assets=assets,
        workspace_id="workspace_demo",
        allowed_asset_types=(),
        top_k=20,
    )

    assert len(results) == 1
    assert results[0].group_kind == "video_segments"
    assert results[0].folded_asset_ids == ["video_a", "video_b"]
    assert results[0].source_locator["end_ms"] == 10_000


def test_database_state_rejects_stale_milvus_revision() -> None:
    asset = _asset("image_a", source_id="source_a")
    stale = ChannelMatch(
        channel="native_multimodal",
        embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
        embedding_id="embedding_current",
        embedding_revision=0,
        rank=1,
        similarity=0.99,
        fusion_contribution=0.1,
    )

    results = SearchResultBuilder().build(
        ranked_hits=[
            FusedHit(
                asset_id=asset.asset_id,
                source_file_id=asset.source_file_id,
                asset_type=asset.asset_type,
                score=0.1,
                matched_channels=[stale],
            )
        ],
        assets={asset.asset_id: asset},
        workspace_id=asset.workspace_id,
        allowed_asset_types=(),
        top_k=20,
    )

    assert results == []


def test_database_state_uses_current_embedding_when_stale_vector_ranks_first() -> None:
    query = QueryVector(
        channel="native_multimodal",
        embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
        vector=[1],
        weight=1,
    )
    recall = ChannelRecall(
        query_vector=query,
        hits=(
            VectorSearchHit(
                embedding_id="embedding_stale",
                asset_id="image_a",
                source_file_id="source_a",
                asset_type="image",
                embedding_revision=0,
                similarity=0.99,
            ),
            VectorSearchHit(
                embedding_id="embedding_current",
                asset_id="image_a",
                source_file_id="source_a",
                asset_type="image",
                embedding_revision=1,
                similarity=0.95,
            ),
        ),
    )
    ranked = FusionEngine(candidate_cap=10).fuse((recall,), FusionMethod.WEIGHTED_RRF)
    asset = _asset("image_a", source_id="source_a")

    results = SearchResultBuilder().build(
        ranked_hits=ranked,
        assets={asset.asset_id: asset},
        workspace_id=asset.workspace_id,
        allowed_asset_types=(),
        top_k=20,
    )

    assert len(results) == 1
    assert results[0].matched_channels[0].embedding_id == "embedding_current"
    assert results[0].matched_channels[0].rank == 2
    assert results[0].file_type == ".png"
    assert results[0].source_file is not None
    assert results[0].source_file.sha256 == "f" * 64


def test_result_builder_preserves_reranked_order_after_database_validation() -> None:
    match = ChannelMatch(
        channel="native_multimodal",
        embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
        embedding_id="embedding_current",
        embedding_revision=1,
        rank=1,
        similarity=0.9,
        fusion_contribution=0.1,
    )
    assets = {
        asset_id: _asset(asset_id, source_id=f"source_{asset_id}")
        for asset_id in ("a", "b")
    }

    results = SearchResultBuilder().build(
        ranked_hits=[
            FusedHit(
                asset_id="b",
                source_file_id="source_b",
                asset_type="image",
                score=0.1,
                matched_channels=[match],
            ),
            FusedHit(
                asset_id="a",
                source_file_id="source_a",
                asset_type="image",
                score=0.9,
                matched_channels=[match],
            ),
        ],
        assets=assets,
        workspace_id="workspace_demo",
        allowed_asset_types=(),
        top_k=20,
    )

    assert [item.asset_id for item in results] == ["b", "a"]


def _asset(
    asset_id: str,
    *,
    source_id: str,
    asset_type: str = "image",
    locator: dict[str, object] | None = None,
) -> SearchAssetRecord:
    created_at = datetime(2026, 7, 29, tzinfo=UTC)
    return SearchAssetRecord(
        asset_id=asset_id,
        workspace_id="workspace_demo",
        project_id="project_default",
        source_file_id=source_id,
        asset_type=asset_type,
        file_name=f"{asset_id}.png",
        file_type=".png",
        asset_key=f"key_{asset_id}",
        content_hash=asset_id.ljust(64, "0"),
        asset_name=asset_id,
        asset_name_source="model",
        asset_description="黄昏素材",
        asset_features={},
        file_tree_context=[],
        source_contexts=[],
        file_info={},
        source_locator=locator or {},
        raw_content=None,
        derived_file_uri=None,
        preview_uri=None,
        processing_status="completed",
        feature_revision=1,
        embedding_revision=1,
        error_message=None,
        created_at=created_at,
        updated_at=created_at,
        source_workspace_id="workspace_demo",
        source_project_id="project_default",
        source_file_name="source.md",
        source_file_type=".md",
        source_mime_type="text/markdown",
        source_relative_path="source.md",
        source_file_tree_context=[],
        source_storage_uri="file:///source.md",
        source_sha256="f" * 64,
        source_file_size_bytes=128,
        source_processing_status="completed",
        source_error_message=None,
        source_created_at=created_at,
        source_updated_at=created_at,
        indexed_embedding_ids=frozenset({"embedding_current"}),
    )
