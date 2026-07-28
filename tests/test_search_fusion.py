from capsule.enums import EmbeddingType
from capsule.search.fusion import WeightedReciprocalRankFusion
from capsule.search.models import ChannelRecall, QueryVector, VectorSearchHit


def hit(asset_id: str, similarity: float) -> VectorSearchHit:
    return VectorSearchHit(
        embedding_id=f"emb_{asset_id}",
        asset_id=asset_id,
        source_file_id=f"source_{asset_id}",
        asset_type="image",
        embedding_revision=1,
        similarity=similarity,
    )


def test_weighted_rrf_merges_assets_and_keeps_channel_evidence() -> None:
    native = ChannelRecall(
        query_vector=QueryVector(
            channel="native_multimodal",
            embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
            vector=[1.0],
            weight=1.0,
        ),
        hits=(hit("a", 0.99), hit("b", 0.90), hit("a", 0.80)),
    )
    description = ChannelRecall(
        query_vector=QueryVector(
            channel="asset_description",
            embedding_type=EmbeddingType.ASSET_DESCRIPTION,
            vector=[1.0],
            weight=0.8,
        ),
        hits=(hit("b", 0.95), hit("a", 0.85)),
    )

    fused = WeightedReciprocalRankFusion(rrf_k=60, candidate_cap=10).fuse((native, description))

    assert [item.asset_id for item in fused] == ["a", "b"]
    assert len(fused[0].matched_channels) == 2
    assert fused[0].score == (1.0 / 61) + (0.8 / 62)
