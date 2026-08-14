from collections.abc import Callable, Sequence

from capsule.search.models import (
    ChannelMatch,
    ChannelRecall,
    FusedHit,
    FusionMethod,
    TextSearchHit,
)

_LOCAL_TEXT_CHANNEL = "local_text"


class WeightedReciprocalRankFusion:
    def __init__(self, *, rrf_k: int = 20, candidate_cap: int = 300) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if candidate_cap < 1:
            raise ValueError("candidate_cap must be positive")
        self._rrf_k = rrf_k
        self._candidate_cap = candidate_cap

    def fuse(self, channels: tuple[ChannelRecall, ...]) -> list[FusedHit]:
        fused: dict[str, FusedHit] = {}
        for recall in channels:
            seen_assets: set[str] = set()
            seen_embeddings: set[str] = set()
            for rank, hit in enumerate(recall.hits, start=1):
                if hit.embedding_id in seen_embeddings:
                    continue
                seen_embeddings.add(hit.embedding_id)
                contribution = recall.query_vector.weight / (self._rrf_k + rank)
                candidate = fused.setdefault(
                    hit.asset_id,
                    FusedHit(
                        asset_id=hit.asset_id,
                        source_file_id=hit.source_file_id,
                        asset_type=hit.asset_type,
                    ),
                )
                if hit.asset_id not in seen_assets:
                    candidate.score += contribution
                    seen_assets.add(hit.asset_id)
                candidate.matched_channels.append(
                    ChannelMatch(
                        channel=recall.query_vector.channel,
                        embedding_type=recall.query_vector.embedding_type,
                        embedding_id=hit.embedding_id,
                        embedding_revision=hit.embedding_revision,
                        rank=rank,
                        similarity=hit.similarity,
                        fusion_contribution=contribution,
                        rrf_contribution=contribution,
                    )
                )
        ranked = sorted(fused.values(), key=lambda item: (-item.score, item.asset_id))
        return ranked[: self._candidate_cap]


class NormalizedWeightedSimilarityFusion:
    def __init__(self, *, candidate_cap: int = 300) -> None:
        if candidate_cap < 1:
            raise ValueError("candidate_cap must be positive")
        self._candidate_cap = candidate_cap

    def fuse(self, channels: tuple[ChannelRecall, ...]) -> list[FusedHit]:
        fused: dict[str, FusedHit] = {}
        for recall in channels:
            if not recall.hits:
                continue
            similarities = [item.similarity for item in recall.hits]
            minimum = min(similarities)
            maximum = max(similarities)
            spread = maximum - minimum
            seen_assets: set[str] = set()
            seen_embeddings: set[str] = set()
            for rank, hit in enumerate(recall.hits, start=1):
                if hit.embedding_id in seen_embeddings:
                    continue
                seen_embeddings.add(hit.embedding_id)
                normalized = 1.0 if spread <= 1e-12 else (hit.similarity - minimum) / spread
                contribution = recall.query_vector.weight * normalized
                candidate = fused.setdefault(
                    hit.asset_id,
                    FusedHit(
                        asset_id=hit.asset_id,
                        source_file_id=hit.source_file_id,
                        asset_type=hit.asset_type,
                    ),
                )
                if hit.asset_id not in seen_assets:
                    candidate.score += contribution
                    seen_assets.add(hit.asset_id)
                candidate.matched_channels.append(
                    ChannelMatch(
                        channel=recall.query_vector.channel,
                        embedding_type=recall.query_vector.embedding_type,
                        embedding_id=hit.embedding_id,
                        embedding_revision=hit.embedding_revision,
                        rank=rank,
                        similarity=hit.similarity,
                        fusion_contribution=contribution,
                    )
                )
        ranked = sorted(fused.values(), key=lambda item: (-item.score, item.asset_id))
        return ranked[: self._candidate_cap]


class FusionEngine:
    def __init__(self, *, rrf_k: int = 20, candidate_cap: int = 300) -> None:
        self._rrf_k = rrf_k
        self._candidate_cap = candidate_cap
        self._rrf = WeightedReciprocalRankFusion(
            rrf_k=rrf_k,
            candidate_cap=candidate_cap,
        )
        self._normalized = NormalizedWeightedSimilarityFusion(
            candidate_cap=candidate_cap,
        )

    def fuse(
        self,
        channels: tuple[ChannelRecall, ...],
        method: FusionMethod,
        *,
        text_hits: Sequence[TextSearchHit] = (),
    ) -> list[FusedHit]:
        if method is FusionMethod.NORMALIZED_WEIGHTED_SIMILARITY:
            ranked = self._normalized.fuse(channels)
            return _merge_text_hits(
                ranked,
                text_hits,
                candidate_cap=self._candidate_cap,
                contribution=lambda score, _rank: score,
            )
        ranked = self._rrf.fuse(channels)
        return _merge_text_hits(
            ranked,
            text_hits,
            candidate_cap=self._candidate_cap,
            contribution=lambda _score, rank: 1 / (self._rrf_k + rank),
        )


def _merge_text_hits(
    ranked: list[FusedHit],
    text_hits: Sequence[TextSearchHit],
    *,
    candidate_cap: int,
    contribution: Callable[[float, int], float],
) -> list[FusedHit]:
    """Merge one unified local-text route with the collective vector route."""

    fused = {item.asset_id: item for item in ranked}
    if not text_hits:
        return ranked
    scores = [item.score for item in text_hits]
    minimum = min(scores)
    maximum = max(scores)
    spread = maximum - minimum
    for rank, hit in enumerate(text_hits, start=1):
        normalized = 1.0 if spread <= 1e-12 else (hit.score - minimum) / spread
        amount = contribution(normalized, rank)
        candidate = fused.setdefault(
            hit.asset_id,
            FusedHit(
                asset_id=hit.asset_id,
                source_file_id=hit.source_file_id,
                asset_type=hit.asset_type,
            ),
        )
        candidate.score += amount
        candidate.matched_channels.append(
            ChannelMatch(
                channel=_LOCAL_TEXT_CHANNEL,
                embedding_type=None,
                embedding_id=None,
                embedding_revision=None,
                rank=rank,
                similarity=hit.score,
                fusion_contribution=amount,
                rrf_contribution=amount,
            )
        )
    return sorted(fused.values(), key=lambda item: (-item.score, item.asset_id))[
        :candidate_cap
    ]
