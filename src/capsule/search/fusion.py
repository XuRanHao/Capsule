from capsule.search.models import ChannelMatch, ChannelRecall, FusedHit, FusionMethod


class WeightedReciprocalRankFusion:
    def __init__(self, *, rrf_k: int = 60, candidate_cap: int = 300) -> None:
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
            for rank, hit in enumerate(recall.hits, start=1):
                if hit.asset_id in seen_assets:
                    continue
                seen_assets.add(hit.asset_id)
                contribution = recall.query_vector.weight / (self._rrf_k + rank)
                candidate = fused.setdefault(
                    hit.asset_id,
                    FusedHit(
                        asset_id=hit.asset_id,
                        source_file_id=hit.source_file_id,
                        asset_type=hit.asset_type,
                    ),
                )
                candidate.score += contribution
                candidate.matched_channels.append(
                    ChannelMatch(
                        channel=recall.query_vector.channel,
                        embedding_type=recall.query_vector.embedding_type,
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
            unique_hits = []
            seen_assets: set[str] = set()
            for hit in recall.hits:
                if hit.asset_id not in seen_assets:
                    unique_hits.append(hit)
                    seen_assets.add(hit.asset_id)
            if not unique_hits:
                continue
            similarities = [item.similarity for item in unique_hits]
            minimum = min(similarities)
            maximum = max(similarities)
            spread = maximum - minimum
            for rank, hit in enumerate(unique_hits, start=1):
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
                candidate.score += contribution
                candidate.matched_channels.append(
                    ChannelMatch(
                        channel=recall.query_vector.channel,
                        embedding_type=recall.query_vector.embedding_type,
                        rank=rank,
                        similarity=hit.similarity,
                        fusion_contribution=contribution,
                    )
                )
        ranked = sorted(fused.values(), key=lambda item: (-item.score, item.asset_id))
        return ranked[: self._candidate_cap]


class FusionEngine:
    def __init__(self, *, rrf_k: int = 60, candidate_cap: int = 300) -> None:
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
    ) -> list[FusedHit]:
        if method is FusionMethod.NORMALIZED_WEIGHTED_SIMILARITY:
            return self._normalized.fuse(channels)
        return self._rrf.fuse(channels)
