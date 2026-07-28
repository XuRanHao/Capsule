import asyncio
from collections.abc import Sequence

from capsule.config import Settings
from capsule.enums import EmbeddingType
from capsule.search.models import QueryEmbeddingPlan, QueryVector, VectorSearchHit
from capsule.search.recall import MultiChannelRecall


class ConcurrentVectorRepository:
    def __init__(self) -> None:
        self.active = 0
        self.max_observed = 0

    async def search(
        self,
        *,
        vector: list[float],
        workspace_id: str,
        embedding_type: str,
        asset_types: tuple[str, ...],
        limit: int,
    ) -> Sequence[VectorSearchHit]:
        self.active += 1
        self.max_observed = max(self.max_observed, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return []


async def test_recall_channels_run_concurrently() -> None:
    repository = ConcurrentVectorRepository()
    recall = MultiChannelRecall(repository, Settings())
    plan = QueryEmbeddingPlan(
        vectors=tuple(
            QueryVector(
                channel=embedding_type.value,
                embedding_type=embedding_type,
                vector=[1.0],
                weight=1.0,
            )
            for embedding_type in EmbeddingType
        )
    )

    result = await recall.search(
        plan=plan,
        workspace_id="workspace_demo",
        asset_types=(),
        top_k=20,
    )

    assert len(result.channels) == 4
    assert repository.max_observed == 4
