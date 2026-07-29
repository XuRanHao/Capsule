"""Run independent HDBSCAN clustering for every configured Embedding Type."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from capsule.config import Settings
from capsule.db.repositories import (
    ClusterEmbeddingAsset,
    ClusterMembershipWrite,
    ClusterRepository,
    EmbeddingRepository,
)
from capsule.enums import (
    ClusterRepresentativeRole,
    ClusterRunStatus,
    EmbeddingType,
)
from capsule.pipeline.cluster_summary import (
    ClusterSummaryRepresentative,
    build_cluster_summary_messages,
)
from capsule.pipeline.clustering import (
    ClusterMemberCandidate,
    InsufficientDataError,
    RepresentativeSelection,
    cluster_vectors,
    dataset_hash,
    dynamic_hdbscan_parameters,
    select_cluster_representatives,
)
from capsule.schemas import ClusterCapsuleWrite, ClusterRepresentativeWrite, ClusterSummary

logger = logging.getLogger(__name__)


class ClusterVectorStore(Protocol):
    async def ensure_collection(self) -> bool: ...

    async def fetch_vectors(self, embedding_ids: Sequence[str]) -> dict[str, list[float]]: ...


class ClusterSummaryClient(Protocol):
    async def summarize_cluster(self, messages: Sequence[Mapping[str, Any]]) -> ClusterSummary: ...


@dataclass(slots=True, frozen=True)
class _LoadedClusterVector:
    asset: ClusterEmbeddingAsset
    vector: list[float]


class EmbeddingTypeClusterResult(BaseModel):
    embedding_type: EmbeddingType
    cluster_run_id: str
    status: ClusterRunStatus
    indexed_asset_count: int
    vector_count: int
    missing_vector_count: int = 0
    cluster_count: int = 0
    noise_count: int = 0
    capsule_ids: list[str] = Field(default_factory=list)
    error: str | None = None


#===========================================
#      One Embedding Type per run
#===========================================


class ClusterService:
    """Cluster exactly one Embedding Type per invocation."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedding_repository: EmbeddingRepository,
        cluster_repository: ClusterRepository,
        vector_store: ClusterVectorStore,
        model_client: ClusterSummaryClient,
    ) -> None:
        self._settings = settings
        self._embedding_repository = embedding_repository
        self._cluster_repository = cluster_repository
        self._vector_store = vector_store
        self._model_client = model_client

    async def run(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType = EmbeddingType.NATIVE_MULTIMODAL,
        cluster_run_id: str | None = None,
        optimize_parameters: bool = False,
    ) -> EmbeddingTypeClusterResult:
        """Run PCA, HDBSCAN, and Capsule generation for one explicit channel."""
        return await self._run_embedding_type(
            workspace_id=workspace_id,
            embedding_type=embedding_type,
            cluster_run_id=cluster_run_id,
            optimize_parameters=optimize_parameters,
        )

    async def _run_embedding_type(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType,
        cluster_run_id: str | None,
        optimize_parameters: bool,
    ) -> EmbeddingTypeClusterResult:
        assets: list[ClusterEmbeddingAsset] = []
        loaded: list[_LoadedClusterVector] = []
        run_id = cluster_run_id
        try:
            assets = await self._embedding_repository.list_indexed_cluster_embeddings(
                workspace_id=workspace_id,
                embedding_type=embedding_type.value,
                model_name=self._settings.embedding_model,
                dimension=self._settings.embedding_dimension,
                milvus_collection=self._settings.milvus_collection,
            )
            loaded = await self._load_vectors(assets)
            preprocessing = {
                "normalization": "l2",
                "pca_max_dimension": 64,
                "post_pca_normalization": "l2",
                "parameter_selection": (
                    "adaptive_dbcv_silhouette"
                    if optimize_parameters
                    else "size_based_default"
                ),
                "indexed_asset_count": len(assets),
                "missing_vector_count": len(assets) - len(loaded),
            }
            embedding_ids = [item.asset.embedding_id for item in loaded]
            if run_id is None:
                run_id = await self._cluster_repository.create_run(
                    workspace_id=workspace_id,
                    embedding_type=embedding_type.value,
                    embedding_ids=embedding_ids,
                    dataset_hash=dataset_hash(embedding_ids),
                    preprocessing=preprocessing,
                    parameters={},
                )
            else:
                await self._cluster_repository.start_pending_run(
                    cluster_run_id=run_id,
                    workspace_id=workspace_id,
                    embedding_type=embedding_type.value,
                    embedding_ids=embedding_ids,
                    dataset_hash=dataset_hash(embedding_ids),
                    preprocessing=preprocessing,
                    parameters={},
                )
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            if run_id is None:
                raise
            logger.exception(
                "could not prepare clustering for workspace=%s embedding_type=%s",
                workspace_id,
                embedding_type.value,
            )
            await self._cluster_repository.fail_run(cluster_run_id=run_id, error=error)
            return EmbeddingTypeClusterResult(
                embedding_type=embedding_type,
                cluster_run_id=run_id,
                status=ClusterRunStatus.FAILED,
                indexed_asset_count=len(assets),
                vector_count=len(loaded),
                missing_vector_count=len(assets) - len(loaded),
                error=error[:2000],
            )

        assert run_id is not None

        try:
            dynamic_hdbscan_parameters(len(loaded))
        except InsufficientDataError:
            await self._cluster_repository.complete_run(
                cluster_run_id=run_id,
                cluster_count=0,
                noise_count=len(loaded),
                noise_ratio=1.0 if loaded else 0.0,
                status=ClusterRunStatus.INSUFFICIENT_DATA,
            )
            return EmbeddingTypeClusterResult(
                embedding_type=embedding_type,
                cluster_run_id=run_id,
                status=ClusterRunStatus.INSUFFICIENT_DATA,
                indexed_asset_count=len(assets),
                vector_count=len(loaded),
                missing_vector_count=len(assets) - len(loaded),
                noise_count=len(loaded),
            )

        try:
            matrix = np.asarray([item.vector for item in loaded], dtype=np.float32)
            clustered = cluster_vectors(
                matrix,
                pca_dimension=64,
                optimize_parameters=optimize_parameters,
            )
            candidates = [
                ClusterMemberCandidate(
                    asset_id=item.asset.asset_id,
                    source_file_id=item.asset.source_file_id,
                    membership_probability=float(clustered.probabilities[index]),
                )
                for index, item in enumerate(loaded)
            ]
            selections = select_cluster_representatives(
                clustered.transformed_vectors,
                clustered.labels,
                candidates,
            )
            capsule_ids = await self._summarize_and_store_capsules(
                run_id=run_id,
                workspace_id=workspace_id,
                embedding_type=embedding_type,
                labels=clustered.labels,
                probabilities=clustered.probabilities,
                transformed_vectors=clustered.transformed_vectors,
                loaded=loaded,
                selections=selections,
            )
            await self._cluster_repository.store_memberships(
                cluster_run_id=run_id,
                memberships=_build_memberships(
                    labels=clustered.labels,
                    probabilities=clustered.probabilities,
                    transformed_vectors=clustered.transformed_vectors,
                    loaded=loaded,
                    selections=selections,
                    capsule_ids=capsule_ids,
                ),
            )
            await self._cluster_repository.complete_run(
                cluster_run_id=run_id,
                cluster_count=clustered.cluster_count,
                noise_count=clustered.noise_count,
                noise_ratio=clustered.noise_ratio,
                preprocessing={
                    **preprocessing,
                    "pca_dimension": clustered.pca_dimension,
                },
                parameters={
                    "min_cluster_size": clustered.parameters.min_cluster_size,
                    "min_samples": clustered.parameters.min_samples,
                    "cluster_selection_method": clustered.parameters.cluster_selection_method,
                    "quality_score": clustered.quality_score,
                    "candidates_evaluated": clustered.parameter_candidates_evaluated,
                },
            )
            return EmbeddingTypeClusterResult(
                embedding_type=embedding_type,
                cluster_run_id=run_id,
                status=ClusterRunStatus.COMPLETED,
                indexed_asset_count=len(assets),
                vector_count=len(loaded),
                missing_vector_count=len(assets) - len(loaded),
                cluster_count=clustered.cluster_count,
                noise_count=clustered.noise_count,
                capsule_ids=[capsule_ids[label] for label in sorted(capsule_ids)],
            )
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            logger.exception(
                "clustering failed for workspace=%s embedding_type=%s",
                workspace_id,
                embedding_type.value,
            )
            await self._cluster_repository.fail_run(cluster_run_id=run_id, error=error)
            return EmbeddingTypeClusterResult(
                embedding_type=embedding_type,
                cluster_run_id=run_id,
                status=ClusterRunStatus.FAILED,
                indexed_asset_count=len(assets),
                vector_count=len(loaded),
                missing_vector_count=len(assets) - len(loaded),
                error=error[:2000],
            )

    async def _load_vectors(
        self,
        assets: list[ClusterEmbeddingAsset],
    ) -> list[_LoadedClusterVector]:
        if not assets:
            return []
        await self._vector_store.ensure_collection()
        vectors = await self._vector_store.fetch_vectors([asset.embedding_id for asset in assets])
        return [
            _LoadedClusterVector(asset=asset, vector=vectors[asset.embedding_id])
            for asset in assets
            if asset.embedding_id in vectors
        ]

    async def _summarize_and_store_capsules(
        self,
        *,
        run_id: str,
        workspace_id: str,
        embedding_type: EmbeddingType,
        labels: NDArray[np.int_],
        probabilities: NDArray[np.float64],
        transformed_vectors: NDArray[np.float32],
        loaded: list[_LoadedClusterVector],
        selections: Mapping[int, list[RepresentativeSelection]],
    ) -> dict[int, str]:
        """Call the naming model with each cluster's selected Asset rows only."""
        assets_by_id = {item.asset.asset_id: item.asset for item in loaded}
        capsule_ids: dict[int, str] = {}
        for label in sorted(selections):
            representatives = selections[label]
            member_indices = np.flatnonzero(labels == label)
            average_probability = float(probabilities[member_indices].mean())
            prompt_representatives = [
                ClusterSummaryRepresentative(
                    asset_id=representative.asset_id,
                    role=ClusterRepresentativeRole(representative.role),
                    asset_type=assets_by_id[representative.asset_id].asset_type,
                    asset_name=assets_by_id[representative.asset_id].asset_name,
                    asset_description=assets_by_id[representative.asset_id].asset_description,
                    asset_features=assets_by_id[representative.asset_id].asset_features,
                    file_tree_context=assets_by_id[representative.asset_id].file_tree_context,
                    membership_probability=representative.membership_probability,
                    distance_to_medoid=representative.distance_to_medoid,
                )
                for representative in representatives
            ]
            summary = await self._model_client.summarize_cluster(
                build_cluster_summary_messages(
                    embedding_type=embedding_type.value,
                    member_count=len(member_indices),
                    average_membership_probability=average_probability,
                    representatives=prompt_representatives,
                )
            )
            stored = await self._cluster_repository.upsert_capsule(
                ClusterCapsuleWrite(
                    cluster_run_id=run_id,
                    workspace_id=workspace_id,
                    embedding_type=embedding_type.value,
                    cluster_label=label,
                    summary=summary,
                    member_count=len(member_indices),
                    average_membership_probability=average_probability,
                    representatives=[
                        ClusterRepresentativeWrite(
                            asset_id=representative.asset_id,
                            role=ClusterRepresentativeRole(representative.role),
                            rank=representative.rank,
                            distance_to_medoid=representative.distance_to_medoid,
                            membership_probability=representative.membership_probability,
                        )
                        for representative in representatives
                    ],
                )
            )
            capsule_ids[label] = stored.cluster_capsule_id
        return capsule_ids


def _build_memberships(
    *,
    labels: NDArray[np.int_],
    probabilities: NDArray[np.float64],
    transformed_vectors: NDArray[np.float32],
    loaded: list[_LoadedClusterVector],
    selections: Mapping[int, list[RepresentativeSelection]],
    capsule_ids: Mapping[int, str],
) -> list[ClusterMembershipWrite]:
    medoid_vectors: dict[int, NDArray[np.float32]] = {}
    asset_index = {item.asset.asset_id: index for index, item in enumerate(loaded)}
    for label, representatives in selections.items():
        medoid = next(
            representative
            for representative in representatives
            if representative.role == ClusterRepresentativeRole.MEDOID.value
        )
        medoid_vectors[label] = transformed_vectors[asset_index[medoid.asset_id]]

    memberships: list[ClusterMembershipWrite] = []
    for index, item in enumerate(loaded):
        label = int(labels[index])
        is_noise = label == -1
        distance = (
            None
            if is_noise
            else float(np.linalg.norm(transformed_vectors[index] - medoid_vectors[label]))
        )
        memberships.append(
            ClusterMembershipWrite(
                asset_id=item.asset.asset_id,
                cluster_capsule_id=None if is_noise else capsule_ids[label],
                hdbscan_label=label,
                membership_probability=float(probabilities[index]),
                is_noise=is_noise,
                distance_to_representative=distance,
            )
        )
    return memberships
