"""Run independent HDBSCAN clustering for every configured Embedding Type."""

import asyncio
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
    CurrentClusterMemberWrite,
    CurrentClusterPublish,
    CurrentClusterRepository,
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
    ensure_path_aware_cluster_summary,
)
from capsule.pipeline.clustering import (
    ClusterMemberCandidate,
    HdbscanParameters,
    InsufficientDataError,
    RepresentativeSelection,
    SemanticMergeParameters,
    SemanticMergeResult,
    cluster_vectors,
    dataset_hash,
    dynamic_hdbscan_parameters,
    merge_semantically_overlapping_clusters,
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


@dataclass(slots=True, frozen=True)
class _StoredClusterCapsule:
    cluster_capsule_id: str
    summary: ClusterSummary


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


# ===========================================
#      One Embedding Type per run
# ===========================================


class ClusterService:
    """Cluster exactly one Embedding Type per invocation."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedding_repository: EmbeddingRepository,
        cluster_repository: ClusterRepository,
        current_cluster_repository: CurrentClusterRepository | None = None,
        vector_store: ClusterVectorStore,
        model_client: ClusterSummaryClient,
    ) -> None:
        self._settings = settings
        self._embedding_repository = embedding_repository
        self._cluster_repository = cluster_repository
        self._current_cluster_repository = current_cluster_repository
        self._vector_store = vector_store
        self._model_client = model_client

    async def run(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType = EmbeddingType.NATIVE_MULTIMODAL,
        cluster_run_id: str | None = None,
        pca_dimension: int = 8,
        min_samples: int = 1,
        min_cluster_size: int = 3,
        optimize_parameters: bool = False,
    ) -> EmbeddingTypeClusterResult:
        """Run PCA, HDBSCAN, and Capsule generation for one explicit channel."""
        return await self._run_embedding_type(
            workspace_id=workspace_id,
            embedding_type=embedding_type,
            cluster_run_id=cluster_run_id,
            pca_dimension=pca_dimension,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            optimize_parameters=optimize_parameters,
        )

    async def _run_embedding_type(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType,
        cluster_run_id: str | None,
        pca_dimension: int,
        min_samples: int,
        min_cluster_size: int,
        optimize_parameters: bool,
    ) -> EmbeddingTypeClusterResult:
        assets: list[ClusterEmbeddingAsset] = []
        loaded: list[_LoadedClusterVector] = []
        run_id = cluster_run_id
        semantic_merge_parameters = SemanticMergeParameters(
            enabled=self._settings.cluster_semantic_merge_enabled,
            centroid_cosine_threshold=(self._settings.cluster_merge_centroid_cosine_threshold),
            cross_cluster_mean_cosine_threshold=(
                self._settings.cluster_merge_cross_mean_cosine_threshold
            ),
            merged_member_min_cosine_threshold=(
                self._settings.cluster_merge_member_min_cosine_threshold
            ),
        )
        try:
            assets = await self._embedding_repository.list_indexed_cluster_embeddings(
                workspace_id=workspace_id,
                embedding_type=embedding_type.value,
                model_name=self._settings.embedding_model,
                dimension=self._settings.embedding_dimension,
                milvus_collection=self._settings.milvus_collection,
            )
            resident_asset_ids = (
                await self._current_cluster_repository.list_resident_asset_ids(
                    workspace_id=workspace_id,
                    embedding_type=embedding_type.value,
                )
                if self._current_cluster_repository is not None
                else set()
            )
            assets = [asset for asset in assets if asset.asset_id not in resident_asset_ids]
            loaded = await self._load_vectors(assets)
            preprocessing = {
                "normalization": "l2",
                "post_pca_normalization": "l2",
                "requested_pca_dimension": pca_dimension,
                "parameter_selection": (
                    "user_defined_selection_optimized" if optimize_parameters else "user_defined"
                ),
                "indexed_asset_count": len(assets),
                "missing_vector_count": len(assets) - len(loaded),
                "resident_excluded_count": len(resident_asset_ids),
            }
            embedding_ids = [item.asset.embedding_id for item in loaded]
            if run_id is None:
                run_id = await self._cluster_repository.create_run(
                    workspace_id=workspace_id,
                    embedding_type=embedding_type.value,
                    embedding_ids=embedding_ids,
                    dataset_hash=dataset_hash(embedding_ids),
                    preprocessing=preprocessing,
                    parameters={
                        "min_cluster_size": min_cluster_size,
                        "min_samples": min_samples,
                        "cluster_selection_epsilon": (
                            self._settings.cluster_selection_epsilon
                        ),
                        "semantic_merge": _semantic_merge_metadata(semantic_merge_parameters),
                    },
                )
            else:
                await self._cluster_repository.start_pending_run(
                    cluster_run_id=run_id,
                    workspace_id=workspace_id,
                    embedding_type=embedding_type.value,
                    embedding_ids=embedding_ids,
                    dataset_hash=dataset_hash(embedding_ids),
                    preprocessing=preprocessing,
                    parameters={
                        "min_cluster_size": min_cluster_size,
                        "min_samples": min_samples,
                        "cluster_selection_epsilon": (
                            self._settings.cluster_selection_epsilon
                        ),
                        "semantic_merge": _semantic_merge_metadata(semantic_merge_parameters),
                    },
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
            if self._current_cluster_repository is not None:
                await self._current_cluster_repository.publish_dynamic_clusters(
                    run_id=run_id,
                    workspace_id=workspace_id,
                    embedding_type=embedding_type.value,
                    clusters=[],
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
                pca_dimension=pca_dimension,
                parameters=HdbscanParameters(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=self._settings.cluster_selection_epsilon,
                ),
                optimize_parameters=optimize_parameters,
            )
            semantic_merge = merge_semantically_overlapping_clusters(
                matrix,
                clustered.labels,
                parameters=semantic_merge_parameters,
            )
            if semantic_merge.decisions:
                logger.info(
                    "merged %s overlapping semantic clusters for workspace=%s "
                    "embedding_type=%s raw_cluster_count=%s final_cluster_count=%s",
                    len(semantic_merge.decisions),
                    workspace_id,
                    embedding_type.value,
                    semantic_merge.raw_cluster_count,
                    semantic_merge.cluster_count,
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
                semantic_merge.labels,
                candidates,
            )
            stored_clusters = await self._summarize_and_store_capsules(
                run_id=run_id,
                workspace_id=workspace_id,
                embedding_type=embedding_type,
                labels=semantic_merge.labels,
                probabilities=clustered.probabilities,
                transformed_vectors=clustered.transformed_vectors,
                loaded=loaded,
                selections=selections,
            )
            capsule_ids = {
                label: stored.cluster_capsule_id
                for label, stored in stored_clusters.items()
            }
            await self._cluster_repository.store_memberships(
                cluster_run_id=run_id,
                memberships=_build_memberships(
                    raw_labels=clustered.labels,
                    capsule_labels=semantic_merge.labels,
                    probabilities=clustered.probabilities,
                    transformed_vectors=clustered.transformed_vectors,
                    loaded=loaded,
                    selections=selections,
                    capsule_ids=capsule_ids,
                ),
            )
            await self._cluster_repository.complete_run(
                cluster_run_id=run_id,
                cluster_count=semantic_merge.cluster_count,
                noise_count=clustered.noise_count,
                noise_ratio=clustered.noise_ratio,
                preprocessing={
                    **preprocessing,
                    "pca_dimension": clustered.pca_dimension,
                    "semantic_merge_vector_space": "original_l2_normalized",
                },
                parameters={
                    "min_cluster_size": clustered.parameters.min_cluster_size,
                    "min_samples": clustered.parameters.min_samples,
                    "cluster_selection_method": clustered.parameters.cluster_selection_method,
                    "cluster_selection_epsilon": (
                        clustered.parameters.cluster_selection_epsilon
                    ),
                    "quality_score": clustered.quality_score,
                    "candidates_evaluated": clustered.parameter_candidates_evaluated,
                    "semantic_merge": _semantic_merge_metadata(
                        semantic_merge_parameters,
                        semantic_merge,
                    ),
                },
            )
            if self._current_cluster_repository is not None:
                await self._current_cluster_repository.publish_dynamic_clusters(
                    run_id=run_id,
                    workspace_id=workspace_id,
                    embedding_type=embedding_type.value,
                    clusters=_build_current_cluster_publish(
                        labels=semantic_merge.labels,
                        probabilities=clustered.probabilities,
                        loaded=loaded,
                        selections=selections,
                        stored_clusters=stored_clusters,
                    ),
                )
            return EmbeddingTypeClusterResult(
                embedding_type=embedding_type,
                cluster_run_id=run_id,
                status=ClusterRunStatus.COMPLETED,
                indexed_asset_count=len(assets),
                vector_count=len(loaded),
                missing_vector_count=len(assets) - len(loaded),
                cluster_count=semantic_merge.cluster_count,
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
    ) -> dict[int, _StoredClusterCapsule]:
        """Call the naming model with each cluster's selected Asset rows only."""
        assets_by_id = {item.asset.asset_id: item.asset for item in loaded}
        concurrency = self._settings.capsule_concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def summarize_and_store(
            label: int,
        ) -> tuple[int, _StoredClusterCapsule]:
            representatives = selections[label]
            member_indices = np.flatnonzero(labels == label)
            average_probability = float(probabilities[member_indices].mean())
            member_source_paths = [
                loaded[int(index)].asset.source_relative_path for index in member_indices
            ]
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
                    source_relative_path=assets_by_id[representative.asset_id].source_relative_path,
                )
                for representative in representatives
            ]
            async with semaphore:
                summary = await self._model_client.summarize_cluster(
                    build_cluster_summary_messages(
                        embedding_type=embedding_type.value,
                        member_count=len(member_indices),
                        average_membership_probability=average_probability,
                        representatives=prompt_representatives,
                        member_source_paths=member_source_paths,
                    )
                )
                if embedding_type in {
                    EmbeddingType.SUBJECT_CONTENT,
                    EmbeddingType.ASSET_USAGE,
                }:
                    summary = ensure_path_aware_cluster_summary(
                        summary,
                        member_source_paths,
                        embedding_type=embedding_type.value,
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
            return label, _StoredClusterCapsule(
                cluster_capsule_id=stored.cluster_capsule_id,
                summary=summary,
            )

        tasks = [
            asyncio.create_task(summarize_and_store(label))
            for label in sorted(selections)
        ]
        try:
            stored_capsules = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return dict(stored_capsules)


def _build_current_cluster_publish(
    *,
    labels: NDArray[np.int_],
    probabilities: NDArray[np.float64],
    loaded: list[_LoadedClusterVector],
    selections: Mapping[int, list[RepresentativeSelection]],
    stored_clusters: Mapping[int, _StoredClusterCapsule],
) -> list[CurrentClusterPublish]:
    publishes: list[CurrentClusterPublish] = []
    for label in sorted(stored_clusters):
        stored = stored_clusters[label]
        medoid = next(
            representative
            for representative in selections[label]
            if representative.role == ClusterRepresentativeRole.MEDOID.value
        )
        member_indices = np.flatnonzero(labels == label)
        publishes.append(
            CurrentClusterPublish(
                name=stored.summary.name,
                description=stored.summary.description,
                representative_asset_id=medoid.asset_id,
                members=[
                    CurrentClusterMemberWrite(
                        asset_id=loaded[int(index)].asset.asset_id,
                        score=float(probabilities[int(index)]),
                    )
                    for index in member_indices
                ],
            )
        )
    return publishes


def _build_memberships(
    *,
    raw_labels: NDArray[np.int_],
    capsule_labels: NDArray[np.int_],
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
        raw_label = int(raw_labels[index])
        capsule_label = int(capsule_labels[index])
        is_noise = raw_label == -1
        distance = (
            None
            if is_noise
            else float(np.linalg.norm(transformed_vectors[index] - medoid_vectors[capsule_label]))
        )
        memberships.append(
            ClusterMembershipWrite(
                asset_id=item.asset.asset_id,
                cluster_capsule_id=None if is_noise else capsule_ids[capsule_label],
                hdbscan_label=raw_label,
                membership_probability=float(probabilities[index]),
                is_noise=is_noise,
                distance_to_representative=distance,
            )
        )
    return memberships


def _semantic_merge_metadata(
    parameters: SemanticMergeParameters,
    result: SemanticMergeResult | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "enabled": parameters.enabled,
        "centroid_cosine_threshold": parameters.centroid_cosine_threshold,
        "cross_cluster_mean_cosine_threshold": (parameters.cross_cluster_mean_cosine_threshold),
        "merged_member_min_cosine_threshold": (parameters.merged_member_min_cosine_threshold),
    }
    if result is None:
        return metadata
    return {
        **metadata,
        "raw_cluster_count": result.raw_cluster_count,
        "merged_cluster_count": result.cluster_count,
        "merge_count": len(result.decisions),
        "raw_to_merged_labels": {
            str(label): merged_label
            for label, merged_label in sorted(result.raw_to_merged_labels.items())
        },
        "decisions": [
            {
                "left_label": decision.left_label,
                "right_label": decision.right_label,
                "target_label": decision.target_label,
                "centroid_cosine": decision.centroid_cosine,
                "cross_cluster_mean_cosine": decision.cross_cluster_mean_cosine,
                "merged_member_min_cosine": decision.merged_member_min_cosine,
            }
            for decision in result.decisions
        ],
    }
