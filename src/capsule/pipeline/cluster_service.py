"""Run one selected clustering algorithm for an Embedding Type."""

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
    ClusterAlgorithm,
    ClusterRepresentativeRole,
    ClusterRunStatus,
    EmbeddingType,
)
from capsule.pipeline.cluster_summary import (
    ClusterSummaryAsset,
    build_cluster_summary_messages,
)
from capsule.pipeline.clustering import (
    ClusterMemberCandidate,
    CompleteLinkParameters,
    HdbscanParameters,
    InsufficientDataError,
    RepresentativeSelection,
    cluster_vectors,
    cluster_vectors_complete_link,
    dataset_hash,
    dynamic_hdbscan_parameters,
    select_cluster_representatives,
)
from capsule.pipeline.vector_fusion import (
    DEFAULT_NATIVE_CONTENT_WEIGHT,
    fuse_native_dimension_vectors,
    validate_native_content_weight,
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
    native_embedding_id: str | None


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
        min_samples: int = 3,
        min_cluster_size: int = 2,
        optimize_parameters: bool = False,
        algorithm: ClusterAlgorithm = ClusterAlgorithm.COMPLETE_LINK,
        distance_threshold: float = 0.5,
        native_content_weight: float = DEFAULT_NATIVE_CONTENT_WEIGHT,
        trigger: str = "user",
    ) -> EmbeddingTypeClusterResult:
        """Run PCA, the selected algorithm, and Capsule generation for one channel."""
        return await self._run_embedding_type(
            workspace_id=workspace_id,
            embedding_type=embedding_type,
            cluster_run_id=cluster_run_id,
            pca_dimension=pca_dimension,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            optimize_parameters=optimize_parameters,
            algorithm=algorithm,
            distance_threshold=distance_threshold,
            native_content_weight=native_content_weight,
            trigger=trigger,
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
        algorithm: ClusterAlgorithm,
        distance_threshold: float,
        native_content_weight: float,
        trigger: str,
    ) -> EmbeddingTypeClusterResult:
        assets: list[ClusterEmbeddingAsset] = []
        loaded: list[_LoadedClusterVector] = []
        run_id = cluster_run_id
        requested_native_content_weight = validate_native_content_weight(native_content_weight)
        effective_native_content_weight = (
            1.0
            if embedding_type is EmbeddingType.NATIVE_MULTIMODAL
            else requested_native_content_weight
        )
        requested_parameters = (
            {
                "algorithm": ClusterAlgorithm.HDBSCAN.value,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_epsilon": self._settings.cluster_selection_epsilon,
            }
            if algorithm is ClusterAlgorithm.HDBSCAN
            else {
                "algorithm": ClusterAlgorithm.COMPLETE_LINK.value,
                "distance_threshold": distance_threshold,
                "min_cluster_size": min_cluster_size,
                "metric": "euclidean",
                "linkage": "complete",
            }
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
            native_assets_by_id: dict[str, ClusterEmbeddingAsset] = {}
            if (
                embedding_type is not EmbeddingType.NATIVE_MULTIMODAL
                and effective_native_content_weight > 0.0
            ):
                native_assets = await self._embedding_repository.list_indexed_cluster_embeddings(
                    workspace_id=workspace_id,
                    embedding_type=EmbeddingType.NATIVE_MULTIMODAL.value,
                    model_name=self._settings.embedding_model,
                    dimension=self._settings.embedding_dimension,
                    milvus_collection=self._settings.milvus_collection,
                )
                native_assets_by_id = {asset.asset_id: asset for asset in native_assets}
            loaded = await self._load_vectors(
                assets,
                native_assets_by_id=native_assets_by_id,
                embedding_type=embedding_type,
                native_content_weight=effective_native_content_weight,
            )
            preprocessing = {
                "trigger": trigger,
                "algorithm": algorithm.value,
                "normalization": "l2",
                "post_pca_normalization": "l2",
                "vector_fusion": _vector_fusion_metadata(
                    embedding_type=embedding_type,
                    requested_native_content_weight=requested_native_content_weight,
                    native_content_weight=effective_native_content_weight,
                ),
                "requested_pca_dimension": pca_dimension,
                "parameter_selection": (
                    "user_defined_selection_optimized"
                    if algorithm is ClusterAlgorithm.HDBSCAN and optimize_parameters
                    else "user_defined"
                ),
                "indexed_asset_count": len(assets),
                "missing_vector_count": len(assets) - len(loaded),
                "resident_excluded_count": len(resident_asset_ids),
            }
            embedding_ids = [item.asset.embedding_id for item in loaded]
            run_dataset_hash = dataset_hash(
                _dataset_hash_inputs(
                    loaded,
                    embedding_type=embedding_type,
                    native_content_weight=effective_native_content_weight,
                )
            )
            if run_id is None:
                run_id = await self._cluster_repository.create_run(
                    workspace_id=workspace_id,
                    embedding_type=embedding_type.value,
                    embedding_ids=embedding_ids,
                    dataset_hash=run_dataset_hash,
                    preprocessing=preprocessing,
                    parameters=requested_parameters,
                )
            else:
                await self._cluster_repository.start_pending_run(
                    cluster_run_id=run_id,
                    workspace_id=workspace_id,
                    embedding_type=embedding_type.value,
                    embedding_ids=embedding_ids,
                    dataset_hash=run_dataset_hash,
                    preprocessing=preprocessing,
                    parameters=requested_parameters,
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
            if self._current_cluster_repository is not None:
                await self._current_cluster_repository.publish_dynamic_clusters(
                    run_id=run_id,
                    workspace_id=workspace_id,
                    embedding_type=embedding_type.value,
                    clusters=[],
                )
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
            if algorithm is ClusterAlgorithm.HDBSCAN:
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
            else:
                clustered = cluster_vectors_complete_link(
                    matrix,
                    pca_dimension=pca_dimension,
                    parameters=CompleteLinkParameters(
                        distance_threshold=distance_threshold,
                        min_cluster_size=min_cluster_size,
                    ),
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
            stored_clusters = await self._summarize_and_store_capsules(
                run_id=run_id,
                workspace_id=workspace_id,
                embedding_type=embedding_type,
                labels=clustered.labels,
                probabilities=clustered.probabilities,
                transformed_vectors=clustered.transformed_vectors,
                loaded=loaded,
                selections=selections,
            )
            capsule_ids = {
                label: stored.cluster_capsule_id for label, stored in stored_clusters.items()
            }
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
            if self._current_cluster_repository is not None:
                await self._current_cluster_repository.publish_dynamic_clusters(
                    run_id=run_id,
                    workspace_id=workspace_id,
                    embedding_type=embedding_type.value,
                    clusters=_build_current_cluster_publish(
                        labels=clustered.labels,
                        probabilities=clustered.probabilities,
                        loaded=loaded,
                        selections=selections,
                        stored_clusters=stored_clusters,
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
                    **_cluster_result_parameters(clustered.parameters),
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
        *,
        native_assets_by_id: Mapping[str, ClusterEmbeddingAsset],
        embedding_type: EmbeddingType,
        native_content_weight: float,
    ) -> list[_LoadedClusterVector]:
        if not assets:
            return []
        await self._vector_store.ensure_collection()
        native_only = embedding_type is EmbeddingType.NATIVE_MULTIMODAL
        dimension_weight = 1.0 - native_content_weight
        vector_ids: set[str] = set()
        if native_only or dimension_weight > 0.0:
            vector_ids.update(asset.embedding_id for asset in assets)
        if not native_only and native_content_weight > 0.0:
            vector_ids.update(
                native_assets_by_id[asset.asset_id].embedding_id
                for asset in assets
                if asset.asset_id in native_assets_by_id
            )
        vectors = await self._vector_store.fetch_vectors(sorted(vector_ids))
        loaded: list[_LoadedClusterVector] = []
        for asset in assets:
            if native_only:
                dimension_vector = vectors.get(asset.embedding_id)
                if dimension_vector is None:
                    continue
                loaded.append(
                    _LoadedClusterVector(
                        asset=asset,
                        vector=dimension_vector,
                        native_embedding_id=asset.embedding_id,
                    )
                )
                continue
            dimension_vector = vectors.get(asset.embedding_id) if dimension_weight > 0.0 else None
            if dimension_weight > 0.0 and dimension_vector is None:
                continue
            native_asset = (
                native_assets_by_id.get(asset.asset_id) if native_content_weight > 0.0 else None
            )
            native_vector = (
                vectors.get(native_asset.embedding_id) if native_asset is not None else None
            )
            if native_content_weight > 0.0 and native_vector is None:
                continue
            try:
                fused = fuse_native_dimension_vectors(
                    native_vector=native_vector,
                    dimension_vector=dimension_vector,
                    native_content_weight=native_content_weight,
                )
            except ValueError as exc:
                logger.warning(
                    "skipping asset with incompatible fusion vectors asset=%s: %s",
                    asset.asset_id,
                    exc,
                )
                continue
            loaded.append(
                _LoadedClusterVector(
                    asset=asset,
                    vector=fused,
                    native_embedding_id=(
                        native_asset.embedding_id if native_asset is not None else None
                    ),
                )
            )
        return loaded

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
        """Call the naming model with complete text evidence from every cluster member."""
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
            prompt_assets = [
                ClusterSummaryAsset(
                    asset_id=loaded[int(index)].asset.asset_id,
                    asset_description=loaded[int(index)].asset.asset_description,
                    asset_features=loaded[int(index)].asset.asset_features,
                    source_relative_path=loaded[int(index)].asset.source_relative_path,
                    file_tree_context=tuple(loaded[int(index)].asset.file_tree_context),
                )
                for index in member_indices
            ]
            async with semaphore:
                summary = await self._model_client.summarize_cluster(
                    build_cluster_summary_messages(
                        embedding_type=embedding_type.value,
                        member_count=len(member_indices),
                        average_membership_probability=average_probability,
                        assets=prompt_assets,
                        member_source_paths=member_source_paths,
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
            return label, _StoredClusterCapsule(
                cluster_capsule_id=stored.cluster_capsule_id,
                summary=summary,
            )

        tasks = [asyncio.create_task(summarize_and_store(label)) for label in sorted(selections)]
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


def _dataset_hash_inputs(
    loaded: Sequence[_LoadedClusterVector],
    *,
    embedding_type: EmbeddingType,
    native_content_weight: float,
) -> list[str]:
    """Return selected eligibility and actual vector inputs for a fusion hash."""
    selected_inputs = [f"selected:{item.asset.embedding_id}" for item in loaded]
    if embedding_type is EmbeddingType.NATIVE_MULTIMODAL:
        return [*selected_inputs, "native_content_weight:1"]
    inputs = [*selected_inputs, f"native_content_weight:{native_content_weight:.12g}"]
    if native_content_weight < 1.0:
        inputs.extend(f"dimension:{item.asset.embedding_id}" for item in loaded)
    if native_content_weight > 0.0:
        inputs.extend(
            f"native:{item.native_embedding_id}"
            for item in loaded
            if item.native_embedding_id is not None
        )
    return inputs


def _cluster_result_parameters(
    parameters: HdbscanParameters | CompleteLinkParameters,
) -> dict[str, Any]:
    if isinstance(parameters, HdbscanParameters):
        return {
            "algorithm": ClusterAlgorithm.HDBSCAN.value,
            "min_cluster_size": parameters.min_cluster_size,
            "min_samples": parameters.min_samples,
            "cluster_selection_method": parameters.cluster_selection_method,
            "cluster_selection_epsilon": parameters.cluster_selection_epsilon,
        }
    return {
        "algorithm": ClusterAlgorithm.COMPLETE_LINK.value,
        "distance_threshold": parameters.distance_threshold,
        "min_cluster_size": parameters.min_cluster_size,
        "metric": "euclidean",
        "linkage": "complete",
    }


def _vector_fusion_metadata(
    *,
    embedding_type: EmbeddingType,
    requested_native_content_weight: float,
    native_content_weight: float,
) -> dict[str, Any]:
    if embedding_type is EmbeddingType.NATIVE_MULTIMODAL:
        return {
            "mode": "native_only",
            "requested_native_content_weight": requested_native_content_weight,
            "effective_native_content_weight": 1.0,
            "native_content_weight": 1.0,
            "dimension_weight": 0.0,
            "components": [EmbeddingType.NATIVE_MULTIMODAL.value],
        }
    return {
        "mode": "l2_normalized_weighted_average",
        "requested_native_content_weight": requested_native_content_weight,
        "effective_native_content_weight": native_content_weight,
        "native_content_weight": native_content_weight,
        "dimension_weight": 1.0 - native_content_weight,
        "components": [EmbeddingType.NATIVE_MULTIMODAL.value, embedding_type.value],
    }
