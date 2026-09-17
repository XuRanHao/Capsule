import asyncio
from dataclasses import asdict, dataclass

from sqlalchemy import select

from capsule.config import Settings
from capsule.db.models import Workspace
from capsule.db.session import Database
from capsule.storage.object_storage import ObjectStorage
from capsule.vectorstore.agent_memory import AgentMemoryMilvusStore
from capsule.vectorstore.milvus import MilvusVectorStore


@dataclass(slots=True, frozen=True)
class BootstrapResult:
    workspace_id: str
    workspace_created: bool
    object_storage_bucket: str
    milvus_collection: str
    milvus_collection_created: bool
    agent_memory_milvus_collection: str
    agent_memory_milvus_collection_created: bool

    def as_dict(self) -> dict[str, str | bool]:
        return asdict(self)


async def bootstrap_runtime(
    settings: Settings,
    *,
    workspace_id: str,
    workspace_name: str,
) -> BootstrapResult:
    """Initialize idempotent runtime resources after database migrations."""

    database = Database(settings)
    storage = ObjectStorage(settings)
    vectors = MilvusVectorStore(settings)
    memory_vectors = AgentMemoryMilvusStore(settings)
    try:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(storage.ensure_bucket())
            collection_task = tasks.create_task(vectors.ensure_collection())
            memory_collection_task = tasks.create_task(memory_vectors.ensure_collection())
            workspace_task = tasks.create_task(
                _ensure_workspace(
                    database,
                    workspace_id=workspace_id,
                    workspace_name=workspace_name,
                )
            )
        return BootstrapResult(
            workspace_id=workspace_id,
            workspace_created=workspace_task.result(),
            object_storage_bucket=settings.object_storage_bucket,
            milvus_collection=settings.milvus_collection,
            milvus_collection_created=collection_task.result(),
            agent_memory_milvus_collection=settings.agent_memory_milvus_collection,
            agent_memory_milvus_collection_created=memory_collection_task.result(),
        )
    finally:
        await database.dispose()


async def _ensure_workspace(
    database: Database,
    *,
    workspace_id: str,
    workspace_name: str,
) -> bool:
    async with database.session() as session:
        existing = await session.scalar(
            select(Workspace).where(Workspace.workspace_id == workspace_id)
        )
        if existing is not None:
            return False
        session.add(Workspace(workspace_id=workspace_id, name=workspace_name))
        await session.commit()
        return True
