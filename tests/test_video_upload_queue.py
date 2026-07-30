import asyncio
from typing import Any
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from capsule.config import Settings
from capsule.pipeline.video_upload_queue import RedisStreamUploadQueue, UploadQueueItem


class _FakeRedis:
    def __init__(self) -> None:
        self.claimed: list[tuple[str, dict[str, str]]] = [
            ("10-0", {"manifest_path": "/spool/manifest.json", "attempt": "1"})
        ]
        self.added: list[dict[str, str]] = []
        self.acked: list[str] = []
        self.deleted: list[str] = []
        self.closed = False

    async def xgroup_create(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def xautoclaim(self, *_args: Any, **_kwargs: Any) -> tuple[str, list[Any], list[Any]]:
        claimed, self.claimed = self.claimed, []
        return ("0-0", claimed, [])

    async def xreadgroup(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        raise AssertionError("XAUTOCLAIM should recover the pending delivery first")

    async def xadd(self, _stream: str, values: dict[str, str]) -> str:
        self.added.append(values)
        return "11-0"

    async def xack(self, _stream: str, _group: str, receipt: str) -> None:
        self.acked.append(receipt)

    async def xdel(self, _stream: str, receipt: str) -> None:
        self.deleted.append(receipt)

    async def aclose(self) -> None:
        self.closed = True


async def test_redis_stream_recovers_and_requeues_abandoned_delivery() -> None:
    client = _FakeRedis()
    queue = RedisStreamUploadQueue(
        redis_url="redis://unused",
        stream="video-stream",
        group="uploaders",
        consumer="worker-2",
        claim_idle_ms=100,
        client=client,
    )
    await queue.start()

    delivery = await queue.receive()
    assert delivery.receipt == "10-0"
    assert delivery.item == UploadQueueItem(
        manifest_path="/spool/manifest.json",
        attempt=1,
    )

    await queue.retry(delivery)
    assert client.added == [{"manifest_path": "/spool/manifest.json", "attempt": "2"}]
    assert client.acked == ["10-0"]
    assert client.deleted == ["10-0"]
    await queue.close()
    assert client.closed


@pytest.mark.integration
async def test_real_redis_xautoclaim_recovers_unacked_delivery() -> None:
    settings = Settings()
    stream = f"capsule:test:video:{uuid4().hex}"
    group = "test-uploaders"
    admin = Redis.from_url(settings.redis_url, decode_responses=True)
    first = RedisStreamUploadQueue(
        redis_url=settings.redis_url,
        stream=stream,
        group=group,
        consumer="worker-that-stopped",
        claim_idle_ms=100,
    )
    recovery = RedisStreamUploadQueue(
        redis_url=settings.redis_url,
        stream=stream,
        group=group,
        consumer="recovery-worker",
        claim_idle_ms=100,
    )
    try:
        try:
            await admin.ping()
        except RedisError:
            pytest.skip("Redis integration service is unavailable")
        await first.start()
        await recovery.start()
        await first.publish(UploadQueueItem(manifest_path="/tmp/recovered.json"))
        abandoned = await first.receive()
        await asyncio.sleep(0.15)
        recovered = await recovery.receive()
        assert recovered.receipt == abandoned.receipt
        await recovery.retry(recovered)
        retried = await recovery.receive()
        assert retried.item.attempt == 1
        await recovery.acknowledge(retried)
        assert (await admin.xpending(stream, group))["pending"] == 0
    finally:
        await first.close()
        await recovery.close()
        try:
            await admin.delete(stream)
        finally:
            await admin.aclose()
