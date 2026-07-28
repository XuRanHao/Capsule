import asyncio

from capsule.model_clients.concurrency import AsyncCallPool


async def test_async_call_pool_limits_concurrency() -> None:
    pool = AsyncCallPool(name="test", concurrency=3, max_attempts=1)
    active = 0
    observed = 0
    lock = asyncio.Lock()

    async def operation() -> int:
        nonlocal active, observed
        async with lock:
            active += 1
            observed = max(observed, active)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return 1

    results = await asyncio.gather(*(pool.run(operation) for _ in range(12)))

    assert results == [1] * 12
    assert observed == 3
    assert pool.max_observed == 3
    assert pool.in_flight == 0
