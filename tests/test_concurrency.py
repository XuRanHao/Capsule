import asyncio
from collections.abc import Awaitable, Callable

import httpx
import pytest

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


async def test_reservation_keeps_slot_across_multiple_calls() -> None:
    pool = AsyncCallPool(name="test", concurrency=1, max_attempts=1)
    first_finished = asyncio.Event()
    allow_repair = asyncio.Event()
    order: list[str] = []

    async def reserved_work() -> None:
        async with pool.reserve() as slot:
            async def first() -> str:
                order.append("first")
                return "invalid"

            assert await slot.run(first) == "invalid"
            first_finished.set()
            await allow_repair.wait()

            async def repair() -> str:
                order.append("repair")
                return "valid"

            assert await slot.run(repair) == "valid"

    async def competing_work() -> None:
        await first_finished.wait()

        async def operation() -> None:
            order.append("competitor")

        await pool.run(operation)

    reserved_task = asyncio.create_task(reserved_work())
    competing_task = asyncio.create_task(competing_work())
    await first_finished.wait()
    await asyncio.sleep(0)
    assert order == ["first"]
    assert pool.in_flight == 1

    allow_repair.set()
    await asyncio.gather(reserved_task, competing_task)

    assert order == ["first", "repair", "competitor"]
    assert pool.max_observed == 1
    assert pool.in_flight == 0


async def test_each_reserved_call_gets_an_independent_retry_budget() -> None:
    pool = AsyncCallPool(name="test", concurrency=1, max_attempts=2)
    attempts = {"first": 0, "repair": 0}

    def retried_operation(name: str) -> Callable[[], Awaitable[str]]:
        async def operation() -> str:
            attempts[name] += 1
            if attempts[name] == 1:
                request = httpx.Request("POST", "https://example.test")
                response = httpx.Response(
                    503,
                    request=request,
                    headers={"Retry-After": "0.001"},
                )
                raise httpx.HTTPStatusError(
                    "temporary failure",
                    request=request,
                    response=response,
                )
            return name

        return operation

    async with pool.reserve() as slot:
        assert await slot.run(retried_operation("first")) == "first"
        assert await slot.run(retried_operation("repair")) == "repair"

    assert attempts == {"first": 2, "repair": 2}
    assert pool.max_observed == 1
    assert pool.in_flight == 0


async def test_reserved_slot_cannot_be_reused_after_context_exit() -> None:
    pool = AsyncCallPool(name="test", concurrency=1, max_attempts=1)

    async with pool.reserve() as slot:
        assert await slot.run(_return_one) == 1

    with pytest.raises(RuntimeError, match="no longer reserved"):
        await slot.run(_return_one)


async def _return_one() -> int:
    return 1
