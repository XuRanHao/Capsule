import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

T = TypeVar("T")

logger = logging.getLogger(__name__)


def _retry_delay(attempt: int, response: httpx.Response | None = None) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                parsed = float(retry_after)
                return parsed if parsed > 0.0 else 0.0
            except ValueError:
                pass
    delay = float(2 ** (attempt - 1)) + random.uniform(0.0, 1.0)
    return 30.0 if delay > 30.0 else delay


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code in {500, 502, 503, 504}


class AsyncCallPool:
    """Bounded asynchronous execution with transient-error retries."""

    def __init__(self, *, name: str, concurrency: int, max_attempts: int) -> None:
        self.name = name
        self._semaphore = asyncio.Semaphore(concurrency)
        self._max_attempts = max_attempts
        self._in_flight = 0
        self._max_observed = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def max_observed(self) -> int:
        return self._max_observed

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T:
        async with self._semaphore:
            self._in_flight += 1
            self._max_observed = max(self._max_observed, self._in_flight)
            try:
                return await self._run_with_retry(operation)
            finally:
                self._in_flight -= 1

    async def _run_with_retry(self, operation: Callable[[], Awaitable[T]]) -> T:
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await operation()
            except httpx.HTTPStatusError as exc:
                if not _is_retryable_status(exc.response.status_code):
                    raise
                if attempt == self._max_attempts:
                    raise
                delay = _retry_delay(attempt, exc.response)
                logger.warning(
                    "%s call received HTTP %s; retrying in %.2fs (%s/%s)",
                    self.name,
                    exc.response.status_code,
                    delay,
                    attempt,
                    self._max_attempts,
                )
                await asyncio.sleep(delay)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == self._max_attempts:
                    raise
                delay = _retry_delay(attempt)
                logger.warning(
                    "%s call failed with %s; retrying in %.2fs (%s/%s)",
                    self.name,
                    type(exc).__name__,
                    delay,
                    attempt,
                    self._max_attempts,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("retry loop exited unexpectedly")
