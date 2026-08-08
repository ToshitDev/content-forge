"""A token-bucket rate limiter for pacing outbound API calls.

Each agent calls acquire() before its request; the bucket refills
continuously (refill_rate tokens/second) up to capacity, and acquire()
only waits once the bucket is actually empty. This smooths out bursts —
e.g. 5 agents from different jobs all calling the API around the same
moment during a concurrent batch run — without slowing down a single
agent working alone, since a full bucket never makes anyone wait.
"""

import asyncio
import time
from collections.abc import Callable


class TokenBucket:
    """Rate limiter: `capacity` tokens, refilling at `refill_rate` tokens/sec.

    The clock is injectable specifically so tests can control the
    passage of time deterministically, without real sleeping — see
    tests/test_rate_limiter.py.
    """

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a bucket that starts full.

        Args:
            capacity: Maximum tokens the bucket can hold.
            refill_rate: Tokens added per second.
            clock: Returns the current time in seconds. Defaults to
                time.monotonic (wall-clock, immune to clock adjustments);
                tests inject a fake, controllable clock instead.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._clock = clock
        self._tokens = float(capacity)
        self._last_refill = clock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it.

        Safe to call concurrently: refilling and consuming a token is
        plain synchronous code with no `await` inside it, so under
        asyncio's cooperative scheduling each check-and-consume always
        runs as one uninterrupted step — no explicit lock is needed to
        protect the shared token count.
        """
        while True:
            self._refill()
            if self._tokens >= 1:
                self._tokens -= 1
                return
            wait_time = (1 - self._tokens) / self.refill_rate
            await asyncio.sleep(wait_time)

    def _refill(self) -> None:
        """Add tokens for elapsed time since the last refill, capped at capacity."""
        now = self._clock()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now
