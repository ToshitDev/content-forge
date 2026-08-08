"""Tests for src/rate_limiter.py's TokenBucket.

Time is controlled with a FakeClock (advanced explicitly by the test)
plus a mocked asyncio.sleep that advances the same FakeClock instead of
actually waiting — so "acquiring more than capacity forces a wait" is
verified by checking the clock moved, not by real wall-clock delay.
"""

import asyncio

import pytest

from src.rate_limiter import TokenBucket


class FakeClock:
    """A controllable clock — advances only when told to, never on its own."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    """A fresh FakeClock, starting at time 0."""
    return FakeClock()


@pytest.fixture(autouse=True)
def fake_sleep(monkeypatch, clock):
    """Replace asyncio.sleep with one that advances `clock` instead of
    actually waiting, so tests that force acquire() to wait run
    instantly rather than taking real wall-clock time."""

    async def _fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)


def test_acquire_consumes_a_token(clock):
    """A single acquire() reduces the available tokens by one."""
    bucket = TokenBucket(capacity=5, refill_rate=1.0, clock=clock)

    asyncio.run(bucket.acquire())

    assert bucket._tokens == pytest.approx(4.0)


def test_bucket_refills_over_time(clock):
    """Tokens regenerate at refill_rate as the injected clock advances."""
    bucket = TokenBucket(capacity=5, refill_rate=2.0, clock=clock)
    asyncio.run(bucket.acquire())
    asyncio.run(bucket.acquire())
    assert bucket._tokens == pytest.approx(3.0)

    clock.advance(1.0)  # 2.0 tokens/sec * 1s = 2 tokens regenerated
    asyncio.run(bucket.acquire())  # this call's own refill check sees them

    assert bucket._tokens == pytest.approx(4.0)  # 3 + 2 refilled - 1 consumed


def test_acquiring_more_than_capacity_forces_a_wait(clock):
    """Draining the bucket makes the next acquire() wait for a refill —
    verified by the clock having advanced, never by real elapsed time."""
    bucket = TokenBucket(capacity=3, refill_rate=1.0, clock=clock)
    for _ in range(3):
        asyncio.run(bucket.acquire())
    assert bucket._tokens == pytest.approx(0.0)

    start = clock.now
    asyncio.run(bucket.acquire())  # bucket is empty: must wait ~1s for one token

    assert clock.now > start
    assert bucket._tokens == pytest.approx(0.0)
