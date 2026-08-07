"""Tests for src/cache.py's content-addressed cache.

Every test gets its own throwaway cache directory (via monkeypatching
cache.CACHE_DIR to pytest's tmp_path), so nothing here ever touches the
real cache/ directory. The last test uses unittest.mock to confirm a
real BaseAgent.run() skips the API on a repeated identical call.
BaseAgent.run() is async (see base.py's module docstring for why), so
it's driven here with asyncio.run() and its API call mocked with
AsyncMock rather than plain MagicMock.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import cache
from src.agents.base import BaseAgent


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    """Redirect every cache read/write in this file to a throwaway directory."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)


def test_cache_key_is_deterministic():
    """The same text always produces the same key."""
    assert cache.cache_key("hello world") == cache.cache_key("hello world")


def test_cache_key_differs_for_different_text():
    """Different text produces a different key."""
    assert cache.cache_key("hello world") != cache.cache_key("goodbye world")


def test_get_returns_none_for_unset_key():
    """A key that was never set() returns None from get()."""
    assert cache.get(cache.cache_key("never cached")) is None


def test_set_then_get_round_trips_the_value():
    """set() followed by get() returns exactly what was stored."""
    key = cache.cache_key("round trip me")
    cache.set(key, "the cached response text")
    assert cache.get(key) == "the cached response text"


def _fake_api_response(text: str) -> MagicMock:
    """Build a MagicMock shaped like a successful anthropic Message response."""
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [MagicMock(type="text", text=text)]
    return response


def test_base_agent_run_hits_cache_on_second_identical_call(monkeypatch):
    """A second run() with identical inputs is served from cache, not the API."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    agent = BaseAgent(prompt_name="hook")
    agent.client.messages.create = AsyncMock(return_value=_fake_api_response("cached reply"))

    inputs = {
        "niche": "student productivity",
        "audience": "college students",
        "platform": "Instagram",
        "brand_voice": "casual",
        "chosen_idea": "why timetables fail",
    }

    first = asyncio.run(agent.run(inputs))
    second = asyncio.run(agent.run(inputs))

    assert first == "cached reply"
    assert second == "cached reply"
    agent.client.messages.create.assert_called_once()
