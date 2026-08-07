"""Content-addressed cache for agent API responses.

Keys are the SHA-256 hex digest of the exact prompt text; values are
stored as JSON files under cache/ so an identical prompt never needs a
fresh API call.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def cache_key(text: str) -> str:
    """Return the SHA-256 hex digest of `text`, used as the cache filename."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get(key: str) -> str | None:
    """Return the cached response text for `key`, or None if there's no entry."""
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return record["text"]


def set(key: str, value: str) -> None:
    """Store `value` under `key`, creating cache/ first if it doesn't exist."""
    CACHE_DIR.mkdir(exist_ok=True)
    record = {
        "text": value,
        # Purely for future debugging (e.g. "how stale is this entry?") —
        # nothing reads this field back yet.
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(record, indent=2))
