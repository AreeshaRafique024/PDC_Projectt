import time
from typing import Any, Dict, Tuple

# Simple in-memory TTL cache for rerank results
_CACHE: Dict[Tuple[str, Tuple[str, ...]], Tuple[float, Any]] = {}

def get_cached(query: str, texts: Tuple[str, ...], ttl: int = 300):
    key = (query, texts)
    item = _CACHE.get(key)
    if not item:
        return None
    ts, value = item
    if time.time() - ts > ttl:
        # expired
        del _CACHE[key]
        return None
    return value


def set_cached(query: str, texts: Tuple[str, ...], value: Any):
    key = (query, texts)
    _CACHE[key] = (time.time(), value)
