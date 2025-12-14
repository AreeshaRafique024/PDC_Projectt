import shelve
import time
import hashlib
from typing import Any, Tuple

DB_PATH = None

def _db_path():
    global DB_PATH
    if DB_PATH:
        return DB_PATH
    DB_PATH = __import__('os').path.join(__import__('os').path.dirname(__file__), 're_ranking_cache.db')
    return DB_PATH

def _make_key(query: str, texts: Tuple[str, ...]) -> str:
    m = hashlib.sha256()
    m.update(query.encode('utf-8'))
    for t in texts:
        m.update(b'\x1f')
        m.update(t.encode('utf-8'))
    return m.hexdigest()

def get_cached(query: str, texts: Tuple[str, ...], ttl: int = 300):
    key = _make_key(query, texts)
    path = _db_path()
    with shelve.open(path) as db:
        item = db.get(key)
        if not item:
            return None
        ts, value = item
        if time.time() - ts > ttl:
            try:
                del db[key]
            except Exception:
                pass
            return None
        return value

def set_cached(query: str, texts: Tuple[str, ...], value: Any):
    key = _make_key(query, texts)
    path = _db_path()
    with shelve.open(path) as db:
        db[key] = (time.time(), value)
