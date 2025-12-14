import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import numpy as np
from .re_ranking_cache import get_cached, set_cached
from .re_ranking_cache_persistent import get_cached as get_cached_persistent, set_cached as set_cached_persistent
import requests


def _get_hf_token() -> Optional[str]:
    backend_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backend', '.env')
    token = None
    if os.path.exists(backend_env_path):
        with open(backend_env_path, 'r') as f:
            for line in f:
                if line.startswith('HUGGINGFACE_API_KEY='):
                    token = line.strip().split('=', 1)[1]
                    break
    return token

try:
    from sentence_transformers import CrossEncoder
    _HAS_CE = True
except Exception:
    CrossEncoder = None
    _HAS_CE = False

_MODEL_CACHE: Dict[str, Any] = {}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 're_rankers.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_model(model_id: str):
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    if _HAS_CE:
        try:
            model = CrossEncoder(model_id)
            _MODEL_CACHE[model_id] = model
            return model
        except Exception:
            # fallback: store None to avoid repeated load attempts
            _MODEL_CACHE[model_id] = None
            return None
    # If no local CrossEncoder available, return None (caller should handle)
    _MODEL_CACHE[model_id] = None
    return None


def _score_with_hf(model_id: str, token: str, query: str, texts: List[str]) -> np.ndarray:
    """Score query-text pairs using HuggingFace Inference API as fallback."""
    if not token:
        return np.zeros(len(texts), dtype=float)
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    # prepare pairs
    inputs = [[query, t] for t in texts]
    try:
        resp = requests.post(url, headers=headers, json={"inputs": inputs, "options": {"wait_for_model": True}}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # expect list of scores per pair; if data is a list of dicts, try extract 'score'
        scores = []
        for item in data:
            if isinstance(item, dict) and 'score' in item:
                scores.append(float(item['score']))
            elif isinstance(item, (int, float)):
                scores.append(float(item))
            else:
                # unknown format
                scores.append(0.0)
        return np.asarray(scores, dtype=float)
    except Exception:
        return np.zeros(len(texts), dtype=float)


def _score_with_model(model_inst, query: str, texts: List[str]) -> np.ndarray:
    if model_inst is None:
        return np.zeros(len(texts), dtype=float)
    pairs = [[query, t] for t in texts]
    scores = model_inst.predict(pairs)
    return np.asarray(scores, dtype=float)


def aggregate_scores(scores_per_model: np.ndarray, aggregation: str = 'rrf', rrf_k: int = 60, weights: Optional[List[float]] = None) -> np.ndarray:
    # scores_per_model: shape (num_models, num_texts)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
    if aggregation == 'mean':
        if weights is None:
            return np.mean(scores_per_model, axis=0)
        wsum = np.sum(scores_per_model * weights[:, None], axis=0)
        return wsum / (np.sum(weights) if np.sum(weights) != 0 else 1.0)
    elif aggregation == 'rrf':
        # Reciprocal Rank Fusion
        # compute ranks per model (0-based: 0 best)
        ranks = np.argsort(-scores_per_model, axis=1).argsort(axis=1)
        rrf = np.sum(1.0 / (rrf_k + ranks), axis=0)
        return rrf
    else:
        return np.mean(scores_per_model, axis=0)


def parallel_rerank(query: str, docs: List[Dict[str, Any]], config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Re-rank `docs` for `query` using models defined in config file.

    docs: list of dicts with at least 'content' (text) field.
    Returns: re-ranked list (top `final_k`) with `score` added to each dict.
    """
    cfg = load_config(config_path)
    models_cfg = cfg.get('models', [])
    aggregation = cfg.get('aggregation', 'rrf')
    rrf_k = cfg.get('rrf_k', 60)
    top_n = cfg.get('top_n', 50)
    final_k = cfg.get('final_k', 5)
    parallelism = cfg.get('parallelism', 4)

    # operate on the provided docs but limit to top_n
    candidates = docs[:top_n]
    texts = [d.get('content') or d.get('text') or '' for d in candidates]
    if not texts:
        return []

    # check cache
    cache_ttl = cfg.get('cache_ttl_seconds', 300)
    cache_key_texts = tuple(texts)
    cached = get_cached(query, cache_key_texts, ttl=cache_ttl)
    if cached is not None:
        return cached

    # load model instances
    model_instances = []
    weights = []
    for m in models_cfg:
        mid = m.get('id')
        w = float(m.get('weight', 1.0))
        inst = _load_model(mid)
        model_instances.append((mid, inst))
        weights.append(w)

    # Score in parallel per model
    scores_list = []
    with ThreadPoolExecutor(max_workers=min(parallelism, max(1, len(model_instances)))) as ex:
        futures = []
        token = _get_hf_token()
        for mid, inst in model_instances:
            if inst is not None:
                futures.append(ex.submit(_score_with_model, inst, query, texts))
            else:
                # fallback to HF inference API
                futures.append(ex.submit(_score_with_hf, mid, token, query, texts))
        for f in as_completed(futures):
            try:
                scores = f.result()
            except Exception:
                scores = np.zeros(len(texts), dtype=float)
            scores_list.append(scores)

    if not scores_list:
        # nothing scored
        for i, d in enumerate(candidates[:final_k]):
            d['score'] = 0.0
        return candidates[:final_k]

    scores_per_model = np.stack(scores_list, axis=0)
    agg = aggregate_scores(scores_per_model, aggregation=aggregation, rrf_k=rrf_k, weights=weights)

    # select top final_k indices by aggregated score
    idx = np.argsort(-agg)[:final_k]
    reranked = []
    for rank, i in enumerate(idx, start=1):
        doc = dict(candidates[i])
        doc['score'] = float(agg[i])
        doc['rerank_rank'] = rank
        reranked.append(doc)

    # cache result
    try:
        set_cached(query, cache_key_texts, reranked)
    except Exception:
        pass

    return reranked
