Parallel Re-ranking
===================

This folder implements parallel re-ranking for the RAG pipeline.

Usage
-----

- Configure models in `config/re_rankers.json`.
- Start the RAG API (`python api.py`).
- Call the retrieve endpoint with `{"query":"...","rerank":true}`.

Benchmarks
----------

Run `python scripts/benchmark_rerank.py` while the API is running to measure latency.

Notes
-----
- The implementation prefers local `sentence-transformers` CrossEncoder models. Install with:

```powershell
.\venv\Scripts\Activate.ps1
pip install -U sentence-transformers
```

- A simple in-memory TTL cache is used to speed up repeated identical queries.
