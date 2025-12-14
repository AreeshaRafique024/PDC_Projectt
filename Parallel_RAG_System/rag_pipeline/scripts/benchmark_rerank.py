"""Simple benchmark script for re-ranking
Run this while the RAG API is running to measure retrieve+rerank latency.
"""
import time
import requests
import statistics

URL = "http://localhost:8000/retrieve"

QUERIES = [
    "What is warp in GPU programming?",
    "Explain threads and blocks in CUDA",
    "How does memory hierarchy affect performance?",
]

def run(iterations=5):
    times = []
    for _ in range(iterations):
        for q in QUERIES:
            payload = {"query": q, "rerank": True}
            t0 = time.time()
            r = requests.post(URL, json=payload, timeout=30)
            dt = time.time() - t0
            times.append(dt)
            print(f"Query '{q[:40]}...' -> {r.status_code} in {dt:.2f}s")
    print("--- stats ---")
    print(f"count: {len(times)}")
    print(f"mean: {statistics.mean(times):.2f}s")
    print(f"p95: {np_percentile(times,95):.2f}s")

def np_percentile(data, p):
    import numpy as _np
    return float(_np.percentile(data, p))

if __name__ == '__main__':
    run()
