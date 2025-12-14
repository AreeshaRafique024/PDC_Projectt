import sys
import os
import numpy as np

# Ensure project root is on sys.path so tests can import rag_pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_pipeline.src.re_ranking import aggregate_scores


def test_aggregate_mean():
    scores = np.array([[0.8, 0.2], [0.6, 0.4]])
    out = aggregate_scores(scores, aggregation='mean')
    assert np.allclose(out, np.array([0.7, 0.3]))


def test_aggregate_rrf():
    scores = np.array([[0.9, 0.1, 0.5], [0.8, 0.7, 0.2]])
    out = aggregate_scores(scores, aggregation='rrf', rrf_k=60)
    # ensure output has correct length and type
    assert out.shape == (3,)
    assert out.dtype == float
