"""
Shared pytest fixtures.

All heavy dependencies (sentence-transformers, FAISS) are replaced with
deterministic stubs so the test suite runs fast and without GPU.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embedding import EmbeddingModel, TextEmbeddingValue
from src.pipeline import RAGPipeline
from src.retrieval import QueryExpander
from src.storage import Document, FaissVectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 8  # tiny dimension to keep tests fast

_RNG = np.random.default_rng(42)


def _fake_embed(texts: List[str]) -> List[TextEmbeddingValue]:
    """Return deterministic unit-norm embeddings seeded by text hash."""
    result = []
    for text in texts:
        seed = abs(hash(text)) % (2**31)
        rng = np.random.default_rng(seed)
        vec = rng.random(EMBEDDING_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        result.append(TextEmbeddingValue(vec))
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_embedding_model() -> EmbeddingModel:
    """EmbeddingModel whose get_embeddings is replaced with _fake_embed."""
    model = EmbeddingModel.__new__(EmbeddingModel)
    model.model_name = "mock"
    model._use_vertex = False
    model.get_embeddings = _fake_embed  # type: ignore[method-assign]
    model.embed_single = lambda text: _fake_embed([text])[0].values  # type: ignore[method-assign]
    return model


@pytest.fixture()
def mock_query_expander() -> QueryExpander:
    """QueryExpander that just appends a fixed suffix."""
    expander = QueryExpander.__new__(QueryExpander)
    expander._use_vertex = False

    def _expand(query: str) -> str:
        return f"{query} load balancing auto-scaling horizontal scaling"

    expander.expand = _expand  # type: ignore[method-assign]
    return expander


@pytest.fixture()
def small_corpus() -> list:
    return [
        {"id": "doc_a", "text": "Horizontal scaling adds more nodes to handle peak load."},
        {"id": "doc_b", "text": "Cosine similarity measures the angle between two vectors."},
        {"id": "doc_c", "text": "Kubernetes HPA scales pod replicas based on CPU metrics."},
        {"id": "doc_d", "text": "Redis caching reduces database pressure at high traffic."},
        {"id": "doc_e", "text": "Circuit breakers prevent cascading failures in microservices."},
    ]


@pytest.fixture()
def populated_pipeline(mock_embedding_model, mock_query_expander, small_corpus) -> RAGPipeline:
    """A fully ingested pipeline backed by mocked components."""
    pipeline = RAGPipeline(
        embedding_dim=EMBEDDING_DIM,
        embedding_model=mock_embedding_model,
        query_expander=mock_query_expander,
        use_faiss=False,  # NumPy backend so FAISS is not required
    )
    pipeline.ingest(small_corpus)
    return pipeline
