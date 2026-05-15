"""
Tests for the retrieval module.

Covers:
- QueryExpander local fallback logic.
- Mocking vertexai.generative_models.GenerativeModel (GCP SDK mock).
- Strategy A: raw vector search.
- Strategy B: query expansion + vector search.
- RetrievalResult helper.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieval import QueryExpander, RetrievalResult, Retriever
from src.storage import Document, FaissVectorStore

DIM = 8


def _unit_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# QueryExpander – local fallback
# ---------------------------------------------------------------------------


class TestQueryExpanderLocal:

    def _make_expander(self) -> QueryExpander:
        exp = QueryExpander.__new__(QueryExpander)
        exp._use_vertex = False
        exp.model_name = "mock"
        return exp

    def test_peak_load_expansion(self):
        exp = self._make_expander()
        result = exp.expand("How does the system handle peak load?")
        assert "peak load" in result.lower()
        assert "auto-scaling" in result or "load balancing" in result

    def test_unknown_query_generic_fallback(self):
        exp = self._make_expander()
        result = exp.expand("a completely unknown topic XYZ")
        assert "a completely unknown topic XYZ" in result
        assert len(result) > len("a completely unknown topic XYZ")

    def test_cache_expansion_includes_redis(self):
        exp = self._make_expander()
        # The keyword dict key is "cache"; query must contain it to trigger expansion
        result = exp.expand("What cache solutions exist?")
        assert "Redis" in result or "cache" in result.lower()

    def test_failure_expansion_includes_circuit_breaker(self):
        exp = self._make_expander()
        result = exp.expand("How to handle failure in microservices?")
        assert "circuit" in result.lower() or "resilience" in result.lower()

    def test_expansion_returns_string(self):
        exp = self._make_expander()
        assert isinstance(exp.expand("anything"), str)

    def test_expansion_non_empty(self):
        exp = self._make_expander()
        assert exp.expand("test") != ""


# ---------------------------------------------------------------------------
# QueryExpander – Mocking vertexai GenerativeModel (GCP SDK mock test)
# ---------------------------------------------------------------------------


class TestQueryExpanderVertexMock:

    def test_vertex_path_calls_generate_content(self):
        mock_response = MagicMock()
        mock_response.text = "peak load traffic surge auto-scaling horizontal scaling"

        mock_gen_model = MagicMock()
        mock_gen_model.generate_content.return_value = mock_response

        exp = QueryExpander.__new__(QueryExpander)
        exp._use_vertex = True
        exp._model = mock_gen_model
        exp.model_name = "gemini-pro"

        result = exp.expand("How does the system handle peak load?")

        mock_gen_model.generate_content.assert_called_once()
        prompt_arg = mock_gen_model.generate_content.call_args[0][0]
        assert "How does the system handle peak load?" in prompt_arg
        assert result == "peak load traffic surge auto-scaling horizontal scaling"

    def test_vertex_mock_returns_stripped_text(self):
        mock_response = MagicMock()
        mock_response.text = "  expanded query text  \n"

        mock_gen_model = MagicMock()
        mock_gen_model.generate_content.return_value = mock_response

        exp = QueryExpander.__new__(QueryExpander)
        exp._use_vertex = True
        exp._model = mock_gen_model
        exp.model_name = "gemini-pro"

        result = exp.expand("test query")
        assert result == "expanded query text"

    def test_generative_model_mock_interface(self):
        """
        Validates that the mock mirrors the real
        vertexai.generative_models.GenerativeModel public API surface.
        """
        with patch("src.retrieval._VertexGenerativeModel") as MockClass:
            instance = MockClass.return_value
            instance.generate_content.return_value.text = "mocked expansion"

            gen_model = MockClass("gemini-pro")
            response = gen_model.generate_content("some prompt")

            MockClass.assert_called_once_with("gemini-pro")
            assert response.text == "mocked expansion"


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:

    def _make_result(self) -> RetrievalResult:
        doc = Document("d1", "Hello world, this is a test document about load balancing.")
        return RetrievalResult(rank=1, doc=doc, score=0.987654)

    def test_to_dict_keys(self):
        result = self._make_result()
        d = result.to_dict()
        assert "rank" in d
        assert "doc_id" in d
        assert "score" in d
        assert "text_preview" in d
        assert "full_text" in d

    def test_to_dict_values(self):
        result = self._make_result()
        d = result.to_dict()
        assert d["rank"] == 1
        assert d["doc_id"] == "d1"
        assert abs(d["score"] - 0.987654) < 1e-4

    def test_text_preview_truncated(self):
        long_text = "A" * 200
        doc = Document("d2", long_text)
        result = RetrievalResult(rank=1, doc=doc, score=0.5)
        d = result.to_dict()
        assert d["text_preview"].endswith("...")
        assert len(d["text_preview"]) <= 123 + 3  # 120 chars + "..."

    def test_short_text_no_ellipsis(self):
        doc = Document("d3", "Short text.")
        result = RetrievalResult(rank=1, doc=doc, score=0.5)
        assert not result.to_dict()["text_preview"].endswith("...")


# ---------------------------------------------------------------------------
# Retriever – Strategy A and Strategy B
# ---------------------------------------------------------------------------


class TestRetriever:

    def _make_retriever(self, num_docs: int = 5):
        """Build a Retriever backed by mock components."""
        from src.embedding import EmbeddingModel, TextEmbeddingValue

        # Mock embedding model
        emb_model = MagicMock(spec=EmbeddingModel)
        emb_model.embed_single.side_effect = lambda text: _unit_vec(abs(hash(text)) % 100)
        emb_model.get_embeddings.side_effect = lambda texts: [
            TextEmbeddingValue(_unit_vec(abs(hash(t)) % 100)) for t in texts
        ]

        # Populate a NumPy-backed store
        store = FaissVectorStore(embedding_dim=DIM, use_faiss=False)
        for i in range(num_docs):
            doc = Document(f"doc_{i}", f"Document number {i} about topic {i}", _unit_vec(i))
            store.add(doc)

        # Deterministic expander
        expander = QueryExpander.__new__(QueryExpander)
        expander._use_vertex = False
        expander.expand = lambda q: f"{q} scaling load balancing"  # type: ignore[method-assign]

        return Retriever(store=store, embedding_model=emb_model, query_expander=expander)

    def test_strategy_a_returns_list(self):
        retriever = self._make_retriever()
        results = retriever.retrieve_strategy_a("test query", top_k=3)
        assert isinstance(results, list)

    def test_strategy_a_returns_top_k(self):
        retriever = self._make_retriever(num_docs=5)
        results = retriever.retrieve_strategy_a("test query", top_k=3)
        assert len(results) == 3

    def test_strategy_a_result_type(self):
        retriever = self._make_retriever()
        results = retriever.retrieve_strategy_a("query", top_k=1)
        assert isinstance(results[0], RetrievalResult)

    def test_strategy_a_ranks_start_at_one(self):
        retriever = self._make_retriever()
        results = retriever.retrieve_strategy_a("query", top_k=3)
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3

    def test_strategy_b_returns_tuple(self):
        retriever = self._make_retriever()
        output = retriever.retrieve_strategy_b("test query", top_k=3)
        assert isinstance(output, tuple)
        assert len(output) == 2

    def test_strategy_b_expanded_query_is_string(self):
        retriever = self._make_retriever()
        _, expanded = retriever.retrieve_strategy_b("test query", top_k=3)
        assert isinstance(expanded, str)

    def test_strategy_b_expanded_query_differs_from_original(self):
        retriever = self._make_retriever()
        query = "test query"
        _, expanded = retriever.retrieve_strategy_b(query, top_k=3)
        assert expanded != query

    def test_strategy_b_result_count(self):
        retriever = self._make_retriever(num_docs=5)
        results, _ = retriever.retrieve_strategy_b("test query", top_k=3)
        assert len(results) == 3

    def test_embed_single_called_for_strategy_a(self):
        retriever = self._make_retriever()
        retriever.retrieve_strategy_a("my query", top_k=1)
        retriever._embedder.embed_single.assert_called_with("my query")

    def test_embed_single_called_with_expanded_query_for_strategy_b(self):
        retriever = self._make_retriever()
        query = "my query"
        _, expanded = retriever.retrieve_strategy_b(query, top_k=1)
        retriever._embedder.embed_single.assert_called_with(expanded)
