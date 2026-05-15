"""
Tests for the RAGPipeline orchestration class.

Validates ingestion, retrieval delegation, document counting, and
error handling – all using mocked embedding / expansion components.
"""

from __future__ import annotations

import pytest

from src.pipeline import RAGPipeline


class TestRAGPipelineIngestion:

    def test_ingest_populates_store(self, populated_pipeline, small_corpus):
        assert populated_pipeline.document_count == len(small_corpus)

    def test_ingest_sets_ingested_flag(self, populated_pipeline):
        assert populated_pipeline._ingested is True

    def test_ingest_creates_retriever(self, populated_pipeline):
        assert populated_pipeline._retriever is not None

    def test_double_ingest_accumulates(
        self, mock_embedding_model, mock_query_expander, small_corpus
    ):
        pipeline = RAGPipeline(
            embedding_dim=8,
            embedding_model=mock_embedding_model,
            query_expander=mock_query_expander,
            use_faiss=False,
        )
        pipeline.ingest(small_corpus)
        pipeline.ingest(small_corpus)
        assert pipeline.document_count == len(small_corpus) * 2


class TestRAGPipelineQueryStrategyA:

    def test_returns_list(self, populated_pipeline):
        results = populated_pipeline.query_strategy_a("peak load", top_k=3)
        assert isinstance(results, list)

    def test_top_k_respected(self, populated_pipeline, small_corpus):
        k = 3
        results = populated_pipeline.query_strategy_a("test", top_k=k)
        assert len(results) == min(k, len(small_corpus))

    def test_results_have_doc_id(self, populated_pipeline):
        results = populated_pipeline.query_strategy_a("test", top_k=1)
        assert hasattr(results[0].doc, "doc_id")

    def test_results_are_ranked(self, populated_pipeline):
        results = populated_pipeline.query_strategy_a("test", top_k=3)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))


class TestRAGPipelineQueryStrategyB:

    def test_returns_tuple(self, populated_pipeline):
        output = populated_pipeline.query_strategy_b("peak load")
        assert isinstance(output, tuple) and len(output) == 2

    def test_expanded_query_is_string(self, populated_pipeline):
        _, expanded = populated_pipeline.query_strategy_b("any query")
        assert isinstance(expanded, str)

    def test_expanded_query_longer_than_original(self, populated_pipeline):
        query = "peak load"
        _, expanded = populated_pipeline.query_strategy_b(query)
        assert len(expanded) >= len(query)

    def test_top_k_respected(self, populated_pipeline, small_corpus):
        k = 2
        results, _ = populated_pipeline.query_strategy_b("test", top_k=k)
        assert len(results) == min(k, len(small_corpus))


class TestRAGPipelineErrors:

    def test_query_before_ingest_raises(
        self, mock_embedding_model, mock_query_expander
    ):
        pipeline = RAGPipeline(
            embedding_dim=8,
            embedding_model=mock_embedding_model,
            query_expander=mock_query_expander,
            use_faiss=False,
        )
        with pytest.raises(RuntimeError, match="not been ingested"):
            pipeline.query_strategy_a("test")

    def test_strategy_b_before_ingest_raises(
        self, mock_embedding_model, mock_query_expander
    ):
        pipeline = RAGPipeline(
            embedding_dim=8,
            embedding_model=mock_embedding_model,
            query_expander=mock_query_expander,
            use_faiss=False,
        )
        with pytest.raises(RuntimeError, match="not been ingested"):
            pipeline.query_strategy_b("test")


class TestRAGPipelinePersistence:

    def test_save_and_load(self, tmp_path, populated_pipeline):
        populated_pipeline.save(str(tmp_path))
        loaded = RAGPipeline(
            embedding_dim=8,
            embedding_model=populated_pipeline._embedder,
            query_expander=populated_pipeline._expander,
            use_faiss=False,
        )
        loaded.load(str(tmp_path))
        assert loaded.document_count == populated_pipeline.document_count

    def test_load_restores_query_capability(self, tmp_path, populated_pipeline):
        populated_pipeline.save(str(tmp_path))
        loaded = RAGPipeline(
            embedding_dim=8,
            embedding_model=populated_pipeline._embedder,
            query_expander=populated_pipeline._expander,
            use_faiss=False,
        )
        loaded.load(str(tmp_path))
        results = loaded.query_strategy_a("test", top_k=1)
        assert len(results) == 1
