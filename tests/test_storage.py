"""
Tests for the vector storage module (FaissVectorStore).

All tests use the NumPy backend (use_faiss=False) to avoid a FAISS
installation requirement in CI.  FAISS-specific paths are tested via
unit-level mocking.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.storage import Document, FaissVectorStore

DIM = 8
RNG = np.random.default_rng(42)


def _unit_vec(seed: int = 0, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_doc(doc_id: str, seed: int) -> Document:
    return Document(doc_id=doc_id, text=f"Document {doc_id}", embedding=_unit_vec(seed))


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class TestDocument:
    def test_to_dict(self):
        doc = Document("d1", "hello world")
        d = doc.to_dict()
        assert d["doc_id"] == "d1"
        assert d["text"] == "hello world"

    def test_no_embedding_by_default(self):
        doc = Document("d2", "text")
        assert doc.embedding is None


# ---------------------------------------------------------------------------
# FaissVectorStore – NumPy backend
# ---------------------------------------------------------------------------


class TestFaissVectorStoreNumpy:

    def _make_store(self) -> FaissVectorStore:
        return FaissVectorStore(embedding_dim=DIM, use_faiss=False)

    def test_empty_store_len(self):
        store = self._make_store()
        assert len(store) == 0
        assert store.is_empty

    def test_add_single_document(self):
        store = self._make_store()
        store.add(_make_doc("d1", 1))
        assert len(store) == 1
        assert not store.is_empty

    def test_add_batch(self):
        store = self._make_store()
        docs = [_make_doc(f"d{i}", i) for i in range(5)]
        store.add_batch(docs)
        assert len(store) == 5

    def test_add_without_embedding_raises(self):
        store = self._make_store()
        doc = Document("d1", "no embedding")
        with pytest.raises(ValueError, match="no embedding"):
            store.add(doc)

    def test_search_returns_top_k(self):
        store = self._make_store()
        docs = [_make_doc(f"d{i}", i) for i in range(5)]
        store.add_batch(docs)
        query = _unit_vec(seed=0)
        results = store.search(query, top_k=3)
        assert len(results) == 3

    def test_search_result_type(self):
        store = self._make_store()
        store.add(_make_doc("d1", 1))
        results = store.search(_unit_vec(1), top_k=1)
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)

    def test_search_returns_highest_similarity_first(self):
        store = self._make_store()
        # doc_a is identical to query → cosine sim = 1.0
        query = _unit_vec(seed=99)
        doc_a = Document("exact", "exact match", embedding=query.copy())
        doc_b = _make_doc("other", seed=1)
        store.add(doc_a)
        store.add(doc_b)
        results = store.search(query, top_k=2)
        assert results[0][0].doc_id == "exact"

    def test_search_scores_in_descending_order(self):
        store = self._make_store()
        docs = [_make_doc(f"d{i}", i) for i in range(6)]
        store.add_batch(docs)
        results = store.search(_unit_vec(seed=3), top_k=6)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_store_returns_empty(self):
        store = self._make_store()
        results = store.search(_unit_vec(), top_k=3)
        assert results == []

    def test_top_k_capped_at_store_size(self):
        store = self._make_store()
        store.add(_make_doc("d1", 1))
        results = store.search(_unit_vec(), top_k=10)
        assert len(results) == 1

    def test_add_batch_empty_list(self):
        store = self._make_store()
        store.add_batch([])  # should not raise
        assert len(store) == 0

    def test_scores_are_cosine_similarity(self):
        """For a query equal to an indexed vector, score should be ~1.0."""
        store = self._make_store()
        v = _unit_vec(seed=7)
        doc = Document("perfect", "perfect match", embedding=v.copy())
        store.add(doc)
        results = store.search(v, top_k=1)
        assert abs(results[0][1] - 1.0) < 1e-5

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def test_save_and_load(self, tmp_path):
        store = self._make_store()
        docs = [_make_doc(f"d{i}", i) for i in range(3)]
        store.add_batch(docs)
        store.save(str(tmp_path))

        # Metadata file must exist
        assert (tmp_path / "metadata.json").exists()

        loaded = FaissVectorStore.load(str(tmp_path), DIM)
        assert len(loaded) == 3
        doc_ids = {d.doc_id for d in loaded._documents}
        assert doc_ids == {"d0", "d1", "d2"}

    def test_save_creates_metadata_json(self, tmp_path):
        store = self._make_store()
        store.add(_make_doc("d1", 1))
        store.save(str(tmp_path))
        with open(tmp_path / "metadata.json") as f:
            meta = json.load(f)
        assert meta[0]["doc_id"] == "d1"
