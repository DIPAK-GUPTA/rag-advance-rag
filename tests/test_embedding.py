"""
Tests for the embedding module.

Covers:
- EmbeddingModel.get_embeddings returns the correct type and shape.
- embed_single convenience wrapper.
- Mocking the vertexai.language_models.TextEmbeddingModel SDK surface.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embedding import EmbeddingModel, TextEmbeddingValue


# ---------------------------------------------------------------------------
# TextEmbeddingValue
# ---------------------------------------------------------------------------


class TestTextEmbeddingValue:
    def test_stores_values(self):
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        tev = TextEmbeddingValue(vec)
        np.testing.assert_array_equal(tev.values, vec)

    def test_values_dtype(self):
        vec = np.ones(16, dtype=np.float64)
        tev = TextEmbeddingValue(vec)
        assert tev.values.dtype == np.float64  # stores as-is


# ---------------------------------------------------------------------------
# EmbeddingModel – local (sentence-transformers) path
# ---------------------------------------------------------------------------


class TestEmbeddingModelLocal:
    """Tests that run without the real Vertex AI SDK."""

    def _make_model(self, fake_st_model):
        """Patch the lazy-loaded _st_model to avoid real downloads."""
        import src.embedding as emb_module

        emb_module._st_model = fake_st_model
        model = EmbeddingModel.__new__(EmbeddingModel)
        model.model_name = "all-MiniLM-L6-v2"
        model._use_vertex = False
        return model

    def _build_fake_st(self, dim: int = 16):
        fake = MagicMock()
        fake.encode.side_effect = lambda texts, **kw: np.random.default_rng(0).random(
            (len(texts), dim)
        ).astype(np.float32)
        return fake

    def test_get_embeddings_returns_list(self):
        fake_st = self._build_fake_st(16)
        model = self._make_model(fake_st)
        results = model.get_embeddings(["hello", "world"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_get_embeddings_type(self):
        fake_st = self._build_fake_st(16)
        model = self._make_model(fake_st)
        results = model.get_embeddings(["test"])
        assert isinstance(results[0], TextEmbeddingValue)

    def test_get_embeddings_shape(self):
        dim = 16
        fake_st = self._build_fake_st(dim)
        model = self._make_model(fake_st)
        results = model.get_embeddings(["test sentence"])
        assert results[0].values.shape == (dim,)

    def test_embed_single_returns_ndarray(self):
        fake_st = self._build_fake_st(16)
        model = self._make_model(fake_st)
        vec = model.embed_single("test")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1

    def test_get_embeddings_dtype_is_float32(self):
        fake_st = self._build_fake_st(16)
        model = self._make_model(fake_st)
        results = model.get_embeddings(["check dtype"])
        assert results[0].values.dtype == np.float32

    def test_encode_called_with_correct_texts(self):
        fake_st = self._build_fake_st(8)
        model = self._make_model(fake_st)
        texts = ["alpha", "beta", "gamma"]
        model.get_embeddings(texts)
        call_args = fake_st.encode.call_args
        assert call_args[0][0] == texts


# ---------------------------------------------------------------------------
# Mocking the vertexai SDK surface (GCP SDK mock test)
# ---------------------------------------------------------------------------


class TestVertexAISDKMock:
    """
    Verifies that EmbeddingModel can transparently delegate to the
    vertexai.language_models.TextEmbeddingModel SDK interface.
    """

    def test_vertex_path_calls_sdk_get_embeddings(self):
        dim = 32
        fake_values = np.random.default_rng(7).random((1, dim)).astype(np.float32)

        # Build a mock that mirrors the real SDK response object
        mock_response = MagicMock()
        mock_response.values = fake_values[0].tolist()

        mock_vertex_model = MagicMock()
        mock_vertex_model.get_embeddings.return_value = [mock_response]

        mock_from_pretrained = MagicMock(return_value=mock_vertex_model)

        with patch("src.embedding._VERTEX_AVAILABLE", True), \
             patch("src.embedding._VertexTextEmbeddingModel") as mock_cls:
            mock_cls.from_pretrained = mock_from_pretrained

            import os
            with patch.dict(os.environ, {"USE_VERTEX_AI": "1"}):
                model = EmbeddingModel(model_name="textembedding-gecko@003")
                model._use_vertex = True
                model._vertex_model = mock_vertex_model

                results = model.get_embeddings(["peak load handling"])

        mock_vertex_model.get_embeddings.assert_called_once_with(["peak load handling"])
        assert len(results) == 1
        assert isinstance(results[0], TextEmbeddingValue)
        np.testing.assert_allclose(results[0].values, fake_values[0], rtol=1e-5)

    def test_vertex_fallback_when_unavailable(self):
        """When USE_VERTEX_AI=0 (default), local model is used."""
        import src.embedding as emb_module

        fake_st = MagicMock()
        fake_st.encode.return_value = np.ones((1, 8), dtype=np.float32)
        emb_module._st_model = fake_st

        model = EmbeddingModel.__new__(EmbeddingModel)
        model.model_name = "all-MiniLM-L6-v2"
        model._use_vertex = False

        results = model.get_embeddings(["test"])
        assert len(results) == 1
        fake_st.encode.assert_called_once()
