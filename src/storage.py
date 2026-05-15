"""
Vector storage module.

Wraps FAISS (with a NumPy fallback) to provide a simple key-value store
where documents are indexed by their embedding vectors and retrieved via
approximate nearest-neighbour (ANN) search.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# FAISS import with graceful NumPy fallback
# ---------------------------------------------------------------------------

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Document:
    """Represents a stored chunk with its metadata."""

    def __init__(self, doc_id: str, text: str, embedding: Optional[np.ndarray] = None) -> None:
        self.doc_id = doc_id
        self.text = text
        self.embedding = embedding  # shape (dim,) float32

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "text": self.text}

    def __repr__(self) -> str:  # pragma: no cover
        return f"Document(doc_id={self.doc_id!r}, text={self.text[:60]!r}...)"


# ---------------------------------------------------------------------------
# FAISS-backed index
# ---------------------------------------------------------------------------

class FaissVectorStore:
    """
    Lightweight FAISS index (Inner Product on L2-normalised vectors ≡ cosine
    similarity).  Falls back to a brute-force NumPy implementation when FAISS
    is not installed – useful for CI environments.
    """

    def __init__(self, embedding_dim: int, use_faiss: bool = True) -> None:
        self.embedding_dim = embedding_dim
        self._use_faiss = use_faiss and _FAISS_AVAILABLE
        self._documents: List[Document] = []
        self._embeddings: List[np.ndarray] = []  # only used by NumPy backend

        if self._use_faiss:
            # IndexFlatIP with pre-normalised vectors == cosine similarity
            self._index = faiss.IndexFlatIP(embedding_dim)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add(self, doc: Document) -> None:
        """
        Add a document with its embedding to the index.
        The embedding must already be L2-normalised (unit norm).
        """
        if doc.embedding is None:
            raise ValueError(f"Document {doc.doc_id!r} has no embedding.")

        vec = doc.embedding.astype(np.float32).reshape(1, -1)
        if self._use_faiss:
            faiss.normalize_L2(vec)  # idempotent if already normalised
            self._index.add(vec)
        else:
            norm = np.linalg.norm(vec)
            self._embeddings.append(vec / norm if norm > 0 else vec)

        self._documents.append(doc)

    def add_batch(self, docs: List[Document]) -> None:
        """Add multiple documents at once (more efficient for FAISS)."""
        if not docs:
            return

        embeddings = np.vstack([d.embedding.astype(np.float32) for d in docs])
        if self._use_faiss:
            faiss.normalize_L2(embeddings)
            self._index.add(embeddings)
        else:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._embeddings = list(embeddings / norms)

        self._documents.extend(docs)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Return the top-k most similar documents and their cosine similarity scores.

        Args:
            query_vector: 1-D float32 numpy array (will be normalised internally).
            top_k:        Number of results to return.

        Returns:
            List of (Document, score) tuples, sorted by descending score.
        """
        if len(self._documents) == 0:
            return []

        k = min(top_k, len(self._documents))
        query = query_vector.astype(np.float32).reshape(1, -1)

        if self._use_faiss:
            faiss.normalize_L2(query)
            scores, indices = self._index.search(query, k)
            results = [
                (self._documents[int(idx)], float(scores[0][rank]))
                for rank, idx in enumerate(indices[0])
                if idx != -1
            ]
        else:
            # NumPy brute-force cosine similarity
            norm = np.linalg.norm(query)
            q = query / norm if norm > 0 else query
            matrix = np.vstack(self._embeddings)  # (N, dim)
            sims = (matrix @ q.T).flatten()        # cosine similarity
            top_indices = np.argsort(sims)[::-1][:k]
            results = [(self._documents[int(i)], float(sims[i])) for i in top_indices]

        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Persist the index and document metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        meta_path = os.path.join(directory, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([d.to_dict() for d in self._documents], f, indent=2)

        if self._use_faiss:
            faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        else:
            np.save(os.path.join(directory, "embeddings.npy"),
                    np.vstack(self._embeddings))

    @classmethod
    def load(cls, directory: str, embedding_dim: int) -> "FaissVectorStore":
        """Restore a previously saved store from disk."""
        store = cls(embedding_dim)
        meta_path = os.path.join(directory, "metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            metas = json.load(f)

        faiss_path = os.path.join(directory, "index.faiss")
        npy_path = os.path.join(directory, "embeddings.npy")

        if store._use_faiss and os.path.exists(faiss_path):
            store._index = faiss.read_index(faiss_path)
            store._documents = [Document(m["doc_id"], m["text"]) for m in metas]
        elif os.path.exists(npy_path):
            embeddings = np.load(npy_path)
            store._use_faiss = False
            for i, m in enumerate(metas):
                doc = Document(m["doc_id"], m["text"], embeddings[i])
                store._documents.append(doc)
                store._embeddings.append(embeddings[i].reshape(1, -1))
        else:
            raise FileNotFoundError(f"No index files found in {directory!r}")

        return store

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._documents)

    @property
    def is_empty(self) -> bool:
        return len(self._documents) == 0
