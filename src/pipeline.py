"""
Orchestration pipeline.

RAGPipeline manages the full lifecycle:
  1. Ingest a list of raw text documents (generate embeddings, index them).
  2. Expose Strategy A and Strategy B retrieval through a clean API.
  3. Optionally persist / reload the vector index from disk.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .embedding import EmbeddingModel
from .retrieval import QueryExpander, RetrievalResult, Retriever
from .storage import Document, FaissVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Context-Aware Retrieval Engine.

    Typical usage::

        pipeline = RAGPipeline()
        pipeline.ingest(CORPUS)
        results_a = pipeline.query_strategy_a("How does the system handle peak load?")
        results_b, expanded = pipeline.query_strategy_b("How does the system handle peak load?")

    Args:
        embedding_dim:   Dimensionality of the embedding vectors.
                         Defaults to 384 (all-MiniLM-L6-v2 output dim).
        embedding_model: Custom EmbeddingModel (injected for testing).
        query_expander:  Custom QueryExpander (injected for testing).
        use_faiss:       Use FAISS index (True) or NumPy fallback (False).
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        embedding_model: Optional[EmbeddingModel] = None,
        query_expander: Optional[QueryExpander] = None,
        use_faiss: bool = True,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._embedder = embedding_model or EmbeddingModel()
        self._expander = query_expander or QueryExpander()
        self._store = FaissVectorStore(embedding_dim=embedding_dim, use_faiss=use_faiss)
        self._retriever: Optional[Retriever] = None
        self._ingested = False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, corpus: List[Dict[str, str]]) -> None:
        """
        Embed and index a list of documents.

        Args:
            corpus: List of dicts with keys ``id`` and ``text``.
        """
        logger.info("Starting ingestion of %d documents …", len(corpus))

        texts = [item["text"] for item in corpus]
        embeddings_objs = self._embedder.get_embeddings(texts)

        docs: List[Document] = []
        for item, emb_obj in zip(corpus, embeddings_objs):
            doc = Document(
                doc_id=item["id"],
                text=item["text"],
                embedding=emb_obj.values,
            )
            docs.append(doc)

        self._store.add_batch(docs)
        self._retriever = Retriever(
            store=self._store,
            embedding_model=self._embedder,
            query_expander=self._expander,
        )
        self._ingested = True
        logger.info("Ingestion complete. %d documents indexed.", len(self._store))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query_strategy_a(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """
        Strategy A – Raw Vector Search.

        Args:
            query: User query string.
            top_k: Number of top results to return.

        Returns:
            Ranked list of RetrievalResult.
        """
        self._ensure_ingested()
        return self._retriever.retrieve_strategy_a(query, top_k=top_k)  # type: ignore[union-attr]

    def query_strategy_b(self, query: str, top_k: int = 3) -> Tuple[List[RetrievalResult], str]:
        """
        Strategy B – AI-Enhanced Retrieval (query expansion).

        Args:
            query: User query string.
            top_k: Number of top results to return.

        Returns:
            Tuple of (ranked RetrievalResult list, expanded_query string).
        """
        self._ensure_ingested()
        return self._retriever.retrieve_strategy_b(query, top_k=top_k)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save the vector index and document metadata to disk."""
        self._store.save(directory)
        logger.info("Pipeline state saved to %r", directory)

    def load(self, directory: str) -> None:
        """Load a previously saved index from disk."""
        self._store = FaissVectorStore.load(directory, self._embedding_dim)
        self._retriever = Retriever(
            store=self._store,
            embedding_model=self._embedder,
            query_expander=self._expander,
        )
        self._ingested = True
        logger.info("Pipeline state loaded from %r (%d docs)", directory, len(self._store))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_ingested(self) -> None:
        if not self._ingested or self._retriever is None:
            raise RuntimeError(
                "Pipeline has not been ingested yet. Call .ingest(corpus) first."
            )

    @property
    def document_count(self) -> int:
        return len(self._store)
