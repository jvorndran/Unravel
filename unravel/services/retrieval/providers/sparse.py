"""Sparse retriever using BM25 keyword search."""

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import RetrieverInfo, RetrieverProvider

if TYPE_CHECKING:
    from unravel.services.embedders import Embedder
    from unravel.services.vector_store import SearchResult, VectorStore

# Lazy import BM25
_BM25Okapi: Any | None = None


def _get_bm25() -> Any:
    """Lazy load BM25 to avoid slow startup."""
    global _BM25Okapi
    if _BM25Okapi is None:
        from rank_bm25 import BM25Okapi

        _BM25Okapi = BM25Okapi
    return _BM25Okapi


class SparseRetriever(RetrieverProvider):
    """Sparse retrieval using BM25 keyword search."""

    @property
    def name(self) -> str:
        return "SparseRetriever"

    def get_available_retrievers(self) -> list[RetrieverInfo]:
        return [
            RetrieverInfo(
                name="SparseRetriever",
                display_name="Sparse (BM25)",
                description="Keyword search using BM25 (rank-bm25)",
                category="Retrieval",
            )
        ]

    def search(
        self,
        retriever_name: str,
        query: str,
        k: int,
        vector_store: "VectorStore",
        embedder: "Embedder",
        **params: Any,
    ) -> list["SearchResult"]:
        """Search using BM25 keyword matching."""
        from unravel.services.vector_store import SearchResult

        # Get BM25 index from params
        bm25_data = params.get("bm25_index_data")
        if not bm25_data:
            raise ValueError("BM25 index not built. Run preprocessing first.")

        bm25_index = bm25_data["bm25_index"]
        texts = vector_store.get_texts()
        metadata = vector_store.get_metadata()

        # Tokenize query (simple whitespace + lowercase)
        tokenized_query = query.lower().split()

        # Get scores for all documents
        scores = bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[-k:][::-1]

        # Build SearchResults
        results = []
        for idx in top_indices:
            results.append(
                SearchResult(
                    index=int(idx),
                    score=float(scores[idx]),
                    text=texts[idx],
                    metadata=metadata[idx],
                )
            )

        return results

    def preprocess(
        self,
        retriever_name: str,
        vector_store: "VectorStore",
        **params: Any,
    ) -> dict[str, Any]:
        """Build BM25 index from vector store texts."""
        BM25Okapi = _get_bm25()

        texts = vector_store.get_texts()

        # Tokenize all texts (simple whitespace + lowercase)
        tokenized_corpus = [text.lower().split() for text in texts]

        # Build BM25 index
        bm25_index = BM25Okapi(tokenized_corpus)

        return {
            "bm25_index": bm25_index,
            "tokenized_corpus": tokenized_corpus,
        }
