"""Backend adapters for different reranking libraries."""

from abc import ABC, abstractmethod
from typing import Any


class RerankerBackend(ABC):
    """Abstract base class for reranking backends."""

    @abstractmethod
    def rerank(
        self, model_name: str, query: str, passages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Rerank passages for a query.

        Args:
            model_name: Name of the model to use
            query: Search query
            passages: List of dicts with 'id' and 'text' keys

        Returns:
            List of dicts with 'id', 'text', and 'score' keys, sorted by score descending
        """
        pass


class FlashRankAdapter(RerankerBackend):
    """Adapter for FlashRank library."""

    def __init__(self):
        self._ranker_cache: dict[str, Any] = {}

    def rerank(
        self, model_name: str, query: str, passages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Rerank using FlashRank."""
        from flashrank import Ranker, RerankRequest

        # Load model (cached)
        if model_name not in self._ranker_cache:
            self._ranker_cache[model_name] = Ranker(model_name=model_name)

        ranker = self._ranker_cache[model_name]

        # Format request
        request = RerankRequest(query=query, passages=passages)

        # Rerank
        results = ranker.rerank(request)

        # Convert to standard format
        return [
            {"id": result["id"], "text": result["text"], "score": result["score"]}
            for result in results
        ]


class SentenceTransformersAdapter(RerankerBackend):
    """Adapter for sentence-transformers CrossEncoder (Mixedbread, Jina)."""

    def __init__(self):
        self._model_cache: dict[str, Any] = {}

    def rerank(
        self, model_name: str, query: str, passages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Rerank using sentence-transformers CrossEncoder."""
        from sentence_transformers import CrossEncoder

        # Load model (cached)
        if model_name not in self._model_cache:
            self._model_cache[model_name] = CrossEncoder(model_name)

        model = self._model_cache[model_name]

        # Format as (query, doc) pairs
        pairs = [(query, passage["text"]) for passage in passages]

        # Get scores
        scores = model.predict(pairs)

        # Combine with passages and sort
        results = [
            {"id": passage["id"], "text": passage["text"], "score": float(score)}
            for passage, score in zip(passages, scores)
        ]

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results


class FlagEmbeddingAdapter(RerankerBackend):
    """Adapter for FlagEmbedding library (BGE models)."""

    def __init__(self):
        self._model_cache: dict[str, Any] = {}

    def rerank(
        self, model_name: str, query: str, passages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Rerank using FlagEmbedding."""
        from FlagEmbedding import FlagReranker

        # Load model (cached)
        if model_name not in self._model_cache:
            self._model_cache[model_name] = FlagReranker(model_name, use_fp16=True)

        reranker = self._model_cache[model_name]

        # Format as [[query, doc], ...] pairs
        pairs = [[query, passage["text"]] for passage in passages]

        # Get scores
        scores = reranker.compute_score(pairs)

        # Handle both single score and list of scores
        if not isinstance(scores, list):
            scores = [scores]

        # Combine with passages and sort
        results = [
            {"id": passage["id"], "text": passage["text"], "score": float(score)}
            for passage, score in zip(passages, scores)
        ]

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results


# Global backend instances (lazy loaded)
_backends: dict[str, RerankerBackend] = {}


def get_backend(backend_name: str) -> RerankerBackend:
    """
    Get a reranking backend instance.

    Args:
        backend_name: Name of the backend ('flashrank', 'sentence-transformers', 'flagembedding')

    Returns:
        RerankerBackend instance

    Raises:
        ValueError: If backend name is unknown
        ImportError: If required library is not installed
    """
    if backend_name in _backends:
        return _backends[backend_name]

    if backend_name == "flashrank":
        _backends[backend_name] = FlashRankAdapter()
    elif backend_name == "sentence-transformers":
        _backends[backend_name] = SentenceTransformersAdapter()
    elif backend_name == "flagembedding":
        _backends[backend_name] = FlagEmbeddingAdapter()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    return _backends[backend_name]
