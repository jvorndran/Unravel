"""Reranking module with support for multiple libraries."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_lens.services.vector_store import SearchResult

# Model registry mapping model names to backend information
RERANKER_MODELS: dict[str, dict[str, Any]] = {
    # FlashRank (existing)
    "ms-marco-MiniLM-L-12-v2": {
        "backend": "flashrank",
        "library": "FlashRank",
        "size": "small",
        "description": "Fast, lightweight reranker",
    },
    "ms-marco-TinyBERT-L-2-v2": {
        "backend": "flashrank",
        "library": "FlashRank",
        "size": "tiny",
        "description": "Fastest reranker",
    },
    # Mixedbread AI
    "mixedbread-ai/mxbai-rerank-xsmall-v1": {
        "backend": "sentence-transformers",
        "library": "Mixedbread AI",
        "size": "xsmall",
        "description": "Ultra-lightweight reranker",
    },
    "mixedbread-ai/mxbai-rerank-base-v1": {
        "backend": "sentence-transformers",
        "library": "Mixedbread AI",
        "size": "base",
        "description": "Balanced speed and quality",
    },
    "mixedbread-ai/mxbai-rerank-large-v1": {
        "backend": "sentence-transformers",
        "library": "Mixedbread AI",
        "size": "large",
        "description": "High quality reranker",
    },
    # BGE Reranker (BAAI)
    "BAAI/bge-reranker-base": {
        "backend": "flagembedding",
        "library": "BAAI",
        "size": "base",
        "description": "BAAI base reranker",
    },
    "BAAI/bge-reranker-large": {
        "backend": "flagembedding",
        "library": "BAAI",
        "size": "large",
        "description": "BAAI large reranker",
    },
    "BAAI/bge-reranker-v2-m3": {
        "backend": "flagembedding",
        "library": "BAAI",
        "size": "base",
        "description": "Multilingual reranker",
    },
    # Jina AI
    "jinaai/jina-reranker-v2-base-multilingual": {
        "backend": "sentence-transformers",
        "library": "Jina AI",
        "size": "base",
        "description": "Multilingual reranker",
    },
}


def list_available_rerankers() -> list[dict[str, Any]]:
    """List all available reranker models with their details.

    Returns:
        List of dicts containing model name and metadata
    """
    return [{"name": name, **info} for name, info in RERANKER_MODELS.items()]


@dataclass
class RerankerConfig:
    """Configuration for reranking."""

    enabled: bool = False
    model: str = "ms-marco-MiniLM-L-12-v2"
    top_n: int = 5


def rerank_results(
    query: str,
    results: list["SearchResult"],
    config: RerankerConfig,
) -> list["SearchResult"]:
    """Rerank search results using cross-encoder.

    Args:
        query: The search query
        results: List of search results to rerank
        config: Reranking configuration

    Returns:
        Reranked and filtered list of search results

    Raises:
        ValueError: If model name is not in registry
        ImportError: If required library is not installed
    """
    from rag_lens.services.vector_store import SearchResult

    if not config.enabled or not results:
        return results

    # Validate model exists
    if config.model not in RERANKER_MODELS:
        raise ValueError(
            f"Unknown reranker model: {config.model}. "
            f"Available models: {', '.join(RERANKER_MODELS.keys())}"
        )

    # Get backend
    model_info = RERANKER_MODELS[config.model]
    backend_name = model_info["backend"]

    # Prepare passages
    passages = [{"id": i, "text": r.text} for i, r in enumerate(results)]

    # Rerank using backend
    from .reranking_backends import get_backend

    backend = get_backend(backend_name)
    reranked = backend.rerank(config.model, query, passages)

    # Map back to SearchResults with new scores
    reranked_results = []
    for item in reranked[: config.top_n]:
        original = results[item["id"]]
        reranked_results.append(
            SearchResult(
                index=original.index,
                score=item["score"],
                text=original.text,
                metadata={
                    **original.metadata,
                    "original_score": original.score,
                    "reranked": True,
                    "reranker_model": config.model,
                },
            )
        )

    return reranked_results
