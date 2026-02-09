"""Retrieval service for RAG-Visualizer.

Provides multiple retrieval strategies:
- Dense: Vector similarity using Qdrant
- Sparse: Keyword search using BM25
- Hybrid: Combined dense + sparse with score fusion
"""

from .core import (
    get_available_retrievers,
    get_retriever,
    preprocess_retriever,
    retrieve,
)
from .providers import RetrieverInfo, RetrieverProvider
from .reranking import RerankerConfig, rerank_results

__all__ = [
    "get_available_retrievers",
    "get_retriever",
    "retrieve",
    "preprocess_retriever",
    "RetrieverProvider",
    "RetrieverInfo",
    "rerank_results",
    "RerankerConfig",
]
