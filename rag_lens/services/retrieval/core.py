"""Core retrieval module with provider registry."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_lens.services.embedders import Embedder
    from rag_lens.services.vector_store import SearchResult, VectorStore

    from .providers import RetrieverProvider

# Provider registry (similar to chunking)
_PROVIDERS: dict[str, "RetrieverProvider"] = {}


def register_provider(provider: "RetrieverProvider") -> None:
    """Register a retrieval provider.

    Args:
        provider: The provider instance to register
    """
    _PROVIDERS[provider.name] = provider


def get_available_retrievers() -> list[str]:
    """Get list of registered retriever names.

    Returns:
        List of retriever provider names
    """
    return list(_PROVIDERS.keys())


def get_retriever(name: str) -> "RetrieverProvider | None":
    """Get a retriever provider by name.

    Args:
        name: The provider name

    Returns:
        The provider instance or None if not found
    """
    return _PROVIDERS.get(name)


def retrieve(
    query: str,
    vector_store: "VectorStore",
    embedder: "Embedder",
    retriever_name: str,
    k: int,
    **params: Any,
) -> list["SearchResult"]:
    """Retrieve chunks using specified strategy.

    Args:
        query: The search query
        vector_store: The vector store to search
        embedder: The embedder for dense retrieval
        retriever_name: Name of the retriever to use
        k: Number of results to return
        **params: Additional parameters for the retriever

    Returns:
        List of search results

    Raises:
        ValueError: If retriever not found
    """
    provider = _PROVIDERS.get(retriever_name)
    if not provider:
        raise ValueError(f"Unknown retriever: {retriever_name}")

    return provider.search(
        retriever_name=retriever_name,
        query=query,
        k=k,
        vector_store=vector_store,
        embedder=embedder,
        **params,
    )


def preprocess_retriever(
    retriever_name: str,
    vector_store: "VectorStore",
    **params: Any,
) -> dict[str, Any]:
    """Build any necessary indices (e.g., BM25).

    Args:
        retriever_name: Name of the retriever
        vector_store: The vector store containing texts
        **params: Additional parameters for preprocessing

    Returns:
        Dictionary containing preprocessed data (e.g., BM25 index)

    Raises:
        ValueError: If retriever not found
    """
    provider = _PROVIDERS.get(retriever_name)
    if not provider:
        raise ValueError(f"Unknown retriever: {retriever_name}")

    return provider.preprocess(retriever_name, vector_store, **params)


def _init_providers() -> None:
    """Initialize and register all providers."""
    from .providers.dense import DenseRetriever
    from .providers.hybrid import HybridRetriever
    from .providers.sparse import SparseRetriever

    register_provider(DenseRetriever())
    register_provider(SparseRetriever())
    register_provider(HybridRetriever())


# Auto-initialize on import
_init_providers()
