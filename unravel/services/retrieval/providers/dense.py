"""Dense retriever using Qdrant vector search."""

from typing import TYPE_CHECKING, Any

from .base import RetrieverInfo, RetrieverProvider

if TYPE_CHECKING:
    from unravel.services.embedders import Embedder
    from unravel.services.vector_store import SearchResult, VectorStore


class DenseRetriever(RetrieverProvider):
    """Dense retrieval using Qdrant vector similarity search."""

    @property
    def name(self) -> str:
        return "DenseRetriever"

    def get_available_retrievers(self) -> list[RetrieverInfo]:
        return [
            RetrieverInfo(
                name="DenseRetriever",
                display_name="Dense (Qdrant)",
                description="Vector similarity search using Qdrant",
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
        """Search using dense vector similarity."""
        query_embedding = embedder.embed_query(query)
        return vector_store.search(query_embedding, k=k)

    def preprocess(
        self,
        retriever_name: str,
        vector_store: "VectorStore",
        **params: Any,
    ) -> dict[str, Any]:
        """No preprocessing needed for dense retrieval."""
        return {}
