"""Base classes for retrieval providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from unravel.services.embedders import Embedder
    from unravel.services.vector_store import SearchResult, VectorStore


@dataclass
class RetrieverInfo:
    """Metadata about a retrieval strategy."""

    name: str
    display_name: str
    description: str
    category: str = "Retrieval"


class RetrieverProvider(ABC):
    """Abstract base class for retrieval providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass

    @abstractmethod
    def get_available_retrievers(self) -> list[RetrieverInfo]:
        """Return list of available retrievers."""
        pass

    @abstractmethod
    def search(
        self,
        retriever_name: str,
        query: str,
        k: int,
        vector_store: "VectorStore",
        embedder: "Embedder",
        **params: Any,
    ) -> list["SearchResult"]:
        """Perform retrieval."""
        pass

    @abstractmethod
    def preprocess(
        self,
        retriever_name: str,
        vector_store: "VectorStore",
        **params: Any,
    ) -> dict[str, Any]:
        """Build any necessary indices."""
        pass
