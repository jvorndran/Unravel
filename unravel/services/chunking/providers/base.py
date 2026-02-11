"""Base classes for chunking providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core import Chunk


@dataclass
class ParameterInfo:
    """Schema for a splitter parameter."""

    name: str
    type: str  # "int", "float", "str", "bool", "multiselect"
    default: Any
    description: str
    min_value: Any | None = None
    max_value: Any | None = None
    options: list[Any] | None = None  # For enum-like or multiselect parameters


@dataclass
class SplitterInfo:
    """Metadata about an available splitter."""

    name: str
    display_name: str
    description: str
    parameters: list[ParameterInfo] = field(default_factory=list)
    category: str = "General"


class ChunkingProvider(ABC):
    """Abstract base class for chunking providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'langchain')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name."""
        pass

    @property
    def attribution(self) -> str | None:
        """Attribution text (e.g., 'Powered by LangChain')."""
        return None

    @abstractmethod
    def get_available_splitters(self) -> list[SplitterInfo]:
        """Return list of available splitters with their parameter schemas."""
        pass

    @abstractmethod
    def chunk(self, splitter_name: str, text: str, **params: Any) -> list["Chunk"]:  # noqa: ANN401
        """Split text into chunks using the specified splitter.

        Must return Chunk objects with accurate start_index and end_index.
        """
        pass
