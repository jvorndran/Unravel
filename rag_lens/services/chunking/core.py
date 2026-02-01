"""Core chunking functionality."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import streamlit as st

if TYPE_CHECKING:
    from .providers.base import ChunkingProvider, SplitterInfo


@dataclass
class Chunk:
    """A chunk of text with position tracking."""

    text: str
    start_index: int
    end_index: int
    metadata: dict[str, Any]


# Provider registry
_PROVIDERS: dict[str, "ChunkingProvider"] = {}


def register_provider(provider: "ChunkingProvider") -> None:
    """Register a chunking provider."""
    _PROVIDERS[provider.name] = provider


def get_available_providers() -> list[str]:
    """Get list of registered provider names."""
    return list(_PROVIDERS.keys())


def get_provider(name: str) -> Optional["ChunkingProvider"]:
    """Get a provider by name."""
    return _PROVIDERS.get(name)


def get_provider_splitters(provider_name: str) -> list["SplitterInfo"]:
    """Get available splitters for a provider with their parameter schemas."""
    provider = _PROVIDERS.get(provider_name)
    if provider:
        return provider.get_available_splitters()
    return []


@st.cache_data(show_spinner="Generating chunks...")
def _get_chunks_cached(
    provider: str,
    splitter: str,
    text: str,
    **params: Any,  # noqa: ANN401
) -> list[dict[str, Any]]:
    """Internal cached function that returns serializable dicts."""
    provider_instance = _PROVIDERS.get(provider)
    if not provider_instance:
        raise ValueError(f"Unknown provider: {provider}")
    chunks = provider_instance.chunk(splitter, text, **params)
    # Convert to dicts for pickle serialization
    return [
        {
            "text": c.text,
            "start_index": c.start_index,
            "end_index": c.end_index,
            "metadata": c.metadata,
        }
        for c in chunks
    ]


def get_chunks(
    provider: str,
    splitter: str,
    text: str,
    **params: Any,  # noqa: ANN401
) -> list[Chunk]:
    """Get chunks using specified provider and splitter.

    Args:
        provider: Provider name (e.g., "LangChain")
        splitter: Splitter name within the provider
        text: Source text to chunk
        **params: Splitter-specific parameters

    Returns:
        List of Chunk objects with text, start_index, end_index, metadata
    """
    chunk_dicts = _get_chunks_cached(provider, splitter, text, **params)
    # Convert back to Chunk objects
    return [
        Chunk(
            text=d["text"],
            start_index=d["start_index"],
            end_index=d["end_index"],
            metadata=d["metadata"],
        )
        for d in chunk_dicts
    ]


def _init_providers() -> None:
    """Initialize and register all providers."""
    from .providers.docling_provider import DoclingProvider

    register_provider(DoclingProvider())


# Auto-initialize on import
_init_providers()
