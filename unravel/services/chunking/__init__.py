"""Chunking module for Unravel."""

from .core import (
    Chunk,
    get_available_providers,
    get_chunks,
    get_provider,
    get_provider_splitters,
    register_provider,
)
from .providers import (
    ChunkingProvider,
    DoclingProvider,
    ParameterInfo,
    SplitterInfo,
)

__all__ = [
    # Core
    "Chunk",
    "get_chunks",
    "get_available_providers",
    "get_provider",
    "get_provider_splitters",
    "register_provider",
    # Providers
    "ChunkingProvider",
    "SplitterInfo",
    "ParameterInfo",
    "DoclingProvider",
]
