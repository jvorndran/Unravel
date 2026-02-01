"""
Embedding models for RAG-Visualizer.

Provides a unified interface for generating embeddings using sentence-transformers.
"""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import streamlit as st

# Lazy import to avoid slow startup
_SentenceTransformer: Any | None = None


def _get_sentence_transformer() -> Any:
    """Lazy load SentenceTransformer to speed up imports."""
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


# Available embedding models with their dimensions
EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "description": "Fast, lightweight model (22M params). Good for experimentation.",
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "description": "Higher quality embeddings (110M params). Best overall quality.",
    },
    "paraphrase-MiniLM-L3-v2": {
        "dimension": 384,
        "description": "Very fast, smallest model (17M params). Quick prototyping.",
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "dimension": 384,
        "description": "Optimized for semantic search and QA retrieval.",
    },
}

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Wrapper around sentence-transformers for generating embeddings.
    """

    def __init__(
        self, model_name: str = DEFAULT_MODEL, device: str | None = None
    ) -> None:
        """Initialize the embedder with a specific model.

        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_name = model_name
        self._model: Any | None = None
        self._device = device

    @property
    def model(self) -> Any:
        """Lazy load the model on first use."""
        if self._model is None:
            sentence_transformer = _get_sentence_transformer()
            self._model = sentence_transformer(
                self.model_name, device=self._device
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension for this model."""
        if self.model_name in EMBEDDING_MODELS:
            return cast(int, EMBEDDING_MODELS[self.model_name]["dimension"])
        # Fallback: get dimension from the model itself
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> NDArray[Any]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show a progress bar
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity)

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return cast(NDArray[Any], embeddings)

    def embed_query(self, query: str, normalize: bool = True) -> NDArray[Any]:
        """Generate embedding for a single query.

        Args:
            query: Query text to embed
            normalize: Whether to L2-normalize the embedding

        Returns:
            numpy array of shape (dimension,)
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return cast(NDArray[Any], embedding)


@st.cache_resource
def get_embedder(
    model_name: str = DEFAULT_MODEL, **kwargs: Any  # noqa: ANN401
) -> Embedder:
    """Factory function to get a cached embedder instance.

    Model is loaded once and reused across reruns.

    Args:
        model_name: Name of the embedding model
        **kwargs: Additional arguments passed to Embedder

    Returns:
        Configured Embedder instance
    """
    return Embedder(model_name=model_name, **kwargs)


def list_available_models() -> list[dict[str, Any]]:
    """List all available embedding models with their details.

    Returns:
        List of dictionaries with model info
    """
    return [
        {"name": name, **info}
        for name, info in EMBEDDING_MODELS.items()
    ]

