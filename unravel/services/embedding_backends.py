"""Backend adapters for embedding models.

Provides a unified interface for different embedding libraries
(sentence-transformers, FlagEmbedding) through an adapter pattern.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class EmbedderBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def load_model(self, model_name: str, device: str | None = None) -> Any:
        """Load and return model instance.

        Args:
            model_name: Model identifier (HuggingFace model name)
            device: Device to use ("cpu", "cuda", None for auto)

        Returns:
            Loaded model instance
        """
        pass

    @abstractmethod
    def embed_texts(
        self, model: Any, texts: list[str], **kwargs: Any  # noqa: ANN401
    ) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts.

        Args:
            model: Loaded model instance
            texts: List of text strings to embed
            **kwargs: Additional backend-specific arguments

        Returns:
            Normalized embeddings array of shape (len(texts), dimension)
        """
        pass

    @abstractmethod
    def embed_query(
        self, model: Any, query: str, **kwargs: Any  # noqa: ANN401
    ) -> NDArray[np.float32]:
        """Generate embedding for a single query.

        Args:
            model: Loaded model instance
            query: Query text to embed
            **kwargs: Additional backend-specific arguments

        Returns:
            Normalized embedding array of shape (dimension,)
        """
        pass


class SentenceTransformersAdapter(EmbedderBackend):
    """Adapter for sentence-transformers library."""

    def load_model(self, model_name: str, device: str | None = None) -> Any:
        """Load model using sentence-transformers.

        Args:
            model_name: Model identifier
            device: Device to use

        Returns:
            SentenceTransformer model instance
        """
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name, device=device)

    def embed_texts(
        self, model: Any, texts: list[str], **kwargs: Any  # noqa: ANN401
    ) -> NDArray[np.float32]:
        """Generate embeddings using sentence-transformers.

        Args:
            model: SentenceTransformer instance
            texts: List of texts to embed
            **kwargs: Additional arguments (batch_size, show_progress, etc.)

        Returns:
            Normalized embeddings array
        """
        # Extract known kwargs
        batch_size = kwargs.get("batch_size", 32)
        show_progress = kwargs.get("show_progress", False)
        normalize = kwargs.get("normalize", True)

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(
        self, model: Any, query: str, **kwargs: Any  # noqa: ANN401
    ) -> NDArray[np.float32]:
        """Generate query embedding using sentence-transformers.

        Args:
            model: SentenceTransformer instance
            query: Query text
            **kwargs: Additional arguments

        Returns:
            Normalized query embedding
        """
        normalize = kwargs.get("normalize", True)

        embedding = model.encode(
            query, normalize_embeddings=normalize, convert_to_numpy=True
        )
        return embedding


class FlagEmbeddingAdapter(EmbedderBackend):
    """Adapter for FlagEmbedding library (BGE models)."""

    def load_model(self, model_name: str, device: str | None = None) -> Any:
        """Load model using FlagEmbedding.

        Args:
            model_name: Model identifier (e.g., "BAAI/bge-base-en-v1.5")
            device: Device to use (FlagModel uses CUDA if available by default)

        Returns:
            FlagModel instance
        """
        from FlagEmbedding import FlagModel

        # FlagModel uses fp16 on GPU for better performance
        use_fp16 = device == "cuda" if device else True
        return FlagModel(model_name, use_fp16=use_fp16)

    def embed_texts(
        self, model: Any, texts: list[str], **kwargs: Any  # noqa: ANN401
    ) -> NDArray[np.float32]:
        """Generate embeddings using FlagEmbedding.

        Args:
            model: FlagModel instance
            texts: List of texts to embed
            **kwargs: Additional arguments (batch_size, etc.)

        Returns:
            Normalized embeddings array
        """
        batch_size = kwargs.get("batch_size", 32)

        embeddings = model.encode(texts, batch_size=batch_size)

        # Normalize embeddings for consistency with sentence-transformers
        embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        return embeddings.astype(np.float32)

    def embed_query(
        self, model: Any, query: str, **kwargs: Any  # noqa: ANN401
    ) -> NDArray[np.float32]:
        """Generate query embedding using FlagEmbedding.

        Args:
            model: FlagModel instance
            query: Query text
            **kwargs: Additional arguments

        Returns:
            Normalized query embedding
        """
        # FlagModel has encode_queries method optimized for queries
        embedding = model.encode_queries([query])

        # Normalize
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding[0].astype(np.float32)


# Backend registry
BACKENDS: dict[str, EmbedderBackend] = {
    "sentence-transformers": SentenceTransformersAdapter(),
    "flagembedding": FlagEmbeddingAdapter(),
}


def get_backend(backend_name: str) -> EmbedderBackend:
    """Get backend adapter by name.

    Args:
        backend_name: Backend identifier ("sentence-transformers" or "flagembedding")

    Returns:
        Backend adapter instance

    Raises:
        ValueError: If backend name is unknown
    """
    if backend_name not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend_name}")
    return BACKENDS[backend_name]
