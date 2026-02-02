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


# Available embedding models with their metadata
EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    # Existing sentence-transformers models
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "description": "Fast, lightweight model (22M params). Good for experimentation.",
        "library": "sentence-transformers",
        "backend": "sentence-transformers",
        "size": "small",
        "use_case": "general",
        "max_seq_length": 512,
        "params_millions": 22,
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "description": "Higher quality embeddings (110M params). Best overall quality.",
        "library": "sentence-transformers",
        "backend": "sentence-transformers",
        "size": "base",
        "use_case": "general",
        "max_seq_length": 512,
        "params_millions": 110,
    },
    "paraphrase-MiniLM-L3-v2": {
        "dimension": 384,
        "description": "Very fast, smallest model (17M params). Quick prototyping.",
        "library": "sentence-transformers",
        "backend": "sentence-transformers",
        "size": "tiny",
        "use_case": "general",
        "max_seq_length": 512,
        "params_millions": 17,
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "dimension": 384,
        "description": "Optimized for semantic search and QA retrieval.",
        "library": "sentence-transformers",
        "backend": "sentence-transformers",
        "size": "small",
        "use_case": "semantic-search",
        "max_seq_length": 512,
        "params_millions": 22,
    },
    # NEW: Additional sentence-transformers models
    "all-MiniLM-L12-v2": {
        "dimension": 384,
        "description": "Middle ground between L6 and mpnet (33M params).",
        "library": "sentence-transformers",
        "backend": "sentence-transformers",
        "size": "small",
        "use_case": "general",
        "max_seq_length": 512,
        "params_millions": 33,
    },
    "paraphrase-multilingual-mpnet-base-v2": {
        "dimension": 768,
        "description": "Multilingual model supporting 50+ languages (110M params).",
        "library": "sentence-transformers",
        "backend": "sentence-transformers",
        "size": "base",
        "use_case": "multilingual",
        "max_seq_length": 512,
        "params_millions": 110,
    },
    "multi-qa-mpnet-base-cos-v1": {
        "dimension": 768,
        "description": "High-quality QA and semantic search (110M params).",
        "library": "sentence-transformers",
        "backend": "sentence-transformers",
        "size": "base",
        "use_case": "semantic-search",
        "max_seq_length": 512,
        "params_millions": 110,
    },
    # NEW: BGE models via FlagEmbedding
    "BAAI/bge-small-en-v1.5": {
        "dimension": 384,
        "description": "Fast BGE model, MTEB optimized (33M params).",
        "library": "BAAI",
        "backend": "flagembedding",
        "size": "small",
        "use_case": "general",
        "max_seq_length": 512,
        "params_millions": 33,
    },
    "BAAI/bge-base-en-v1.5": {
        "dimension": 768,
        "description": "Balanced BGE model, excellent MTEB scores (109M params).",
        "library": "BAAI",
        "backend": "flagembedding",
        "size": "base",
        "use_case": "general",
        "max_seq_length": 512,
        "params_millions": 109,
    },
    "BAAI/bge-large-en-v1.5": {
        "dimension": 1024,
        "description": "Highest quality BGE model, top MTEB leaderboard (335M params).",
        "library": "BAAI",
        "backend": "flagembedding",
        "size": "large",
        "use_case": "general",
        "max_seq_length": 512,
        "params_millions": 335,
    },
    "BAAI/bge-m3": {
        "dimension": 1024,
        "description": "Multilingual BGE model, 100+ languages (560M params).",
        "library": "BAAI",
        "backend": "flagembedding",
        "size": "large",
        "use_case": "multilingual",
        "max_seq_length": 8192,
        "params_millions": 560,
    },
    # NEW: Specialized models
    "mixedbread-ai/mxbai-embed-large-v1": {
        "dimension": 1024,
        "description": "State-of-the-art general-purpose embeddings (335M params).",
        "library": "Mixedbread AI",
        "backend": "sentence-transformers",
        "size": "large",
        "use_case": "general",
        "max_seq_length": 512,
        "params_millions": 335,
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "dimension": 768,
        "description": "Long context support (8192 tokens), excellent retrieval (137M params).",
        "library": "Nomic AI",
        "backend": "sentence-transformers",
        "size": "base",
        "use_case": "general",
        "max_seq_length": 8192,
        "params_millions": 137,
    },
    "jinaai/jina-embeddings-v2-base-en": {
        "dimension": 768,
        "description": "Long context (8192 tokens), strong semantic search (137M params).",
        "library": "Jina AI",
        "backend": "sentence-transformers",
        "size": "base",
        "use_case": "semantic-search",
        "max_seq_length": 8192,
        "params_millions": 137,
    },
    "Salesforce/SFR-Embedding-Mistral": {
        "dimension": 4096,
        "description": "Highest dimension embeddings, Mistral-based (7B params).",
        "library": "Salesforce",
        "backend": "sentence-transformers",
        "size": "xlarge",
        "use_case": "general",
        "max_seq_length": 4096,
        "params_millions": 7000,
    },
}

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """Embedding model wrapper with multi-backend support.

    Supports sentence-transformers and FlagEmbedding backends.
    Uses lazy loading - model is only loaded on first use.
    """

    def __init__(
        self, model_name: str = DEFAULT_MODEL, device: str | None = None
    ) -> None:
        """Initialize the embedder with a specific model.

        Args:
            model_name: Model identifier from EMBEDDING_MODELS registry
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_name = model_name
        self._model: Any | None = None
        self._device = device

        # Determine backend from registry
        if model_name in EMBEDDING_MODELS:
            self.backend_name = EMBEDDING_MODELS[model_name]["backend"]
        else:
            # Fallback to sentence-transformers for unknown models
            self.backend_name = "sentence-transformers"

    @property
    def model(self) -> Any:
        """Lazy load the model on first use."""
        if self._model is None:
            from rag_lens.services.embedding_backends import get_backend

            backend = get_backend(self.backend_name)
            self._model = backend.load_model(self.model_name, self._device)
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension for this model."""
        if self.model_name in EMBEDDING_MODELS:
            return cast(int, EMBEDDING_MODELS[self.model_name]["dimension"])

        # Fallback: query the loaded model
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            return self.model.get_sentence_embedding_dimension()
        elif hasattr(self.model, "dim"):
            return self.model.dim
        else:
            raise ValueError(
                f"Cannot determine dimension for model: {self.model_name}"
            )

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

        from rag_lens.services.embedding_backends import get_backend

        backend = get_backend(self.backend_name)
        embeddings = backend.embed_texts(
            self.model,
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=normalize,
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
        from rag_lens.services.embedding_backends import get_backend

        backend = get_backend(self.backend_name)
        embedding = backend.embed_query(
            self.model, query, normalize=normalize
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

