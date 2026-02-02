"""Services module for RAG Lens."""

from rag_lens.services.chunking import (
    Chunk,
    get_available_providers,
    get_chunks,
    get_provider_splitters,
)
from rag_lens.services.embedding_backends import (
    EmbedderBackend,
    get_backend,
)
from rag_lens.services.embedders import (
    DEFAULT_MODEL,
    EMBEDDING_MODELS,
    Embedder,
    get_embedder,
    list_available_models,
)
from rag_lens.services.llm import (
    DEFAULT_SYSTEM_PROMPT,
    LLM_PROVIDERS,
    LLMConfig,
    ModelWrapper,
    RAGContext,
    generate_response,
    generate_response_stream,
    get_api_key_from_env,
    get_model,
    get_provider_models,
    list_providers,
    validate_config,
)
from rag_lens.services.vector_store import (
    SearchResult,
    VectorStore,
    create_vector_store,
)

__all__ = [
    # Chunking
    "Chunk",
    "get_chunks",
    "get_available_providers",
    "get_provider_splitters",
    # Embedding Backends
    "EmbedderBackend",
    "get_backend",
    # Embeddings
    "Embedder",
    "get_embedder",
    "list_available_models",
    "EMBEDDING_MODELS",
    "DEFAULT_MODEL",
    # Vector Store
    "VectorStore",
    "SearchResult",
    "create_vector_store",
    # LLM
    "LLMConfig",
    "RAGContext",
    "LLM_PROVIDERS",
    "DEFAULT_SYSTEM_PROMPT",
    "generate_response",
    "generate_response_stream",
    "get_api_key_from_env",
    "list_providers",
    "get_provider_models",
    "validate_config",
    "get_model",
    "ModelWrapper",
]
