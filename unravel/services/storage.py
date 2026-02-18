"""Local file storage management for Unravel.

Manages the ~/.unravel/ directory for storing:
- Uploaded documents
- Processed chunks
- Cached embeddings
- Vector store indices
"""

import json
import shutil
from pathlib import Path
from typing import Any, cast

import streamlit as st

# Default storage location
DEFAULT_STORAGE_DIR = Path.home() / ".unravel"


def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and other security issues.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for use in file operations
    """
    # Remove any path components (e.g., ../, /, \)
    filename = Path(filename).name
    # Remove any null bytes
    filename = filename.replace("\x00", "")
    # Remove any control characters
    filename = "".join(c for c in filename if ord(c) >= 32 or c in "\n\r\t")
    # Ensure filename is not empty
    if not filename or filename.strip() == "":
        filename = "unnamed_file"
    return filename


def get_storage_dir() -> Path:
    """Get the storage directory path.

    Returns:
        Path to the storage directory (~/.unravel/)
    """
    return DEFAULT_STORAGE_DIR


def ensure_storage_dir() -> Path:
    """Ensure the storage directory and subdirectories exist.

    Creates the following structure:
        ~/.unravel/
        ├── .env           # API keys configuration
        ├── documents/     # Uploaded raw documents
        ├── chunks/        # Processed chunk data
        ├── embeddings/    # Cached embeddings
        └── indices/       # Vector indices (Qdrant storage)

    Returns:
        Path to the storage directory
    """
    storage_dir = get_storage_dir()

    # Create main directory and subdirectories
    subdirs = ["documents", "chunks", "embeddings", "indices"]
    for subdir in subdirs:
        (storage_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Create .env template if it doesn't exist
    env_file = storage_dir / ".env"
    if not env_file.exists():
        env_template = """# Unravel - API Keys Configuration
#
# Add your API keys below by uncommenting the relevant line and adding your key.

# OpenAI API Key (for GPT models)
# OPENAI_API_KEY=sk-...

# Anthropic API Key (for Claude models)
# ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini API Key (for Gemini models)
# GEMINI_API_KEY=...

# OpenRouter API Key (for OpenRouter models)
# OPENROUTER_API_KEY=sk-or-...

# Groq API Key (for Groq models - fast inference)
# GROQ_API_KEY=gsk_...

# For local models (Ollama, LM Studio, etc.), no API key is usually needed

# --- Cloud Deployment (Optional) ---
# Qdrant Cloud (for demo deployments without Docker)
# QDRANT_URL=https://your-cluster.qdrant.io
# QDRANT_API_KEY=your-api-key

# Demo Mode (auto-load sample document on startup)
# DEMO_MODE=true
"""
        env_file.write_text(env_template)

    return storage_dir


def get_documents_dir() -> Path:
    """Get the documents storage directory.

    Returns:
        Path to the documents directory
    """
    return get_storage_dir() / "documents"


def get_chunks_dir() -> Path:
    """Get the chunks storage directory.

    Returns:
        Path to the chunks directory
    """
    return get_storage_dir() / "chunks"


def get_embeddings_dir() -> Path:
    """Get the embeddings storage directory.

    Returns:
        Path to the embeddings directory
    """
    return get_storage_dir() / "embeddings"


def get_indices_dir() -> Path:
    """Get the indices storage directory.

    Returns:
        Path to the indices directory
    """
    return get_storage_dir() / "indices"


def save_document(filename: str, content: bytes) -> Path:
    """Save an uploaded document to storage, replacing any existing document.

    Only one document is stored at a time. Any existing documents are deleted
    before saving the new one.

    Args:
        filename: Original filename
        content: File content as bytes

    Returns:
        Path to the saved file
    """
    ensure_storage_dir()
    # Clear all existing documents (only keep one at a time)
    clear_documents()
    safe_filename = _sanitize_filename(filename)
    doc_path = get_documents_dir() / safe_filename
    doc_path.write_bytes(content)
    return doc_path


def clear_documents() -> None:
    """Clear all documents from storage."""
    docs_dir = get_documents_dir()
    if docs_dir.exists():
        for f in docs_dir.iterdir():
            if f.is_file():
                f.unlink()


def get_current_document() -> tuple[str, bytes] | None:
    """Get the current (only) stored document.

    Returns:
        Tuple of (filename, content) or None if no document exists
    """
    docs = list_documents()
    if not docs:
        return None
    filename = docs[0]
    content = load_document(filename)
    if content is None:
        return None
    return (filename, content)


def load_document(filename: str) -> bytes | None:
    """Load a document from storage.

    Args:
        filename: Document filename

    Returns:
        File content as bytes, or None if not found
    """
    safe_filename = _sanitize_filename(filename)
    doc_path = get_documents_dir() / safe_filename
    if doc_path.exists():
        return doc_path.read_bytes()
    return None


def delete_document(filename: str) -> bool:
    """Delete a document from storage.

    Args:
        filename: Document filename

    Returns:
        True if deleted, False if not found
    """
    safe_filename = _sanitize_filename(filename)
    doc_path = get_documents_dir() / safe_filename
    if doc_path.exists():
        doc_path.unlink()
        return True
    return False


@st.cache_data(ttl=2)
def list_documents() -> list[str]:
    """List all stored documents.

    Cached for 2 seconds to avoid repeated filesystem scans during reruns.

    Returns:
        List of document filenames
    """
    docs_dir = get_documents_dir()
    if not docs_dir.exists():
        return []
    return [f.name for f in docs_dir.iterdir() if f.is_file()]


def save_json(
    subdir: str, filename: str, data: dict[str, Any] | list[Any] | Any  # noqa: ANN401
) -> Path:
    """Save JSON data to a subdirectory.

    Args:
        subdir: Subdirectory name (chunks, embeddings, etc.)
        filename: JSON filename
        data: Data to serialize

    Returns:
        Path to the saved file
    """
    ensure_storage_dir()
    safe_filename = _sanitize_filename(filename)
    file_path = get_storage_dir() / subdir / safe_filename
    file_path.write_text(json.dumps(data, indent=2, default=str))
    return file_path


def load_json(
    subdir: str, filename: str
) -> dict[str, Any] | list[Any] | Any | None:  # noqa: ANN401
    """Load JSON data from a subdirectory.

    Args:
        subdir: Subdirectory name
        filename: JSON filename

    Returns:
        Deserialized data, or None if not found
    """
    safe_filename = _sanitize_filename(filename)
    file_path = get_storage_dir() / subdir / safe_filename
    if file_path.exists():
        return json.loads(file_path.read_text())
    return None


def clear_storage() -> None:
    """Clear all stored data.

    Warning: This deletes all documents, chunks, embeddings, and indices!
    """
    storage_dir = get_storage_dir()
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    ensure_storage_dir()


def get_storage_stats() -> dict[str, int]:
    """Get statistics about stored data.

    Returns:
        Dictionary with counts of documents, chunks, embeddings, and indices
    """
    ensure_storage_dir()

    def count_files(directory: Path) -> int:
        if not directory.exists():
            return 0
        return len([f for f in directory.iterdir() if f.is_file()])

    return {
        "documents": count_files(get_documents_dir()),
        "chunks": count_files(get_chunks_dir()),
        "embeddings": count_files(get_embeddings_dir()),
        "indices": count_files(get_indices_dir()),
    }


# --- Session State Persistence ---

SESSION_STATE_FILE = "session_state.json"
VECTOR_STORE_DIR = "current_vector_store"
BM25_INDEX_FILE = "bm25_index.pkl"


def save_bm25_index(index_data: dict[str, Any]) -> None:
    """Save BM25 index to disk.

    Args:
        index_data: Dictionary containing BM25 index and metadata
    """
    import pickle

    ensure_storage_dir()
    state_dir = get_storage_dir() / "session"
    state_dir.mkdir(parents=True, exist_ok=True)

    with (state_dir / BM25_INDEX_FILE).open("wb") as f:
        pickle.dump(index_data, f)


def load_bm25_index() -> dict[str, Any] | None:
    """Load BM25 index from disk.

    Returns:
        Dictionary containing BM25 index and metadata, or None if not found
    """
    import pickle

    state_dir = get_storage_dir() / "session"
    index_file = state_dir / BM25_INDEX_FILE

    if not index_file.exists():
        return None

    try:
        with index_file.open("rb") as f:
            return pickle.load(f)  # noqa: S301
    except (OSError, pickle.UnpicklingError):
        return None


def save_session_state(state_data: dict[str, Any]) -> None:
    """Save session state data to disk for persistence across refreshes.

    Args:
        state_data: Dictionary containing session state to persist
    """
    import os

    import numpy as np

    # Skip saving in demo mode
    if os.getenv("DEMO_MODE") == "true":
        return

    ensure_storage_dir()
    state_dir = get_storage_dir() / "session"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for JSON serialization
    serializable: dict[str, Any] = {}

    # Save basic fields
    if "doc_name" in state_data:
        serializable["doc_name"] = state_data["doc_name"]
    if "embedding_model_name" in state_data:
        serializable["embedding_model_name"] = state_data["embedding_model_name"]
    if "chunking_params" in state_data:
        serializable["chunking_params"] = state_data["chunking_params"]
    if "parsing_params" in state_data:
        serializable["parsing_params"] = state_data["parsing_params"]

    # Save chunks as JSON
    if "chunks" in state_data and state_data["chunks"]:
        chunks_data = []
        for chunk in state_data["chunks"]:
            chunks_data.append(
                {
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "start_index": getattr(chunk, "start_index", 0),
                    "end_index": getattr(chunk, "end_index", len(chunk.text)),
                }
            )
        serializable["chunks"] = chunks_data

    # Save embeddings result metadata
    if "last_embeddings_result" in state_data:
        emb_result = state_data["last_embeddings_result"]
        # NOTE: embedder is NOT saved - PyTorch models aren't serializable
        # The embedder will be recreated from the model name on-demand
        serializable["embeddings_key"] = emb_result.get("key", "")
        serializable["embeddings_model"] = emb_result.get("model", "")

        embeddings = emb_result.get("embeddings")
        if embeddings is not None:
            np.save(state_dir / "embeddings.npy", embeddings)
            serializable["has_embeddings"] = True

        # Save reduced embeddings from projections dict (new format) or legacy key (old format)
        # Try new projections format first
        projections = emb_result.get("projections", {})
        if projections:
            # Save the first available projection (typically 3D)
            for n_components, (reduced_emb, _) in projections.items():
                if reduced_emb is not None:
                    np.save(
                        state_dir / "reduced_embeddings.npy",
                        reduced_emb,
                    )
                    serializable["has_reduced_embeddings"] = True
                    serializable["reduced_embeddings_components"] = n_components
                    break
        else:
            # Fallback to legacy format for backwards compatibility
            reduced_emb = emb_result.get("reduced_embeddings")
            if reduced_emb is not None:
                np.save(
                    state_dir / "reduced_embeddings.npy",
                    reduced_emb,
                )
                serializable["has_reduced_embeddings"] = True
                serializable["reduced_embeddings_components"] = emb_result.get(
                    "reduced_embeddings_components"
                )

        # Save vector store
        if "vector_store" in emb_result and emb_result["vector_store"] is not None:
            vs_path = state_dir / VECTOR_STORE_DIR
            vector_store = emb_result["vector_store"]
            if vs_path.exists() and vector_store.storage_path.resolve() != vs_path.resolve():
                shutil.rmtree(vs_path)
            vector_store.save(vs_path)
            serializable["has_vector_store"] = True

    # Save retrieval config
    if "retrieval_config" in state_data:
        serializable["retrieval_config"] = state_data["retrieval_config"]

    # Save reranking config
    if "reranking_config" in state_data:
        serializable["reranking_config"] = state_data["reranking_config"]

    # Save BM25 index if exists
    if "bm25_index_data" in state_data:
        save_bm25_index(state_data["bm25_index_data"])
        serializable["has_bm25_index"] = True

    # Save query step settings
    query_settings = {}
    if "query_top_k" in state_data:
        query_settings["top_k"] = state_data["query_top_k"]
    if "query_threshold" in state_data:
        query_settings["threshold"] = state_data["query_threshold"]
    if "query_expansion_enabled" in state_data:
        query_settings["expansion_enabled"] = state_data["query_expansion_enabled"]
    if "query_expansion_count" in state_data:
        query_settings["expansion_count"] = state_data["query_expansion_count"]
    if "query_rewrite_prompt" in state_data:
        query_settings["rewrite_prompt"] = state_data["query_rewrite_prompt"]
    if "query_system_prompt" in state_data:
        query_settings["system_prompt"] = state_data["query_system_prompt"]
    if "api_endpoint_enabled" in state_data:
        query_settings["api_endpoint_enabled"] = state_data["api_endpoint_enabled"]

    if query_settings:
        serializable["query_settings"] = query_settings

    # Write JSON state
    (state_dir / SESSION_STATE_FILE).write_text(json.dumps(serializable, indent=2))


def load_session_state() -> dict[str, Any] | None:
    """Load persisted session state from disk.

    Returns:
        Dictionary with restored state data, or None if no saved state exists
    """
    import numpy as np

    from unravel.services.chunking import Chunk
    from unravel.services.vector_store import VectorStore

    state_dir = get_storage_dir() / "session"
    state_file = state_dir / SESSION_STATE_FILE

    if not state_file.exists():
        return None

    try:
        serializable = cast(dict[str, Any], json.loads(state_file.read_text()))
    except (OSError, json.JSONDecodeError):
        return None

    state_data: dict[str, Any] = {}

    # Restore basic fields
    if "doc_name" in serializable:
        state_data["doc_name"] = serializable["doc_name"]
    if "embedding_model_name" in serializable:
        state_data["embedding_model_name"] = serializable["embedding_model_name"]
    if "chunking_params" in serializable:
        state_data["chunking_params"] = serializable["chunking_params"]
    if "parsing_params" in serializable:
        state_data["parsing_params"] = serializable["parsing_params"]

    # Restore chunks
    if "chunks" in serializable:
        chunks: list[Chunk] = []
        for cd in serializable["chunks"]:
            chunks.append(
                Chunk(
                    text=cd["text"],
                    metadata=cd["metadata"],
                    start_index=cd.get("start_index", 0),
                    end_index=cd.get("end_index", len(cd["text"])),
                )
            )
        state_data["chunks"] = chunks

    # Restore embeddings result
    if serializable.get("embeddings_key"):
        emb_result = {
            "key": serializable["embeddings_key"],
            "model": serializable.get("embeddings_model", ""),
            # NOTE: embedder is NOT restored - will be recreated on-demand
        }

        # Load reduced embeddings
        embeddings_path = state_dir / "embeddings.npy"
        if serializable.get("has_embeddings") and embeddings_path.exists():
            emb_result["embeddings"] = np.load(embeddings_path)

        reduced_path = state_dir / "reduced_embeddings.npy"
        if serializable.get("has_reduced_embeddings") and reduced_path.exists():
            emb_result["reduced_embeddings"] = np.load(reduced_path)
            emb_result["reducer"] = None  # UMAP reducer can't be easily serialized
            emb_result["reduced_embeddings_components"] = serializable.get(
                "reduced_embeddings_components"
            )

        # Load vector store
        vs_path = state_dir / VECTOR_STORE_DIR
        if serializable.get("has_vector_store") and vs_path.exists():
            qdrant_url = st.session_state.get("qdrant_url")
            if not qdrant_url:
                # Docker not available, skip loading vector store
                # Embeddings will need regeneration when Docker becomes available
                pass
            else:
                try:
                    qdrant_api_key = st.session_state.get("qdrant_api_key")
                    emb_result["vector_store"] = VectorStore.load(
                        vs_path, url=qdrant_url, api_key=qdrant_api_key
                    )
                except Exception as exc:
                    # Log but don't crash - vector store can be regenerated
                    emb_result["vector_store_error"] = str(exc)

        # Include chunks reference
        if "chunks" in state_data:
            emb_result["chunks"] = state_data["chunks"]

        state_data["last_embeddings_result"] = emb_result

    # Restore retrieval config
    if "retrieval_config" in serializable:
        state_data["retrieval_config"] = serializable["retrieval_config"]

    # Restore reranking config
    if "reranking_config" in serializable:
        state_data["reranking_config"] = serializable["reranking_config"]

    # Restore BM25 index
    if serializable.get("has_bm25_index"):
        bm25_data = load_bm25_index()
        if bm25_data:
            state_data["bm25_index_data"] = bm25_data

    # Restore query step settings
    if "query_settings" in serializable:
        qs = serializable["query_settings"]
        if "top_k" in qs:
            state_data["query_top_k"] = qs["top_k"]
        if "threshold" in qs:
            state_data["query_threshold"] = qs["threshold"]
        if "expansion_enabled" in qs:
            state_data["query_expansion_enabled"] = qs["expansion_enabled"]
        if "expansion_count" in qs:
            state_data["query_expansion_count"] = qs["expansion_count"]
        if "rewrite_prompt" in qs:
            state_data["query_rewrite_prompt"] = qs["rewrite_prompt"]
        if "system_prompt" in qs:
            state_data["query_system_prompt"] = qs["system_prompt"]
        if "api_endpoint_enabled" in qs:
            state_data["api_endpoint_enabled"] = qs["api_endpoint_enabled"]

    return state_data


def clear_session_state() -> None:
    """Clear persisted session state."""
    import shutil

    state_dir = get_storage_dir() / "session"
    if state_dir.exists():
        shutil.rmtree(state_dir)


# --- LLM Configuration Persistence ---

LLM_CONFIG_FILE = "llm_config.json"


def save_llm_config(config_data: dict[str, Any]) -> None:
    """Save LLM configuration to disk for persistence across sessions.

    Args:
        config_data: Dictionary containing LLM config to persist.
                    Should include: provider, model, base_url (optional),
                    temperature, max_tokens, system_prompt (optional)

    Note:
        API keys are NEVER saved to this file. They must be set in the
        ~/.unravel/.env file using environment variable names like
        OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
    """
    import os

    # Skip saving in demo mode
    if os.getenv("DEMO_MODE") == "true":
        return

    ensure_storage_dir()
    config_dir = get_storage_dir() / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Remove API key from config if present (security measure)
    safe_config = config_data.copy()
    safe_config.pop("api_key", None)

    (config_dir / LLM_CONFIG_FILE).write_text(json.dumps(safe_config, indent=2))


def load_llm_config() -> dict[str, Any] | None:
    """Load persisted LLM configuration from disk.

    Returns:
        Dictionary with restored config data, or None if no saved config exists

    Note:
        API keys are never loaded from this file. They are always loaded from
        environment variables via the .env file.
    """
    config_dir = get_storage_dir() / "config"
    config_file = config_dir / LLM_CONFIG_FILE

    if not config_file.exists():
        return None

    try:
        config = cast(dict[str, Any], json.loads(config_file.read_text()))
        # Remove any API keys that might have been saved in older versions
        config.pop("api_key", None)
        config.pop("api_key_stored", None)
        return config
    except (OSError, json.JSONDecodeError):
        return None


def clear_llm_config() -> None:
    """Clear persisted LLM configuration."""
    config_dir = get_storage_dir() / "config"
    config_file = config_dir / LLM_CONFIG_FILE
    if config_file.exists():
        config_file.unlink()


# --- RAG Configuration Persistence ---

RAG_CONFIG_FILE = "rag_config.json"


def save_rag_config(config_data: dict[str, Any]) -> None:
    """Save RAG configuration to disk for persistence across refreshes.

    Args:
        config_data: Dictionary containing RAG config to persist.
                    Should include: doc_name, embedding_model_name,
                    chunking_params, parsing_params
    """
    import os

    # Skip saving in demo mode
    if os.getenv("DEMO_MODE") == "true":
        return

    ensure_storage_dir()
    config_dir = get_storage_dir() / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    (config_dir / RAG_CONFIG_FILE).write_text(json.dumps(config_data, indent=2))


def load_rag_config() -> dict[str, Any] | None:
    """Load persisted RAG configuration from disk.

    Returns:
        Dictionary with restored config data, or None if no saved config exists
    """
    config_dir = get_storage_dir() / "config"
    config_file = config_dir / RAG_CONFIG_FILE

    if not config_file.exists():
        return None

    try:
        return cast(dict[str, Any], json.loads(config_file.read_text()))
    except (OSError, json.JSONDecodeError):
        return None
