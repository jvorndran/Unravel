"""Caching utilities for parsed document text.

Provides functions to cache and retrieve parsed document text to avoid
re-parsing documents with the same parameters.
"""

import hashlib
import json
from typing import Any

from rag_lens.services.storage import get_storage_dir


def get_parsed_text_key(doc_name: str, parsing_params: dict[str, Any]) -> str:
    """Get a stable key for parsed text storage.
    
    Args:
        doc_name: Name of the document
        parsing_params: Dictionary of parsing parameters
        
    Returns:
        Stable cache key string
    """
    # Use JSON serialization with sorted keys for stability
    params_str = json.dumps(parsing_params, sort_keys=True)
    return f"parsed_text_{doc_name}_{params_str}"


def get_parsed_text_filename(doc_name: str, parsing_params: dict[str, Any]) -> str:
    """Get a safe filename for parsed text storage.
    
    Args:
        doc_name: Name of the document
        parsing_params: Dictionary of parsing parameters
        
    Returns:
        Safe filename for filesystem storage
    """
    key = get_parsed_text_key(doc_name, parsing_params)
    # Use hash for filename to avoid filesystem issues with special chars
    key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
    # Basic sanitization: remove path separators and use hash
    safe_doc_name = doc_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"{safe_doc_name}_{key_hash}.txt"


def save_parsed_text(
    doc_name: str, parsing_params: dict[str, Any], parsed_text: str
) -> None:
    """Save parsed text to persistent storage.
    
    Args:
        doc_name: Name of the document
        parsing_params: Dictionary of parsing parameters used
        parsed_text: The parsed text content to cache
        
    Raises:
        OSError: If the file cannot be written (e.g., permission denied, disk full)
    """
    storage_dir = get_storage_dir()
    parsed_dir = storage_dir / "parsed_texts"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    
    filename = get_parsed_text_filename(doc_name, parsing_params)
    file_path = parsed_dir / filename
    
    try:
        file_path.write_text(parsed_text, encoding="utf-8")
    except (OSError, PermissionError) as e:
        # Re-raise with more context
        raise OSError(f"Failed to save cached parsed text for {doc_name}: {e}") from e


def load_parsed_text(
    doc_name: str, parsing_params: dict[str, Any]
) -> str | None:
    """Load parsed text from persistent storage.
    
    Args:
        doc_name: Name of the document
        parsing_params: Dictionary of parsing parameters
        
    Returns:
        Cached parsed text if found, None otherwise
    """
    storage_dir = get_storage_dir()
    parsed_dir = storage_dir / "parsed_texts"
    
    filename = get_parsed_text_filename(doc_name, parsing_params)
    file_path = parsed_dir / filename
    
    if file_path.exists():
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception:
            return None
    return None

