"""Shared fixtures and helpers for sidebar tests.

These fixtures provide common setup for testing Streamlit sidebar components
using streamlit.testing.v1.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest


def element_exists(at: AppTest, element_type: str, key: str) -> bool:
    """Check if an element with the given key exists in the app.

    Streamlit testing API raises KeyError when element doesn't exist,
    so we use try/except to check for presence.

    Args:
        at: AppTest instance
        element_type: Element type string (e.g., "selectbox", "checkbox", "text_input")
        key: The widget key to look for

    Returns:
        True if element exists, False otherwise
    """
    try:
        element_getter = getattr(at, element_type)
        element_getter(key=key)
        return True
    except KeyError:
        return False


def get_form_submit_button(at: AppTest, label: str = "Save & Apply") -> Any | None:
    """Find and return the form submit button by label.

    Args:
        at: AppTest instance
        label: The button label to search for

    Returns:
        The button element if found, None otherwise
    """
    for btn in at.button:
        if btn.label == label:
            return btn
    return None


@pytest.fixture
def mock_storage_dir(tmp_path: Path) -> Path:
    """Create a temporary storage directory with test documents.

    Creates:
        - documents/document_a.pdf
        - documents/document_b.pdf
        - documents/document_c.txt
        - config/ (for save_rag_config)
    """
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True)

    # Create test documents
    (docs_dir / "document_a.pdf").write_bytes(b"test content a")
    (docs_dir / "document_b.pdf").write_bytes(b"test content b")
    (docs_dir / "document_c.txt").write_text("test content c")

    # Create config dir (needed for save_rag_config)
    (tmp_path / "config").mkdir(parents=True)

    return tmp_path


@pytest.fixture
def sidebar_app_script() -> str:
    """Return the app script string for testing the RAG config sidebar.

    This fixture provides a minimal app script that renders the RAG config
    sidebar. Session state initialization mirrors production behavior.

    Use with AppTest.from_string(sidebar_app_script).
    """
    return '''
import streamlit as st
from unravel.ui.sidebar import render_rag_config_sidebar

# Initialize required session state (mirrors production initialization)
if "chunking_params" not in st.session_state:
    st.session_state.chunking_params = {
        "provider": "Docling",
        "splitter": "HybridChunker",
        "max_tokens": 512,
        "chunk_overlap": 50,
        "tokenizer": "cl100k_base",
    }
if "embedding_model_name" not in st.session_state:
    st.session_state.embedding_model_name = "all-MiniLM-L6-v2"
if "parsing_params" not in st.session_state:
    st.session_state.parsing_params = {
        # Docling options
        "docling_enable_ocr": False,
        "docling_table_structure": True,
        "docling_threads": 4,
        "docling_filter_labels": ["PAGE_HEADER", "PAGE_FOOTER"],
        "docling_extract_images": False,
        "docling_enable_captioning": False,
        "docling_device": "auto",
        # Output options
        "output_format": "markdown",
        "normalize_whitespace": True,
        "remove_special_chars": False,
    }
if "applied_parsing_params" not in st.session_state:
    st.session_state.applied_parsing_params = st.session_state.parsing_params.copy()
if "applied_chunking_params" not in st.session_state:
    st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()

with st.sidebar:
    render_rag_config_sidebar()
'''


@pytest.fixture
def chunking_config_app_script() -> str:
    """Return the app script string for testing the chunking config UI."""
    return '''
import streamlit as st
from unravel.ui.components.chunking_config import render_chunking_configuration

# Initialize required session state (mirrors production initialization)
if "chunking_params" not in st.session_state:
    st.session_state.chunking_params = {
        "provider": "Docling",
        "splitter": "HybridChunker",
        "max_tokens": 512,
        "chunk_overlap": 50,
        "tokenizer": "cl100k_base",
    }
if "parsing_params" not in st.session_state:
    st.session_state.parsing_params = {
        # Docling options
        "docling_enable_ocr": False,
        "docling_table_structure": True,
        "docling_threads": 4,
        "docling_filter_labels": ["PAGE_HEADER", "PAGE_FOOTER"],
        "docling_extract_images": False,
        "docling_enable_captioning": False,
        "docling_device": "auto",
        # Output options
        "output_format": "markdown",
        "normalize_whitespace": True,
        "remove_special_chars": False,
    }
if "applied_parsing_params" not in st.session_state:
    st.session_state.applied_parsing_params = st.session_state.parsing_params.copy()
if "applied_chunking_params" not in st.session_state:
    st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()

render_chunking_configuration()
'''


@pytest.fixture
def patched_storage(mock_storage_dir: Path) -> Any:
    """Context manager that patches the storage directory.

    Usage:
        with patched_storage:
            at = AppTest.from_string(sidebar_app_script).run()
    """
    return patch(
        "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
    )
