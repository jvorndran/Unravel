"""Main Streamlit application for Unravel."""

import os
from pathlib import Path

import streamlit as st
import streamlit_shadcn_ui as ui

from unravel.services.embedders import DEFAULT_MODEL
from unravel.services.llm import DEFAULT_SYSTEM_PROMPT
from unravel.services.storage import (
    ensure_storage_dir,
    list_documents,
    load_llm_config,
    load_rag_config,
    load_session_state,
    save_document,
)
from unravel.ui.sidebar import render_sidebar
from unravel.ui.steps import (
    render_chunks_step,
    render_embeddings_step,
    render_export_step,
    render_query_step,
    render_upload_step,
)
from unravel.utils.qdrant_manager import ensure_qdrant_server
from unravel.utils.ui import apply_custom_styles, render_step_nav

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def load_demo_document() -> None:
    """Load demo document if in demo mode and no document exists."""
    if os.getenv("DEMO_MODE") != "true":
        return

    # Check if a document already exists
    if list_documents():
        return

    # Load the demo document from docs/demo.md
    demo_path = Path(__file__).parent.parent / "docs" / "demo.md"
    if not demo_path.exists():
        st.warning("Demo mode enabled but demo document not found at docs/demo.md")
        return

    try:
        demo_content = demo_path.read_bytes()
        save_document("demo.md", demo_content)
        st.session_state.doc_name = "demo.md"
        st.session_state.document_metadata = {
            "name": "demo.md",
            "format": "MD",
            "size_bytes": len(demo_content),
            "path": str(demo_path),
            "source": "demo",
        }
    except Exception as e:
        st.error(f"Failed to load demo document: {e}")

# Page configuration
st.set_page_config(
    page_title="Unravel",
    page_icon="U",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure storage directory exists
ensure_storage_dir()

# Load demo document if in demo mode (before session state restoration)
load_demo_document()

# Ensure Qdrant server is available (Cloud or local Docker)
if "qdrant_url" not in st.session_state:
    try:
        with st.spinner("Starting Qdrant vector database..."):
            qdrant_url, qdrant_api_key = ensure_qdrant_server()
            st.session_state.qdrant_url = qdrant_url
            st.session_state.qdrant_api_key = qdrant_api_key
    except RuntimeError as e:
        st.session_state.qdrant_url = None
        st.session_state.qdrant_api_key = None
        st.session_state.qdrant_startup_status = "unavailable"

# --- Restore persisted session state on refresh ---
if "session_restored" not in st.session_state:
    st.session_state.session_restored = True
    saved_state = load_session_state()
    if saved_state:
        # Restore all saved state keys
        for key, value in saved_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

    # Migration: Remove old embedder objects from session state (fix for meta tensor error)
    if "last_embeddings_result" in st.session_state:
        emb_result = st.session_state["last_embeddings_result"]
        if isinstance(emb_result, dict) and "embedder" in emb_result:
            # Remove the embedder key - it will be recreated on-demand
            del emb_result["embedder"]

    # Load LLM config (API keys are never loaded here - they come from .env)
    saved_llm_config = load_llm_config()
    if saved_llm_config:
        if "llm_provider" not in st.session_state:
            st.session_state.llm_provider = saved_llm_config.get("provider", "OpenAI")
        if "llm_model" not in st.session_state:
            st.session_state.llm_model = saved_llm_config.get("model", "")
        if "llm_base_url" not in st.session_state:
            st.session_state.llm_base_url = saved_llm_config.get("base_url", "")
        if "llm_temperature" not in st.session_state:
            st.session_state.llm_temperature = saved_llm_config.get("temperature", 0.7)
        if "llm_max_tokens" not in st.session_state:
            st.session_state.llm_max_tokens = saved_llm_config.get("max_tokens", 1024)
        if "llm_system_prompt" not in st.session_state:
            st.session_state.llm_system_prompt = saved_llm_config.get(
                "system_prompt", DEFAULT_SYSTEM_PROMPT
            )

    # Initialize llm_api_key as empty (never stored, always loaded from .env)
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""

    # Load RAG config (sidebar options)
    saved_rag_config = load_rag_config()
    if saved_rag_config:
        if "chunking_params" not in st.session_state and "chunking_params" in saved_rag_config:
            st.session_state.chunking_params = saved_rag_config["chunking_params"]
        if "parsing_params" not in st.session_state and "parsing_params" in saved_rag_config:
            st.session_state.parsing_params = saved_rag_config["parsing_params"]
        if (
            "embedding_model_name" not in st.session_state
            and "embedding_model_name" in saved_rag_config
        ):
            st.session_state.embedding_model_name = saved_rag_config["embedding_model_name"]
        if "doc_name" not in st.session_state and "doc_name" in saved_rag_config:
            st.session_state.doc_name = saved_rag_config["doc_name"]

    # Initialize RAG config defaults if not present
    if "doc_name" not in st.session_state:
        # Check for uploaded files first
        docs = list_documents()
        if docs:
            st.session_state.doc_name = docs[0]
        else:
            st.session_state.doc_name = None
    # If doc_name is "Sample Text" and files exist, switch to first file
    elif st.session_state.doc_name == "Sample Text":
        docs = list_documents()
        if docs:
            st.session_state.doc_name = docs[0]

    if "chunking_params" not in st.session_state:
        st.session_state.chunking_params = {
            "provider": "Docling",
            "splitter": "HybridChunker",
            "max_tokens": 512,
            "chunk_overlap": 50,
            "tokenizer": "cl100k_base",
        }
    if "embedding_model_name" not in st.session_state:
        st.session_state.embedding_model_name = DEFAULT_MODEL

# Apply styles
apply_custom_styles()

# Render Sidebar
render_sidebar()

# Handle routing via session state (primary) and query parameters (initial load)
if "current_step" not in st.session_state:
    if "step" in st.query_params:
        st.session_state.current_step = st.query_params["step"]
    else:
        st.session_state.current_step = "chunks"  # Default step

# Update query parameters to match session state (without reload in modern Streamlit)
st.query_params["step"] = st.session_state.current_step

# Render Main Title Section
st.markdown(
    """<div class="title-container">
<div class="main-title">Unravel</div>
</div>""",
    unsafe_allow_html=True,
)

# Show demo mode banner if enabled
if os.getenv("DEMO_MODE") == "true":
    st.info(
        "**Demo Mode** – You're viewing a demo with a pre-loaded document. "
        "[Install locally](https://github.com/jvorndran/unravel) to use your own documents.",
        icon="ℹ️",
    )


@st.fragment
def render_main_content() -> None:
    """Render step navigation and content as a fragment.

    Using a fragment isolates reruns to this section only,
    preventing the sidebar from re-rendering on tab switches.
    """
    # Render Step Navigation
    render_step_nav(active_step=st.session_state.current_step)

    # Dispatcher for steps
    if st.session_state.current_step == "upload":
        render_upload_step()
    elif st.session_state.current_step == "chunks":
        render_chunks_step()
    elif st.session_state.current_step == "embeddings":
        render_embeddings_step()
    elif st.session_state.current_step == "query":
        render_query_step()
    elif st.session_state.current_step == "export":
        render_export_step()
    else:
        # Home card if no valid step
        with ui.card(key="home_card"):
            st.markdown("### Welcome to Unravel")
            st.markdown(
                "This interactive tool allows you to visualize and experiment with different stages of a Retrieval-Augmented Generation pipeline."
            )
            st.markdown(
                "Choose a step above to get started or navigate to the **Upload** section to add your own documents."
            )


# Render main content as a fragment (tab switches only re-run this section)
render_main_content()
