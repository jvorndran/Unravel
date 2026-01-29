"""Main Streamlit application for RAG Visualizer."""

import streamlit as st
import streamlit_shadcn_ui as ui

from rag_visualizer.services.embedders import DEFAULT_MODEL
from rag_visualizer.services.llm import DEFAULT_SYSTEM_PROMPT
from rag_visualizer.services.storage import (
    ensure_storage_dir,
    list_documents,
    load_llm_config,
    load_rag_config,
    load_session_state,
)
from rag_visualizer.ui.sidebar import render_sidebar
from rag_visualizer.ui.steps import (
    render_chunks_step,
    render_embeddings_step,
    render_export_step,
    render_query_step,
    render_upload_step,
)
from rag_visualizer.utils.ui import apply_custom_styles, render_step_nav

# Page configuration
st.set_page_config(
    page_title="RAG Play",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure storage directory exists
ensure_storage_dir()

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
            st.session_state.llm_system_prompt = saved_llm_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

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
        if "embedding_model_name" not in st.session_state and "embedding_model_name" in saved_rag_config:
            st.session_state.embedding_model_name = saved_rag_config["embedding_model_name"]
        if "doc_name" not in st.session_state and "doc_name" in saved_rag_config:
            st.session_state.doc_name = saved_rag_config["doc_name"]
        if "retrieval_config" not in st.session_state and "retrieval_config" in saved_rag_config:
            st.session_state.retrieval_config = saved_rag_config["retrieval_config"]
        if "reranking_config" not in st.session_state and "reranking_config" in saved_rag_config:
            st.session_state.reranking_config = saved_rag_config["reranking_config"]

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
        st.session_state.current_step = "chunks" # Default step

# Update query parameters to match session state (without reload in modern Streamlit)
st.query_params["step"] = st.session_state.current_step

# Render Main Title Section
st.markdown(
    """<div class="title-container">
<div class="main-title">RAG Playground</div>
<div class="main-subtitle">Explore each step of the RAG pipeline through interactive visualizations</div>
</div>""",
    unsafe_allow_html=True,
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
            st.markdown("### Welcome to RAG Playground")
            st.markdown("This interactive tool allows you to visualize and experiment with different stages of a Retrieval-Augmented Generation pipeline.")
            st.markdown("Choose a step above to get started or navigate to the **Upload** section to add your own documents.")


# Render main content as a fragment (tab switches only re-run this section)
render_main_content()
