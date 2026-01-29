from typing import cast

import streamlit as st
import streamlit_shadcn_ui as ui

from rag_visualizer.services.embedders import DEFAULT_MODEL, list_available_models
from rag_visualizer.ui.components.sidebar_config import render_sidebar_config
from rag_visualizer.services.llm import (
    LLM_PROVIDERS,
    LLMConfig,
    get_api_key_from_env,
    get_provider_models,
    validate_config,
)
from rag_visualizer.services.storage import (
    clear_session_state,
    list_documents,
    save_llm_config,
    save_rag_config,
)


def render_rag_config_sidebar() -> None:
    """Render RAG configuration in the sidebar."""

    # Initialize session state if not present
    docs = list_documents()
    
    if "doc_name" not in st.session_state:
        # If files exist, use first file; otherwise use None (will show empty state)
        if docs:
            st.session_state.doc_name = docs[0]
        else:
            st.session_state.doc_name = None
    
    # Auto-select first file if currently on "Sample Text" and files exist
    if st.session_state.doc_name == "Sample Text" and docs:
        st.session_state.doc_name = docs[0]
        # Invalidate chunks and embeddings when switching from sample text
        if "chunks" in st.session_state:
            del st.session_state["chunks"]
        if "last_embeddings_result" in st.session_state:
            del st.session_state["last_embeddings_result"]
        if "search_results" in st.session_state:
            del st.session_state["search_results"]
    
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

    # Initialize parsing_params if not present
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
            "max_characters": 40_000,
        }

    # Track applied parameters (what was actually used for processing)
    if "applied_parsing_params" not in st.session_state:
        st.session_state.applied_parsing_params = st.session_state.parsing_params.copy()
    if "applied_chunking_params" not in st.session_state:
        st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()

    # Document selection - only show "Sample Text" if no files exist
    if docs:
        all_docs = docs
    else:
        all_docs = ["Sample Text"] if st.session_state.doc_name == "Sample Text" else []
    
    # Ensure current selection is valid
    if st.session_state.doc_name and st.session_state.doc_name not in all_docs:
        if docs:
            st.session_state.doc_name = docs[0]
        else:
            st.session_state.doc_name = None

    if not all_docs:
        st.info("No documents available. Upload a file in the Upload step.")
        st.session_state.doc_name = None
        return

    # Get current retrieval config or set defaults
    retrieval_config_raw = st.session_state.get("retrieval_config")
    if retrieval_config_raw is None or not isinstance(retrieval_config_raw, dict):
        current_retrieval_config = {"strategy": "DenseRetriever", "params": {}}
    else:
        current_retrieval_config = retrieval_config_raw

    # Ensure current_retrieval_config is always a dict (defensive check)
    if not isinstance(current_retrieval_config, dict):
        current_retrieval_config = {"strategy": "DenseRetriever", "params": {}}

    # Get current reranking config or set defaults
    reranking_config_raw = st.session_state.get("reranking_config")
    if reranking_config_raw is None or not isinstance(reranking_config_raw, dict):
        current_reranking_config = {"enabled": False}
    else:
        current_reranking_config = reranking_config_raw

    # Ensure current_reranking_config is always a dict (defensive check)
    if not isinstance(current_reranking_config, dict):
        current_reranking_config = {"enabled": False}

    # Embedding model selection
    models = list_available_models()
    model_names = [m["name"] for m in models]

    current_model_name = st.session_state.embedding_model_name
    if model_names and current_model_name not in model_names:
        current_model_name = model_names[0]

    component_payload = render_sidebar_config(
        docs=all_docs,
        current_doc=st.session_state.doc_name,
        retrieval_config=current_retrieval_config.copy(),
        reranking_config=current_reranking_config.copy(),
        model_names=model_names,
        current_model=current_model_name,
    )

    apply_clicked = component_payload is not None
    form_doc_name = (
        component_payload.get("doc_name", st.session_state.doc_name)
        if component_payload
        else st.session_state.doc_name
    )
    new_embedding_model = (
        component_payload.get("embedding_model_name", current_model_name)
        if component_payload
        else current_model_name
    )
    new_retrieval_config = (
        component_payload.get("retrieval_config", current_retrieval_config)
        if component_payload
        else current_retrieval_config
    )
    new_reranking_config = (
        component_payload.get("reranking_config", current_reranking_config)
        if component_payload
        else current_reranking_config
    )

    if not isinstance(new_retrieval_config, dict):
        new_retrieval_config = {"strategy": "DenseRetriever", "params": {}}
    if not isinstance(new_reranking_config, dict):
        new_reranking_config = {"enabled": False}

    if apply_clicked:
        doc_changed = form_doc_name != st.session_state.doc_name
        model_changed = new_embedding_model != st.session_state.embedding_model_name
        retrieval_changed = new_retrieval_config != st.session_state.get("retrieval_config", {})

        st.session_state.doc_name = form_doc_name
        st.session_state.embedding_model_name = new_embedding_model
        st.session_state.retrieval_config = new_retrieval_config
        st.session_state.reranking_config = new_reranking_config

        # Invalidate caches based on what changed
        if doc_changed:
            # Document changed - invalidate everything
            for key in ["chunks", "last_embeddings_result", "bm25_index_data", "search_results"]:
                if key in st.session_state:
                    del st.session_state[key]

        if model_changed or retrieval_changed:
            # Model or retrieval changed - invalidate embeddings and search results
            if "last_embeddings_result" in st.session_state:
                del st.session_state["last_embeddings_result"]
            if "bm25_index_data" in st.session_state:
                del st.session_state["bm25_index_data"]
            if "search_results" in st.session_state:
                del st.session_state["search_results"]

        # Save configuration (use current parsing/chunking params from session state)
        current_rag_config = {
            "doc_name": st.session_state.doc_name,
            "embedding_model_name": st.session_state.embedding_model_name,
            "chunking_params": st.session_state.get("chunking_params", {}),
            "parsing_params": st.session_state.get("parsing_params", {}),
            "retrieval_config": st.session_state.get("retrieval_config", {}),
            "reranking_config": st.session_state.get("reranking_config", {}),
        }
        save_rag_config(current_rag_config)
        st.session_state["_last_saved_rag_config"] = current_rag_config.copy()

        st.success("Changes applied!")
        st.rerun()

    # Clear session state button
    st.write("")
    st.write("")
    st.divider()
    st.markdown("**Troubleshooting**")
    st.caption("Clear cached embeddings and session data if you encounter errors.")

    if ui.button("Clear Session State", variant="outline", key="clear_session_btn"):
        # Clear persisted session state files
        clear_session_state()

        # Clear in-memory session state (keep only essential keys)
        keys_to_keep = {
            "session_restored", "doc_name", "chunking_params", "embedding_model_name",
            "llm_provider", "llm_model", "llm_api_key", "llm_base_url",
            "llm_temperature", "llm_max_tokens",
            "current_step"  # Keep current step to avoid navigation issues
        }
        keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
        for key in keys_to_delete:
            del st.session_state[key]

        st.success("✓ Session state cleared! Refresh the page to regenerate embeddings.")


def render_llm_sidebar() -> None:
    """Render LLM configuration in the sidebar."""
    st.markdown("### LLM Configuration")
    
    # Initialize session state if not present
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "OpenAI"
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = ""
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""
    if "llm_base_url" not in st.session_state:
        st.session_state.llm_base_url = ""
    if "llm_temperature" not in st.session_state:
        st.session_state.llm_temperature = 0.7
    if "llm_max_tokens" not in st.session_state:
        st.session_state.llm_max_tokens = 1024
    
    # Provider selection
    providers = list(LLM_PROVIDERS.keys())
    current_provider = cast(str, st.session_state.llm_provider)
    if current_provider not in providers:
        current_provider = providers[0]

    # Pre-select based on state
    if "sidebar_provider" not in st.session_state:
        st.session_state["sidebar_provider"] = current_provider

    provider = cast(
        str,
        ui.select(
        options=providers,
        key="sidebar_provider",
        label="Provider"
        ),
    )
    st.session_state.llm_provider = provider
    
    st.write("")
    
    # Show explanation for OpenAI-Compatible
    if provider == "OpenAI-Compatible":
        with ui.card(key="openai_compat_info"):
            st.markdown(
                """
                **OpenAI-Compatible** allows local models (Ollama, LM Studio, etc).
                
                **Common Base URLs:**
                - **Ollama**: `http://localhost:11434/v1`
                - **LM Studio**: `http://localhost:1234/v1`
                """
            )
        st.write("")
    
    # Model selection
    models = get_provider_models(provider)
    if provider == "OpenAI-Compatible":
        st.markdown("**Model Name**")
        model = ui.input(
            default_value=st.session_state.llm_model or "llama2",
            placeholder="e.g., llama2, mistral",
            key="sidebar_model_input"
        )
        st.session_state.llm_model = model
    else:
        default_model = cast(str, LLM_PROVIDERS[provider]["default"])
        current_model = st.session_state.llm_model or default_model
        if current_model not in models:
            current_model = default_model
            st.session_state.llm_model = default_model
        
        # Pre-select based on state
        if "sidebar_model_select" not in st.session_state:
            st.session_state["sidebar_model_select"] = current_model

        model = ui.select(
            options=models,
            key="sidebar_model_select",
            label="Model"
        )
        st.session_state.llm_model = model
    
    st.write("")

    # API Key status
    st.markdown("**API Key**")
    env_key = get_api_key_from_env(provider)
    if env_key:
        st.success("✓ API key loaded")
        api_key = env_key
    else:
        if provider == "OpenAI-Compatible":
            st.info("No API key required for most local models")
            api_key = "not-needed"
        else:
            st.warning("⚠️ No API key found")
            st.caption(f"Set `{LLM_PROVIDERS[provider]['env_key']}` in `.env` file")
            api_key = ""

    st.session_state.llm_api_key = ""  # Never store API keys in session state

    st.write("")

    # Base URL (for OpenAI-Compatible)
    if provider == "OpenAI-Compatible":
        st.markdown("**Base URL**")
        base_url = ui.input(
            default_value=st.session_state.llm_base_url or "http://localhost:11434/v1",
            placeholder="http://localhost:11434/v1",
            key="sidebar_base_url"
        )
        st.session_state.llm_base_url = base_url
    else:
        base_url = None
        st.session_state.llm_base_url = ""
    
    # Advanced settings
    with st.expander("Advanced Settings", expanded=False):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.llm_temperature,
            step=0.1,
            key="sidebar_temperature",
            help="Higher values make output more random, lower values more deterministic."
        )
        st.session_state.llm_temperature = temperature
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=st.session_state.llm_max_tokens,
            step=100,
            key="sidebar_max_tokens",
            help="Maximum number of tokens in the response."
        )
        st.session_state.llm_max_tokens = max_tokens
        
    
    st.write("")

    # Save button
    if ui.button("Save Configuration", variant="primary", key="save_config_btn"):
        config_data = {
            "provider": provider,
            "model": model,
            "base_url": base_url if provider == "OpenAI-Compatible" else None,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        save_llm_config(config_data)
        st.success("✓ Configuration saved")
    
    # Validation status
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    is_valid, error_msg = validate_config(config)
    if not is_valid:
        st.warning(f"⚠️ {error_msg}")


def render_sidebar() -> None:
    """Render the main sidebar with tabs."""
    with st.sidebar:
        # Using ui.tabs in sidebar doesn't work well due to container width issues
        # Stick to st.tabs for top-level sidebar navigation for now, 
        # but usage of shadcn components inside tabs is fine.
        tab1, tab2 = st.tabs(["RAG Config", "LLM Config"])
        
        with tab1:
            render_rag_config_sidebar()
        
        with tab2:
            render_llm_sidebar()

