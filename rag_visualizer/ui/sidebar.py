from typing import cast

import streamlit as st
import streamlit_shadcn_ui as ui

from rag_visualizer.services.embedders import DEFAULT_MODEL, list_available_models
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

    if all_docs:
        if "sidebar_doc_selector" not in st.session_state:
            st.session_state["sidebar_doc_selector"] = st.session_state.doc_name or all_docs[0]
        elif st.session_state.get("sidebar_doc_selector") != st.session_state.doc_name:
            if st.session_state.doc_name in all_docs:
                st.session_state["sidebar_doc_selector"] = st.session_state.doc_name
    else:
        st.info("No documents available. Upload a file in the Upload step.")
        st.session_state.doc_name = None
        return  # Exit early before the form to avoid missing submit button error

    with st.form("rag_config_form"):
        st.write("")
        st.markdown("**Document**")
        form_doc_name = st.selectbox(
            "Document",
            options=all_docs,
            key="sidebar_doc_selector",
            label_visibility="collapsed"
        )

        st.write("")
        st.markdown("**Retrieval Strategy**")

        # Get current retrieval config or set defaults
        retrieval_config_raw = st.session_state.get("retrieval_config")
        if retrieval_config_raw is None or not isinstance(retrieval_config_raw, dict):
            current_retrieval_config = {
                "strategy": "DenseRetriever",
                "params": {}
            }
        else:
            current_retrieval_config = retrieval_config_raw
        
        # Ensure current_retrieval_config is always a dict (defensive check)
        if not isinstance(current_retrieval_config, dict):
            current_retrieval_config = {
                "strategy": "DenseRetriever",
                "params": {}
            }

        retrieval_strategies = ["Dense (FAISS)", "Sparse (BM25)", "Hybrid"]
        strategy_map = {
            "Dense (FAISS)": "DenseRetriever",
            "Sparse (BM25)": "SparseRetriever",
            "Hybrid": "HybridRetriever"
        }
        reverse_strategy_map = {v: k for k, v in strategy_map.items()}

        current_strategy_display = reverse_strategy_map.get(
            current_retrieval_config.get("strategy", "DenseRetriever"),
            "Dense (FAISS)"
        )

        if "sidebar_retrieval_strategy" not in st.session_state:
            st.session_state["sidebar_retrieval_strategy"] = current_strategy_display

        retrieval_strategy = st.selectbox(
            "Strategy",
            options=retrieval_strategies,
            key="sidebar_retrieval_strategy",
            help="Choose how to retrieve relevant chunks"
        )

        new_retrieval_params = {}

        if retrieval_strategy == "Hybrid":
            with st.expander("Hybrid Settings", expanded=False):
                dense_weight = st.slider(
                    "Dense weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_retrieval_config.get("params", {}).get("dense_weight", 0.7),
                    step=0.05,
                    help="Weight for vector similarity (1-weight goes to BM25)",
                    key="sidebar_dense_weight"
                )

                current_fusion = current_retrieval_config.get("params", {}).get("fusion_method", "weighted_sum")
                fusion_display_map = {"weighted_sum": "Weighted Sum", "rrf": "Reciprocal Rank Fusion"}
                reverse_fusion_map = {v: k for k, v in fusion_display_map.items()}

                fusion_method = st.radio(
                    "Fusion method",
                    options=["Weighted Sum", "Reciprocal Rank Fusion"],
                    index=0 if current_fusion == "weighted_sum" else 1,
                    key="sidebar_fusion_method"
                )

                new_retrieval_params = {
                    "dense_weight": dense_weight,
                    "fusion_method": reverse_fusion_map.get(fusion_method, "weighted_sum")
                }
        elif retrieval_strategy == "Sparse (BM25)":
            with st.expander("BM25 Settings", expanded=False):
                st.markdown("_Using rank-bm25 library with Okapi BM25_")

        new_retrieval_config = {
            "strategy": strategy_map.get(retrieval_strategy, "DenseRetriever"),
            "params": new_retrieval_params
        }

        st.write("")
        st.markdown("**Reranking (Optional)**")

        # Get current reranking config or set defaults
        reranking_config_raw = st.session_state.get("reranking_config")
        if reranking_config_raw is None or not isinstance(reranking_config_raw, dict):
            current_reranking_config = {"enabled": False}
        else:
            current_reranking_config = reranking_config_raw
        
        # Ensure current_reranking_config is always a dict (defensive check)
        if not isinstance(current_reranking_config, dict):
            current_reranking_config = {"enabled": False}

        enable_reranking = st.checkbox(
            "Enable reranking",
            value=current_reranking_config.get("enabled", False),
            key="sidebar_enable_reranking",
            help="Use cross-encoder to rerank results"
        )

        new_reranking_config = {"enabled": False}
        if enable_reranking:
            with st.expander("Reranking Settings", expanded=False):
                st.markdown("_Using FlashRank library_")

                rerank_models = [
                    "ms-marco-MiniLM-L-12-v2",
                    "ms-marco-TinyBERT-L-2-v2",
                ]

                rerank_model = st.selectbox(
                    "Model",
                    options=rerank_models,
                    index=rerank_models.index(current_reranking_config.get("model", rerank_models[0])) if current_reranking_config.get("model") in rerank_models else 0,
                    key="sidebar_rerank_model"
                )

                rerank_top_n = st.slider(
                    "Keep top N after reranking",
                    min_value=1,
                    max_value=20,
                    value=current_reranking_config.get("top_n", 5),
                    key="sidebar_rerank_top_n"
                )

            new_reranking_config = {
                "enabled": True,
                "model": rerank_model,
                "top_n": rerank_top_n
            }

        st.write("")
        st.markdown("**Embedding Model**")

        # Embedding model selection
        models = list_available_models()
        model_names = [m["name"] for m in models]

        current_model_name = st.session_state.embedding_model_name
        if current_model_name not in model_names:
            current_model_name = model_names[0]

        if "sidebar_embedding_model" not in st.session_state:
            st.session_state["sidebar_embedding_model"] = current_model_name
        elif st.session_state.get("sidebar_embedding_model") not in model_names:
            st.session_state["sidebar_embedding_model"] = current_model_name

        new_embedding_model = st.selectbox(
            "Embedding Model",
            options=model_names,
            key="sidebar_embedding_model"
        )

        apply_clicked = st.form_submit_button("Save & Apply")

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

    # API Key
    st.markdown("**API Key**")
    env_key = get_api_key_from_env(provider)
    if env_key:
        st.success("✓ API key from environment")
        api_key = env_key
        st.session_state.llm_api_key = ""  # Clear stored key if env var exists
    else:
        api_key = ui.input(
            default_value=st.session_state.llm_api_key,
            placeholder=f"Enter your {provider} API key",
            key="sidebar_api_key",
            type="password"
        )
        st.session_state.llm_api_key = api_key
    
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
            "api_key": api_key if not env_key else "",  # Don't save if from env
            "base_url": base_url if provider == "OpenAI-Compatible" else None,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        save_llm_config(config_data)
    
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

