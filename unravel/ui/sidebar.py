from typing import cast

import streamlit as st
import streamlit_shadcn_ui as ui

from unravel.services.embedders import DEFAULT_MODEL, list_available_models
from unravel.ui.constants import WidgetKeys
from unravel.services.llm import (
    LLM_PROVIDERS,
    LLMConfig,
    get_api_key_from_env,
    get_provider_models,
    validate_config,
)
from unravel.services.storage import (
    clear_session_state,
    list_documents,
    load_llm_config,
    load_rag_config,
    save_llm_config,
    save_rag_config,
)


def render_rag_config_sidebar() -> None:
    """Render RAG configuration in the sidebar."""

    # Load saved RAG config on first run
    if "rag_config_loaded" not in st.session_state:
        saved_config = load_rag_config()
        if saved_config:
            st.session_state.doc_name = saved_config.get("doc_name")
            st.session_state.embedding_model_name = saved_config.get(
                "embedding_model_name", DEFAULT_MODEL
            )
            st.session_state.chunking_params = saved_config.get("chunking_params", {})
            st.session_state.parsing_params = saved_config.get("parsing_params", {})
            st.session_state.retrieval_config = saved_config.get(
                "retrieval_config", {"strategy": "DenseRetriever", "params": {}}
            )
            st.session_state.reranking_config = saved_config.get(
                "reranking_config", {"enabled": False}
            )
        st.session_state.rag_config_loaded = True

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
        }

    # Track applied parameters (what was actually used for processing)
    if "applied_parsing_params" not in st.session_state:
        st.session_state.applied_parsing_params = st.session_state.parsing_params.copy()
    if "applied_chunking_params" not in st.session_state:
        st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()

    # Document selection
    with st.container(border=True):
        st.markdown("**Document**")
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
            if WidgetKeys.SIDEBAR_DOC_SELECTOR not in st.session_state:
                st.session_state[WidgetKeys.SIDEBAR_DOC_SELECTOR] = st.session_state.doc_name or all_docs[0]
            elif st.session_state.get(WidgetKeys.SIDEBAR_DOC_SELECTOR) != st.session_state.doc_name:
                if st.session_state.doc_name in all_docs:
                    st.session_state[WidgetKeys.SIDEBAR_DOC_SELECTOR] = st.session_state.doc_name
            
            st.selectbox(
                "Document",
                options=all_docs,
                key=WidgetKeys.SIDEBAR_DOC_SELECTOR,
                label_visibility="collapsed",
            )
        else:
            st.info("No documents available. Upload a file in the Upload step.")
            st.session_state.doc_name = None
            return  # Exit early before the form to avoid missing submit button error

    # Retrieval Strategy
    with st.container(border=True):
        st.markdown("**Retrieval Strategy**")

        # Get current retrieval config or set defaults
        retrieval_config_raw = st.session_state.get("retrieval_config")
        if retrieval_config_raw is None or not isinstance(retrieval_config_raw, dict):
            current_retrieval_config = {"strategy": "DenseRetriever", "params": {}}
        else:
            current_retrieval_config = retrieval_config_raw

        # Ensure current_retrieval_config is always a dict (defensive check)
        if not isinstance(current_retrieval_config, dict):
            current_retrieval_config = {"strategy": "DenseRetriever", "params": {}}

        retrieval_strategies = ["Dense (Qdrant)", "Sparse (BM25)", "Hybrid"]
        strategy_map = {
            "Dense (Qdrant)": "DenseRetriever",
            "Sparse (BM25)": "SparseRetriever",
            "Hybrid": "HybridRetriever",
        }
        reverse_strategy_map = {v: k for k, v in strategy_map.items()}

        # Fusion method mappings (needed for pending config building)
        fusion_display_map = {
            "weighted_sum": "Weighted Sum",
            "rrf": "Reciprocal Rank Fusion",
        }
        reverse_fusion_map = {v: k for k, v in fusion_display_map.items()}

        current_strategy_display = reverse_strategy_map.get(
            current_retrieval_config.get("strategy", "DenseRetriever"), "Dense (Qdrant)"
        )

        if WidgetKeys.SIDEBAR_RETRIEVAL_STRATEGY not in st.session_state:
            st.session_state[WidgetKeys.SIDEBAR_RETRIEVAL_STRATEGY] = current_strategy_display

        retrieval_strategy = st.selectbox(
            "Strategy",
            options=retrieval_strategies,
            key=WidgetKeys.SIDEBAR_RETRIEVAL_STRATEGY,
            help="Choose how to retrieve relevant chunks",
            label_visibility="collapsed",
        )

        if retrieval_strategy == "Hybrid":
            st.markdown("---")
            st.caption("Hybrid Settings")
            st.slider(
                "Dense weight",
                min_value=0.0,
                max_value=1.0,
                value=current_retrieval_config.get("params", {}).get("dense_weight", 0.7),
                step=0.05,
                help="Weight for vector similarity (1-weight goes to BM25)",
                key=WidgetKeys.SIDEBAR_DENSE_WEIGHT,
            )

            current_fusion = current_retrieval_config.get("params", {}).get(
                "fusion_method", "weighted_sum"
            )

            st.radio(
                "Fusion method",
                options=["Weighted Sum", "Reciprocal Rank Fusion"],
                index=0 if current_fusion == "weighted_sum" else 1,
                key=WidgetKeys.SIDEBAR_FUSION_METHOD,
            )
        elif retrieval_strategy == "Sparse (BM25)":
            st.caption("_Using rank-bm25 library with Okapi BM25_")

    # Reranking
    with st.container(border=True):
        st.markdown("**Reranking**")

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
            key=WidgetKeys.SIDEBAR_ENABLE_RERANKING,
            help="Use cross-encoder to rerank results",
        )

        if enable_reranking:
            st.markdown("---")
            from unravel.services.retrieval.reranking import list_available_rerankers

            # Get all available models
            all_models = list_available_rerankers()

            # Group models by library for better UX
            libraries = {}
            for model in all_models:
                lib = model.get("library", "Other")
                if lib not in libraries:
                    libraries[lib] = []
                libraries[lib].append(model)

            # Show library selector
            st.caption("Model Library")
            library_names = list(libraries.keys())

            # Find current library
            current_model = current_reranking_config.get("model", "ms-marco-MiniLM-L-12-v2")
            current_library = next(
                (
                    lib
                    for lib, models in libraries.items()
                    if any(m["name"] == current_model for m in models)
                ),
                library_names[0],
            )

            selected_library = st.radio(
                "Library",
                options=library_names,
                index=library_names.index(current_library),
                horizontal=True,
                label_visibility="collapsed",
                key=WidgetKeys.SIDEBAR_RERANK_LIBRARY,
            )

            # Show models from selected library
            available_models = [m["name"] for m in libraries[selected_library]]
            current_model_in_lib = (
                current_model if current_model in available_models else available_models[0]
            )

            st.selectbox(
                "Model",
                options=available_models,
                index=available_models.index(current_model_in_lib),
                key=WidgetKeys.SIDEBAR_RERANK_MODEL,
                help="Select reranking model",
            )

            # Show model description
            selected_model_name = st.session_state.get(WidgetKeys.SIDEBAR_RERANK_MODEL)
            model_info = next((m for m in all_models if m["name"] == selected_model_name), None)
            if model_info:
                st.caption(f"_{model_info.get('description', '')}_")

            st.slider(
                "Keep top N after reranking",
                min_value=1,
                max_value=20,
                value=current_reranking_config.get("top_n", 5),
                key=WidgetKeys.SIDEBAR_RERANK_TOP_N,
            )

    # Embedding Model
    with st.container(border=True):
        st.markdown("**Embedding Model**")

        # Embedding model selection
        models = list_available_models()
        model_names = [m["name"] for m in models]

        # Create simple display options
        display_options = []
        option_to_model = {}

        for model in models:
            # Strip any prefix from model name for display
            display_name = model["name"]
            if "/" in display_name:
                display_name = display_name.split("/")[-1]

            display_option = f"{display_name}"
            display_options.append(display_option)
            option_to_model[display_option] = model["name"]

        # Get current model name
        current_model_name = st.session_state.embedding_model_name
        if current_model_name not in model_names:
            current_model_name = model_names[0]

        # Find current selection in display format
        current_display_name = current_model_name
        if "/" in current_display_name:
            current_display_name = current_display_name.split("/")[-1]
        current_selection = f"{current_display_name}"

        # Initialize widget state if not present
        if WidgetKeys.SIDEBAR_EMBEDDING_MODEL not in st.session_state:
            st.session_state[WidgetKeys.SIDEBAR_EMBEDDING_MODEL] = current_selection
        elif st.session_state.get(WidgetKeys.SIDEBAR_EMBEDDING_MODEL) not in display_options:
            st.session_state[WidgetKeys.SIDEBAR_EMBEDDING_MODEL] = current_selection

        # Simple selectbox
        selected_option = st.selectbox(
            "Embedding Model",
            options=display_options,
            key=WidgetKeys.SIDEBAR_EMBEDDING_MODEL,
            label_visibility="collapsed",
        )

        # Extract model name from selection
        if selected_option in option_to_model:
            selected_model_name = option_to_model[selected_option]
        else:
            selected_model_name = current_model_name

        # Display model details as caption
        selected_model_info = next((m for m in models if m["name"] == selected_model_name), None)
        if selected_model_info:
            provider = selected_model_info.get("library", "N/A")
            dimension = selected_model_info.get("dimension", "N/A")
            size = selected_model_info.get("size", "N/A").title()
            use_case = selected_model_info.get("use_case", "general").replace("-", " ").title()
            max_tokens = selected_model_info.get("max_seq_length", 512)
            params = selected_model_info.get("params_millions", 0)
            if params >= 1000:
                params_str = f"{params / 1000:.1f}B"
            else:
                params_str = f"{params}M"

            details = (
                f"**Provider:** {provider}  \n"
                f"**Dimension:** {dimension}d  \n"
                f"**Size:** {size}  \n"
                f"**Use Case:** {use_case}  \n"
                f"**Max Tokens:** {max_tokens}  \n"
                f"**Parameters:** {params_str}"
            )
            st.caption(details)

    # Build pending configuration from widget values
    pending_doc_name = st.session_state.get(WidgetKeys.SIDEBAR_DOC_SELECTOR)
    pending_embedding_model = selected_model_name

    # Build retrieval config from widgets
    retrieval_strategy = st.session_state.get(
        WidgetKeys.SIDEBAR_RETRIEVAL_STRATEGY,
        "Dense (Qdrant)",
    )
    pending_retrieval_params = {}

    if retrieval_strategy == "Hybrid":
        pending_retrieval_params = {
            "dense_weight": st.session_state.get(WidgetKeys.SIDEBAR_DENSE_WEIGHT, 0.7),
            "fusion_method": reverse_fusion_map.get(
                st.session_state.get(WidgetKeys.SIDEBAR_FUSION_METHOD, "Weighted Sum"),
                "weighted_sum",
            ),
        }

    pending_retrieval_config = {
        "strategy": strategy_map.get(retrieval_strategy, "DenseRetriever"),
        "params": pending_retrieval_params,
    }

    # Build reranking config from widgets
    enable_reranking = st.session_state.get(WidgetKeys.SIDEBAR_ENABLE_RERANKING, False)
    pending_reranking_config = {"enabled": False}

    if enable_reranking:
        pending_reranking_config = {
            "enabled": True,
            "model": st.session_state.get(WidgetKeys.SIDEBAR_RERANK_MODEL, "ms-marco-MiniLM-L-12-v2"),
            "top_n": st.session_state.get(WidgetKeys.SIDEBAR_RERANK_TOP_N, 5),
        }

    # Check for changes between pending and applied configuration
    default_retrieval = {"strategy": "DenseRetriever", "params": {}}
    default_reranking = {"enabled": False}
    has_changes = (
        pending_doc_name != st.session_state.doc_name
        or pending_embedding_model != st.session_state.embedding_model_name
        or pending_retrieval_config != st.session_state.get("retrieval_config", default_retrieval)
        or pending_reranking_config != st.session_state.get("reranking_config", default_reranking)
    )

    # Show status badge and save button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if has_changes:
        st.warning("Changes pending", icon="⚠️")
    else:
        st.success("Configuration applied", icon="✅")

    if st.button(
        "Save & Apply",
        type="primary",
        disabled=not has_changes,
        key=WidgetKeys.SIDEBAR_SAVE_RAG_CONFIG_BTN,
        use_container_width=True,
    ):
        # Detect specific changes for targeted cache invalidation
        doc_changed = pending_doc_name != st.session_state.doc_name
        model_changed = pending_embedding_model != st.session_state.embedding_model_name

        old_retrieval_config = st.session_state.get("retrieval_config", {})
        retrieval_strategy_changed = pending_retrieval_config.get(
            "strategy"
        ) != old_retrieval_config.get("strategy")

        # Update session state with pending configuration
        st.session_state.doc_name = pending_doc_name
        st.session_state.embedding_model_name = pending_embedding_model
        st.session_state.retrieval_config = pending_retrieval_config
        st.session_state.reranking_config = pending_reranking_config

        # Invalidate caches based on what changed
        if doc_changed:
            # Document changed - invalidate everything
            for key in [
                "chunks",
                "last_embeddings_result",
                "bm25_index_data",
                "search_results",
            ]:
                if key in st.session_state:
                    del st.session_state[key]

        if model_changed:
            # Embedding model changed - invalidate embeddings and downstream
            for key in ["last_embeddings_result", "bm25_index_data", "search_results"]:
                if key in st.session_state:
                    del st.session_state[key]

        if retrieval_strategy_changed:
            # Retrieval strategy changed (e.g., Dense -> Sparse -> Hybrid)
            # Only need to invalidate BM25 index if switching to/from strategies that use it
            old_strategy = old_retrieval_config.get("strategy", "DenseRetriever")
            new_strategy = pending_retrieval_config.get("strategy", "DenseRetriever")

            old_uses_bm25 = old_strategy in ["SparseRetriever", "HybridRetriever"]
            new_uses_bm25 = new_strategy in ["SparseRetriever", "HybridRetriever"]

            # Only invalidate BM25 index if BM25 usage changed
            if old_uses_bm25 != new_uses_bm25 and "bm25_index_data" in st.session_state:
                del st.session_state["bm25_index_data"]

            # Always invalidate search results when strategy changes
            if "search_results" in st.session_state:
                del st.session_state["search_results"]

        # Note: Changing fusion method or weights doesn't require cache invalidation
        # The query step will use new settings on next search

        # Save configuration to disk
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
   
    st.markdown("**Troubleshooting**")
    st.caption("Clear cached embeddings and session data if you encounter errors.")

    if ui.button("Clear Session State", variant="outline", key=WidgetKeys.SIDEBAR_CLEAR_SESSION_BTN):
        # Clear persisted session state files
        clear_session_state()

        # Clear in-memory session state (keep only essential keys)
        keys_to_keep = {
            "session_restored",
            "doc_name",
            "chunking_params",
            "embedding_model_name",
            "llm_provider",
            "llm_model",
            "llm_api_key",
            "llm_base_url",
            "llm_temperature",
            "current_step",  # Keep current step to avoid navigation issues
        }
        keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
        for key in keys_to_delete:
            del st.session_state[key]

        st.success("Session state cleared. Refresh the page to regenerate embeddings.")


def render_llm_sidebar() -> None:
    """Render LLM configuration in the sidebar."""

    # Load saved LLM config on first run
    if "llm_config_loaded" not in st.session_state:
        saved_config = load_llm_config()
        if saved_config:
            st.session_state.llm_provider = saved_config.get("provider", "OpenAI")
            st.session_state.llm_model = saved_config.get("model", "")
            st.session_state.llm_base_url = saved_config.get("base_url", "")
            st.session_state.llm_temperature = saved_config.get("temperature", 0.7)
        st.session_state.llm_config_loaded = True

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

    # Provider selection
    providers = list(LLM_PROVIDERS.keys())
    current_provider = cast(str, st.session_state.llm_provider)
    if current_provider not in providers:
        current_provider = providers[0]

    if WidgetKeys.SIDEBAR_PROVIDER not in st.session_state:
        st.session_state[WidgetKeys.SIDEBAR_PROVIDER] = current_provider

    # Provider & Model Settings
    with st.container(border=True):
        st.markdown("**Provider**")
        provider = cast(
            str,
            st.selectbox(
                "Provider",
                options=providers,
                key=WidgetKeys.SIDEBAR_PROVIDER,
                label_visibility="collapsed",
            ),
        )

        # Detect provider change and reset model to the new provider's default
        previous_provider = st.session_state.get("_last_llm_provider")
        if previous_provider and previous_provider != provider:
            default_model = cast(str, LLM_PROVIDERS[provider].get("default", ""))
            st.session_state.llm_model = default_model
            st.session_state.llm_base_url = ""
            # Clear model widget keys so they re-initialize with new options/defaults
            for key in (WidgetKeys.SIDEBAR_MODEL_SELECT, WidgetKeys.SIDEBAR_MODEL_INPUT):
                if key in st.session_state:
                    del st.session_state[key]
            if WidgetKeys.SIDEBAR_BASE_URL in st.session_state:
                del st.session_state[WidgetKeys.SIDEBAR_BASE_URL]
        st.session_state._last_llm_provider = provider
        st.session_state.llm_provider = provider

        # Show explanation for OpenAI-Compatible
        if provider == "OpenAI-Compatible":
            st.info(
                "**OpenAI-Compatible** allows local models (Ollama, LM Studio, etc).\n\n"
                "**Common Base URLs:**\n"
                "- Ollama: `http://localhost:11434/v1`\n"
                "- LM Studio: `http://localhost:1234/v1`"
            )

        if provider == "OpenRouter":
            st.info(
                "**OpenRouter** provides access to hundreds of models via a single API.\n\n"
                "Enter any model identifier from [openrouter.ai/models](https://openrouter.ai/models)."
            )

        if provider == "Vertex AI":
            st.info(
                "**Vertex AI** uses Google Cloud authentication.\n\n"
                "**Setup Steps:**\n"
                "1. Authenticate: `gcloud auth application-default login`\n"
                "2. Set `VERTEXAI_PROJECT=your-project-id` in `~/.unravel/.env`\n"
                "3. (Optional) Set `VERTEXAI_LOCATION=us-central1` (default: us-central1)"
            )

        # Model selection
        models = get_provider_models(provider)
        if provider == "OpenAI-Compatible":
            model = st.text_input(
                "Model Name",
                value=st.session_state.llm_model or "llama2",
                placeholder="e.g., llama2, mistral",
                key=WidgetKeys.SIDEBAR_MODEL_INPUT,
            )
            st.session_state.llm_model = model
        elif provider == "OpenRouter":
            model = st.text_input(
                "Model Name",
                value=st.session_state.llm_model or "anthropic/claude-opus-4-6",
                placeholder="e.g., anthropic/claude-opus-4-6",
                key=WidgetKeys.SIDEBAR_MODEL_INPUT,
            )
            st.session_state.llm_model = model
        else:
            default_model = cast(str, LLM_PROVIDERS[provider]["default"])
            current_model = st.session_state.llm_model or default_model
            if current_model not in models:
                current_model = default_model
                st.session_state.llm_model = default_model

            if WidgetKeys.SIDEBAR_MODEL_SELECT not in st.session_state:
                st.session_state[WidgetKeys.SIDEBAR_MODEL_SELECT] = current_model

            model = cast(
                str,
                st.selectbox(
                    "Model",
                    options=models,
                    key=WidgetKeys.SIDEBAR_MODEL_SELECT,
                    label_visibility="collapsed",
                ),
            )
            st.session_state.llm_model = model

        # API Key status
        api_key_from_env = get_api_key_from_env(provider)
        if api_key_from_env:
            if provider == "Vertex AI":
                st.caption(f"✅ Project ID loaded: {api_key_from_env[:20]}...")
            else:
                st.caption("✅ API key loaded")
            api_key = api_key_from_env
        else:
            if provider == "OpenAI-Compatible":
                api_key = "not-needed"
            elif provider == "Vertex AI":
                st.caption("⚠️ Set `VERTEXAI_PROJECT` in `~/.unravel/.env`")
                st.caption("💡 Run: `gcloud auth application-default login`")
                api_key = ""
            else:
                st.caption(f"⚠️ Set `{LLM_PROVIDERS[provider]['env_key']}` in `~/.unravel/.env`")
                api_key = ""

        st.session_state.llm_api_key = ""  # Never store API keys in session state

        # Base URL (for OpenAI-Compatible only; OpenRouter uses a hardcoded URL)
        if provider == "OpenAI-Compatible":
            default_base_url = "http://localhost:11434/v1"
            base_url = st.text_input(
                "Base URL",
                value=st.session_state.llm_base_url or default_base_url,
                placeholder=default_base_url,
                key=WidgetKeys.SIDEBAR_BASE_URL,
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
            key=WidgetKeys.SIDEBAR_TEMPERATURE,
            help="Higher values make output more random, lower values more deterministic.",
        )
        st.session_state.llm_temperature = temperature

    # Save button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(
        "Save Configuration",
        key=WidgetKeys.SIDEBAR_SAVE_CONFIG_BTN,
        use_container_width=True,
    ):
        config_data = {
            "provider": provider,
            "model": model,
            "base_url": base_url if provider == "OpenAI-Compatible" else None,
            "temperature": temperature,
        }
        save_llm_config(config_data)
        st.success("Configuration saved")

    # Validation status
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
    is_valid, error_msg = validate_config(config)
    if not is_valid:
        st.warning(error_msg, icon="⚠️")


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
