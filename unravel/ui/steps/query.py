"""Query tester page for Unravel.

Allows users to test the full RAG pipeline with retrieval and LLM response generation.
"""

import platform
import subprocess
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit_shadcn_ui as ui

from unravel.services.embedders import DEFAULT_MODEL, get_embedder
from unravel.ui.constants import WidgetKeys, get_threshold_slider_key
from unravel.services.llm import (
    DEFAULT_QUERY_REWRITE_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    LLMConfig,
    RAGContext,
    get_model,
    strip_think_blocks,
)
from unravel.services.storage import get_storage_dir, save_session_state
from unravel.ui.components.api_endpoint import render_api_endpoint_expander
from unravel.ui.components.chunk_viewer import (
    prepare_chunk_display_data,
    render_chunk_cards,
)

def _render_model_response(text: str) -> None:
    answer = strip_think_blocks(text)
    if answer:
        st.markdown(answer)
    else:
        st.caption("No answer content returned.")


def _open_folder_in_explorer(folder_path: Path) -> None:
    """Open a folder in the system file explorer.

    Args:
        folder_path: Path to the folder to open
    """
    try:
        system = platform.system()
        if system == "Windows":
            # Use os.startfile for more reliable Windows folder opening
            import os

            os.startfile(str(folder_path))
        elif system == "Darwin":  # macOS
            subprocess.run(["open", str(folder_path)], check=False)
        else:  # Linux
            subprocess.run(["xdg-open", str(folder_path)], check=False)
    except Exception:
        pass  # Silently fail if we can't open the folder


def _ensure_env_file_exists() -> Path:
    """Ensure the .env file exists in the storage directory with helpful template.

    Returns:
        Path to the .env file
    """
    env_path = get_storage_dir() / ".env"

    if not env_path.exists():
        template = """# Unravel API Keys
# Set your API keys below (uncomment and add your keys)

# OpenAI API Key
# OPENAI_API_KEY=sk-your-key-here

# Anthropic API Key
# ANTHROPIC_API_KEY=sk-ant-your-key-here

# Google AI Studio (for Gemini provider)
# GEMINI_API_KEY=your-key-here

# Google Cloud Vertex AI (for Vertex AI provider)
# VERTEXAI_PROJECT=your-project-id
# VERTEXAI_LOCATION=us-central1

# Groq API Key
# GROQ_API_KEY=your-key-here

# OpenRouter API Key
# OPENROUTER_API_KEY=your-key-here

# For local models (Ollama, LM Studio), no API key is usually needed
"""
        get_storage_dir().mkdir(parents=True, exist_ok=True)
        env_path.write_text(template)

    return env_path


def _render_api_key_setup_message(provider: str, env_key_name: str) -> None:
    """Render a helpful message when API key is not configured.

    Args:
        provider: LLM provider name
        env_key_name: Environment variable name for the API key
    """
    with st.container(border=True):
        st.markdown("### API Key Required")
        st.markdown(
            f"To use **{provider}** for answer generation, you need to configure your API key."
        )

       
        st.markdown("**Setup Steps:**")
        st.markdown(f"""
            1. Click the button below to open the configuration folder
            2. Edit the `.env` file in a text editor
            3. Add your API key: `{env_key_name}=your-key-here`
            4. Save the file and refresh this page
            """)

       
        col1, col2 = st.columns([1, 2])
        with col1:
            if ui.button("Open Config Folder", variant="secondary", key=WidgetKeys.QUERY_OPEN_CONFIG_FOLDER_BTN):
                _ensure_env_file_exists()
                _open_folder_in_explorer(get_storage_dir())
                st.success("✓ Folder opened! Edit the .env file and refresh.")

        with col2:
            st.caption(
                f"The folder contains a `.env` file where you can securely store your {provider} API key."
            )


def _get_embeddings_data() -> dict[str, Any] | None:
    """Retrieve embeddings data from session state."""
    if "last_embeddings_result" not in st.session_state:
        return None
    return st.session_state["last_embeddings_result"]


def _render_empty_state() -> None:
    """Render the empty state when no embeddings are available."""
    with st.container(border=True):
        st.markdown("### No embeddings found")
        st.caption(
            "Please go to the Embeddings step to generate vector representations of your document chunks first."
        )

       
        if ui.button("Go to Embeddings Step", key=WidgetKeys.QUERY_GOTO_EMBEDDINGS):
            st.session_state.current_step = "embeddings"
            st.rerun(scope="app")


def _render_retrieved_chunks(results: list[Any], show_scores: bool = True) -> None:
    """Render retrieved chunks using the chunk viewer component."""
    if not results:
        return

    st.markdown(f"#### Retrieved Context ({len(results)} chunks)")

    # Convert search results to chunks for the viewer
    from dataclasses import dataclass

    @dataclass
    class ChunkAdapter:
        """Adapter to make SearchResult compatible with chunk viewer."""

        text: str
        metadata: dict[str, Any]
        start_index: int = 0
        end_index: int = 0

    retrieved_chunks = [
        ChunkAdapter(
            text=res.text,
            metadata=res.metadata,
            start_index=i,
            end_index=i,
        )
        for i, res in enumerate(results)
    ]

    # Prepare display data
    retrieved_display_data = prepare_chunk_display_data(
        chunks=retrieved_chunks,
        source_text=None,
        calculate_overlap=False,
    )

    custom_badges: list[list[dict[str, Any]]] | None = None
    if show_scores:
        custom_badges = []
        for res in results:
            base_score = res.metadata.get("original_score", res.score)
            badges: list[dict[str, Any]] = [
                {
                    "label": "Score",
                    "value": f"{base_score:.4f}",
                    "color": "#d1fae5",
                }
            ]

            dense_rrf_contribution = res.metadata.get("dense_rrf_contribution")
            if dense_rrf_contribution is not None:
                badges.append(
                    {
                        "label": "Dense Contribution",
                        "value": f"{dense_rrf_contribution:.4f}",
                        "color": "#93c5fd",
                    }
                )

            sparse_rrf_contribution = res.metadata.get("sparse_rrf_contribution")
            if sparse_rrf_contribution is not None:
                badges.append(
                    {
                        "label": "Sparse Contribution",
                        "value": f"{sparse_rrf_contribution:.4f}",
                        "color": "#fecdd3",
                    }
                )

            custom_badges.append(badges)

    # Render using the reusable component in card mode
    render_chunk_cards(
        chunk_display_data=retrieved_display_data,
        custom_badges=custom_badges,
        show_overlap=False,
        display_mode="card",
    )


def _render_query_variations(variations: list[str]) -> None:
    if not variations:
        return

    with st.container(border=True):
        st.markdown(f"#### Query Variations ({len(variations)})")
        for variation in variations:
            st.markdown(f"- {variation}")
        st.caption("Retrieved chunks are the union of results from all variations.")


def _get_llm_config_from_sidebar() -> tuple[LLMConfig, str]:
    """Get LLM configuration from sidebar session state."""
    provider = st.session_state.get("llm_provider", "OpenAI")
    model = st.session_state.get("llm_model", "")
    api_key = st.session_state.get("llm_api_key", "")
    base_url = st.session_state.get("llm_base_url", "")
    temperature = st.session_state.get("llm_temperature", 0.7)
    system_prompt = st.session_state.get("llm_system_prompt", DEFAULT_SYSTEM_PROMPT)

    # Check for env var API key
    from unravel.services.llm import get_api_key_from_env

    env_key = get_api_key_from_env(provider)
    if env_key:
        api_key = env_key

    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url if base_url else None,
        temperature=temperature,
    )

    return config, system_prompt


def _save_query_settings() -> None:
    """Save query step settings to session state for persistence."""
    try:
        save_session_state(
            {
                "doc_name": st.session_state.get("doc_name"),
                "embedding_model_name": st.session_state.get("embedding_model_name"),
                "chunking_params": st.session_state.get("chunking_params"),
                "chunks": st.session_state.get("chunks"),
                "last_embeddings_result": st.session_state.get("last_embeddings_result"),
                "retrieval_config": st.session_state.get("retrieval_config"),
                "reranking_config": st.session_state.get("reranking_config"),
                "bm25_index_data": st.session_state.get("bm25_index_data"),
                # Query step settings (use the actual stored values, not widget keys)
                "query_top_k": st.session_state.get("query_top_k"),
                "query_threshold": st.session_state.get("query_threshold"),
                "query_expansion_enabled": st.session_state.get(WidgetKeys.QUERY_EXPANSION_ENABLED, False),
                "query_expansion_count": st.session_state.get(WidgetKeys.QUERY_EXPANSION_COUNT, 4),
                "query_rewrite_prompt": st.session_state.get(WidgetKeys.QUERY_REWRITE_PROMPT, ""),
                "query_system_prompt": st.session_state.get(WidgetKeys.QUERY_SYSTEM_PROMPT, ""),
                "api_endpoint_enabled": st.session_state.get("api_endpoint_enabled", False),
            }
        )
    except Exception:
        pass  # Silent fail - not critical


def _merge_search_results(results_by_query: list[list[Any]]) -> list[Any]:
    """Merge results from multiple queries using Reciprocal Rank Fusion.

    RRF is used instead of max score because similarity scores from different
    query embeddings are not comparable - they exist in different vector spaces.
    Rank-based fusion provides more reliable multi-query merging.
    """
    if not results_by_query:
        return []

    if len(results_by_query) == 1:
        return results_by_query[0]

    rrf_k = 60  # Standard RRF constant
    rrf_scores: dict[int, float] = {}

    # Calculate RRF scores by summing 1/(k + rank) for each query
    for results in results_by_query:
        for rank, result in enumerate(results, 1):
            current_score = rrf_scores.get(result.index, 0.0)
            rrf_scores[result.index] = current_score + (1.0 / (rrf_k + rank))

    # Build merged results with RRF scores
    result_map = {r.index: r for results in results_by_query for r in results}
    merged = []

    for idx, score in rrf_scores.items():
        original = result_map[idx]
        merged.append(
            type(original)(
                index=idx,
                score=score,
                text=original.text,
                metadata={
                    **original.metadata,
                    "multi_query_rrf_score": score,
                    "original_score": original.score,
                },
            )
        )

    return sorted(merged, key=lambda r: r.score, reverse=True)


def render_query_step() -> None:
    """Render the query tester step with unified RAG pipeline."""

    # --- Check for embeddings data ---
    embeddings_data = _get_embeddings_data()

    if not embeddings_data:
        _render_empty_state()
        return

    # Extract data from state
    vector_store_error = embeddings_data.get("vector_store_error")
    if vector_store_error:
        st.warning("Stored vector data could not be loaded. " "Please regenerate embeddings.")
        _render_empty_state()
        return

    vector_store = embeddings_data.get("vector_store")
    model_name = embeddings_data.get("model", DEFAULT_MODEL)

    # Always recreate embedder on-demand (don't rely on stored object)
    embedder = get_embedder(model_name)

    if not vector_store or vector_store.size == 0:
        _render_empty_state()
        return

    # --- Initialize session state ---
    if "current_query" not in st.session_state:
        st.session_state.current_query = None
    if "current_response" not in st.session_state:
        st.session_state.current_response = None
    if "last_search_results" not in st.session_state:
        st.session_state.last_search_results = None
    if "last_query_variations" not in st.session_state:
        st.session_state.last_query_variations = []

    # Initialize API endpoint state
    if "api_endpoint_enabled" not in st.session_state:
        st.session_state.api_endpoint_enabled = False

    # Check if server is actually running on startup
    if "api_server_manager" not in st.session_state:
        st.session_state.api_server_manager = None

        # If API was enabled in saved state, check if server is actually up
        if st.session_state.api_endpoint_enabled:
            import socket

            def is_port_in_use(port: int = 8000) -> bool:
                """Check if a port is in use."""
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    return s.connect_ex(("127.0.0.1", port)) == 0

            if not is_port_in_use(8000):
                # Port is not in use, server is not running
                st.session_state.api_endpoint_enabled = False

    # === Header & Metrics ===
    st.subheader("Query & Retrieval")
    st.caption("Test your RAG pipeline with real-time retrieval and generation.")

   

    # === Check API Key Configuration ===
    provider = st.session_state.get("llm_provider", "OpenAI")
    from unravel.services.llm import LLM_PROVIDERS, get_api_key_from_env

    # Check if API key is configured (skip check for OpenAI-Compatible)
    if provider != "OpenAI-Compatible":
        env_key_name = LLM_PROVIDERS[provider].get("env_key", "")
        api_key = get_api_key_from_env(provider)

        if not api_key:
            _render_api_key_setup_message(provider, env_key_name)
            return



    # === Query Input Section ===
    with st.container(border=True):
        col_input, col_btn = st.columns([6, 1])

        with col_input:
            query_text = st.text_input(
                "Enter your query",
                placeholder="Ask a question about your documents...",
                key=WidgetKeys.QUERY_INPUT,
            )

        with col_btn:
            # Add spacing to align with text input label
            st.markdown('<div style="margin-top: 29px;"></div>', unsafe_allow_html=True)
            ask_clicked = st.button("Ask", type="primary", key=WidgetKeys.QUERY_ASK_BUTTON, width="stretch")

        # Configuration
        with st.expander("Retrieval Settings", expanded=False):
            col_k, col_threshold = st.columns(2)
            with col_k:
                # Use saved value if available
                default_top_k = st.session_state.get("query_top_k", min(5, vector_store.size))
                top_k = st.slider(
                    "Top K Results",
                    min_value=1,
                    max_value=min(20, vector_store.size),
                    value=min(default_top_k, vector_store.size),
                    key=WidgetKeys.QUERY_TOP_K_SLIDER,
                )
                # Store in session state for persistence
                st.session_state.query_top_k = top_k

            with col_threshold:
                # Get retrieval strategy for threshold configuration
                retrieval_config_raw = st.session_state.get("retrieval_config")
                if retrieval_config_raw is None or not isinstance(retrieval_config_raw, dict):
                    retrieval_strategy = "DenseRetriever"
                    fusion_method = None
                else:
                    retrieval_strategy = retrieval_config_raw.get("strategy", "DenseRetriever")
                    fusion_method = (
                        retrieval_config_raw.get("params", {}).get("fusion_method", "weighted_sum")
                        if retrieval_strategy == "HybridRetriever"
                        else None
                    )

                # Strategy-specific threshold defaults and ranges
                THRESHOLD_CONFIG = {
                    "DenseRetriever": {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                    "SparseRetriever": {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.5,
                    },
                    "HybridRetriever": {
                        "weighted_sum": {
                            "default": 0.2,
                            "min": 0.0,
                            "max": 1.0,
                            "step": 0.05,
                        },
                        "rrf": {"default": 0.0, "min": 0.0, "max": 0.1, "step": 0.005},
                    },
                }

                # Get appropriate threshold config
                if retrieval_strategy == "HybridRetriever":
                    threshold_cfg = THRESHOLD_CONFIG[retrieval_strategy][fusion_method]
                else:
                    threshold_cfg = THRESHOLD_CONFIG.get(
                        retrieval_strategy, THRESHOLD_CONFIG["DenseRetriever"]
                    )

                # Use strategy and fusion method in key to make slider reactive to changes
                # This ensures the slider regenerates with new config when strategy changes
                strategy_key = (
                    f"{retrieval_strategy}_{fusion_method}" if fusion_method else retrieval_strategy
                )

                # Use saved value if available and matches current strategy
                saved_threshold = st.session_state.get("query_threshold")
                default_threshold = threshold_cfg["default"]
                if saved_threshold is not None and st.session_state.get("last_threshold_strategy") == strategy_key:
                    default_threshold = saved_threshold

                threshold = st.slider(
                    "Minimum Similarity Score",
                    min_value=threshold_cfg["min"],
                    max_value=threshold_cfg["max"],
                    value=default_threshold,
                    step=threshold_cfg["step"],
                    key=get_threshold_slider_key(retrieval_strategy, fusion_method),
                    help=f"Filter results below this threshold ({retrieval_strategy}{', ' + fusion_method if fusion_method else ''})",
                )
                # Store in session state for persistence
                st.session_state.query_threshold = threshold
                st.session_state.last_threshold_strategy = strategy_key

        with st.expander("Query Expansion", expanded=False):
            enable_query_expansion = st.checkbox(
                "Generate query variations with the LLM",
                value=False,
                key=WidgetKeys.QUERY_EXPANSION_ENABLED,
                help="Creates multiple alternate phrasings and unions their results.",
            )
            variation_count = st.slider(
                "Number of variations",
                min_value=3,
                max_value=5,
                value=4,
                key=WidgetKeys.QUERY_EXPANSION_COUNT,
            )
            rewrite_prompt = st.text_area(
                "Rewrite Prompt",
                value=st.session_state.get(
                    WidgetKeys.QUERY_REWRITE_PROMPT,
                    DEFAULT_QUERY_REWRITE_PROMPT,
                ),
                height=120,
                key=WidgetKeys.QUERY_REWRITE_PROMPT,
                help="Prompt used to generate query variations for retrieval.",
            )

        with st.expander("System Prompt", expanded=False):
            query_system_prompt = st.text_area(
                "System Prompt",
                value=st.session_state.get(WidgetKeys.QUERY_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
                height=150,
                key=WidgetKeys.QUERY_SYSTEM_PROMPT,
                label_visibility="collapsed",
                help="Instructions for how the model should behave.",
            )

        # API Endpoint expander
        render_api_endpoint_expander(
            vector_store=vector_store,
            embedder=embedder,
            query_system_prompt=query_system_prompt,
        )

    # Variations display placeholder (kept near the top so it doesn't get lost).
    variations_placeholder = st.empty()

    # === Update API server state if enabled ===
    if st.session_state.api_endpoint_enabled and st.session_state.api_server_manager:
        from unravel.services.api_server import update_pipeline_state

        llm_config, system_prompt = _get_llm_config_from_sidebar()
        retrieval_config = st.session_state.get("retrieval_config", {
            "strategy": "DenseRetriever",
            "params": {}
        })
        reranking_config = st.session_state.get("reranking_config", {
            "enabled": False
        })
        bm25_data = st.session_state.get("bm25_index_data")

        # Get current slider values for API
        top_k = st.session_state.get("query_top_k", 5)
        threshold = st.session_state.get("query_threshold", 0.3)

        # Note: BM25 index will be built on-demand by API server if needed

        update_pipeline_state(
            vector_store=vector_store,
            embedder=embedder,
            llm_config=llm_config,
            system_prompt=query_system_prompt,
            retrieval_config=retrieval_config,
            reranking_config=reranking_config,
            bm25_index_data=bm25_data,
            top_k=top_k,
            threshold=threshold,
        )

    # === Process Query: Retrieve + Generate ===
    # Detect if retrieval parameters changed (to re-run query automatically)
    should_requery = False
    if st.session_state.current_query:
        last_top_k = st.session_state.get("last_top_k")
        last_threshold = st.session_state.get("last_threshold")

        if last_top_k != top_k or last_threshold != threshold:
            should_requery = True

    if (ask_clicked and query_text.strip()) or should_requery:
        # Clear previous results
        if ask_clicked:
            st.session_state.current_query = query_text.strip()
        st.session_state.current_response = None
        st.session_state.last_query_variations = []

        # Store current parameters for change detection
        st.session_state.last_top_k = top_k
        st.session_state.last_threshold = threshold

        # Use current query if re-querying
        query_text = st.session_state.current_query if should_requery else query_text

        # Get LLM config from sidebar
        llm_config, _ = _get_llm_config_from_sidebar()
        system_prompt = query_system_prompt

        # Validate LLM config
        from unravel.services.llm import validate_config

        is_valid, error_msg = validate_config(llm_config)

        if not is_valid:
            st.error(f"Configuration Error: {error_msg}")
            st.info("Please configure your LLM settings in the sidebar.")
            return

        # Step 1: Build query variations (optional)
        query_variations: list[str] = []
        if enable_query_expansion:
            from unravel.services.llm import rewrite_query_variations

            with st.spinner("Generating query variations..."):
                try:
                    query_variations = rewrite_query_variations(
                        llm_config,
                        query_text.strip(),
                        count=variation_count,
                        prompt=rewrite_prompt,
                    )
                except Exception as e:
                    st.warning(f"Query expansion failed: {str(e)}")
                    query_variations = []

            # Persist variations even if empty, so UI state matches what happened.
            st.session_state.last_query_variations = query_variations

            # Render variations immediately so the user can see them even if they
            # scroll past the retrieved chunks.
            with variations_placeholder.container():
                _render_query_variations(query_variations)

        # Step 2: Retrieve chunks using configured strategy
        with st.spinner("Retrieving context..."):
            from unravel.services.retrieval import retrieve
            from unravel.services.retrieval.reranking import (
                RerankerConfig,
                rerank_results,
            )

            retrieval_config_raw = st.session_state.get("retrieval_config")
            if retrieval_config_raw is None or not isinstance(retrieval_config_raw, dict):
                retrieval_config = {"strategy": "DenseRetriever", "params": {}}
            else:
                retrieval_config = retrieval_config_raw

            # Add BM25 data to params if needed
            params = retrieval_config.get("params", {}).copy()
            if retrieval_config.get("strategy") in [
                "SparseRetriever",
                "HybridRetriever",
            ]:
                bm25_data = st.session_state.get("bm25_index_data")

                # Try to build BM25 index if missing
                if not bm25_data:
                    with st.spinner("Building BM25 index for sparse/hybrid retrieval..."):
                        try:
                            from unravel.services.retrieval import preprocess_retriever

                            bm25_data = preprocess_retriever(
                                "SparseRetriever",
                                vector_store,
                                **params,
                            )
                            st.session_state["bm25_index_data"] = bm25_data
                        except Exception as e:
                            st.warning(
                                f"Failed to build BM25 index: {str(e)}. Falling back to dense retrieval."
                            )
                            retrieval_config = {
                                "strategy": "DenseRetriever",
                                "params": {},
                            }
                            params = {}
                            bm25_data = None

                # Add to params if we have it
                if bm25_data:
                    params["bm25_index_data"] = bm25_data

            queries_to_search = [query_text.strip()]
            if query_variations:
                queries_to_search.extend(
                    [q for q in query_variations if q.strip() and q.strip() != query_text.strip()]
                )

            # Retrieve (multi-query)
            try:
                results_by_query = []
                for query in queries_to_search:
                    results_by_query.append(
                        retrieve(
                            query=query,
                            vector_store=vector_store,
                            embedder=embedder,
                            retriever_name=retrieval_config["strategy"],
                            k=top_k,
                            **params,
                        )
                    )
                all_results = _merge_search_results(results_by_query)
            except Exception as e:
                st.error(f"Retrieval failed: {str(e)}")
                all_results = []

            # Apply threshold filter
            search_results = [r for r in all_results if r.score >= threshold]
            search_results = search_results[:top_k]

            # Optional reranking (uses original query)
            reranking_config_raw = st.session_state.get("reranking_config")
            reranking_config = (
                reranking_config_raw
                if isinstance(reranking_config_raw, dict)
                else {"enabled": False}
            )
            if reranking_config.get("enabled", False) and search_results:
                with st.spinner("Reranking results..."):
                    try:
                        rerank_cfg = RerankerConfig(
                            enabled=True,
                            model=reranking_config.get("model", "ms-marco-MiniLM-L-12-v2"),
                            top_n=reranking_config.get("top_n", 5),
                        )
                        search_results = rerank_results(
                            query_text.strip(), search_results, rerank_cfg
                        )
                    except ImportError:
                        st.warning(
                            "FlashRank not installed. Install with: pip install unravel[reranking]"
                        )
                    except Exception as e:
                        st.error(f"Reranking failed: {str(e)}")

            # Store for later display
            st.session_state.last_search_results = search_results

        # Prepare containers for layout: Response (top) -> Chunks (bottom)
       
        response_container = st.container()
       
        chunks_container = st.container()

        # Display retrieved chunks immediately in the bottom container
        with chunks_container:
            if search_results:
                _render_retrieved_chunks(search_results)
            else:
                st.warning(
                    "No chunks found matching the similarity threshold. "
                    "Try lowering the Minimum Score or rephrasing your query."
                )
                return  # Stop if no context

        # Step 2: Generate response with retrieved chunks in the top container
        with response_container:
            st.markdown("#### Model Response")

            # Use a nice card for the response
            with st.container(border=True):
                answer_placeholder = st.empty()
                raw_response = ""
                visible_answer = ""

                try:
                    # Build RAG context
                    context = RAGContext(
                        query=query_text.strip(),
                        chunks=[r.text for r in search_results],
                        scores=[r.score for r in search_results],
                    )

                    # Get model instance
                    model = get_model(
                        provider=llm_config.provider,
                        model=llm_config.model,
                        api_key=llm_config.api_key,
                        base_url=llm_config.base_url,
                        temperature=llm_config.temperature,
                    )

                    # Stream response
                    with st.spinner("Generating response..."):
                        for chunk in model.stream(context, system_prompt):
                            raw_response += chunk

                            # If we have a complete response, filter and stream the answer only.
                            # This keeps behavior simple and avoids showing any <think> blocks.
                            visible_answer = strip_think_blocks(raw_response)
                            answer_placeholder.markdown(visible_answer + "▌")

                    # Final render (answer only).
                    answer_placeholder.empty()
                    with st.container():
                        _render_model_response(raw_response)

                    st.session_state.current_response = raw_response

                except ImportError as e:
                    st.error(f"Missing Dependency: {str(e)}")
                    st.info("Install the required library with: `pip install unravel[llm]`")
                except Exception as e:
                    st.error(f"Generation Error: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())

    # === Display Previous Results ===
    elif st.session_state.current_query:
        # Show previously generated variations near the top.
        with variations_placeholder.container():
            _render_query_variations(st.session_state.last_query_variations)

        # Show previous response first
       
        if st.session_state.current_response:
            st.markdown("#### Model Response")
            with st.container(border=True):
                _render_model_response(st.session_state.current_response)
           

        # Show previous chunks below
        if "last_search_results" in st.session_state:
            _render_retrieved_chunks(st.session_state.last_search_results)

    # Save query settings for persistence
    _save_query_settings()
