"""Query tester page for RAG Visualizer.

Allows users to test the full RAG pipeline with retrieval and LLM response generation.
"""

from typing import Any

import streamlit as st
import streamlit_shadcn_ui as ui

from rag_visualizer.services.embedders import DEFAULT_MODEL, get_embedder
from rag_visualizer.services.llm import (
    DEFAULT_SYSTEM_PROMPT,
    LLMConfig,
    RAGContext,
    get_model,
)
from rag_visualizer.ui.components.chunk_viewer import (
    prepare_chunk_display_data,
    render_chunk_cards,
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
        
        st.write("")
        if ui.button("Go to Embeddings Step", key="goto_embeddings"):
            st.session_state.current_step = "embeddings"
            st.rerun()


def _render_retrieved_chunks(
    results: list[Any], show_scores: bool = True
) -> None:
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
    
    # Add similarity score as custom badge if requested
    custom_badges: list[dict[str, Any]] | None = None
    if show_scores:
        custom_badges = [
            {
                "label": "Score",
                "value": f"{res.score:.3f}",
                "color": "#d1fae5"  # Green tint for similarity
            }
            for res in results
        ]
    
    # Render using the reusable component in card mode
    render_chunk_cards(
        chunk_display_data=retrieved_display_data,
        custom_badges=custom_badges,
        show_overlap=False,
        display_mode="card",
    )


def _get_llm_config_from_sidebar() -> tuple[LLMConfig, str]:
    """Get LLM configuration from sidebar session state."""
    provider = st.session_state.get("llm_provider", "OpenAI")
    model = st.session_state.get("llm_model", "")
    api_key = st.session_state.get("llm_api_key", "")
    base_url = st.session_state.get("llm_base_url", "")
    temperature = st.session_state.get("llm_temperature", 0.7)
    max_tokens = st.session_state.get("llm_max_tokens", 1024)
    system_prompt = st.session_state.get("llm_system_prompt", DEFAULT_SYSTEM_PROMPT)
    
    # Check for env var API key
    from rag_visualizer.services.llm import get_api_key_from_env
    env_key = get_api_key_from_env(provider)
    if env_key:
        api_key = env_key
    
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url if base_url else None,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return config, system_prompt


def render_query_step() -> None:
    """Render the query tester step with unified RAG pipeline."""
    
    # --- Check for embeddings data ---
    embeddings_data = _get_embeddings_data()
    
    if not embeddings_data:
        _render_empty_state()
        return
    
    # Extract data from state
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
    
    # === Header & Metrics ===
    st.subheader("Query & Retrieval")
    st.caption("Test your RAG pipeline with real-time retrieval and generation.")
    
    cols = st.columns(3)
    with cols[0]:
        ui.metric_card(
            title="Indexed Chunks",
            content=str(vector_store.size),
            description="Total chunks in store",
            key="metric_chunks"
        )
    with cols[1]:
        ui.metric_card(
            title="Embedding Model",
            content=model_name.split("/")[-1], # Shorten model name if path
            description="Used for retrieval",
            key="metric_embed_model"
        )
    with cols[2]:
        llm_model = st.session_state.get("llm_model", "Default")
        ui.metric_card(
            title="LLM Model",
            content=llm_model,
            description="Generation model",
            key="metric_llm_model"
        )

    st.write("")
    
    # === Query Input Section ===
    with st.container(border=True):
        col_input, col_btn = st.columns([6, 1])
        
        with col_input:
            query_text = st.text_input(
                "Enter your query",
                placeholder="Ask a question about your documents...",
                key="query_input"
            )
            
        with col_btn:
            # Add spacing to align with text input label
            st.markdown('<div style="margin-top: 29px;"></div>', unsafe_allow_html=True)
            ask_clicked = st.button(
                "Ask", 
                type="primary",
                key="ask_button", 
                use_container_width=True
            )

        # Configuration
        with st.expander("Retrieval Settings", expanded=False):
            col_k, col_threshold = st.columns(2)
            with col_k:
                top_k = st.slider(
                    "Top K Results",
                    min_value=1,
                    max_value=min(20, vector_store.size),
                    value=min(5, vector_store.size),
                    key="top_k_slider"
                )
            
            with col_threshold:
                threshold = st.slider(
                    "Minimum Similarity Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    key="threshold_slider"
                )

        with st.expander("System Prompt", expanded=False):
            query_system_prompt = st.text_area(
                "System Prompt",
                value=st.session_state.get("llm_system_prompt", DEFAULT_SYSTEM_PROMPT),
                height=150,
                key="query_system_prompt",
                label_visibility="collapsed",
                help="Instructions for how the model should behave."
            )
    
    # === Process Query: Retrieve + Generate ===
    if ask_clicked and query_text.strip():
        # Clear previous results
        st.session_state.current_query = query_text.strip()
        st.session_state.current_response = None
        
        # Get LLM config from sidebar
        llm_config, _ = _get_llm_config_from_sidebar()
        system_prompt = query_system_prompt
        
        # Validate LLM config
        from rag_visualizer.services.llm import validate_config
        is_valid, error_msg = validate_config(llm_config)
        
        if not is_valid:
            st.error(f"Configuration Error: {error_msg}")
            st.info("Please configure your LLM settings in the sidebar.")
            return
        
        # Step 1: Retrieve chunks using configured strategy
        with st.spinner("Retrieving context..."):
            from rag_visualizer.services.retrieval import retrieve
            from rag_visualizer.services.retrieval.reranking import (
                RerankerConfig,
                rerank_results,
            )

            retrieval_config = st.session_state.get("retrieval_config", {
                "strategy": "DenseRetriever",
                "params": {}
            })

            # Add BM25 data to params if needed
            params = retrieval_config.get("params", {}).copy()
            if retrieval_config["strategy"] in ["SparseRetriever", "HybridRetriever"]:
                params["bm25_index_data"] = st.session_state.get("bm25_index_data")

                if not params["bm25_index_data"]:
                    st.warning("BM25 index not found. Falling back to dense retrieval.")
                    retrieval_config["strategy"] = "DenseRetriever"
                    params = {}

            # Retrieve
            try:
                all_results = retrieve(
                    query=query_text.strip(),
                    vector_store=vector_store,
                    embedder=embedder,
                    retriever_name=retrieval_config["strategy"],
                    k=top_k,
                    **params
                )
            except Exception as e:
                st.error(f"Retrieval failed: {str(e)}")
                all_results = []

            # Apply threshold filter
            search_results = [r for r in all_results if r.score >= threshold]

            # Optional reranking
            reranking_config = st.session_state.get("reranking_config", {"enabled": False})
            if reranking_config["enabled"] and search_results:
                with st.spinner("Reranking results..."):
                    try:
                        rerank_cfg = RerankerConfig(
                            enabled=True,
                            model=reranking_config["model"],
                            top_n=reranking_config["top_n"]
                        )
                        search_results = rerank_results(query_text.strip(), search_results, rerank_cfg)
                    except ImportError:
                        st.warning("FlashRank not installed. Install with: pip install rag-visualizer[reranking]")
                    except Exception as e:
                        st.error(f"Reranking failed: {str(e)}")

            # Store for later display
            st.session_state.last_search_results = search_results

        # Prepare containers for layout: Response (top) -> Chunks (bottom)
        st.write("")
        response_container = st.container()
        st.write("") 
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
                return # Stop if no context

        # Step 2: Generate response with retrieved chunks in the top container
        with response_container:
            st.markdown("#### Model Response")
            
            # Use a nice card for the response
            with st.container(border=True):
                response_placeholder = st.empty()
                full_response = ""
                
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
                        max_tokens=llm_config.max_tokens,
                    )
                    
                    # Stream response
                    with st.spinner("Generating response..."):
                        for chunk in model.stream(context, system_prompt):
                            full_response += chunk
                            # Update placeholder for streaming effect
                            response_placeholder.markdown(full_response + "▌")
                    
                    # Final update without cursor
                    response_placeholder.markdown(full_response)
                    st.session_state.current_response = full_response
                    
                except ImportError as e:
                    st.error(f"Missing Dependency: {str(e)}")
                    st.info("Install the required library with: `pip install rag-visualizer[llm]`")
                except Exception as e:
                    st.error(f"Generation Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # === Display Previous Results ===
    elif st.session_state.current_query:
        # Show previous response first
        st.write("")
        if st.session_state.current_response:
            st.markdown("#### Model Response")
            with st.container(border=True):
                st.markdown(st.session_state.current_response)
            st.write("")

        # Show previous chunks below
        if "last_search_results" in st.session_state:
            _render_retrieved_chunks(st.session_state.last_search_results)
