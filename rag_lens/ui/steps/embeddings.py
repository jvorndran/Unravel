from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit_shadcn_ui as ui
from numpy.typing import NDArray

from rag_lens.services.chunking import get_chunks
from rag_lens.services.embedders import (
    DEFAULT_MODEL,
    get_embedder,
)
from rag_lens.services.retrieval import retrieve
from rag_lens.services.retrieval import (
    preprocess_retriever,
    retrieve,
)
from rag_lens.services.storage import (
    get_storage_dir,
    load_document,
    save_session_state,
)
from rag_lens.services.vector_store import create_vector_store
from rag_lens.ui.components.chunk_viewer import (
    prepare_chunk_display_data,
    render_chunk_cards,
)
from rag_lens.ui.components.qdrant_dashboard import render_qdrant_dashboard
from rag_lens.utils.parsers import parse_document
from rag_lens.utils.qdrant_manager import (
    get_qdrant_status,
    restart_qdrant_server,
)
from rag_lens.utils.visualization import (
    cluster_embeddings,
    create_embedding_plot,
    reduce_dimensions,
)


@st.cache_data(show_spinner="Generating embeddings...")
def generate_embeddings(
    texts: list[str], model_name: str
) -> tuple[NDArray[Any], int]:
    """Generate embeddings for texts using specified model.

    Cached based on texts content and model name.

    Args:
        texts: List of text strings to embed
        model_name: Name of the embedding model

    Returns:
        Tuple of (embeddings array, embedding dimension)
    """
    embedder = get_embedder(model_name)
    embeddings = embedder.embed_texts(texts)
    return embeddings, embedder.dimension


def render_embeddings_step() -> None:
    
    # Read configuration from session state (set in sidebar)
    chunks = st.session_state.get("chunks", [])
    doc_name = st.session_state.get("doc_name")
    selected_model = st.session_state.get("embedding_model_name", DEFAULT_MODEL)
    selected_doc = doc_name  # Use doc_name from session state

    # --- Main Header ---
    st.markdown("## Embeddings & Similarity")
    st.caption("Visualize how your document chunks are represented in vector space and test semantic search.")

    # Check if document is selected
    if not selected_doc:
        st.info(
            "No document selected. Upload a file in the **Upload** step or select "
            "a document in the sidebar (RAG Config tab)."
        )
        if ui.button("Go to Upload Step", key="goto_upload_embeddings"):
            st.session_state.current_step = "upload"
            st.rerun()
        return

    # Document Processing Logic (Runs if chunks are missing)
    if not chunks:
        with st.spinner(f"Processing {selected_doc}..."):
            # 1. Load Text
            source_text = ""
            try:
                content = load_document(selected_doc)
                if content:
                    # Use applied params, not current
                    parsing_params = st.session_state.get("applied_parsing_params", st.session_state.get("parsing_params", {}))
                    source_text, _, _ = parse_document(selected_doc, content, parsing_params)
                else:
                    st.error(f"Failed to load document: {selected_doc}")
                    return
            except Exception as e:
                st.error(f"Error parsing document: {str(e)}")
                return

            # 2. Chunk Text
            if source_text:
                # Use applied params, not current
                params = st.session_state.get("applied_chunking_params", st.session_state.get("chunking_params", {
                    "provider": "Docling",
                    "splitter": "HybridChunker",
                    "max_tokens": 512,
                    "chunk_overlap": 50,
                    "tokenizer": "cl100k_base",
                }))

                try:
                    provider = params.get("provider", "Docling")
                    splitter = params.get("splitter", "HybridChunker")
                    splitter_params = {k: v for k, v in params.items()
                                       if k not in ["provider", "splitter"]}
                    new_chunks = get_chunks(
                        provider=provider,
                        splitter=splitter,
                        text=source_text,
                        **splitter_params
                    )

                    # Update State
                    st.session_state["chunks"] = new_chunks
                    st.session_state["doc_name"] = selected_doc
                    
                    # Invalidate embeddings
                    if "last_embeddings_result" in st.session_state:
                        del st.session_state["last_embeddings_result"]
                    if "search_results" in st.session_state:
                        del st.session_state["search_results"]
                        
                    st.rerun()
                except Exception as e:
                    st.error(f"Error chunking document: {str(e)}")
            else:
                st.warning("Selected document is empty.")

    st.write("")

    # --- Main Content Area ---

    # 1. Empty State Check
    chunks = st.session_state.get("chunks", [])
    if not chunks:
        st.info("👋 No chunks available. Configure document and chunking parameters in the sidebar (RAG Config tab), then go to the **Chunks** step to generate chunks.")
        if ui.button("Go to Chunks Step", key="goto_chunks"):
             st.session_state.current_step = "chunks"
             st.rerun()
        return

    # 2. Embedding Generation (if needed)
    # Include content hash to detect chunking parameter changes
    import hashlib
    content_hash = hashlib.md5(''.join([c.text for c in chunks]).encode()).hexdigest()[:8]
    current_state_key = f"embeddings_{content_hash}_{selected_model}_{doc_name}"
    
    # Initialize or recover state structure if it's old version
    if "last_embeddings_result" in st.session_state:
        state_data = st.session_state["last_embeddings_result"]
        # Migration: Add projections dict if missing
        if "projections" not in state_data:
            # If we have legacy reduced_embeddings, move it to projections by dimension
            if "reduced_embeddings" in state_data:
                components = state_data.get("reduced_embeddings_components")
                if not isinstance(components, int):
                    components = int(state_data["reduced_embeddings"].shape[1])
                state_data["projections"] = {
                    components: (
                        state_data["reduced_embeddings"],
                        state_data["reducer"],
                    )
                }
                # Clean up legacy keys
                del state_data["reduced_embeddings"]
                del state_data["reducer"]
                if "reduced_embeddings_components" in state_data:
                    del state_data["reduced_embeddings_components"]
            else:
                 state_data["projections"] = {}
        
        # Migration: Clean up any old embedder objects
        if "embedder" in state_data:
            del state_data["embedder"]
            
        st.session_state["last_embeddings_result"] = state_data

    # Generate if key changed, data missing, or store invalid
    needs_regeneration = False
    vector_store_locked = False
    if "last_embeddings_result" not in st.session_state:
        needs_regeneration = True
    else:
        state_data = st.session_state["last_embeddings_result"]
        if state_data.get("key") != current_state_key:
            needs_regeneration = True
        elif state_data.get("vector_store_error"):
            error_message = str(state_data.get("vector_store_error"))
            if "already accessed by another instance" in error_message:
                embeddings = state_data.get("embeddings")
                if embeddings is None or embeddings.shape[0] == 0:
                    st.error(
                        "Stored vector data is locked by another process. "
                        "Stop the other instance or run a Qdrant server for "
                        "concurrent access, then refresh."
                    )
                    return
                st.warning(
                    "Stored vector data is locked by another process. "
                    "Semantic search is disabled until the lock is released."
                )
                vector_store_locked = True
            if not vector_store_locked:
                st.warning(
                    "Stored vector data could not be loaded. Regenerating embeddings."
                )
                needs_regeneration = True
        else:
            vector_store = state_data.get("vector_store")
            if not vector_store or vector_store.size == 0:
                needs_regeneration = True
            else:
                embeddings = state_data.get("embeddings")
                if embeddings is None or embeddings.shape[0] == 0:
                    embeddings = vector_store.get_all_embeddings()
                    if embeddings.shape[0] == 0:
                        needs_regeneration = True
                    else:
                        state_data["embeddings"] = embeddings
                        st.session_state["last_embeddings_result"] = state_data
                if not needs_regeneration and embeddings.shape[0] != len(chunks):
                    needs_regeneration = True

    if needs_regeneration:
        
        with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
            try:
                texts = [c.text for c in chunks]
                embeddings, dimension = generate_embeddings(texts, selected_model)

                vector_store_path = get_storage_dir() / "session" / "current_vector_store"
                try:
                    vector_store = create_vector_store(
                        dimension=dimension,
                        storage_path=vector_store_path,
                    )
                    vector_store.clear()
                except Exception as exc:
                    if "already accessed by another instance" in str(exc):
                        st.error(
                            "The local Qdrant storage is already in use by another "
                            "process. Stop the other instance or run a Qdrant server "
                            "for concurrent access."
                        )
                        return
                    raise
                vector_store.add(embeddings, texts, metadata=[c.metadata for c in chunks])

                # Initial 2D projection - just to have it, though we will default to 3D now
                reduced_embeddings, reducer = reduce_dimensions(embeddings, n_components=3)

                st.session_state["last_embeddings_result"] = {
                    "key": current_state_key,
                    "vector_store": vector_store,
                    "embeddings": embeddings, # Keep original embeddings for re-projection
                    "projections": {
                        3: (reduced_embeddings, reducer)
                    },
                    "reduced_embeddings": reduced_embeddings,
                    "reduced_embeddings_components": 3,
                    "chunks": chunks,
                    "model": selected_model,
                }

                # Build BM25 index if needed for sparse/hybrid retrieval
                retrieval_config_raw = st.session_state.get("retrieval_config")
                retrieval_config = retrieval_config_raw if isinstance(retrieval_config_raw, dict) else {}
                if retrieval_config.get("strategy") in ["SparseRetriever", "HybridRetriever"]:
                    with st.spinner("Building BM25 index for sparse/hybrid retrieval..."):
                        try:
                            from rag_lens.services.retrieval import preprocess_retriever

                            bm25_data = preprocess_retriever(
                                "SparseRetriever",
                                vector_store,
                                **retrieval_config.get("params", {})
                            )
                            st.session_state["bm25_index_data"] = bm25_data
                        except Exception as e:
                            st.warning(f"Failed to build BM25 index: {str(e)}")

                # Persist session state to disk for refresh resilience
                save_session_state({
                    "doc_name": st.session_state.get("doc_name"),
                    "embedding_model_name": selected_model,
                    "chunking_params": st.session_state.get("chunking_params"),
                    "chunks": chunks,
                    "last_embeddings_result": st.session_state["last_embeddings_result"],
                    "retrieval_config": st.session_state.get("retrieval_config"),
                    "reranking_config": st.session_state.get("reranking_config"),
                    "bm25_index_data": st.session_state.get("bm25_index_data"),
                })
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return

    # Load data from state
    state_data = st.session_state["last_embeddings_result"]
    vector_store = state_data.get("vector_store")
    model_name = state_data.get("model", selected_model)
    
    # Ensure raw embeddings are available (might be missing if loaded from disk/legacy)
    if "embeddings" not in state_data:
        if vector_store is None:
            st.error("Embeddings are missing. Please regenerate embeddings.")
            return
        state_data["embeddings"] = vector_store.get_all_embeddings()
        st.session_state["last_embeddings_result"] = state_data
    embeddings = state_data["embeddings"]

    qdrant_url = st.session_state.get("qdrant_url")
    if qdrant_url and vector_store is not None:
        if getattr(vector_store, "url", None) != qdrant_url:
            with st.spinner("Syncing embeddings to Qdrant server..."):
                try:
                    vector_store_path = get_storage_dir() / "session" / "current_vector_store"
                    server_store = create_vector_store(
                        dimension=embeddings.shape[1],
                        storage_path=vector_store_path,
                    )
                    server_store.clear()
                    texts = [c.text for c in chunks]
                    server_store.add(
                        embeddings,
                        texts,
                        metadata=[c.metadata for c in chunks],
                    )
                    state_data["vector_store"] = server_store
                    st.session_state["last_embeddings_result"] = state_data
                    vector_store = server_store
                except Exception as err:
                    st.warning(f"Failed to sync embeddings to Qdrant server: {err}")

    # Recreate embedder on-demand (lightweight - model loads lazily)
    embedder = get_embedder(model_name)

    # --- Qdrant Status Header ---
    qdrant_status = get_qdrant_status()
    is_running = qdrant_status.get("running", False)
    
    # Modern, minimal status card
    with st.container(border=True):
        # Layout: Status & URL (80%) | Action (20%)
        # vertical_alignment="center" requires Streamlit >= 1.31
        col_status, col_action = st.columns([4, 1], vertical_alignment="center")

        with col_status:
            if is_running:
                url = qdrant_status.get('url', '')
                if url:
                    st.markdown(
                        f"**Qdrant Status** &nbsp; <span style='color: #16a34a; background-color: #dcfce7; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;'>Running</span> &nbsp; <a href='{url + "/dashboard#/collections"}' target='_blank' style='color: #4b5563; text-decoration: none; border-bottom: 1px dotted #9ca3af;'>{url + "/dashboard#/collections"} ↗</a>",
                        unsafe_allow_html=True
                    )
            else:
                error_msg = "Docker not found" if not qdrant_status.get("docker_available") else "Connection lost"
                st.markdown(
                    f"**Qdrant Status** &nbsp; <span style='color: #dc2626; background-color: #fee2e2; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;'>Stopped</span> &nbsp; <span style='color: #ef4444; font-size: 0.9em;'>{error_msg}</span>",
                    unsafe_allow_html=True
                )

        with col_action:
            if st.button("Restart", key="restart_qdrant_header_btn", use_container_width=True):
                with st.spinner("..."):
                    restart_qdrant_server()
                st.rerun()

        # Contextual Warnings/Errors
        if not is_running and qdrant_status.get("error"):
             st.markdown(f"<div style='margin-top: 8px; font-size: 0.85em; color: #dc2626; background: #fef2f2; padding: 8px; border-radius: 4px;'>{qdrant_status['error']}</div>", unsafe_allow_html=True)
        
        if is_running and qdrant_status.get("mount_type") == "bind":
             st.markdown(f"<div style='margin-top: 8px; font-size: 0.85em; color: #854d0e; background: #fef9c3; padding: 8px; border-radius: 4px;'>⚠️ <b>Performance Note:</b> Using bind mount on Windows (slower I/O).</div>", unsafe_allow_html=True)

    # 4. Tabs for Content Organization
    active_tab = ui.tabs(options=["Visual Explorer", "Vector Analysis"], default_value="Visual Explorer", key="embed_main_tabs")

    # --- TAB 1: Visual Explorer ---
    if active_tab == "Visual Explorer":
        
        # 3. Logic & Calculation
        # Enforce 3D mode
        n_components = 3
        
        # Ensure projection exists for selected mode
        if n_components not in state_data["projections"]:
            with st.spinner(f"Calculating 3D projection..."):
                reduced, reducer = reduce_dimensions(embeddings, n_components=n_components)
                state_data["projections"][n_components] = (reduced, reducer)
                st.session_state["last_embeddings_result"] = state_data # Update session state
        
        reduced_embeddings, reducer = state_data["projections"][n_components]

        # Calculate Clusters
        if "cluster_labels" not in state_data:
            with st.spinner("Analyzing semantic clusters..."):
                # Don't cluster very small or empty datasets
                if len(chunks) < 10 or embeddings.shape[0] == 0:
                    labels = np.zeros(len(chunks), dtype=int)
                else:
                    # Determine optimal clusters - simple heuristic: sqrt(N/2) capped at 10
                    n_clusters = min(10, max(2, int(np.sqrt(len(chunks) / 2))))
                    labels = cluster_embeddings(embeddings, n_clusters=n_clusters)
                state_data["cluster_labels"] = labels
                st.session_state["last_embeddings_result"] = state_data
        
        cluster_labels = state_data["cluster_labels"]

        # Query Interface
        st.write("")
        st.markdown("#### Semantic Search")
        
        # Streamlit reruns the script on every keystroke for text inputs.
        # Wrapping the input in a form prevents the expensive rerun until the user clicks Search.
        if "query_input" not in st.session_state:
            st.session_state["query_input"] = ""

        with st.form("semantic_search_form", clear_on_submit=False):
            c1, c2 = st.columns([4, 1])
            with c1:
                query_text = st.text_input(
                    label="Query",
                    placeholder="e.g., What is the main challenge of RAG?",
                    key="query_input",
                    label_visibility="collapsed",
                )
            with c2:
                submit_button = st.form_submit_button("Search", type="primary", width='stretch')

        # Process Query
        if "search_results" not in st.session_state:
            st.session_state.search_results = None

        if submit_button and query_text and query_text.strip():
            if vector_store is None or vector_store_locked:
                st.warning(
                    "Semantic search is unavailable while the vector store is locked."
                )
                st.session_state.search_results = None
                return
            with st.spinner("Calculating similarity..."):
                # Embedder is already created above - no fallback needed
                query_embedding = embedder.embed_query(query_text)

                retrieval_config = st.session_state.get("retrieval_config")
                if not retrieval_config or not isinstance(retrieval_config, dict):
                    retrieval_config = {"strategy": "DenseRetriever", "params": {}}

                strategy = retrieval_config.get("strategy", "DenseRetriever")
                params = retrieval_config.get("params", {})

                # Ensure BM25 data exists if needed
                if strategy in ("SparseRetriever", "HybridRetriever"):
                    bm25_data = st.session_state.get("bm25_index_data")
                    if not bm25_data:
                        try:
                            bm25_data = preprocess_retriever(
                                "SparseRetriever",
                                vector_store,
                                **params,
                            )
                            st.session_state["bm25_index_data"] = bm25_data
                        except Exception as preprocess_err:
                            st.warning(
                                f"Failed to build BM25 index for retrieval: {preprocess_err}"
                            )
                    if bm25_data:
                        params = {**params, "bm25_index_data": bm25_data}

                try:
                    search_results = retrieve(
                        query=query_text,
                        vector_store=vector_store,
                        embedder=embedder,
                        retriever_name=strategy,
                        k=5,
                        **params,
                    )
                except Exception as err:
                    st.error(f"Retrieval failed: {err}")
                    search_results = []

                st.session_state.search_results = {
                    "query": query_text,
                    "embedding": query_embedding,
                    "neighbors": search_results,
                }

        # Retrieve results
        results = st.session_state.search_results
        query_embedding = results.get("embedding") if results else None
        neighbors = results.get("neighbors", []) if results else []
        neighbor_indices = [r.index for r in neighbors]
        
        # Project Query Point (Dynamic based on current reducer)
        query_point_viz = None
        if query_embedding is not None and reducer is not None:
             query_reshaped = query_embedding.reshape(1, -1)
             try:
                 query_proj = reducer.transform(query_reshaped)
                 # Enforce 3D logic
                 query_point_viz = {"x": query_proj[0][0], "y": query_proj[0][1], "z": query_proj[0][2]}
             except Exception:
                 st.warning("Could not project query point.")

        # 4. Render Chart (Visual Location: Below Search)
        with st.container(border=True):
            help_text = "Points: Each dot is a chunk. Colors represent semantic clusters. Pink Star = Query."
            st.markdown(f"##### Embedding Space (3D UMAP)", help=help_text)
            
            # Prepare Data for Plotly
            df = pd.DataFrame(reduced_embeddings, columns=["x", "y", "z"])
            df["text_preview"] = [c.text[:150] + "..." for c in chunks]
            df["chunk_index"] = range(len(chunks))
            df["cluster_label"] = cluster_labels  # Numeric for color
            df["Cluster"] = [f"Cluster {l}" for l in cluster_labels] # String for hover
            
            fig = create_embedding_plot(
                df,
                x_col="x",
                y_col="y",
                z_col="z",
                color_col="cluster_label",
                hover_data=["text_preview", "Cluster"],
                title="",
                query_point=query_point_viz,
                neighbors_indices=neighbor_indices
            )
            st.plotly_chart(fig, width='stretch')
            
        # Nearest Neighbors Section
        st.write("")
        st.markdown("##### Nearest Neighbors")
        if neighbors:
            # Convert search results to chunks for the viewer
            from dataclasses import dataclass
            
            @dataclass
            class ChunkAdapter:
                """Adapter to make SearchResult compatible with chunk viewer."""
                text: str
                metadata: dict[str, Any]
                start_index: int = 0
                end_index: int = 0
            
            neighbor_chunks = [
                ChunkAdapter(
                    text=res.text,
                    metadata=res.metadata,
                    start_index=i,
                    end_index=i,
                )
                for i, res in enumerate(neighbors)
            ]
            
            # Prepare display data
            neighbor_display_data = prepare_chunk_display_data(
                chunks=neighbor_chunks,
                source_text=None,
                calculate_overlap=False,
            )
            
            # Add similarity score and strategy-specific badges
            custom_badges = []
            for res in neighbors:
                badges = []
                # Main Score
                badges.append({
                    "label": "Score",
                    "value": f"{res.score:.4f}",
                    "color": "#d1fae5"  # Green tint for similarity
                })
                
                dense_rrf_contribution = res.metadata.get("dense_rrf_contribution")
                if dense_rrf_contribution is not None:
                    badges.append({
                        "label": "Dense Contribution",
                        "value": f"{dense_rrf_contribution:.4f}",
                        "color": "#93c5fd"
                    })

                sparse_rrf_contribution = res.metadata.get("sparse_rrf_contribution")
                if sparse_rrf_contribution is not None:
                    badges.append({
                        "label": "Sparse Contribution",
                        "value": f"{sparse_rrf_contribution:.4f}",
                        "color": "#fecdd3"
                    })

                custom_badges.append(badges)
            
            # Render using the reusable component in card mode
            render_chunk_cards(
                chunk_display_data=neighbor_display_data,
                custom_badges=custom_badges,
                show_overlap=False,
                display_mode="card",
            )
        else:
            st.info("Enter a query above to see the most similar chunks here.")

    # --- TAB 2: Vector Analysis ---
    elif active_tab == "Vector Analysis":
        render_qdrant_dashboard()
