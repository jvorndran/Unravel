from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit_shadcn_ui as ui
from numpy.typing import NDArray

from rag_visualizer.services.chunking import get_chunks
from rag_visualizer.services.embedders import (
    DEFAULT_MODEL,
    get_embedder,
)
from rag_visualizer.services.storage import (
    load_document,
    save_session_state,
)
from rag_visualizer.services.vector_store import create_vector_store
from rag_visualizer.ui.components.chunk_viewer import (
    prepare_chunk_display_data,
    render_chunk_cards,
)
from rag_visualizer.utils.parsers import parse_document
from rag_visualizer.utils.visualization import (
    calculate_similarity_matrix,
    create_embedding_plot,
    create_similarity_histogram,
    find_outliers,
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
    current_state_key = f"embeddings_{len(chunks)}_{selected_model}_{doc_name}"
    
    if "last_embeddings_result" not in st.session_state or \
       st.session_state["last_embeddings_result"].get("key") != current_state_key:
        
        with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
            try:
                texts = [c.text for c in chunks]
                embeddings, dimension = generate_embeddings(texts, selected_model)

                vector_store = create_vector_store(dimension=dimension)
                vector_store.add(embeddings, texts, metadata=[c.metadata for c in chunks])

                reduced_embeddings, reducer = reduce_dimensions(embeddings)

                st.session_state["last_embeddings_result"] = {
                    "key": current_state_key,
                    "vector_store": vector_store,
                    "reduced_embeddings": reduced_embeddings,
                    "reducer": reducer,
                    "chunks": chunks,
                    "model": selected_model,
                    # Don't store embedder - PyTorch models aren't serializable
                    # It will be recreated on-demand from the model name
                }

                # Build BM25 index if needed for sparse/hybrid retrieval
                retrieval_config_raw = st.session_state.get("retrieval_config")
                retrieval_config = retrieval_config_raw if isinstance(retrieval_config_raw, dict) else {}
                if retrieval_config.get("strategy") in ["SparseRetriever", "HybridRetriever"]:
                    with st.spinner("Building BM25 index for sparse/hybrid retrieval..."):
                        try:
                            from rag_visualizer.services.retrieval import preprocess_retriever

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

    # Migration: Clean up any old embedder objects (fix for meta tensor error)
    if "embedder" in st.session_state["last_embeddings_result"]:
        del st.session_state["last_embeddings_result"]["embedder"]

    # Load data from state
    state_data = st.session_state["last_embeddings_result"]
    vector_store = state_data["vector_store"]
    reduced_embeddings = state_data["reduced_embeddings"]
    reducer = state_data["reducer"]
    model_name = state_data.get("model", selected_model)

    # Restore a usable reducer when session state was loaded from disk
    # (UMAP reducers are not serialized, so query projection would fail).
    if reducer is None:
        vectors = vector_store.get_all_embeddings()
        if vectors.size:
            reduced_embeddings, reducer = reduce_dimensions(vectors)
            state_data["reduced_embeddings"] = reduced_embeddings
            state_data["reducer"] = reducer
            st.session_state["last_embeddings_result"] = state_data

    # Recreate embedder on-demand (lightweight - model loads lazily)
    embedder = get_embedder(model_name)

    # 3. KPIs / Top Metrics
    col_kpi1, col_kpi2 = st.columns(2)
    with col_kpi1:
        ui.metric_card(title="Total Chunks", content=len(chunks), key="metric_total_chunks")
    with col_kpi2:
        ui.metric_card(title="Vector Dimension", content=getattr(embedder, "dimension", "N/A"), key="metric_vec_dim")

    st.write("") # Spacer

    # 4. Tabs for Content Organization
    active_tab = ui.tabs(options=["Visual Explorer", "Vector Analysis"], default_value="Visual Explorer", key="embed_main_tabs")

    # --- TAB 1: Visual Explorer ---
    if active_tab == "Visual Explorer":
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
            with st.spinner("Calculating similarity..."):
                # Embedder is already created above - no fallback needed
                query_embedding = embedder.embed_query(query_text)
                search_results = vector_store.search(query_embedding, k=5)
                
                # Project Query to 2D
                query_point_2d = None
                if reducer is not None:
                    query_reshaped = query_embedding.reshape(1, -1)
                    try:
                        query_2d = reducer.transform(query_reshaped)
                        query_point_2d = {"x": query_2d[0][0], "y": query_2d[0][1]}
                    except Exception:
                        st.warning("Could not project query point.")

                st.session_state.search_results = {
                    "query": query_text,
                    "embedding": query_embedding,
                    "neighbors": search_results,
                    "query_point_2d": query_point_2d
                }

        # Retrieve results
        results = st.session_state.search_results
        query_point_2d = results.get("query_point_2d") if results else None
        neighbors = results.get("neighbors", []) if results else []
        neighbor_indices = [r.index for r in neighbors]

        # Visualization Section
        help_text = "Points: Each dot is a chunk. Proximity = Similarity. Pink Star = Query."
        
        st.write("")
        with st.container(border=True):
            st.markdown("##### Embedding Space (UMAP Projection)", help=help_text)
            
            # Prepare Data for Plotly
            df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
            df["text_preview"] = [c.text[:150] + "..." for c in chunks]
            df["chunk_index"] = range(len(chunks))
            
            fig = create_embedding_plot(
                df,
                x_col="x",
                y_col="y",
                hover_data=["text_preview"],
                title="",
                query_point=query_point_2d,
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
            
            # Add similarity score as custom badge
            custom_badges = [
                {
                    "label": "Score",
                    "value": f"{res.score:.4f}",
                    "color": "#d1fae5"  # Green tint for similarity
                }
                for res in neighbors
            ]
            
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
        st.write("")
        st.markdown("#### Semantic Cohesion & Outliers")
        st.caption("Analyze how similar your chunks are to each other and identify outliers")
        
        # Get embeddings from vector store
        vectors = vector_store.get_all_embeddings()
        
        if len(vectors) < 2:
            st.info("Need at least 2 chunks for similarity analysis.")
        else:
            # Calculate similarity matrix
            with st.spinner("Analyzing vector space..."):
                similarity_matrix = calculate_similarity_matrix(vectors)
                
                # Calculate global metrics
                # Get upper triangle (excluding diagonal) for pairwise similarities
                n = similarity_matrix.shape[0]
                upper_triangle_indices = np.triu_indices(n, k=1)
                pairwise_similarities = similarity_matrix[upper_triangle_indices]
                
                avg_similarity = float(pairwise_similarities.mean())
                embedding_variance = float(vectors.var())
                
                # Find outliers
                outliers = find_outliers(vectors, chunks, n_outliers=5)
            
            # Display global metrics
            st.write("")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                ui.metric_card(
                    title="Avg. Similarity",
                    content=f"{avg_similarity:.3f}",
                    description="Higher = more cohesive",
                    key="metric_avg_sim"
                )
            with col_m2:
                ui.metric_card(
                    title="Embedding Variance",
                    content=f"{embedding_variance:.4f}",
                    description="Spread in vector space",
                    key="metric_variance"
                )
            with col_m3:
                ui.metric_card(
                    title="Total Chunks",
                    content=len(chunks),
                    description="analyzed",
                    key="metric_chunks_analyzed"
                )
            
            # Similarity Distribution
            st.write("")
            with st.container(border=True):
                st.markdown("##### Similarity Distribution")
                st.caption("How similar are chunks to each other? Peaks near 1.0 indicate high cohesion.")
                
                fig_hist = create_similarity_histogram(similarity_matrix)
                st.plotly_chart(fig_hist, width='stretch')
            
            # Outlier Analysis
            st.write("")
            st.markdown("##### Least Similar Chunks (Potential Outliers)")
            st.caption("These chunks are semantically distant from the rest. They may represent unique concepts, noise, or structural elements like headers.")
            
            if outliers:
                # Prepare data for chunk viewer component
                outlier_chunks = [o['chunk'] for o in outliers]
                outlier_display_data = prepare_chunk_display_data(
                    chunks=outlier_chunks,
                    source_text=None,
                    calculate_overlap=False,
                )
                
                # Add similarity score as custom badge
                custom_badges = [
                    {
                        "label": "Sim",
                        "value": f"{o['avg_similarity']:.3f}",
                        "color": "#fef3c7"
                    }
                    for o in outliers
                ]
                
                # Render using the reusable component in card mode
                render_chunk_cards(
                    chunk_display_data=outlier_display_data,
                    custom_badges=custom_badges,
                    show_overlap=False,
                    display_mode="card",
                )
            else:
                st.info("No outliers detected.")
