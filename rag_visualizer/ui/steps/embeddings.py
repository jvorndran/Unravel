import numpy as np
import pandas as pd
import streamlit as st
import streamlit_shadcn_ui as ui

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
from rag_visualizer.utils.parsers import parse_document
from rag_visualizer.utils.visualization import create_embedding_plot, reduce_dimensions


def _extract_header_metadata(metadata: dict) -> dict:
    """Extract header metadata from chunk metadata."""
    headers = {}
    for key in ["Header 1", "Header 2", "Header 3"]:
        if key in metadata and metadata[key]:
            headers[key] = metadata[key]
    return headers


@st.cache_data(show_spinner="Generating embeddings...")
def generate_embeddings(
    texts: list[str], model_name: str
) -> tuple[np.ndarray, int]:
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
        st.info("👋 No document selected. Upload a file in the **Upload** step or select a document in the sidebar (RAG Config tab).")
        if ui.button("Go to Upload Step", key="goto_upload_embeddings"):
            st.session_state.current_step = "upload"
            st.rerun()
        return

    # Display current configuration
    st.write("")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Embedding Model:** {selected_model}")
        with c2:
            st.markdown(f"**Document:** {selected_doc}")
        
        st.caption("💡 Configure these settings in the sidebar (RAG Config tab)")

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
                
                # Persist session state to disk for refresh resilience
                save_session_state({
                    "doc_name": st.session_state.get("doc_name"),
                    "embedding_model_name": selected_model,
                    "chunking_params": st.session_state.get("chunking_params"),
                    "chunks": chunks,
                    "last_embeddings_result": st.session_state["last_embeddings_result"],
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
    active_tab = ui.tabs(options=["Visual Explorer", "Data Inspector"], default_value="Visual Explorer", key="embed_main_tabs")

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
                submit_button = st.form_submit_button("Search", type="primary", use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)
            
        # Nearest Neighbors Section
        st.write("")
        st.markdown("##### Nearest Neighbors")
        if neighbors:
            for i, res in enumerate(neighbors):
                with st.container(border=True):
                    st.markdown(f"**{i+1}. Score: {res.score:.4f}**")
                    # Show header breadcrumb if available
                    header_metadata = _extract_header_metadata(res.metadata) if res.metadata else {}
                    if header_metadata:
                        header_parts = [header_metadata[level] for level in ["Header 1", "Header 2", "Header 3"] if level in header_metadata]
                        if header_parts:
                            breadcrumb = " > ".join(header_parts)
                            st.caption(f"{breadcrumb}")
                    st.caption(res.text)
        else:
            st.info("Enter a query above to see the most similar chunks here.")

    # --- TAB 2: Data Inspector ---
    elif active_tab == "Data Inspector":
        st.write("")
        st.markdown("#### Under the Hood")
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("**Chunk Data**")
            # Create a clean dataframe for display
            chunk_data = []
            for i, c in enumerate(chunks):
                chunk_data.append({
                    "Index": i,
                    "Length": len(c.text),
                    "Preview": c.text[:50] + "..."
                })
            # Using st.dataframe as it's better for scrolling through data
            st.dataframe(pd.DataFrame(chunk_data), use_container_width=True, hide_index=True)
            
        with col_d2:
            st.markdown("**Vector Representations**")
            vectors = vector_store.get_all_embeddings()
            st.write(f"Shape: {len(vectors)} chunks × {len(vectors[0])} dimensions")
            
            # Show a few dimensions of the first few vectors
            vec_data = []
            for i in range(min(10, len(vectors))):
                vec_preview = vectors[i][:5].tolist() # First 5 dims
                vec_data.append({
                    "Chunk Index": i,
                    "Vector Preview (First 5 dims)": [round(v, 4) for v in vec_preview]
                })
            st.dataframe(pd.DataFrame(vec_data), use_container_width=True, hide_index=True)
