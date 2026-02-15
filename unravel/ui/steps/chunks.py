import streamlit as st
import streamlit_shadcn_ui as ui

from unravel.services.chunking import get_chunks
from unravel.services.storage import load_document, save_rag_config
from unravel.ui.constants import WidgetKeys
from unravel.ui.components.chunk_viewer import (
    prepare_chunk_display_data,
    render_chunk_cards,
)
from unravel.ui.components.chunking_config import render_chunking_configuration
from unravel.utils.cache import (
    get_parsed_text_key,
    load_parsed_text,
    save_parsed_text,
)
from unravel.utils.parsers import parse_document


def render_chunks_step() -> None:
    st.markdown("## Document Processing & Text Splitting")
    st.caption("Configure document parsing and text splitting")

    # Render chunking configuration
   
    with st.expander("Configuration", expanded=False):
        new_parsing_params, new_chunking_params, has_changes = render_chunking_configuration()

       
        if st.button(
            "Apply Configuration",
            type="primary",
            disabled=not has_changes,
            key=WidgetKeys.CHUNKS_APPLY_BTN,
        ):
            # Update session state
            st.session_state.parsing_params = new_parsing_params
            st.session_state.chunking_params = new_chunking_params
            st.session_state.applied_parsing_params = new_parsing_params.copy()
            st.session_state.applied_chunking_params = new_chunking_params.copy()

            # Invalidate downstream caches
            for key in [
                "chunks",
                "last_embeddings_result",
                "search_results",
                "bm25_index_data",
            ]:
                if key in st.session_state:
                    del st.session_state[key]

            # Persist to disk
            save_rag_config(
                {
                    "doc_name": st.session_state.doc_name,
                    "parsing_params": new_parsing_params,
                    "chunking_params": new_chunking_params,
                    "embedding_model_name": st.session_state.embedding_model_name,
                }
            )

            st.success("Configuration applied successfully!")
            st.rerun()

    # Read configuration from session state
    selected_doc = st.session_state.get("doc_name")
    chunking_params = st.session_state.get(
        "applied_chunking_params",
        st.session_state.get(
            "chunking_params",
            {
                "provider": "Docling",
                "splitter": "HybridChunker",
                "max_tokens": 512,
                "chunk_overlap": 50,
                "tokenizer": "cl100k_base",
            },
        ),
    )
    # Get current parsing params (from sidebar) and applied params (what was used)
    current_parsing_params = st.session_state.get("parsing_params", {})
    applied_parsing_params = st.session_state.get(
        "applied_parsing_params", current_parsing_params.copy()
    )

    provider = chunking_params.get("provider", "Docling")
    splitter = chunking_params.get("splitter", "HybridChunker")
    max_tokens = chunking_params.get("max_tokens", 512)
    overlap_size = chunking_params.get("chunk_overlap", 50)

    # Check if document is selected
    if not selected_doc:
        st.info(
            "No document selected. Upload a file in the Upload step or "
            "select a document in the sidebar (RAG Config tab)."
        )
        if st.button("Go to Upload Step", key=WidgetKeys.CHUNKS_GOTO_UPLOAD, type="primary"):
            st.session_state.current_step = "upload"
            st.rerun()
        return

    # Check if parsing settings have changed (compare current vs applied)
    parsing_settings_changed = current_parsing_params != applied_parsing_params

    # Always use applied params for lookup (what was actually used for the current parsed text)
    # This ensures we show the existing parsed document even after "Save & Apply"
    params_for_lookup = applied_parsing_params
    parsed_text_key = get_parsed_text_key(selected_doc, params_for_lookup)

    # Try to load parsed text from session state first, then from persistent storage
    # Use applied params to get the currently displayed parsed text
    source_text = st.session_state.get(parsed_text_key, "")
    if not source_text:
        # Try loading from persistent storage using applied params
        source_text = load_parsed_text(selected_doc, params_for_lookup) or ""
        if source_text:
            # Restore to session state
            st.session_state[parsed_text_key] = source_text

    doc_display_name = selected_doc

    # Show parse button if no parsed text OR if parsing settings have changed
    needs_parsing = not source_text or parsing_settings_changed
    if needs_parsing:
       
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            button_text = (
                "Reparse Document" if parsing_settings_changed and source_text else "Parse Document"
            )
            if st.button(button_text, key=WidgetKeys.CHUNKS_PARSE_BTN, type="primary"):
                with st.spinner("Parsing document..."):
                    try:
                        content = load_document(selected_doc)
                        if content:
                            # Use current parsing params for parsing
                            parsed_text, _, _ = parse_document(
                                selected_doc, content, current_parsing_params
                            )
                            # Update applied params to match current
                            # (so they're in sync after reparse)
                            st.session_state["applied_parsing_params"] = (
                                current_parsing_params.copy()
                            )
                            new_applied_params = current_parsing_params.copy()
                            new_parsed_text_key = get_parsed_text_key(
                                selected_doc, new_applied_params
                            )
                            # Cache parsed text in session state
                            st.session_state[new_parsed_text_key] = parsed_text
                            # Save to persistent storage
                            save_parsed_text(selected_doc, new_applied_params, parsed_text)
                            st.rerun()
                        else:
                            st.error(f"Failed to load document: {selected_doc}")
                    except Exception as e:
                        st.error(f"Error parsing document: {str(e)}")

        if parsing_settings_changed and source_text:
            st.info(
                "Parsing settings have changed. Click the button above to reparse the document with new settings."
            )
        else:
            st.info("Click the button above to parse the document and generate chunks.")

        # If we have no parsed text, return early until the user parses.
        # Otherwise, continue to show the existing parsed text even if settings changed.
        if not source_text:
            return

    # Generate Chunks (only if we have parsed text)
    output_format = applied_parsing_params.get("output_format", "markdown")
    if source_text:
        # Extract splitter-specific params (exclude provider/splitter keys)
        splitter_params = {
            k: v for k, v in chunking_params.items() if k not in ["provider", "splitter"]
        }
        chunks = get_chunks(
            provider=provider,
            splitter=splitter,
            text=source_text,
            output_format=output_format,
            **splitter_params,
        )

        # Save chunks to session state (config already set in sidebar)
        st.session_state["chunks"] = chunks
    else:
        chunks = []
        st.session_state["chunks"] = []

    # Main Visualization
   
    st.markdown("### Generated Chunks")
    st.caption(f"Source: {doc_display_name} | Total characters: {len(source_text)}")

    # Prepare chunk display data (includes overlap calculation)
    chunk_display_data = prepare_chunk_display_data(
        chunks=chunks,
        source_text=source_text,
        calculate_overlap=True,
    )

    view_mode = ui.tabs(
        options=["Visual View", "Raw JSON"],
        default_value="Visual View",
        key=WidgetKeys.CHUNKS_VIEW_TAB,
    )

    # Render based on selected view
   
    if view_mode == "Visual View":
        with st.container(border=True):
            render_chunk_cards(
                chunk_display_data=chunk_display_data,
                show_overlap=True,
                display_mode="continuous",
                render_format=output_format,
            )
    else:  # Raw JSON
        # Convert chunks to serializable format for JSON display
        chunks_json = []
        for chunk in chunks:
            chunk_dict = {
                "text": chunk.text,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
                "metadata": chunk.metadata,
            }
            chunks_json.append(chunk_dict)

        with st.container(border=True):
            st.json(chunks_json, expanded=True)
