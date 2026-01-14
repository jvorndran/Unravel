import hashlib
import html
import json

import streamlit as st
import streamlit_shadcn_ui as ui

try:
    import markdown as markdown_lib
except ImportError:
    markdown_lib = None

from rag_visualizer.services.chunking import get_chunks
from rag_visualizer.services.storage import get_storage_dir, load_document
from rag_visualizer.utils.parsers import parse_document


def _extract_header_metadata(metadata: dict) -> dict:
    """Extract header metadata from chunk metadata."""
    headers = {}
    for key in ["Header 1", "Header 2", "Header 3"]:
        if key in metadata and metadata[key]:
            headers[key] = metadata[key]
    return headers


def _get_parsed_text_key(doc_name: str, parsing_params: dict) -> str:
    """Get a stable key for parsed text storage."""
    # Use JSON serialization with sorted keys for stability
    params_str = json.dumps(parsing_params, sort_keys=True)
    return f"parsed_text_{doc_name}_{params_str}"


def _get_parsed_text_filename(doc_name: str, parsing_params: dict) -> str:
    """Get a safe filename for parsed text storage."""
    key = _get_parsed_text_key(doc_name, parsing_params)
    # Use hash for filename to avoid filesystem issues with special chars
    key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
    # Basic sanitization: remove path separators and use hash
    safe_doc_name = doc_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"{safe_doc_name}_{key_hash}.txt"


def _save_parsed_text(doc_name: str, parsing_params: dict, parsed_text: str) -> None:
    """Save parsed text to persistent storage."""
    storage_dir = get_storage_dir()
    parsed_dir = storage_dir / "parsed_texts"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    
    filename = _get_parsed_text_filename(doc_name, parsing_params)
    file_path = parsed_dir / filename
    
    file_path.write_text(parsed_text, encoding="utf-8")


def _load_parsed_text(doc_name: str, parsing_params: dict) -> str | None:
    """Load parsed text from persistent storage."""
    storage_dir = get_storage_dir()
    parsed_dir = storage_dir / "parsed_texts"
    
    filename = _get_parsed_text_filename(doc_name, parsing_params)
    file_path = parsed_dir / filename
    
    if file_path.exists():
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception:
            return None
    return None


def render_chunks_step() -> None:
    st.markdown("## Text Splitting")
    st.caption(
        "Visualize how documents are split into meaningful chunks "
        "while preserving semantic coherence"
    )

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
    # Use applied params for display/chunking, but check current for changes
    parsing_params = applied_parsing_params
    use_markdown_output = parsing_params.get("output_format") == "markdown"

    provider = chunking_params.get("provider", "Docling")
    splitter = chunking_params.get("splitter", "HybridChunker")
    max_tokens = chunking_params.get("max_tokens", 512)
    overlap_size = chunking_params.get("chunk_overlap", 50)

    # Check if document is selected
    if not selected_doc:
        st.info(
            "👋 No document selected. Upload a file in the **Upload** step or "
            "select a document in the sidebar (RAG Config tab)."
        )
        if ui.button("Go to Upload Step", key="goto_upload_chunks"):
            st.session_state.current_step = "upload"
            st.rerun()
        return

    # Display current configuration
    st.write("")
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"**Document:** {selected_doc}")
            # Get display name from splitter info
            from rag_visualizer.services.chunking import get_provider_splitters
            splitter_infos = get_provider_splitters(provider)
            splitter_display = next(
                (info.display_name for info in splitter_infos if info.name == splitter),
                splitter,
            )
            st.markdown(f"**Strategy:** {splitter_display}")

        with col2:
            st.markdown(f"**Max Tokens:** {max_tokens}")

        with col3:
            st.markdown(f"**Overlap:** {overlap_size}")

        st.caption(f"Using {provider} | Configure in sidebar (RAG Config tab)")

    # Check if parsing settings have changed (compare current vs applied)
    parsing_settings_changed = current_parsing_params != applied_parsing_params
    
    # Always use applied params for lookup (what was actually used for the current parsed text)
    # This ensures we show the existing parsed document even after "Save & Apply"
    params_for_lookup = applied_parsing_params
    parsed_text_key = _get_parsed_text_key(selected_doc, params_for_lookup)
    
    # Try to load parsed text from session state first, then from persistent storage
    # Use applied params to get the currently displayed parsed text
    source_text = st.session_state.get(parsed_text_key, "")
    if not source_text:
        # Try loading from persistent storage using applied params
        source_text = _load_parsed_text(selected_doc, params_for_lookup) or ""
        if source_text:
            # Restore to session state
            st.session_state[parsed_text_key] = source_text
    
    doc_display_name = selected_doc
    
    # Show parse button if no parsed text OR if parsing settings have changed
    needs_parsing = not source_text or parsing_settings_changed
    if needs_parsing:
        st.write("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            button_text = (
                "Reparse Document"
                if parsing_settings_changed and source_text
                else "Parse Document"
            )
            if st.button(button_text, key="parse_document_btn", type="primary"):
                with st.spinner("Parsing document..."):
                    try:
                        content = load_document(selected_doc)
                        if content:
                            # Use current parsing params (from sidebar) for parsing
                            parsed_text, _, _ = parse_document(
                                selected_doc, content, current_parsing_params
                            )
                            # Update applied params to match current
                            # (so they're in sync after reparse)
                            st.session_state["applied_parsing_params"] = (
                                current_parsing_params.copy()
                            )
                            new_applied_params = current_parsing_params.copy()
                            new_parsed_text_key = _get_parsed_text_key(
                                selected_doc, new_applied_params
                            )
                            # Cache parsed text in session state
                            st.session_state[new_parsed_text_key] = parsed_text
                            # Save to persistent storage
                            _save_parsed_text(selected_doc, new_applied_params, parsed_text)
                            st.rerun()
                        else:
                            st.error(f"Failed to load document: {selected_doc}")
                    except Exception as e:
                        st.error(f"Error parsing document: {str(e)}")

        if parsing_settings_changed and source_text:
            st.info("Parsing settings have changed. Click the button above to reparse the document with new settings.")
        else:
            st.info("Click the button above to parse the document and generate chunks.")
        
        # If we have no parsed text, return early until the user parses.
        # Otherwise, continue to show the existing parsed text even if settings changed.
        if not source_text:
            return
    
    # Generate Chunks (only if we have parsed text)
    if source_text:
        # Extract splitter-specific params (exclude provider/splitter keys)
        splitter_params = {k: v for k, v in chunking_params.items()
                          if k not in ["provider", "splitter"]}
        chunks = get_chunks(
            provider=provider,
            splitter=splitter,
            text=source_text,
            **splitter_params
        )

        # Save chunks to session state (config already set in sidebar)
        st.session_state["chunks"] = chunks
    else:
        chunks = []
        st.session_state["chunks"] = []

    # Main Visualization
    st.write("")
    st.markdown("### Generated Chunks")
    st.caption(f"Source: {doc_display_name} | Total characters: {len(source_text)}")

    # Calculate real stats based on generated chunks
    num_chunks = len(chunks)
    avg_size = int(sum(len(c.text) for c in chunks) / num_chunks) if num_chunks > 0 else 0
    
    chunks_with_overlap = 0
    total_overlap_len = 0
    
    sorted_chunks = sorted(chunks, key=lambda c: c.start_index)
    chunk_display_data = []
    
    for i, chunk in enumerate(sorted_chunks):
        overlap_text = ""
        main_text = chunk.text
        overlap_len = 0
        
        if i > 0:
            prev_chunk = sorted_chunks[i-1]
            calc_overlap = prev_chunk.end_index - chunk.start_index
            if calc_overlap > 0:
                overlap_len = calc_overlap
                overlap_text = chunk.text[:overlap_len]
                main_text = chunk.text[overlap_len:]
                chunks_with_overlap += 1
                total_overlap_len += overlap_len
        
        chunk_display_data.append({
            "chunk": chunk,
            "overlap_text": overlap_text,
            "main_text": main_text,
            "len": len(chunk.text),
            "header_metadata": _extract_header_metadata(chunk.metadata),
        })

    avg_overlap_size = (
        int(total_overlap_len / chunks_with_overlap)
        if chunks_with_overlap > 0
        else 0
    )

    # Stats bar using metric cards
    cols = st.columns(4)
    with cols[0]:
        ui.metric_card(
            title="Chunks", content=num_chunks, description="total", key="stat_chunks"
        )
    with cols[1]:
        ui.metric_card(
            title="Avg. Size",
            content=f"{avg_size}",
            description="chars",
            key="stat_size",
        )
    with cols[2]:
        ui.metric_card(
            title="With Overlap",
            content=chunks_with_overlap,
            description="chunks",
            key="stat_overlap_cnt",
        )
    with cols[3]:
        ui.metric_card(
            title="Avg. Overlap",
            content=f"{avg_overlap_size}",
            description="chars",
            key="stat_overlap_sz",
        )

    # Palette for chunk backgrounds (pastel colors)
    colors = [
        "#fef2f2", # Reddish
        "#eff6ff", # Blueish
        "#f0fdf4", # Greenish
        "#faf5ff", # Purpleish
        "#fffbeb", # Yellowish
    ]

    # Build continuous HTML
    chunks_html_parts = []
    chunks_html_parts.append(
        '<div style="font-family: -apple-system, BlinkMacSystemFont, '
        'sans-serif; line-height: 1.6; color: #111;">'
    )

    def render_chunk_body(text: str) -> str:
        """Render chunk text as markdown when enabled, fallback to escaped HTML."""
        if use_markdown_output and markdown_lib:
            try:
                return markdown_lib.markdown(text, extensions=['tables'])
            except Exception:
                return html.escape(text)
        return html.escape(text)
    
    for i, data in enumerate(chunk_display_data):
        color_idx = i % len(colors)
        bg_color = colors[color_idx]
        
        chunks_html_parts.append(
            f'<div style="background-color: {bg_color}; padding: 4px 12px; '
            f'margin: 0; position: relative;" title="Chunk {i+1}">'
        )
        
        # Char count badge (floating right)
        chunks_html_parts.append(
            f'<span style="float: right; background: rgba(0,0,0,0.1); '
            f'color: #555; font-size: 0.7rem; padding: 1px 6px; '
            f'border-radius: 10px; margin-left: 8px; user-select: none;">'
            f'{data["len"]}</span>'
        )

        # Header breadcrumb (when available)
        if data["header_metadata"]:
            header_parts = []
            for level in ["Header 1", "Header 2", "Header 3"]:
                if level in data["header_metadata"]:
                    header_parts.append(html.escape(data["header_metadata"][level]))
            if header_parts:
                breadcrumb = " > ".join(header_parts)
                chunks_html_parts.append(
                    f'<div style="font-size: 0.8rem; color: #6b7280; '
                    f'margin-bottom: 4px; font-weight: 500;">{breadcrumb}</div>'
                )

        # Render overlap text inline with dashed box style
        if data["overlap_text"]:
            chunks_html_parts.append(
                '<span style="display: inline-block; border: 1px dashed #f97316; '
                'background-color: rgba(255, 247, 237, 0.8); color: #c2410c; '
                'border-radius: 4px; padding: 0 4px; margin-right: 4px;">'
            )
            chunks_html_parts.append(
                '<span style="user-select: none; margin-right: 2px; '
                'font-weight: bold;">↪</span>'
            )
            chunks_html_parts.append(html.escape(data["overlap_text"]))
            chunks_html_parts.append('</span>')
        
        # Render main text
        chunks_html_parts.append(render_chunk_body(data["main_text"]))
        
        chunks_html_parts.append('</div>')
        
    chunks_html_parts.append('</div>')
    chunks_html = "".join(chunks_html_parts)

    if not chunks:
        chunks_html = (
            '<div style="color: #666; font-style: italic;">No chunks generated.</div>'
        )

    st.write("")
    with st.container(border=True):
        st.markdown(chunks_html, unsafe_allow_html=True)
