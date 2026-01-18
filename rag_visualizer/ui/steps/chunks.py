import hashlib
import html
import json

import streamlit as st
import streamlit_shadcn_ui as ui

from rag_visualizer.services.chunking import get_chunks
from rag_visualizer.services.storage import get_storage_dir, load_document
from rag_visualizer.utils.parsers import parse_document

def _find_text_overlap(prev_text: str, curr_text: str) -> int:
    """Find max length where prev_text suffix matches curr_text prefix."""
    if not prev_text or not curr_text:
        return 0
    max_len = min(len(prev_text), len(curr_text))
    for size in range(max_len, 0, -1):
        if prev_text.endswith(curr_text[:size]):
            return size
    return 0


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _normalize_for_overlap(text: str) -> str:
    """Normalize text for overlap comparison (whitespace-insensitive)."""
    return " ".join(text.split()).strip()


def _normalized_prefix_length(text: str, normalized_len: int) -> int:
    """Map normalized prefix length back to original text prefix length."""
    if normalized_len <= 0:
        return 0
    normalized_count = 0
    prefix_len = 0
    in_space = False
    for i, ch in enumerate(text):
        prefix_len = i + 1
        if ch.isspace():
            if not in_space and normalized_count > 0:
                normalized_count += 1
            in_space = True
        else:
            normalized_count += 1
            in_space = False
        if normalized_count >= normalized_len:
            break
    return prefix_len


def _extract_docling_metadata(metadata: dict) -> dict:
    """Extract Docling-specific metadata from chunk metadata.

    Returns a dict with:
        - section_hierarchy: List of section headers
        - element_type: Type of document element (paragraph, header, code, etc.)
        - strategy: Chunking strategy used (Hierarchical or Hybrid)
        - size: Character count
        - page_number: Source page (if available)
    """
    result = {}

    # Section hierarchy (Docling format)
    if "section_hierarchy" in metadata:
        result["section_hierarchy"] = metadata["section_hierarchy"]
    # Legacy LangChain format fallback
    elif any(f"Header {i}" in metadata for i in range(1, 4)):
        headers = []
        for i in range(1, 4):
            key = f"Header {i}"
            if key in metadata and metadata[key]:
                headers.append(metadata[key])
        if headers:
            result["section_hierarchy"] = headers

    # Element type
    if "element_type" in metadata:
        result["element_type"] = metadata["element_type"]

    # Chunking strategy
    if "strategy" in metadata:
        result["strategy"] = metadata["strategy"]

    # Size
    if "size" in metadata:
        result["size"] = metadata["size"]

    # Page number (if available from Docling)
    if "page_number" in metadata:
        result["page_number"] = metadata["page_number"]
    elif "page" in metadata:
        result["page_number"] = metadata["page"]

    return result


def _contextualize_chunk(chunk_text: str, metadata: dict) -> str:
    """Create contextualized text for embedding by prepending section context.

    This mimics Docling's chunker.contextualize() method, which produces
    metadata-enriched text suitable for embedding models.

    Args:
        chunk_text: The raw chunk text
        metadata: Docling metadata extracted via _extract_docling_metadata

    Returns:
        Contextualized text with section headers prepended
    """
    context_parts = []

    # Add section hierarchy as context
    if "section_hierarchy" in metadata:
        hierarchy = metadata["section_hierarchy"]
        if hierarchy:
            context_parts.append(" > ".join(hierarchy))

    # Add element type indicator for non-paragraph content
    element_type = metadata.get("element_type", "")
    if element_type and element_type not in ("paragraph", "text", "merged"):
        type_label = element_type.replace("_", " ").title()
        context_parts.append(f"[{type_label}]")

    if context_parts:
        context_prefix = " | ".join(context_parts)
        return f"{context_prefix}\n\n{chunk_text}"

    return chunk_text


def _format_element_type(element_type: str) -> tuple[str, str]:
    """Format element type for display, returning (label, color).

    Returns:
        Tuple of (display_label, background_color)
    """
    type_styles = {
        "paragraph": ("Para", "#e0e7ff"),
        "text": ("Text", "#e0e7ff"),
        "header_1": ("H1", "#fce7f3"),
        "header_2": ("H2", "#fce7f3"),
        "header_3": ("H3", "#fce7f3"),
        "header_4": ("H4", "#fce7f3"),
        "header_5": ("H5", "#fce7f3"),
        "header_6": ("H6", "#fce7f3"),
        "list_item": ("List", "#d1fae5"),
        "code": ("Code", "#fef3c7"),
        "table": ("Table", "#e0f2fe"),
        "figure": ("Figure", "#f3e8ff"),
        "merged": ("Merged", "#f1f5f9"),
    }
    if element_type in type_styles:
        return type_styles[element_type]
    return (element_type.replace("_", " ").title(), "#f1f5f9")


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
            prev_chunk = sorted_chunks[i - 1]
            calc_overlap = prev_chunk.end_index - chunk.start_index
            if calc_overlap > 0:
                index_overlap_len = min(calc_overlap, len(chunk.text))
            else:
                index_overlap_len = 0

            if index_overlap_len >= len(chunk.text):
                index_overlap_len = 0

            text_overlap_len = _find_text_overlap(prev_chunk.text, chunk.text)
            source_overlap_text = ""
            normalized_prefix_match = False
            if index_overlap_len > 0:
                source_overlap_text = source_text[
                    chunk.start_index : chunk.start_index + index_overlap_len
                ]
                normalized_prefix_match = _normalize_whitespace(
                    source_overlap_text
                ) == _normalize_whitespace(chunk.text[:index_overlap_len])

            use_index_overlap = index_overlap_len > 0 and (
                prev_chunk.text.endswith(chunk.text[:index_overlap_len])
                or normalized_prefix_match
            )
            overlap_len = index_overlap_len if use_index_overlap else text_overlap_len

            if overlap_len > 0:
                overlap_text = chunk.text[:overlap_len]
                main_text = chunk.text[overlap_len:]
                chunks_with_overlap += 1
                total_overlap_len += overlap_len

            if index_overlap_len > 0 and not use_index_overlap:
                normalized_prev = _normalize_for_overlap(prev_chunk.text)
                normalized_curr = _normalize_for_overlap(chunk.text)
                normalized_overlap_len = _find_text_overlap(
                    normalized_prev,
                    normalized_curr,
                )
                if normalized_overlap_len > 0:
                    normalized_mapped_len = _normalized_prefix_length(
                        chunk.text, normalized_overlap_len
                    )
                    if normalized_mapped_len > overlap_len:
                        overlap_len = normalized_mapped_len
                        overlap_text = chunk.text[:overlap_len]
                        main_text = chunk.text[overlap_len:]
                        chunks_with_overlap += 1
                        total_overlap_len += overlap_len
            elif index_overlap_len == 0 and text_overlap_len == 0:
                normalized_prev = _normalize_for_overlap(prev_chunk.text)
                normalized_curr = _normalize_for_overlap(chunk.text)
                normalized_overlap_len = _find_text_overlap(
                    normalized_prev,
                    normalized_curr,
                )
                if normalized_overlap_len > 0:
                    normalized_mapped_len = _normalized_prefix_length(
                        chunk.text, normalized_overlap_len
                    )
                    if 0 < normalized_mapped_len < len(chunk.text):
                        overlap_len = normalized_mapped_len
                        overlap_text = chunk.text[:overlap_len]
                        main_text = chunk.text[overlap_len:]
                        chunks_with_overlap += 1
                        total_overlap_len += overlap_len
        
        docling_meta = _extract_docling_metadata(chunk.metadata)
        chunk_display_data.append({
            "chunk": chunk,
            "overlap_text": overlap_text,
            "main_text": main_text,
            "len": len(chunk.text),
            "docling_metadata": docling_meta,
            "contextualized_text": _contextualize_chunk(chunk.text, docling_meta),
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

    # Build continuous HTML with expandable chunks
    chunks_html_parts = []
    chunks_html_parts.append(
        '<div style="font-family: -apple-system, BlinkMacSystemFont, '
        'sans-serif; line-height: 1.6; color: #111;">'
    )
    # Add CSS for details/summary styling
    chunks_html_parts.append('''
        <style>
            .chunk-details summary { cursor: pointer; list-style: none; }
            .chunk-details summary::-webkit-details-marker { display: none; }
            .chunk-details[open] .chunk-expand-icon { transform: rotate(180deg); }
            .chunk-details .chunk-expand-icon {
                transition: transform 0.15s ease;
                display: inline-block;
                color: #9ca3af;
            }
            .chunk-expanded-content {
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid rgba(0,0,0,0.08);
            }
            .chunk-context-box {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 10px;
                font-family: monospace;
                font-size: 0.8rem;
                white-space: pre-wrap;
                word-break: break-word;
            }
            .chunk-context-label {
                font-size: 0.7rem;
                color: #6b7280;
                margin-bottom: 6px;
                font-weight: 500;
            }
        </style>
    ''')

    for i, data in enumerate(chunk_display_data):
        color_idx = i % len(colors)
        bg_color = colors[color_idx]
        meta = data["docling_metadata"]

        # Calculate extra context (prefix added during contextualization)
        chunk_text = data["chunk"].text
        contextualized = data["contextualized_text"]
        extra_context = ""
        if contextualized != chunk_text and contextualized.endswith(chunk_text):
            extra_context = contextualized[: len(contextualized) - len(chunk_text)].strip()

        # Wrap each chunk in a details element for expand/collapse
        chunks_html_parts.append(
            f'<details class="chunk-details" style="background-color: {bg_color}; '
            f'padding: 4px 12px; margin: 0; position: relative;">'
        )

        # Summary (always visible part - shows full chunk text)
        chunks_html_parts.append('<summary style="outline: none; display: block;">')

        # Metadata badges row (floating right)
        chunks_html_parts.append(
            '<span style="float: right; display: flex; gap: 4px; align-items: center;">'
        )

        # Page number badge (if available)
        if "page_number" in meta:
            chunks_html_parts.append(
                f'<span style="background: #dbeafe; color: #1e40af; '
                f'font-size: 0.65rem; padding: 1px 5px; border-radius: 8px; '
                f'user-select: none;">p.{meta["page_number"]}</span>'
            )

        # Element type badge
        if "element_type" in meta:
            type_label, type_color = _format_element_type(meta["element_type"])
            chunks_html_parts.append(
                f'<span style="background: {type_color}; color: #374151; '
                f'font-size: 0.65rem; padding: 1px 5px; border-radius: 8px; '
                f'user-select: none;">{type_label}</span>'
            )

        # Char count badge
        chunks_html_parts.append(
            f'<span style="background: rgba(0,0,0,0.1); '
            f'color: #555; font-size: 0.65rem; padding: 1px 5px; '
            f'border-radius: 8px; user-select: none;">'
            f'{data["len"]} chars</span>'
        )

        chunks_html_parts.append('</span>')

        # Section hierarchy breadcrumb (when available)
        if "section_hierarchy" in meta and meta["section_hierarchy"]:
            hierarchy = meta["section_hierarchy"]
            breadcrumb = " > ".join(html.escape(h) for h in hierarchy)
            chunks_html_parts.append(
                f'<div style="font-size: 0.75rem; color: #6b7280; '
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

        # Render full main text
        chunks_html_parts.append(html.escape(data["main_text"]))

        # Expand icon at bottom of chunk
        chunks_html_parts.append(
            '<div style="text-align: center; margin-top: 6px;">'
            '<span class="chunk-expand-icon" style="font-size: 0.6rem;" '
            'title="Click to show added context">▼</span></div>'
        )

        chunks_html_parts.append('</summary>')

        # Expanded content (shows ONLY the extra context prefix)
        chunks_html_parts.append('<div class="chunk-expanded-content">')

        if extra_context:
            chunks_html_parts.append(
                '<div class="chunk-context-label">Added Context (prepended for embedding)</div>'
            )
            chunks_html_parts.append(
                f'<div class="chunk-context-box">{html.escape(extra_context)}</div>'
            )
        else:
            chunks_html_parts.append(
                '<div style="font-size: 0.8rem; color: #9ca3af; font-style: italic;">'
                'No extra context added to this chunk.</div>'
            )

        chunks_html_parts.append('</div>')  # End expanded content
        chunks_html_parts.append('</details>')

    chunks_html_parts.append('</div>')
    chunks_html = "".join(chunks_html_parts)

    if not chunks:
        chunks_html = (
            '<div style="color: #666; font-style: italic;">No chunks generated.</div>'
        )

    st.write("")
    with st.container(border=True):
        st.markdown(chunks_html, unsafe_allow_html=True)
