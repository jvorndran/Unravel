"""Reusable chunk visualization component for displaying document chunks."""

import html
from typing import Any

import markdown  # type: ignore[import-untyped]
import streamlit as st


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


def _extract_docling_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
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


def _normalize_element_type(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, str) and item:
                return item
        return ""
    return str(value)


def _contextualize_chunk(chunk_text: str, metadata: dict[str, Any]) -> str:
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
    element_type = _normalize_element_type(metadata.get("element_type", ""))
    if element_type and element_type not in ("paragraph", "text", "merged"):
        type_label = element_type.replace("_", " ").title()
        context_parts.append(f"[{type_label}]")

    if context_parts:
        context_prefix = " | ".join(context_parts)
        return f"{context_prefix}\n\n{chunk_text}"

    return chunk_text


def _format_element_type(element_type: Any) -> tuple[str, str]:
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
    normalized_type = _normalize_element_type(element_type)
    if normalized_type in type_styles:
        return type_styles[normalized_type]
    if normalized_type:
        return (normalized_type.replace("_", " ").title(), "#f1f5f9")
    return ("Unknown", "#f1f5f9")


def prepare_chunk_display_data(
    chunks: list[Any],
    source_text: str | None = None,
    calculate_overlap: bool = False,
) -> list[dict[str, Any]]:
    """Prepare chunk data for visualization with optional overlap calculation.

    Args:
        chunks: List of chunk objects with .text, .start_index, .end_index, .metadata
        source_text: Full source text (required if calculate_overlap=True)
        calculate_overlap: Whether to calculate sequential overlap between chunks

    Returns:
        List of dicts with keys: chunk, overlap_text, main_text, len, docling_metadata
    """
    if not chunks:
        return []

    sorted_chunks = sorted(chunks, key=lambda c: c.start_index)
    chunk_display_data = []

    for i, chunk in enumerate(sorted_chunks):
        overlap_text = ""
        main_text = chunk.text
        overlap_len = 0

        if calculate_overlap and i > 0 and source_text:
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

        docling_meta = _extract_docling_metadata(chunk.metadata)
        chunk_display_data.append(
            {
                "chunk": chunk,
                "overlap_text": overlap_text,
                "main_text": main_text,
                "len": len(chunk.text),
                "docling_metadata": docling_meta,
            }
        )

    return chunk_display_data


def render_chunk_cards(
    chunk_display_data: list[dict[str, Any]],
    custom_badges: list[dict[str, Any]] | None = None,
    show_overlap: bool = True,
    display_mode: str = "continuous",
) -> None:
    """Render chunk cards as HTML with expandable metadata.

    Args:
        chunk_display_data: List of dicts from prepare_chunk_display_data()
        custom_badges: Optional list of dicts with custom badge info per chunk.
                      Each dict can have keys like {"label": "Score", "value": "0.85", "color": "#..."}
        show_overlap: Whether to render overlap highlighting (default: True)
        display_mode: Display style - "continuous" (colored flow) or "card" (individual cards with borders)
    """
    if not chunk_display_data:
        st.markdown(
            '<div style="color: #666; font-style: italic;">No chunks to display.</div>',
            unsafe_allow_html=True,
        )
        return

    # Palette for chunk backgrounds (pastel colors) - used in continuous mode
    colors = [
        "#fef2f2",  # Reddish
        "#eff6ff",  # Blueish
        "#f0fdf4",  # Greenish
        "#faf5ff",  # Purpleish
        "#fffbeb",  # Yellowish
    ]

    # Build HTML with expandable chunks
    chunks_html_parts = []
    
    # Container style depends on display mode
    if display_mode == "card":
        # Card mode: Container with gap for spacing between cards
        chunks_html_parts.append(
            '<div style="font-family: -apple-system, BlinkMacSystemFont, '
            'sans-serif; line-height: 1.6; color: #111; display: flex; '
            'flex-direction: column; gap: 12px;">'
        )
    else:
        # Continuous mode: No gap, chunks flow together
        chunks_html_parts.append(
            '<div style="font-family: -apple-system, BlinkMacSystemFont, '
            'sans-serif; line-height: 1.6; color: #111;">'
        )
    # Add CSS for details/summary styling
    chunks_html_parts.append(
        """
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
            /* Rendered markdown styles within summary */
            .chunk-rendered {
                display: inline-block;
                width: 100%;
                margin-top: 4px;
            }
            .chunk-rendered p {
                margin: 0 0 8px 0;
            }
            .chunk-rendered h1, .chunk-rendered h2, .chunk-rendered h3, 
            .chunk-rendered h4, .chunk-rendered h5, .chunk-rendered h6 {
                margin: 12px 0 8px 0;
                font-size: 1em;
                font-weight: 600;
            }
            .chunk-rendered ul, .chunk-rendered ol {
                margin: 4px 0 8px 20px;
                padding: 0;
            }
            .chunk-rendered pre {
                background: #f1f5f9;
                padding: 4px 8px;
                border-radius: 4px;
                margin: 4px 0;
                white-space: pre-wrap;
            }
        </style>
    """
    )

    for i, data in enumerate(chunk_display_data):
        meta = data["docling_metadata"]

        # Style depends on display mode
        if display_mode == "card":
            # Card mode: White background with border and shadow
            chunk_style = (
                'background-color: #ffffff; '
                'border: 1px solid #e5e7eb; '
                'border-radius: 8px; '
                'padding: 12px 16px; '
                'box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); '
                'position: relative;'
            )
        else:
            # Continuous mode: Alternating colors, no border
            color_idx = i % len(colors)
            bg_color = colors[color_idx]
            chunk_style = (
                f'background-color: {bg_color}; '
                'padding: 4px 12px; '
                'margin: 0; '
                'position: relative;'
            )

        # Wrap each chunk in a details element for expand/collapse
        chunks_html_parts.append(
            f'<details class="chunk-details" style="{chunk_style}">'
        )

        # Summary (always visible part - shows full chunk text)
        chunks_html_parts.append('<summary style="outline: none; display: block;">')

        # Metadata badges row (floating right)
        chunks_html_parts.append(
            '<span style="float: right; display: flex; gap: 4px; align-items: center;">'
        )

        # Custom badges (e.g., similarity score)
        if custom_badges and i < len(custom_badges):
            badge_info = custom_badges[i]
            if badge_info:
                label = badge_info.get("label", "")
                value = badge_info.get("value", "")
                badge_color = badge_info.get("color", "#e0e7ff")
                if label and value:
                    chunks_html_parts.append(
                        f'<span style="background: {badge_color}; color: #374151; '
                        f'font-size: 0.65rem; padding: 1px 5px; border-radius: 8px; '
                        f'user-select: none;">{html.escape(label)}: {html.escape(str(value))}</span>'
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

        chunks_html_parts.append("</span>")

        # Section hierarchy breadcrumb (when available)
        if "section_hierarchy" in meta and meta["section_hierarchy"]:
            hierarchy = meta["section_hierarchy"]
            breadcrumb = " > ".join(html.escape(h) for h in hierarchy)
            chunks_html_parts.append(
                f'<div style="font-size: 0.75rem; color: #6b7280; '
                f'margin-bottom: 4px; font-weight: 500;">{breadcrumb}</div>'
            )

        # Render rendered markdown in summary (Default View)
        try:
            rendered_html = markdown.markdown(data["chunk"].text)
            chunks_html_parts.append(
                f'<div class="chunk-rendered">{rendered_html}</div>'
            )
        except Exception:
            # Fallback to plain text if markdown fails
            chunks_html_parts.append(html.escape(data["chunk"].text))

        # Expand icon at bottom of chunk
        chunks_html_parts.append(
            '<div style="text-align: center; margin-top: 6px;">'
            '<span class="chunk-expand-icon" style="font-size: 0.6rem;" '
            'title="Click to show metadata">▼</span></div>'
        )

        chunks_html_parts.append("</summary>")

        # Expanded content
        chunks_html_parts.append('<div class="chunk-expanded-content">')

        chunks_html_parts.append('<div class="chunk-context-label">Chunk Metadata</div>')
        chunks_html_parts.append(
            '<div class="chunk-context-box" style="display: grid; '
            'grid-template-columns: auto 1fr; gap: 4px 12px;">'
        )

        # Get full metadata from the chunk
        chunk_metadata = data["chunk"].metadata

        # Display metadata fields in a structured way
        metadata_display = [
            ("Strategy", chunk_metadata.get("strategy")),
            ("Chunk Index", chunk_metadata.get("chunk_index")),
            ("Size", f"{chunk_metadata.get('size', len(data['chunk'].text))} chars"),
            ("Token Count", chunk_metadata.get("token_count")),
            ("Element Type", chunk_metadata.get("element_type")),
            ("Section Hierarchy", chunk_metadata.get("section_hierarchy")),
            ("Heading Text", chunk_metadata.get("heading_text")),
            ("Start Index", chunk_metadata.get("start_index")),
            ("End Index", chunk_metadata.get("end_index")),
        ]

        for label, value in metadata_display:
            if value is not None:
                # Format lists nicely
                if isinstance(value, list):
                    if len(value) == 0:
                        continue
                    value_str = " > ".join(str(v) for v in value)
                else:
                    value_str = str(value)
                chunks_html_parts.append(
                    f'<span style="color: #6b7280; font-weight: 500;">{html.escape(label)}:</span>'
                    f'<span style="color: #374151;">{html.escape(value_str)}</span>'
                )

        chunks_html_parts.append("</div>")  # End metadata grid
        chunks_html_parts.append("</div>")  # End expanded content
        chunks_html_parts.append("</details>")

    chunks_html_parts.append("</div>")
    chunks_html = "".join(chunks_html_parts)

    st.markdown(chunks_html, unsafe_allow_html=True)

