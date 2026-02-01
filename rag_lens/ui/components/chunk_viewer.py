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

    # Heading text
    if "heading_text" in metadata:
        result["heading_text"] = metadata["heading_text"]

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
        List of dicts with keys: chunk, overlap_text, main_text, len,
        docling_metadata, display_text, used_fallback, fallback_reason
    """
    if not chunks:
        return []

    sorted_chunks = chunks
    chunk_display_data = []

    for i, chunk in enumerate(sorted_chunks):
        overlap_text = ""
        main_text = chunk.text
        overlap_len = 0
        display_text = chunk.text
        used_fallback = False
        fallback_reason = ""

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

        display_text = chunk.text

        docling_meta = _extract_docling_metadata(chunk.metadata)
        chunk_display_data.append(
            {
                "chunk": chunk,
                "overlap_text": overlap_text,
                "main_text": main_text,
                "len": len(chunk.text),
                "docling_metadata": docling_meta,
                "display_text": display_text,
                "used_fallback": used_fallback,
                "fallback_reason": fallback_reason,
            }
        )

    return chunk_display_data


def render_chunk_cards(
    chunk_display_data: list[dict[str, Any]],
    custom_badges: list[list[dict[str, Any]] | dict[str, Any]] | None = None,
    show_overlap: bool = True,
    display_mode: str = "continuous",
    render_format: str = "markdown",
) -> None:
    """Render chunk cards as HTML with expandable metadata.

    Args:
        chunk_display_data: List of dicts from prepare_chunk_display_data()
        custom_badges: Optional list of badges per chunk.
                      Can be a list of single badges (dict) or a list of lists of badges.
                      Each badge dict keys: {"label": "Score", "value": "0.85", "color": "#..."}
        show_overlap: Whether to render overlap highlighting (default: True)
        display_mode: Display style - "continuous" (colored flow) or "card" (individual cards with borders)
        render_format: Chunk render format (markdown, html, doctags, json)
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
            .chunk-rendered table {
                border-collapse: collapse;
                margin: 6px 0 10px 0;
                width: 100%;
                font-size: 0.85rem;
            }
            .chunk-rendered th,
            .chunk-rendered td {
                border: 1px solid #e2e8f0;
                padding: 4px 6px;
                text-align: left;
                vertical-align: top;
            }
            .chunk-rendered th {
                background: #f8fafc;
                font-weight: 600;
            }
            .chunk-raw {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 8px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
                    "Liberation Mono", "Courier New", monospace;
                font-size: 0.8rem;
                white-space: pre-wrap;
                word-break: break-word;
            }
        </style>
    """
    )

    normalized_format = (render_format or "markdown").strip().lower()

    for i, data in enumerate(chunk_display_data):
        meta = data["docling_metadata"]
        display_text = data.get("display_text", data["chunk"].text)
        used_fallback = bool(data.get("used_fallback"))
        fallback_reason = data.get("fallback_reason", "")

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
            badge_entry = custom_badges[i]
            
            # Normalize to list of badges
            badges_list = []
            if isinstance(badge_entry, list):
                badges_list = badge_entry
            elif isinstance(badge_entry, dict) and badge_entry:
                badges_list = [badge_entry]
            
            for badge_info in badges_list:
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

        # Fallback badge when rendering from non-source text
        if used_fallback:
            fallback_titles = {
                "empty_source_slice": "Source slice was empty",
                "invalid_source_indices": "Source indices were invalid",
                "missing_source_text": "Source text was unavailable",
            }
            fallback_title = fallback_titles.get(
                fallback_reason, "Rendered using chunk text fallback"
            )
            chunks_html_parts.append(
                '<span style="background: #fee2e2; color: #991b1b; '
                'font-size: 0.65rem; padding: 1px 5px; border-radius: 8px; '
                f'user-select: none;" title="{html.escape(fallback_title)}">'
                "Fallback</span>"
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

        # Section heading label (when available)
        heading_text = meta.get("heading_text")
        if heading_text:
            chunks_html_parts.append(
                f'<div style="font-size: 0.75rem; color: #6b7280; '
                f'margin-bottom: 4px; font-weight: 500;">{html.escape(str(heading_text))}</div>'
            )

        # Render chunk content by format
        if normalized_format == "html":
            rendered_html = display_text
            render_failed = False
        elif normalized_format in ("json", "doctags"):
            rendered_html = f'<pre class="chunk-raw">{html.escape(display_text)}</pre>'
            render_failed = False
        else:
            try:
                rendered_html = markdown.markdown(
                    display_text,
                    extensions=[
                        "nl2br",
                        "tables",
                        "fenced_code",
                        "sane_lists",
                    ],
                )
                render_failed = False
            except Exception:
                # Fallback to plain text if markdown fails
                rendered_html = html.escape(display_text)
                render_failed = True
        if render_failed:
            chunks_html_parts.append(
                '<div style="font-size: 0.7rem; color: #b45309; margin-top: 4px;">'
                "Rendered as plain text</div>"
            )
        chunks_html_parts.append(
            f'<div class="chunk-rendered">{rendered_html}</div>'
        )

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

        # Add RRF Fusion Details if available
        if chunk_metadata.get("fusion_method") == "rrf":
            dense_rank = chunk_metadata.get("dense_rank")
            sparse_rank = chunk_metadata.get("sparse_rank")
            dense_score = chunk_metadata.get("dense_rrf_score", 0)
            sparse_score = chunk_metadata.get("sparse_rrf_score", 0)
            rrf_k = chunk_metadata.get("rrf_k", 60)

            # Calculate percentages
            total = dense_score + sparse_score
            dense_pct = (dense_score / total * 100) if total > 0 else 0
            sparse_pct = (sparse_score / total * 100) if total > 0 else 0

            chunks_html_parts.append(
                '<div class="chunk-context-label" style="margin-top: 12px;">RRF Fusion Breakdown</div>'
            )
            chunks_html_parts.append('<div style="font-size: 0.75rem; line-height: 1.5;">')

            # Dense contribution
            if dense_rank is not None:
                chunks_html_parts.append(
                    f'<div style="margin-bottom: 8px;">'
                    f'<div style="display: flex; justify-content: space-between; margin-bottom: 2px;">'
                    f'<span style="color: #374151;">Dense (Semantic): Rank #{dense_rank}</span>'
                    f'<span style="color: #6b7280; font-family: monospace;">'
                    f'1/({rrf_k}+{dense_rank}) = {dense_score:.4f}</span>'
                    f'</div>'
                    f'<div style="background: #e0e7ff; height: 8px; border-radius: 4px; overflow: hidden;">'
                    f'<div style="background: #4f46e5; height: 100%; width: {dense_pct}%;"></div>'
                    f'</div>'
                    f'<div style="color: #6b7280; font-size: 0.7rem; margin-top: 1px;">'
                    f'Contributes {dense_pct:.1f}% to final score</div>'
                    f'</div>'
                )
            else:
                chunks_html_parts.append(
                    '<div style="margin-bottom: 8px; color: #9ca3af;">'
                    'Dense (Semantic): Not found in top results'
                    '</div>'
                )

            # Sparse contribution
            if sparse_rank is not None:
                chunks_html_parts.append(
                    f'<div style="margin-bottom: 4px;">'
                    f'<div style="display: flex; justify-content: space-between; margin-bottom: 2px;">'
                    f'<span style="color: #374151;">Sparse (BM25): Rank #{sparse_rank}</span>'
                    f'<span style="color: #6b7280; font-family: monospace;">'
                    f'1/({rrf_k}+{sparse_rank}) = {sparse_score:.4f}</span>'
                    f'</div>'
                    f'<div style="background: #fef3c7; height: 8px; border-radius: 4px; overflow: hidden;">'
                    f'<div style="background: #f59e0b; height: 100%; width: {sparse_pct}%;"></div>'
                    f'</div>'
                    f'<div style="color: #6b7280; font-size: 0.7rem; margin-top: 1px;">'
                    f'Contributes {sparse_pct:.1f}% to final score</div>'
                    f'</div>'
                )
            else:
                chunks_html_parts.append(
                    '<div style="margin-bottom: 4px; color: #9ca3af;">'
                    'Sparse (BM25): Not found in top results'
                    '</div>'
                )

            # Summary
            chunks_html_parts.append(
                f'<div style="margin-top: 8px; padding-top: 8px; '
                f'border-top: 1px solid rgba(0,0,0,0.08); color: #374151;">'
                f'<strong>Combined RRF Score:</strong> {dense_score + sparse_score:.4f} '
                f'<span style="color: #6b7280;">(normalized in final ranking)</span>'
                f'</div>'
            )

            chunks_html_parts.append("</div>")  # End RRF breakdown

        chunks_html_parts.append("</div>")  # End expanded content
        chunks_html_parts.append("</details>")

    chunks_html_parts.append("</div>")
    chunks_html = "".join(chunks_html_parts)

    st.markdown(chunks_html, unsafe_allow_html=True)

