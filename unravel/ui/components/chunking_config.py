"""Chunking configuration component for the chunks step."""

import os

import streamlit as st

from unravel.services.chunking import (
    get_available_providers,
    get_provider,
    get_provider_splitters,
)
from unravel.services.chunking.providers.docling_provider import METADATA_OPTIONS


def render_chunking_configuration() -> tuple[dict, dict, bool]:
    """
    Render chunking configuration UI.

    Returns:
        tuple: (new_parsing_params, new_chunking_params, has_changes)
    """
    # Get current and applied parameters from session state
    current_parsing_params = st.session_state.get("parsing_params", {})
    current_chunking_params = st.session_state.get("chunking_params", {})
    applied_parsing_params = st.session_state.get(
        "applied_parsing_params", current_parsing_params.copy()
    )
    applied_chunking_params = st.session_state.get(
        "applied_chunking_params", current_chunking_params.copy()
    )

    # Output Format selector
    st.markdown("**Output Format**")
    format_display_map = {
        "Markdown": "markdown",
        "HTML": "html",
        "DocTags": "doctags",
        "JSON (Lossless)": "json",
    }
    output_formats = list(format_display_map.keys())
    current_output_format = current_parsing_params.get("output_format", "markdown")

    format_value_map = {v: k for k, v in format_display_map.items()}
    current_format_display = format_value_map.get(current_output_format, "Markdown")

    # Sync widget state only if needed
    if "chunking_config_output_format" not in st.session_state:
        st.session_state["chunking_config_output_format"] = current_format_display
    elif st.session_state.get("chunking_config_output_format") not in output_formats:
        st.session_state["chunking_config_output_format"] = current_format_display

    output_format = st.selectbox(
        "Output Format",
        options=output_formats,
        key="chunking_config_output_format",
        label_visibility="collapsed",
    )

    st.write("")
    st.markdown("**Parsing Options**")

    # Global page cap (Docling formats)
    max_pages = st.number_input(
        "Max pages to parse",
        min_value=1,
        max_value=1_000,
        step=1,
        value=int(current_parsing_params.get("max_pages", 50) or 50),
        key="chunking_config_max_pages",
        help="Limits Docling parsing to the first N pages. Set higher to process more content.",
    )

    max_threads = max(1, min((os.cpu_count() or 4), 16))

    docling_enable_ocr = st.checkbox(
        "Enable OCR (slower, use for scanned PDFs)",
        value=current_parsing_params.get("docling_enable_ocr", False),
        key="chunking_config_docling_enable_ocr",
        help="Skip this for digital PDFs to speed up parsing.",
    )
    docling_table_structure = st.checkbox(
        "Extract tables and layout",
        value=current_parsing_params.get("docling_table_structure", True),
        key="chunking_config_docling_table_structure",
        help="Disable to speed up parsing when table fidelity is not needed.",
    )
    docling_threads = st.slider(
        "Docling worker threads",
        min_value=1,
        max_value=max_threads,
        value=min(
            max(1, current_parsing_params.get("docling_threads", 4)),
            max_threads,
        ),
        key="chunking_config_docling_threads",
        help="Increase on larger CPUs to process pages in parallel.",
    )

    # Advanced Table Options
    with st.expander("Advanced Table Options", expanded=False):
        enable_table_merging = st.checkbox(
            "Enable table merging",
            value=current_parsing_params.get("enable_table_merging", True),
            key="chunking_config_enable_table_merging",
            help="Merge adjacent table cells that span multiple rows or columns.",
        )
        enable_table_reconstruction = st.checkbox(
            "Enable table reconstruction",
            value=current_parsing_params.get("enable_table_reconstruction", True),
            key="chunking_config_enable_table_reconstruction",
            help="Reconstruct table structure from detected layout elements.",
        )

    st.write("")
    st.markdown("**Content Filtering**")

    # All available DocItemLabel options with human-readable names
    docling_filter_options = {
        "Page Header": "PAGE_HEADER",
        "Page Footer": "PAGE_FOOTER",
        "Section Header": "SECTION_HEADER",
        "Caption": "CAPTION",
        "Chart": "CHART",
        "Checkbox (Selected)": "CHECKBOX_SELECTED",
        "Checkbox (Unselected)": "CHECKBOX_UNSELECTED",
        "Code": "CODE",
        "Document Index": "DOCUMENT_INDEX",
        "Empty Value": "EMPTY_VALUE",
        "Footnote": "FOOTNOTE",
        "Form": "FORM",
        "Formula": "FORMULA",
        "Grading Scale": "GRADING_SCALE",
        "Handwritten Text": "HANDWRITTEN_TEXT",
        "Key-Value Region": "KEY_VALUE_REGION",
        "List Item": "LIST_ITEM",
        "Paragraph": "PARAGRAPH",
        "Picture": "PICTURE",
        "Reference": "REFERENCE",
        "Table": "TABLE",
        "Text": "TEXT",
        "Title": "TITLE",
    }

    # Reverse mapping for display
    filter_value_to_display = {v: k for k, v in docling_filter_options.items()}

    # Default filters
    default_filters = ["PAGE_HEADER", "PAGE_FOOTER"]

    # Get current selection from session state
    current_filter_labels = current_parsing_params.get("docling_filter_labels", default_filters)

    # Convert stored values to display names
    current_display_labels = [
        filter_value_to_display.get(label, label)
        for label in current_filter_labels
        if label in filter_value_to_display
    ]

    # Multi-select for filter labels
    selected_display_labels = st.multiselect(
        "Filter document items",
        options=sorted(docling_filter_options.keys()),
        default=current_display_labels,
        key="chunking_config_docling_filter_labels",
        help="Selected items will be excluded from the parsed output.",
    )

    # Convert display names back to internal values
    docling_filter_labels = [docling_filter_options[label] for label in selected_display_labels]

    st.write("")
    st.markdown("**Image Extraction**")

    docling_extract_images = st.checkbox(
        "Extract images from PDF",
        value=current_parsing_params.get("docling_extract_images", False),
        key="chunking_config_docling_extract_images",
        help="Extract embedded images from PDF. May increase parsing time.",
    )

    docling_enable_captioning = False
    docling_use_native_description = False
    if docling_extract_images:
        # Determine initial strategy from params, defaulting to Native if none previously set
        current_strategy = "Docling Native"
        if current_parsing_params.get("docling_enable_captioning", False):
            current_strategy = "LLM Captioning"

        strategy_options = ["Docling Native", "LLM Captioning"]
        try:
            strategy_index = strategy_options.index(current_strategy)
        except ValueError:
            strategy_index = 0

        description_strategy = st.radio(
            "Description Strategy",
            options=strategy_options,
            index=strategy_index,
            key="chunking_config_description_strategy",
            help="Choose how to describe extracted images. Native uses Docling's local VLM (no API cost).",
        )

        docling_use_native_description = description_strategy == "Docling Native"
        docling_enable_captioning = description_strategy == "LLM Captioning"

        # Show warning if captioning enabled but no vision model configured
        if docling_enable_captioning:
            from unravel.services.llm import VISION_CAPABLE_MODELS

            llm_config_raw = st.session_state.get("llm_config")
            llm_config = llm_config_raw if isinstance(llm_config_raw, dict) else {}
            llm_provider = llm_config.get("provider", "")
            llm_model = llm_config.get("model", "")
            llm_api_key = llm_config.get("api_key", "")

            if not llm_api_key:
                st.warning("Configure a vision-capable LLM in the LLM Config tab.")
            elif llm_provider != "OpenAI-Compatible" and llm_model not in VISION_CAPABLE_MODELS.get(
                llm_provider, []
            ):
                st.warning(f"Model '{llm_model}' may not support vision. Use GPT-4o or Claude 3.")

    # Update parsing params
    new_parsing_params = {
        "output_format": format_display_map.get(output_format, "markdown"),
        "docling_device": "auto",
        "docling_enable_ocr": docling_enable_ocr,
        "docling_table_structure": docling_table_structure,
        "docling_threads": docling_threads,
        "enable_table_merging": enable_table_merging,
        "enable_table_reconstruction": enable_table_reconstruction,
        "docling_filter_labels": docling_filter_labels,
        "docling_extract_images": docling_extract_images,
        "docling_enable_captioning": docling_enable_captioning,
        "docling_use_native_description": docling_use_native_description,
        "max_pages": max_pages,
    }

    st.divider()
    st.markdown("**Text Splitting**")

    # Provider selection
    providers = get_available_providers()
    current_provider = current_chunking_params.get("provider", "Docling")
    if current_provider not in providers:
        current_provider = providers[0] if providers else "Docling"

    if "chunking_config_provider" not in st.session_state:
        st.session_state["chunking_config_provider"] = current_provider

    provider = st.selectbox("Library", options=providers, key="chunking_config_provider")

    # Get splitters for selected provider
    splitter_infos = get_provider_splitters(provider)
    splitter_options = [info.display_name for info in splitter_infos]
    splitter_name_map = {info.display_name: info.name for info in splitter_infos}
    splitter_display_map = {info.name: info.display_name for info in splitter_infos}

    current_splitter = current_chunking_params.get("splitter", "HybridChunker")
    current_display = splitter_display_map.get(
        current_splitter, splitter_options[0] if splitter_options else ""
    )

    if "chunking_config_splitter" not in st.session_state:
        st.session_state["chunking_config_splitter"] = current_display
    elif st.session_state.get("chunking_config_splitter") not in splitter_options:
        st.session_state["chunking_config_splitter"] = current_display

    st.write("")
    splitter_display = st.selectbox(
        "Strategy", options=splitter_options, key="chunking_config_splitter"
    )

    splitter_name = splitter_name_map.get(splitter_display, "HybridChunker")

    # Detailed explanations for each splitter strategy
    splitter_details = {
        "HierarchicalChunker": {
            "title": "Hierarchical Chunker",
            "when_to_use": "Use when preserving document structure is important. Creates one chunk per logical element (paragraph, header, list, code block).",
            "how_it_works": "Analyzes document structure and splits at natural boundaries like paragraphs, headers, and code blocks. Optionally merges very small chunks and tracks section hierarchy for better context.",
            "best_for": "Structured documents, documentation, articles with clear sections",
        },
        "HybridChunker": {
            "title": "Hybrid Chunker",
            "when_to_use": "Best default choice for embedding-based RAG. Respects structure while ensuring chunks fit within model token limits.",
            "how_it_works": "Combines structure-aware splitting with token counting. Accumulates structural elements until reaching the token limit, then creates a chunk. Supports overlap for continuity.",
            "best_for": "RAG applications, embedding models with fixed context windows",
        },
    }

    # Show detailed explanation for selected splitter
    if splitter_name in splitter_details:
        details = splitter_details[splitter_name]
        with st.expander(f"About {details['title']}", expanded=False):
            st.markdown(f"**When to use:** {details['when_to_use']}")
            st.markdown(f"**How it works:** {details['how_it_works']}")
            st.markdown(f"**Best for:** {details['best_for']}")

    # Get parameter schema for selected splitter
    selected_info = next((info for info in splitter_infos if info.name == splitter_name), None)

    # Render dynamic parameters
    splitter_params = {}
    if selected_info:
        st.write("")

        for param in selected_info.parameters:
            current_value = current_chunking_params.get(param.name, param.default)

            if param.type == "int":
                value = st.number_input(
                    param.name.replace("_", " ").title(),
                    min_value=param.min_value or 0,
                    max_value=param.max_value or 10000,
                    value=(int(current_value) if current_value is not None else param.default),
                    step=50 if "size" in param.name else 10,
                    help=param.description,
                    key=f"chunking_config_param_{param.name}",
                )
            elif param.type == "str":
                if param.options:
                    if f"chunking_config_param_{param.name}" not in st.session_state:
                        st.session_state[f"chunking_config_param_{param.name}"] = str(current_value)
                    value = st.selectbox(
                        param.name.replace("_", " ").title(),
                        options=param.options,
                        key=f"chunking_config_param_{param.name}",
                    )
                else:
                    value = st.text_input(
                        param.name.replace("_", " ").title(),
                        value=str(current_value),
                        help=param.description,
                        key=f"chunking_config_param_{param.name}",
                    )
            elif param.type == "bool":
                value = st.checkbox(
                    param.name.replace("_", " ").title(),
                    value=bool(current_value),
                    help=param.description,
                    key=f"chunking_config_param_{param.name}",
                )
            elif param.type == "multiselect" and param.options:
                # Handle multiselect parameters with display-to-value mapping
                metadata_value_to_display = {v: k for k, v in METADATA_OPTIONS.items()}

                # Convert stored internal values to display names for the widget
                current_list = current_value if isinstance(current_value, list) else param.default
                current_display_list = [metadata_value_to_display.get(v, v) for v in current_list]
                # Filter to only valid options
                current_display_list = [v for v in current_display_list if v in param.options]
                if not current_display_list:
                    current_display_list = param.default

                selected_display = st.multiselect(
                    param.name.replace("_", " ").title(),
                    options=param.options,
                    default=current_display_list,
                    help=param.description,
                    key=f"chunking_config_param_{param.name}",
                )

                # Convert display names back to internal values for storage
                value = [METADATA_OPTIONS.get(d, d) for d in selected_display]
            else:
                value = current_value

            splitter_params[param.name] = value

    # Show provider attribution
    provider_instance = get_provider(provider)
    if provider_instance and provider_instance.attribution:
        st.caption(f"_{provider_instance.attribution}_")

    # Update chunking params
    new_chunking_params = {
        "provider": provider,
        "splitter": splitter_name,
        **splitter_params,
    }

    # Calculate has_changes based on actual widget values
    has_changes = (
        new_parsing_params != applied_parsing_params
        or new_chunking_params != applied_chunking_params
    )

    # Show status badge at the end, after all widgets are rendered
    st.write("")
    status_badge_html = (
        '<span style="background-color: #f59e0b; color: white; padding: 2px 8px; '
        'border-radius: 4px; font-size: 12px; font-weight: 500;">Changes pending</span>'
        if has_changes
        else '<span style="background-color: #10b981; color: white; padding: 2px 8px; '
        'border-radius: 4px; font-size: 12px; font-weight: 500;">Configuration applied</span>'
    )
    st.markdown(status_badge_html, unsafe_allow_html=True)

    return new_parsing_params, new_chunking_params, has_changes
