import os

import streamlit as st
import streamlit_shadcn_ui as ui

from rag_visualizer.services.chunking import (
    get_available_providers,
    get_provider,
    get_provider_splitters,
)
from rag_visualizer.services.embedders import DEFAULT_MODEL, list_available_models
from rag_visualizer.services.llm import (
    DEFAULT_SYSTEM_PROMPT,
    LLM_PROVIDERS,
    LLMConfig,
    get_api_key_from_env,
    get_provider_models,
    validate_config,
)
from rag_visualizer.services.storage import (
    clear_session_state,
    list_documents,
    save_llm_config,
    save_rag_config,
)


def render_rag_config_sidebar() -> None:
    """Render RAG configuration in the sidebar."""
    st.markdown("### 📄 RAG Configuration")

    # Initialize session state if not present
    docs = list_documents()
    
    if "doc_name" not in st.session_state:
        # If files exist, use first file; otherwise use None (will show empty state)
        if docs:
            st.session_state.doc_name = docs[0]
        else:
            st.session_state.doc_name = None
    
    # Auto-select first file if currently on "Sample Text" and files exist
    if st.session_state.doc_name == "Sample Text" and docs:
        st.session_state.doc_name = docs[0]
        # Invalidate chunks and embeddings when switching from sample text
        if "chunks" in st.session_state:
            del st.session_state["chunks"]
        if "last_embeddings_result" in st.session_state:
            del st.session_state["last_embeddings_result"]
        if "search_results" in st.session_state:
            del st.session_state["search_results"]
    
    if "chunking_params" not in st.session_state:
        st.session_state.chunking_params = {
            "provider": "Docling",
            "splitter": "HybridChunker",
            "max_tokens": 512,
            "chunk_overlap": 50,
            "tokenizer": "cl100k_base",
        }
    if "embedding_model_name" not in st.session_state:
        st.session_state.embedding_model_name = DEFAULT_MODEL

    # Initialize parsing_params if not present
    if "parsing_params" not in st.session_state:
        st.session_state.parsing_params = {
            # Docling options
            "docling_enable_ocr": False,
            "docling_table_structure": True,
            "docling_threads": 4,
            "docling_filter_labels": ["PAGE_HEADER", "PAGE_FOOTER"],
            "docling_extract_images": False,
            "docling_enable_captioning": False,
            "docling_device": "auto",
            # Output options
            "output_format": "markdown",
            "normalize_whitespace": True,
            "remove_special_chars": False,
            "max_characters": 40_000,
        }

    # Track applied parameters (what was actually used for processing)
    if "applied_parsing_params" not in st.session_state:
        st.session_state.applied_parsing_params = st.session_state.parsing_params.copy()
    if "applied_chunking_params" not in st.session_state:
        st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()

    # Document selection - only show "Sample Text" if no files exist
    if docs:
        all_docs = docs
    else:
        all_docs = ["Sample Text"] if st.session_state.doc_name == "Sample Text" else []
    
    # Ensure current selection is valid
    if st.session_state.doc_name and st.session_state.doc_name not in all_docs:
        if docs:
            st.session_state.doc_name = docs[0]
        else:
            st.session_state.doc_name = None

    form_doc_name = st.session_state.doc_name
    new_parsing_params = st.session_state.parsing_params
    new_chunking_params = st.session_state.chunking_params
    new_embedding_model = st.session_state.embedding_model_name

    if all_docs:
        if "sidebar_doc_selector" not in st.session_state:
            st.session_state["sidebar_doc_selector"] = st.session_state.doc_name or all_docs[0]
        elif st.session_state.get("sidebar_doc_selector") != st.session_state.doc_name:
            if st.session_state.doc_name in all_docs:
                st.session_state["sidebar_doc_selector"] = st.session_state.doc_name
    else:
        st.info("No documents available. Upload a file in the Upload step.")
        st.session_state.doc_name = None

    with st.form("rag_config_form"):
        st.write("")
        st.markdown("**Document**")
        if all_docs:
            form_doc_name = st.selectbox(
                "Document",
                options=all_docs,
                key="sidebar_doc_selector"
            )
        else:
            st.stop()

        st.write("")
        st.markdown("**Document Parsing & Text Splitting**")

        # Output Format selector
        output_formats = ["Markdown (Recommended)", "Original Format", "Plain Text"]
        current_output_format = st.session_state.parsing_params.get("output_format", "markdown")

        # Map display names to internal values
        format_display_map = {
            "Markdown (Recommended)": "markdown",
            "Original Format": "original",
            "Plain Text": "plain_text"
        }
        format_value_map = {v: k for k, v in format_display_map.items()}
        current_format_display = format_value_map.get(current_output_format, "Markdown (Recommended)")

        # Sync widget state only if needed
        if "sidebar_output_format" not in st.session_state:
            st.session_state["sidebar_output_format"] = current_format_display
        elif st.session_state.get("sidebar_output_format") not in output_formats:
            st.session_state["sidebar_output_format"] = current_format_display

        output_format = st.selectbox(
            "Output Format",
            options=output_formats,
            key="sidebar_output_format"
        )

        st.write("")

        # Normalize whitespace checkbox
        normalize_whitespace = st.checkbox(
            "Normalize Whitespace",
            value=st.session_state.parsing_params.get("normalize_whitespace", True),
            key="sidebar_normalize_whitespace",
            help="Remove excessive spaces and newlines"
        )

        # Remove special chars checkbox
        remove_special_chars = st.checkbox(
            "Remove Special Characters",
            value=st.session_state.parsing_params.get("remove_special_chars", False),
            key="sidebar_remove_special_chars",
            help="Strip non-alphanumeric characters except basic punctuation"
        )

        # Advanced parsing options (Docling)
        with st.expander("Advanced Parsing Options", expanded=False):
            from rag_visualizer.utils.parsers import (
                get_available_devices,
                get_device_info,
            )

            # Global character cap (affects all steps)
            max_characters = st.number_input(
                "Max characters to parse",
                min_value=1_000,
                max_value=1_000_000,
                step=1_000,
                value=int(st.session_state.parsing_params.get("max_characters", 40_000) or 40_000),
                key="sidebar_max_characters",
                help="Only the first N characters will be parsed. Set to a higher value to process more content.",
            )

            max_threads = max(1, min((os.cpu_count() or 4), 16))

            # Device selector for GPU acceleration
            available_devices = get_available_devices()
            device_info = get_device_info()
            current_device = st.session_state.parsing_params.get("docling_device", "auto")
            if current_device not in available_devices:
                current_device = "auto"

            docling_device = st.selectbox(
                "Compute Device",
                options=available_devices,
                index=available_devices.index(current_device),
                key="sidebar_docling_device",
                help="Use GPU for faster processing. 'auto' will try to detect GPU automatically.",
            )

            # Show device status and guidance
            if docling_device == "cpu":
                st.caption("Using CPU only (slower)")
            elif docling_device == "auto":
                if device_info["cuda_available"]:
                    st.caption(f"Auto-detected: {device_info['gpu_name']} (CUDA {device_info['cuda_version']})")
                elif device_info["mps_available"]:
                    st.caption("Auto-detected: Apple Metal GPU")
                else:
                    st.caption("No GPU detected, will use CPU")
            elif docling_device.startswith("cuda"):
                if device_info["cuda_available"]:
                    st.caption(f"Using: {device_info['gpu_name']} (CUDA {device_info['cuda_version']})")
                elif not device_info["torch_installed"]:
                    st.warning("PyTorch not installed. Install with CUDA support:")
                    st.code("pip install torch --index-url https://download.pytorch.org/whl/cu121", language="bash")
                else:
                    st.warning("CUDA not available. Ensure you have:")
                    st.markdown("""
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
""")
            elif docling_device == "mps":
                if device_info["mps_available"]:
                    st.caption("Using: Apple Metal GPU")
                else:
                    st.warning("Apple Metal not available. Requires macOS with Apple Silicon.")

            docling_enable_ocr = st.checkbox(
                "Enable OCR (slower, use for scanned PDFs)",
                value=st.session_state.parsing_params.get("docling_enable_ocr", False),
                key="sidebar_docling_enable_ocr",
                help="Skip this for digital PDFs to speed up parsing.",
            )
            docling_table_structure = st.checkbox(
                "Extract tables and layout",
                value=st.session_state.parsing_params.get("docling_table_structure", True),
                key="sidebar_docling_table_structure",
                help="Disable to speed up parsing when table fidelity is not needed.",
            )
            docling_threads = st.slider(
                "Docling worker threads",
                min_value=1,
                max_value=max_threads,
                value=min(
                    max(1, st.session_state.parsing_params.get("docling_threads", 4)),
                    max_threads,
                ),
                key="sidebar_docling_threads",
                help="Increase on larger CPUs to process pages in parallel.",
            )

            st.divider()
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
            filter_value_to_display = {
                v: k for k, v in docling_filter_options.items()
            }

            # Default filters (common noise)
            default_filters = ["PAGE_HEADER", "PAGE_FOOTER"]

            # Get current selection from session state
            current_filter_labels = st.session_state.parsing_params.get(
                "docling_filter_labels", default_filters
            )

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
                key="sidebar_docling_filter_labels",
                help="Selected items will be excluded from the parsed output.",
            )

            # Convert display names back to internal values
            docling_filter_labels = [
                docling_filter_options[label]
                for label in selected_display_labels
            ]

            st.divider()
            st.markdown("**Image Extraction**")

            docling_extract_images = st.checkbox(
                "Extract images from PDF",
                value=st.session_state.parsing_params.get("docling_extract_images", False),
                key="sidebar_docling_extract_images",
                help="Extract embedded images from PDF. May increase parsing time.",
            )

            docling_enable_captioning = False
            if docling_extract_images:
                docling_enable_captioning = st.checkbox(
                    "Caption images with LLM",
                    value=st.session_state.parsing_params.get("docling_enable_captioning", False),
                    key="sidebar_docling_enable_captioning",
                    help="Generate searchable captions using configured LLM.",
                )

                # Show warning if captioning enabled but no vision model configured
                if docling_enable_captioning:
                    from rag_visualizer.services.llm import VISION_CAPABLE_MODELS
                    llm_config = st.session_state.get("llm_config", {})
                    llm_provider = llm_config.get("provider", "")
                    llm_model = llm_config.get("model", "")
                    llm_api_key = llm_config.get("api_key", "")

                    if not llm_api_key:
                        st.warning("Configure a vision-capable LLM in the LLM Config tab.")
                    elif llm_provider != "OpenAI-Compatible" and llm_model not in VISION_CAPABLE_MODELS.get(llm_provider, []):
                        st.warning(f"Model '{llm_model}' may not support vision. Use GPT-4o or Claude 3.")

        # Update parsing params (but don't apply yet)
        new_parsing_params = {
            "output_format": format_display_map.get(output_format, "markdown"),
            "normalize_whitespace": normalize_whitespace,
            "remove_special_chars": remove_special_chars,
            "docling_device": docling_device,
            "docling_enable_ocr": docling_enable_ocr,
            "docling_table_structure": docling_table_structure,
            "docling_threads": docling_threads,
            "docling_filter_labels": docling_filter_labels,
            "docling_extract_images": docling_extract_images,
            "docling_enable_captioning": docling_enable_captioning,
            "max_characters": max_characters,
        }

        st.write("")
        st.markdown("**Text Splitting**")

        # Provider selection
        providers = get_available_providers()
        current_provider = st.session_state.chunking_params.get("provider", "Docling")
        if current_provider not in providers:
            current_provider = providers[0] if providers else "Docling"

        if "sidebar_chunking_provider" not in st.session_state:
            st.session_state["sidebar_chunking_provider"] = current_provider

        provider = st.selectbox(
            "Library",
            options=providers,
            key="sidebar_chunking_provider"
        )

        # Get splitters for selected provider
        splitter_infos = get_provider_splitters(provider)
        splitter_options = [info.display_name for info in splitter_infos]
        splitter_name_map = {info.display_name: info.name for info in splitter_infos}
        splitter_display_map = {info.name: info.display_name for info in splitter_infos}

        current_splitter = st.session_state.chunking_params.get("splitter", "HybridChunker")
        current_display = splitter_display_map.get(current_splitter, splitter_options[0] if splitter_options else "")

        if "sidebar_splitter" not in st.session_state:
            st.session_state["sidebar_splitter"] = current_display
        elif st.session_state.get("sidebar_splitter") not in splitter_options:
            st.session_state["sidebar_splitter"] = current_display

        st.write("")
        splitter_display = st.selectbox(
            "Strategy",
            options=splitter_options,
            key="sidebar_splitter"
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
                current_value = st.session_state.chunking_params.get(param.name, param.default)

                if param.type == "int":
                    value = st.number_input(
                        param.name.replace("_", " ").title(),
                        min_value=param.min_value or 0,
                        max_value=param.max_value or 10000,
                        value=int(current_value) if current_value is not None else param.default,
                        step=50 if "size" in param.name else 10,
                        help=param.description,
                        key=f"sidebar_param_{param.name}"
                    )
                elif param.type == "str":
                    if param.options:
                        if f"sidebar_param_{param.name}" not in st.session_state:
                            st.session_state[f"sidebar_param_{param.name}"] = str(current_value)
                        value = st.selectbox(
                            param.name.replace("_", " ").title(),
                            options=param.options,
                            key=f"sidebar_param_{param.name}"
                        )
                    else:
                        value = st.text_input(
                            param.name.replace("_", " ").title(),
                            value=str(current_value),
                            help=param.description,
                            key=f"sidebar_param_{param.name}"
                        )
                elif param.type == "bool":
                    value = st.checkbox(
                        param.name.replace("_", " ").title(),
                        value=bool(current_value),
                        help=param.description,
                        key=f"sidebar_param_{param.name}"
                    )
                else:
                    value = current_value

                splitter_params[param.name] = value

        # Show provider attribution
        provider_instance = get_provider(provider)
        if provider_instance and provider_instance.attribution:
            st.caption(f"_{provider_instance.attribution}_")

        # Update chunking params (but don't apply yet)
        new_chunking_params = {
            "provider": provider,
            "splitter": splitter_name,
            **splitter_params,
        }

        st.write("")
        st.markdown("**Embedding Model**")

        # Embedding model selection
        models = list_available_models()
        model_names = [m["name"] for m in models]

        current_model_name = st.session_state.embedding_model_name
        if current_model_name not in model_names:
            current_model_name = model_names[0]

        if "sidebar_embedding_model" not in st.session_state:
            st.session_state["sidebar_embedding_model"] = current_model_name
        elif st.session_state.get("sidebar_embedding_model") not in model_names:
            st.session_state["sidebar_embedding_model"] = current_model_name

        new_embedding_model = st.selectbox(
            "Embedding Model",
            options=model_names,
            key="sidebar_embedding_model"
        )

        apply_clicked = st.form_submit_button("Save & Apply")

    if apply_clicked:
        doc_changed = form_doc_name != st.session_state.doc_name
        model_changed = new_embedding_model != st.session_state.embedding_model_name
        parsing_changed = new_parsing_params != st.session_state.applied_parsing_params
        chunking_changed = new_chunking_params != st.session_state.applied_chunking_params

        st.session_state.doc_name = form_doc_name
        st.session_state.embedding_model_name = new_embedding_model
        st.session_state.parsing_params = new_parsing_params
        st.session_state.chunking_params = new_chunking_params
        st.session_state.applied_parsing_params = new_parsing_params.copy()
        st.session_state.applied_chunking_params = new_chunking_params.copy()

        if doc_changed or parsing_changed or chunking_changed:
            if "chunks" in st.session_state:
                del st.session_state["chunks"]
        if doc_changed or parsing_changed or chunking_changed or model_changed:
            if "last_embeddings_result" in st.session_state:
                del st.session_state["last_embeddings_result"]
            if "search_results" in st.session_state:
                del st.session_state["search_results"]

        current_rag_config = {
            "doc_name": st.session_state.doc_name,
            "embedding_model_name": st.session_state.embedding_model_name,
            "chunking_params": st.session_state.chunking_params,
            "parsing_params": st.session_state.parsing_params,
        }
        save_rag_config(current_rag_config)
        st.session_state["_last_saved_rag_config"] = current_rag_config.copy()

        st.success("✓ Changes applied! Processing will use new settings.")
        st.rerun()

    # Clear session state button
    st.write("")
    st.write("")
    st.divider()
    st.markdown("**Troubleshooting**")
    st.caption("Clear cached embeddings and session data if you encounter errors.")

    if ui.button("Clear Session State", variant="outline", key="clear_session_btn"):
        # Clear persisted session state files
        clear_session_state()

        # Clear in-memory session state (keep only essential keys)
        keys_to_keep = {
            "session_restored", "doc_name", "chunking_params", "embedding_model_name",
            "llm_provider", "llm_model", "llm_api_key", "llm_base_url",
            "llm_temperature", "llm_max_tokens", "llm_system_prompt",
            "current_step"  # Keep current step to avoid navigation issues
        }
        keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
        for key in keys_to_delete:
            del st.session_state[key]

        st.success("✓ Session state cleared! Refresh the page to regenerate embeddings.")


def render_llm_sidebar() -> None:
    """Render LLM configuration in the sidebar."""
    st.markdown("### 🤖 LLM Configuration")
    
    # Initialize session state if not present
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "OpenAI"
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = ""
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""
    if "llm_base_url" not in st.session_state:
        st.session_state.llm_base_url = ""
    if "llm_temperature" not in st.session_state:
        st.session_state.llm_temperature = 0.7
    if "llm_max_tokens" not in st.session_state:
        st.session_state.llm_max_tokens = 1024
    if "llm_system_prompt" not in st.session_state:
        st.session_state.llm_system_prompt = DEFAULT_SYSTEM_PROMPT
    
    # Provider selection
    providers = list(LLM_PROVIDERS.keys())
    current_provider = st.session_state.llm_provider
    if current_provider not in providers:
        current_provider = providers[0]

    # Pre-select based on state
    if "sidebar_provider" not in st.session_state:
        st.session_state["sidebar_provider"] = current_provider

    provider = ui.select(
        options=providers,
        key="sidebar_provider",
        label="Provider"
    )
    st.session_state.llm_provider = provider
    
    st.write("")
    
    # Show explanation for OpenAI-Compatible
    if provider == "OpenAI-Compatible":
        with ui.card(key="openai_compat_info"):
            st.markdown(
                """
                **OpenAI-Compatible** allows local models (Ollama, LM Studio, etc).
                
                **Common Base URLs:**
                - **Ollama**: `http://localhost:11434/v1`
                - **LM Studio**: `http://localhost:1234/v1`
                """
            )
        st.write("")
    
    # Model selection
    models = get_provider_models(provider)
    if provider == "OpenAI-Compatible":
        st.markdown("**Model Name**")
        model = ui.input(
            default_value=st.session_state.llm_model or "llama2",
            placeholder="e.g., llama2, mistral",
            key="sidebar_model_input"
        )
        st.session_state.llm_model = model
    else:
        default_model = LLM_PROVIDERS[provider]["default"]
        current_model = st.session_state.llm_model or default_model
        if current_model not in models:
            current_model = default_model
            st.session_state.llm_model = default_model
        
        # Pre-select based on state
        if "sidebar_model_select" not in st.session_state:
            st.session_state["sidebar_model_select"] = current_model

        model = ui.select(
            options=models,
            key="sidebar_model_select",
            label="Model"
        )
        st.session_state.llm_model = model
    
    st.write("")

    # API Key
    st.markdown("**API Key**")
    env_key = get_api_key_from_env(provider)
    if env_key:
        st.success("✓ API key from environment")
        api_key = env_key
        st.session_state.llm_api_key = ""  # Clear stored key if env var exists
    else:
        api_key = ui.input(
            default_value=st.session_state.llm_api_key,
            placeholder=f"Enter your {provider} API key",
            key="sidebar_api_key",
            type="password"
        )
        st.session_state.llm_api_key = api_key
    
    st.write("")

    # Base URL (for OpenAI-Compatible)
    if provider == "OpenAI-Compatible":
        st.markdown("**Base URL**")
        base_url = ui.input(
            default_value=st.session_state.llm_base_url or "http://localhost:11434/v1",
            placeholder="http://localhost:11434/v1",
            key="sidebar_base_url"
        )
        st.session_state.llm_base_url = base_url
    else:
        base_url = None
        st.session_state.llm_base_url = ""
    
    # Advanced settings
    with st.expander("Advanced Settings", expanded=False):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.llm_temperature,
            step=0.1,
            key="sidebar_temperature",
            help="Higher values make output more random, lower values more deterministic."
        )
        st.session_state.llm_temperature = temperature
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=st.session_state.llm_max_tokens,
            step=100,
            key="sidebar_max_tokens",
            help="Maximum number of tokens in the response."
        )
        st.session_state.llm_max_tokens = max_tokens
        
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.llm_system_prompt,
            height=100,
            key="sidebar_system_prompt",
            help="Instructions for how the model should behave."
        )
        st.session_state.llm_system_prompt = system_prompt
    
    st.write("")

    # Save button
    if ui.button("Save Configuration", variant="primary", key="save_config_btn"):
        config_data = {
            "provider": provider,
            "model": model,
            "api_key": api_key if not env_key else "",  # Don't save if from env
            "base_url": base_url if provider == "OpenAI-Compatible" else None,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
        }
        save_llm_config(config_data)
    
    # Validation status
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    is_valid, error_msg = validate_config(config)
    if not is_valid:
        st.warning(f"⚠️ {error_msg}")


def render_sidebar() -> None:
    """Render the main sidebar with tabs."""
    with st.sidebar:
        # Using ui.tabs in sidebar doesn't work well due to container width issues
        # Stick to st.tabs for top-level sidebar navigation for now, 
        # but usage of shadcn components inside tabs is fine.
        tab1, tab2 = st.tabs(["RAG Config", "LLM Config"])
        
        with tab1:
            render_rag_config_sidebar()
        
        with tab2:
            render_llm_sidebar()

