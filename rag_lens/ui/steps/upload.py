from pathlib import Path

import streamlit as st
import streamlit_shadcn_ui as ui

from rag_lens.services.storage import (
    clear_documents,
    get_current_document,
    save_document,
)


def render_upload_step() -> None:
    # Initialize session state for document metadata
    if "document_metadata" not in st.session_state:
        st.session_state.document_metadata = None

    # Check if stored document matches session state
    current_doc = get_current_document()
    if current_doc:
        doc_name, _ = current_doc
        # If session state has different document, clear it
        if st.session_state.document_metadata and st.session_state.document_metadata.get("name") != doc_name:
            st.session_state.document_metadata = None
    else:
        # No document stored, clear metadata
        st.session_state.document_metadata = None

    # File uploader section
    with st.container(border=True):
        st.markdown("### Upload Document")
        st.caption("Supported formats: PDF, DOCX, PPTX, XLSX, HTML, MD, TXT, PNG, JPG")

        uploaded_file = st.file_uploader(
            "Choose a file to upload",
            type=["pdf", "docx", "pptx", "xlsx", "html", "htm", "md", "txt", "png", "jpg", "jpeg", "bmp", "tiff", "tif"],
            accept_multiple_files=False,
            label_visibility="collapsed"
        )

        # Handle file upload
        if uploaded_file is not None:
            # Check if this is a new file (different from current)
            is_new_file = (
                st.session_state.document_metadata is None
                or st.session_state.document_metadata.get("name") != uploaded_file.name
            )

            if is_new_file:
                try:
                    content = uploaded_file.read()

                    # Save document (replaces any existing document)
                    doc_path = save_document(uploaded_file.name, content)

                    # Get basic metadata
                    file_format = Path(uploaded_file.name).suffix.upper().lstrip(".")
                    size_bytes = len(content)

                    # Store metadata in session state (single document)
                    st.session_state.document_metadata = {
                        "name": uploaded_file.name,
                        "format": file_format,
                        "size_bytes": size_bytes,
                        "path": str(doc_path),
                    }

                    # Set as current document
                    st.session_state.doc_name = uploaded_file.name

                    # Invalidate chunks and embeddings for new file
                    for key in ["chunks", "last_embeddings_result", "search_results"]:
                        if key in st.session_state:
                            del st.session_state[key]

                    st.success(f"Uploaded: {uploaded_file.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to upload {uploaded_file.name}: {str(e)}")

    # Load metadata for existing document if not in session state
    if current_doc and st.session_state.document_metadata is None:
        doc_name, content = current_doc
        file_format = Path(doc_name).suffix.upper().lstrip(".")
        size_bytes = len(content)

        st.session_state.document_metadata = {
            "name": doc_name,
            "format": file_format,
            "size_bytes": size_bytes,
            "path": "",
        }
        st.session_state.doc_name = doc_name

    # Display current document
    metadata = st.session_state.document_metadata
    if metadata:
        st.write("")
        st.markdown("### Current Document")

        with st.container(border=True):
            col1, col2 = st.columns([5, 1])

            with col1:
                file_format = metadata.get("format", "Unknown")
                size_bytes = metadata.get("size_bytes", 0)

                # Format file size
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

                st.markdown(f"**{metadata.get('name', 'Unknown')}**")
                st.caption(f"{file_format} | {size_str}")

            with col2:
                if ui.button("Delete", variant="destructive", key="delete_current_doc"):
                    clear_documents()
                    st.session_state.document_metadata = None
                    st.session_state.doc_name = None
                    for key in ["chunks", "last_embeddings_result", "search_results"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

    else:
        st.info("No document uploaded yet. Use the file uploader above to add a document.")
