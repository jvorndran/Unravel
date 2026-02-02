"""Export step UI for generating pipeline code snippets."""

import streamlit as st
import streamlit_shadcn_ui as ui

from rag_lens.services.embedders import DEFAULT_MODEL
from rag_lens.services.export import (
    ExportConfig,
    generate_chunking_code,
    generate_embedding_code,
    generate_installation_command,
    generate_llm_code,
    generate_parsing_code,
    generate_reranking_code,
    generate_retrieval_code,
    get_config_summary,
)


def render_export_step() -> None:
    """Render the export code step."""
    st.markdown("## Export Code")
    st.caption("Export your configured RAG pipeline as Python code snippets")

    # Check if we have configuration to export
    chunking_params = st.session_state.get(
        "applied_chunking_params",
        st.session_state.get("chunking_params")
    )
    parsing_params = st.session_state.get(
        "applied_parsing_params",
        st.session_state.get("parsing_params", {})
    )
    embedding_model = st.session_state.get("embedding_model_name", DEFAULT_MODEL)

    if not chunking_params:
        st.info("Configure your pipeline first. Go to the Text Splitting or Vector Embedding steps to set up your configuration.")
        if ui.button("Go to Text Splitting", key="goto_chunks_export"):
            st.session_state.current_step = "chunks"
            st.rerun()
        return

    # Determine file format from current document
    doc_name = st.session_state.get("doc_name", "")
    file_format = _get_file_format(doc_name)

    # Get retrieval configuration
    retrieval_config = st.session_state.get("retrieval_config")
    retrieval_strategy = None
    retrieval_params = None
    if retrieval_config:
        retrieval_strategy = retrieval_config.get("strategy")
        params = retrieval_config.get("params", {})
        # Add top_k to params
        retrieval_params = {**params, "top_k": 5}
        # For hybrid, calculate sparse_weight if not provided
        if retrieval_strategy == "HybridRetriever" and "dense_weight" in retrieval_params:
            if "sparse_weight" not in retrieval_params:
                retrieval_params["sparse_weight"] = 1.0 - retrieval_params["dense_weight"]

    # Get reranking configuration
    reranking_config = st.session_state.get("reranking_config")

    # Get LLM configuration
    llm_config = None
    if all(key in st.session_state for key in ["llm_provider", "llm_model"]):
        llm_config = {
            "provider": st.session_state.get("llm_provider"),
            "model": st.session_state.get("llm_model"),
            "temperature": st.session_state.get("llm_temperature", 0.7),
            "max_tokens": st.session_state.get("llm_max_tokens", 1024),
            "base_url": st.session_state.get("llm_base_url"),
            "system_prompt": st.session_state.get("llm_system_prompt", "You are a helpful assistant. Answer questions based on the provided context."),
        }

    # Build export config
    config = ExportConfig(
        parsing_params=parsing_params,
        chunking_params=chunking_params,
        embedding_model=embedding_model,
        file_format=file_format,
        retrieval_strategy=retrieval_strategy,
        retrieval_params=retrieval_params,
        reranking_config=reranking_config,
        llm_config=llm_config,
    )


    # Installation section
    st.markdown("### Installation")
    st.caption("Install the required dependencies")
    st.code(generate_installation_command(config), language="bash")

    st.write("")

    # Parsing section
    st.markdown("### Document Parsing")
    st.caption("Parse documents and extract text content")
    st.code(generate_parsing_code(config), language="python")

    st.write("")

    # Chunking section
    st.markdown("### Text Chunking")
    st.caption("Split text into overlapping chunks for embedding")
    st.code(generate_chunking_code(config), language="python")

    st.write("")

    # Embedding section
    st.markdown("### Embedding Generation")
    st.caption("Generate vector embeddings for semantic search")
    st.code(generate_embedding_code(config), language="python")

    st.write("")

    # Retrieval section (if configured)
    retrieval_code = generate_retrieval_code(config)
    if retrieval_code:
        st.markdown("### Retrieval Strategy")
        st.caption("Search for relevant chunks using your configured strategy")
        st.code(retrieval_code, language="python")
        st.write("")

    # Reranking section (if configured)
    reranking_code = generate_reranking_code(config)
    if reranking_code:
        st.markdown("### Reranking")
        st.caption("Improve retrieval quality with cross-encoder reranking")
        st.code(reranking_code, language="python")
        st.write("")

    # LLM section (if configured)
    llm_code = generate_llm_code(config)
    if llm_code:
        st.markdown("### RAG Response Generation")
        st.caption("Generate answers using retrieved context and LLM")
        st.code(llm_code, language="python")
        st.write("")

    # Full pipeline section
    with st.expander("View Full Pipeline Script", expanded=False):
        st.caption("Combined script with all components")
        full_script = _generate_full_pipeline(config)
        st.code(full_script, language="python")


def _get_file_format(filename: str) -> str:
    """Determine file format from filename."""
    if not filename:
        return "PDF"

    filename_lower = filename.lower()
    if filename_lower.endswith(".pdf"):
        return "PDF"
    elif filename_lower.endswith(".docx"):
        return "DOCX"
    elif filename_lower.endswith((".md", ".markdown")):
        return "Markdown"
    elif filename_lower.endswith(".txt"):
        return "Text"
    else:
        return "PDF"


def _generate_full_pipeline(config: ExportConfig) -> str:
    """Generate a complete pipeline script."""
    summary = get_config_summary(config)
    chunk_size_label = summary.get('chunk_size_key', 'chunk_size')
    chunk_size_display = "Max Tokens" if chunk_size_label == "max_tokens" else "Chunk Size"

    script = f'''"""RAG Pipeline - Full Script

Configuration exported from RAG Lens:
- Parser: {summary['parser']}
- Splitter: {summary['splitter']}
- {chunk_size_display}: {summary['chunk_size']}
- Chunk Overlap: {summary['chunk_overlap']}
- Embedding Model: {summary['embedding_model']}
"""

# =============================================================================
# STEP 1: Document Parsing
# =============================================================================

{_strip_docstring(generate_parsing_code(config))}

# =============================================================================
# STEP 2: Text Chunking
# =============================================================================

{_strip_docstring(generate_chunking_code(config))}

# =============================================================================
# STEP 3: Embedding Generation
# =============================================================================

{_strip_docstring(generate_embedding_code(config))}'''

    # Add retrieval section if configured
    retrieval_code = generate_retrieval_code(config)
    if retrieval_code:
        script += f'''

# =============================================================================
# STEP 4: Retrieval Strategy
# =============================================================================

{_strip_docstring(retrieval_code)}'''

    # Add reranking section if configured
    reranking_code = generate_reranking_code(config)
    if reranking_code:
        script += f'''

# =============================================================================
# STEP 5: Reranking
# =============================================================================

{_strip_docstring(reranking_code)}'''

    # Add LLM section if configured
    llm_code = generate_llm_code(config)
    if llm_code:
        script += f'''

# =============================================================================
# STEP 6: RAG Response Generation
# =============================================================================

{_strip_docstring(llm_code)}'''

    # Determine the final step number
    final_step_num = 4
    if retrieval_code:
        final_step_num += 1
    if reranking_code:
        final_step_num += 1
    if llm_code:
        final_step_num += 1

    script += f'''

# =============================================================================
# STEP {final_step_num}: Full Pipeline Execution
# =============================================================================

def run_pipeline(file_path: str, query: str = None):
    """Run the complete RAG pipeline.

    Args:
        file_path: Path to the document file
        query: Optional query for similarity search

    Returns:
        dict with chunks, embeddings, and optional search results
    """
    # Parse document
    print(f"Parsing {{file_path}}...")
    text = parse_{_get_parse_function_name(config)}(file_path)
    print(f"Extracted {{len(text)}} characters")

    # Chunk text
    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {{len(chunks)}} chunks")

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    print(f"Generated embeddings with shape: {{embeddings.shape}}")

    result = {{
        "text": text,
        "chunks": chunks,
        "embeddings": embeddings,
    }}

    # Optional: RAG query processing
    if query:
        print(f"Processing query: {{query}}")'''

    # Build the query processing logic based on configured features
    has_retrieval = retrieval_code is not None
    has_reranking = reranking_code is not None
    has_llm = llm_code is not None

    if has_retrieval:
        # Use configured retrieval method
        if config.retrieval_strategy == "DenseRetriever":
            script += '''
        retrieved = retrieve_dense(query, chunks, embeddings)'''
        elif config.retrieval_strategy == "SparseRetriever":
            script += '''
        retrieved = retrieve_sparse(query, chunks)'''
        elif config.retrieval_strategy == "HybridRetriever":
            script += '''
        retrieved = retrieve_hybrid(query, chunks, embeddings)'''
    else:
        # Default to simple dense retrieval
        script += '''
        query_embedding = embed_query(query)
        similarities = embeddings @ query_embedding
        top_indices = similarities.argsort()[::-1][:5]
        retrieved = [(chunks[i], float(similarities[i])) for i in top_indices]'''

    if has_reranking:
        script += '''
        candidate_chunks = [chunk for chunk, _ in retrieved]
        reranked = rerank(query, candidate_chunks)
        result["top_chunks"] = reranked'''
    else:
        script += '''
        result["top_chunks"] = retrieved'''

    if has_llm:
        script += '''
        top_chunks = [chunk for chunk, _ in result["top_chunks"]]
        response = generate_rag_response(query, top_chunks)
        result["response"] = response
        print(f"\\nAnswer: {{response}}")'''
    else:
        script += '''

        print("\\nTop results:")
        for i, (chunk, score) in enumerate(result["top_chunks"]):
            print(f"{{i+1}}. Score: {{score:.4f}}")
            print(f"   {{chunk[:100]}}...\\n")'''

    script += '''

    return result


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <file_path> [query]")
        sys.exit(1)

    file_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else None

    result = run_pipeline(file_path, query)
    print(f"\\nPipeline complete! {{len(result['chunks'])}} chunks embedded.")
'''

    return script


def _strip_docstring(code: str) -> str:
    """Remove the module docstring from generated code."""
    lines = code.split('\n')
    in_docstring = False
    result_lines = []

    for i, line in enumerate(lines):
        if i == 0 and line.strip().startswith('"""'):
            in_docstring = True
            # Check if docstring ends on same line
            if line.strip().endswith('"""') and len(line.strip()) > 3:
                continue
            continue

        if in_docstring:
            if '"""' in line:
                in_docstring = False
            continue

        result_lines.append(line)

    return '\n'.join(result_lines).strip()


def _get_parse_function_name(config: ExportConfig) -> str:
    """Get the parse function name based on file format."""
    file_format = config.file_format
    if file_format == "DOCX":
        return "docx"
    elif file_format in ["Text", "Markdown"]:
        return "text"
    else:
        return "pdf"
