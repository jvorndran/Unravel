"""Tests for the Export step UI.

Tests code generation functionality for exporting the configured RAG pipeline
as Python code snippets (installation, parsing, chunking, embedding, retrieval, etc.).
"""

import pytest
from streamlit.testing.v1 import AppTest

from unravel.ui.constants import WidgetKeys


@pytest.fixture
def export_app_script() -> str:
    """Return the app script string for testing the export step."""
    return '''
import streamlit as st
from unravel.ui.steps.export import render_export_step

# Initialize required session state (mirrors production initialization)
if "chunking_params" not in st.session_state:
    st.session_state.chunking_params = {
        "provider": "Docling",
        "splitter": "HybridChunker",
        "max_tokens": 512,
        "chunk_overlap": 50,
        "tokenizer": "cl100k_base",
    }
if "applied_chunking_params" not in st.session_state:
    st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()
if "parsing_params" not in st.session_state:
    st.session_state.parsing_params = {
        "docling_enable_ocr": False,
        "docling_table_structure": True,
        "docling_threads": 4,
        "docling_filter_labels": ["PAGE_HEADER", "PAGE_FOOTER"],
        "docling_extract_images": False,
        "docling_enable_captioning": False,
        "docling_device": "auto",
        "output_format": "markdown",
        "normalize_whitespace": True,
        "remove_special_chars": False,
    }
if "applied_parsing_params" not in st.session_state:
    st.session_state.applied_parsing_params = st.session_state.parsing_params.copy()
if "embedding_model_name" not in st.session_state:
    st.session_state.embedding_model_name = "all-MiniLM-L6-v2"
if "doc_name" not in st.session_state:
    st.session_state.doc_name = "sample_document.pdf"

render_export_step()
'''


class TestExportStepInitialization:
    """Test export step initialization and configuration requirements."""

    def test_shows_info_when_no_config(self):
        """Test that the step shows an info message when no chunking config is set."""
        at = AppTest.from_string('''
import streamlit as st
from unravel.ui.steps.export import render_export_step

# Initialize with NO chunking params
if "chunking_params" not in st.session_state:
    st.session_state.chunking_params = None

render_export_step()
''').run(timeout=30)

        assert not at.exception
        # Should show info message
        assert len(at.info) > 0
        assert any("Configure your pipeline" in str(info.value) for info in at.info)
        # Note: Button uses streamlit_shadcn_ui which may not be detected by AppTest

    def test_loads_chunking_params_from_session_state(self, export_app_script: str):
        """Test that the step loads chunking params from session state."""
        at = AppTest.from_string(export_app_script).run(timeout=30)

        assert not at.exception
        # Should not show error/info about missing config
        assert not any("Configure your pipeline" in str(info.value) for info in at.info)

    def test_loads_embedding_model_from_session_state(self, export_app_script: str):
        """Test that the step loads embedding model from session state."""
        at = AppTest.from_string(export_app_script).run(timeout=30)

        assert not at.exception
        # Should not show error/info about missing config
        assert not any("Configure your pipeline" in str(info.value) for info in at.info)


class TestExportStepCodeGeneration:
    """Test code generation for different pipeline components."""

    def test_generates_installation_commands(self, export_app_script: str):
        """Test that installation commands are generated correctly."""
        at = AppTest.from_string(export_app_script).run(timeout=30)

        assert not at.exception
        # Should have Installation section (check both markdown and caption)
        all_text = [str(md.value) for md in at.markdown] + [str(c.value) for c in at.caption]
        assert any("Installation" in text for text in all_text), f"Expected 'Installation' section. Found: {all_text}"
        # Should have code block with pip install
        code_blocks = [c.value for c in at.code]
        assert any("pip install" in code for code in code_blocks), f"Expected 'pip install' in code. Found: {code_blocks}"

    def test_generates_parsing_code(self, export_app_script: str):
        """Test that document parsing code is generated."""
        at = AppTest.from_string(export_app_script).run(timeout=30)

        assert not at.exception
        # Should have Document Parsing section (check both markdown and caption)
        all_text = [str(md.value) for md in at.markdown] + [str(c.value) for c in at.caption]
        assert any("Parsing" in text for text in all_text)
        # Should have code block with parsing logic
        code_blocks = [c.value for c in at.code]
        assert any("docling" in code.lower() or "parse" in code.lower() for code in code_blocks)

    def test_generates_chunking_code(self, export_app_script: str):
        """Test that text chunking code is generated."""
        at = AppTest.from_string(export_app_script).run(timeout=30)

        assert not at.exception
        # Should have Text Chunking section (check both markdown and caption)
        all_text = [str(md.value) for md in at.markdown] + [str(c.value) for c in at.caption]
        assert any("Chunking" in text for text in all_text)
        # Should have code block with chunking logic
        code_blocks = [c.value for c in at.code]
        assert any(
            "HybridChunker" in code or "chunk" in code.lower() for code in code_blocks
        ), "Should generate chunking code"

    def test_generates_embedding_code(self, export_app_script: str):
        """Test that embedding generation code is generated."""
        at = AppTest.from_string(export_app_script).run(timeout=30)

        assert not at.exception
        # Should have Embedding Generation section (check both markdown and caption)
        all_text = [str(md.value) for md in at.markdown] + [str(c.value) for c in at.caption]
        assert any("Embedding" in text for text in all_text)
        # Should have code block with embedding logic
        code_blocks = [c.value for c in at.code]
        assert any(
            "all-MiniLM-L6-v2" in code or "embed" in code.lower() for code in code_blocks
        ), "Should generate embedding code"

    def test_generates_retrieval_code_when_configured(self):
        """Test that retrieval code is generated when retrieval config is present."""
        app_script = '''
import streamlit as st
from unravel.ui.steps.export import render_export_step

# Initialize with retrieval config
st.session_state.chunking_params = {
    "provider": "Docling",
    "splitter": "HybridChunker",
    "max_tokens": 512,
    "chunk_overlap": 50,
}
st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()
st.session_state.parsing_params = {"output_format": "markdown"}
st.session_state.applied_parsing_params = st.session_state.parsing_params.copy()
st.session_state.embedding_model_name = "all-MiniLM-L6-v2"
st.session_state.doc_name = "test.pdf"
st.session_state.retrieval_config = {
    "strategy": "DenseRetriever",
    "params": {"top_k": 5}
}

render_export_step()
'''
        at = AppTest.from_string(app_script).run(timeout=30)

        assert not at.exception
        # Should have Retrieval Strategy section (check both markdown and caption)
        all_text = [str(md.value) for md in at.markdown] + [str(c.value) for c in at.caption]
        assert any("Retrieval" in text for text in all_text)
        # Should have code block with retrieval logic
        code_blocks = [c.value for c in at.code]
        assert any("retriev" in code.lower() for code in code_blocks)

    def test_generates_reranking_code_when_configured(self):
        """Test that reranking code is generated when reranking is enabled."""
        app_script = '''
import streamlit as st
from unravel.ui.steps.export import render_export_step

# Initialize with reranking config
st.session_state.chunking_params = {
    "provider": "Docling",
    "splitter": "HybridChunker",
    "max_tokens": 512,
}
st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()
st.session_state.parsing_params = {"output_format": "markdown"}
st.session_state.applied_parsing_params = st.session_state.parsing_params.copy()
st.session_state.embedding_model_name = "all-MiniLM-L6-v2"
st.session_state.doc_name = "test.pdf"
st.session_state.reranking_config = {
    "enabled": True,
    "model": "ms-marco-MiniLM-L-12-v2",
    "top_n": 5
}

render_export_step()
'''
        at = AppTest.from_string(app_script).run(timeout=30)

        assert not at.exception
        # Should have Reranking section (check both markdown and caption)
        all_text = [str(md.value) for md in at.markdown] + [str(c.value) for c in at.caption]
        assert any("rerank" in text.lower() for text in all_text)

    def test_generates_llm_code_when_configured(self):
        """Test that LLM/RAG code is generated when LLM is configured."""
        app_script = '''
import streamlit as st
from unravel.ui.steps.export import render_export_step

# Initialize with LLM config
st.session_state.chunking_params = {
    "provider": "Docling",
    "splitter": "HybridChunker",
    "max_tokens": 512,
}
st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()
st.session_state.parsing_params = {"output_format": "markdown"}
st.session_state.applied_parsing_params = st.session_state.parsing_params.copy()
st.session_state.embedding_model_name = "all-MiniLM-L6-v2"
st.session_state.doc_name = "test.pdf"
st.session_state.llm_provider = "OpenAI"
st.session_state.llm_model = "gpt-4"
st.session_state.llm_temperature = 0.7
st.session_state.llm_max_tokens = 1024

render_export_step()
'''
        at = AppTest.from_string(app_script).run(timeout=30)

        assert not at.exception
        # Should have RAG Response Generation section (check both markdown and caption)
        all_text = [str(md.value) for md in at.markdown] + [str(c.value) for c in at.caption]
        assert any("RAG" in text or "Response" in text or "Generation" in text for text in all_text)

    def test_generates_full_pipeline_script(self, export_app_script: str):
        """Test that a full pipeline script is generated."""
        at = AppTest.from_string(export_app_script).run(timeout=30)

        assert not at.exception
        # Should have Full Pipeline expander
        expanders = [str(e.label) for e in at.expander]
        assert any("Full Pipeline" in exp for exp in expanders)


class TestExportStepFileFormatDetection:
    """Test file format detection for different document types."""

    def test_detects_pdf_format(self):
        """Test that PDF format is detected from filename."""
        app_script = '''
import streamlit as st
from unravel.ui.steps.export import render_export_step

st.session_state.chunking_params = {"splitter": "HybridChunker", "max_tokens": 512}
st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()
st.session_state.parsing_params = {"output_format": "markdown"}
st.session_state.embedding_model_name = "all-MiniLM-L6-v2"
st.session_state.doc_name = "document.pdf"

render_export_step()
'''
        at = AppTest.from_string(app_script).run(timeout=30)

        assert not at.exception
        # Should generate parsing code for PDF
        code_blocks = [c.value for c in at.code]
        # Docling is the default PDF parser
        assert any("docling" in code.lower() for code in code_blocks)

    def test_detects_docx_format(self):
        """Test that DOCX format is detected from filename."""
        app_script = '''
import streamlit as st
from unravel.ui.steps.export import render_export_step

st.session_state.chunking_params = {"splitter": "HybridChunker", "max_tokens": 512}
st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()
st.session_state.parsing_params = {"output_format": "markdown"}
st.session_state.embedding_model_name = "all-MiniLM-L6-v2"
st.session_state.doc_name = "document.docx"

render_export_step()
'''
        at = AppTest.from_string(app_script).run(timeout=30)

        assert not at.exception
        # Should generate parsing code for DOCX
        code_blocks = [c.value for c in at.code]
        # DOCX uses python-docx
        assert any("docx" in code.lower() for code in code_blocks)

    def test_detects_markdown_format(self):
        """Test that Markdown format is detected from filename."""
        app_script = '''
import streamlit as st
from unravel.ui.steps.export import render_export_step

st.session_state.chunking_params = {"splitter": "HybridChunker", "max_tokens": 512}
st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()
st.session_state.parsing_params = {"output_format": "markdown"}
st.session_state.embedding_model_name = "all-MiniLM-L6-v2"
st.session_state.doc_name = "document.md"

render_export_step()
'''
        at = AppTest.from_string(app_script).run(timeout=30)

        assert not at.exception
        # Should generate simple text parsing code for markdown
        code_blocks = [c.value for c in at.code]
        # Markdown/text files use simple text parsing (open, read)
        assert any("open" in code.lower() or "read" in code.lower() for code in code_blocks)


class TestExportStepEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_missing_doc_name(self):
        """Test that the step handles missing document name gracefully."""
        app_script = '''
import streamlit as st
from unravel.ui.steps.export import render_export_step

st.session_state.chunking_params = {"splitter": "HybridChunker", "max_tokens": 512}
st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()
st.session_state.parsing_params = {"output_format": "markdown"}
st.session_state.embedding_model_name = "all-MiniLM-L6-v2"
# No doc_name set

render_export_step()
'''
        at = AppTest.from_string(app_script).run(timeout=30)

        assert not at.exception
        # Should still generate code, defaulting to PDF
        code_blocks = [c.value for c in at.code]
        assert len(code_blocks) > 0

    def test_handles_missing_embedding_model(self):
        """Test that the step handles missing embedding model gracefully."""
        app_script = '''
import streamlit as st
from unravel.ui.steps.export import render_export_step

st.session_state.chunking_params = {"splitter": "HybridChunker", "max_tokens": 512}
st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()
st.session_state.parsing_params = {"output_format": "markdown"}
# No embedding_model_name set
st.session_state.doc_name = "test.pdf"

render_export_step()
'''
        at = AppTest.from_string(app_script).run(timeout=30)

        assert not at.exception
        # Should use default embedding model
        code_blocks = [c.value for c in at.code]
        assert any("embed" in code.lower() for code in code_blocks)


class TestExportStepDifferentSplitters:
    """Test code generation for different chunking splitters."""

    def test_generates_code_for_hierarchical_chunker(self):
        """Test code generation for HierarchicalChunker."""
        app_script = '''
import streamlit as st
from unravel.ui.steps.export import render_export_step

st.session_state.chunking_params = {
    "provider": "Docling",
    "splitter": "HierarchicalChunker",
    "include_headers": True,
}
st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()
st.session_state.parsing_params = {"output_format": "markdown"}
st.session_state.embedding_model_name = "all-MiniLM-L6-v2"
st.session_state.doc_name = "test.pdf"

render_export_step()
'''
        at = AppTest.from_string(app_script).run(timeout=30)

        assert not at.exception
        code_blocks = [c.value for c in at.code]
        assert any("HierarchicalChunker" in code for code in code_blocks)

    def test_generates_code_for_hybrid_chunker(self):
        """Test code generation for HybridChunker."""
        app_script = '''
import streamlit as st
from unravel.ui.steps.export import render_export_step

st.session_state.chunking_params = {
    "provider": "Docling",
    "splitter": "HybridChunker",
    "max_tokens": 512,
    "chunk_overlap": 50,
    "tokenizer": "cl100k_base",
}
st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()
st.session_state.parsing_params = {"output_format": "markdown"}
st.session_state.embedding_model_name = "all-MiniLM-L6-v2"
st.session_state.doc_name = "test.pdf"

render_export_step()
'''
        at = AppTest.from_string(app_script).run(timeout=30)

        assert not at.exception
        code_blocks = [c.value for c in at.code]
        assert any("HybridChunker" in code for code in code_blocks)
        assert any("max_tokens" in code.lower() for code in code_blocks)
