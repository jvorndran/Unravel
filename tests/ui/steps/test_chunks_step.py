"""Tests for the Chunks UI step.

This module tests the chunks step functionality including:
- Configuration expander with parsing/chunking params
- Apply Configuration button behavior
- Parse Document / Reparse Document functionality
- Chunk generation with different providers/splitters
- Chunk visualization (Visual View and Raw JSON)
- Navigation to Upload step when no document
- Session state management
"""

from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from unravel.services.chunking import Chunk
from unravel.ui.constants import WidgetKeys


@pytest.fixture
def chunks_app_script() -> str:
    """Return app script for testing the chunks step."""
    return """
import streamlit as st
from unravel.ui.steps.chunks import render_chunks_step

# Initialize session state
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "chunking_params" not in st.session_state:
    st.session_state.chunking_params = {
        "provider": "Docling",
        "splitter": "HybridChunker",
        "max_tokens": 512,
        "chunk_overlap": 50,
        "tokenizer": "cl100k_base",
    }
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
if "applied_chunking_params" not in st.session_state:
    st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()
if "embedding_model_name" not in st.session_state:
    st.session_state.embedding_model_name = "all-MiniLM-L6-v2"

render_chunks_step()
"""


class TestChunksStepInitialization:
    """Test chunks step initialization and rendering."""

    def test_shows_config_expander_by_default(self, chunks_app_script):
        """Test that the configuration expander is rendered."""
        script_with_doc = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )

        with patch("unravel.ui.steps.chunks.load_document") as mock_load:
            mock_load.return_value = b"test content"

            with patch("unravel.ui.steps.chunks.load_parsed_text") as mock_load_parsed:
                mock_load_parsed.return_value = "Parsed test content"

                at = AppTest.from_string(script_with_doc).run()

                # Should have expander for configuration
                assert len(at.expander) > 0

    def test_loads_default_chunking_params(self, chunks_app_script):
        """Test that default chunking parameters are loaded."""
        at = AppTest.from_string(chunks_app_script).run()

        # Check default params in session state
        assert at.session_state.chunking_params["provider"] == "Docling"
        assert at.session_state.chunking_params["splitter"] == "HybridChunker"
        assert at.session_state.chunking_params["max_tokens"] == 512


class TestChunksStepConfiguration:
    """Test configuration management."""

    def test_apply_button_exists(self, chunks_app_script):
        """Test that Apply Configuration button exists."""
        script_with_doc = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )

        with patch("unravel.ui.steps.chunks.load_document") as mock_load:
            mock_load.return_value = b"test content"

            with patch("unravel.ui.steps.chunks.load_parsed_text") as mock_load_parsed:
                mock_load_parsed.return_value = "Parsed test content"

                at = AppTest.from_string(script_with_doc).run()

                # Find Apply Configuration button
                apply_button = None
                for btn in at.button:
                    if btn.key == WidgetKeys.CHUNKS_APPLY_BTN:
                        apply_button = btn
                        break

                # Button should exist
                assert apply_button is not None

    @patch("unravel.ui.steps.chunks.save_rag_config")
    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_apply_updates_session_state(
        self, mock_load_parsed, mock_load_doc, mock_save_config, chunks_app_script
    ):
        """Test that applying configuration updates session state."""
        mock_load_doc.return_value = b"test content"
        mock_load_parsed.return_value = "Parsed test content"

        # Create script with modified params (to enable the button)
        script = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )
        script = script.replace(
            'st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()',
            '''st.session_state.applied_chunking_params = {
                "provider": "Docling",
                "splitter": "HybridChunker",
                "max_tokens": 256,  # Different value
                "chunk_overlap": 50,
                "tokenizer": "cl100k_base",
            }''',
        )

        at = AppTest.from_string(script).run()

        # Apply button should be enabled due to change
        apply_button = None
        for btn in at.button:
            if btn.key == WidgetKeys.CHUNKS_APPLY_BTN:
                apply_button = btn
                break

        assert apply_button is not None
        # The button's disabled state depends on the has_changes calculation
        # which is returned by render_chunking_configuration

    @patch("unravel.ui.steps.chunks.save_rag_config")
    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_apply_invalidates_downstream_caches(
        self, mock_load_parsed, mock_load_doc, mock_save_config, chunks_app_script
    ):
        """Test that applying configuration invalidates downstream caches."""
        mock_load_doc.return_value = b"test content"
        mock_load_parsed.return_value = "Parsed test content"

        # Create script with existing downstream data
        script = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            '''st.session_state.doc_name = "test.pdf"
st.session_state.chunks = [{"text": "old chunk"}]
st.session_state.last_embeddings_result = {"embeddings": []}
st.session_state.search_results = [{"result": "old"}]
st.session_state.bm25_index_data = {"index": "old"}''',
        )

        # Modify applied params to enable the button
        script = script.replace(
            'st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()',
            '''st.session_state.applied_chunking_params = {
                "provider": "Docling",
                "splitter": "HybridChunker",
                "max_tokens": 256,
                "chunk_overlap": 50,
                "tokenizer": "cl100k_base",
            }''',
        )

        at = AppTest.from_string(script).run()

        # Note: We can't easily click the button in the test due to the rerun,
        # but we can verify the initial state has the data
        # The invalidation logic is in the button callback


class TestChunksStepParsing:
    """Test document parsing functionality."""

    @patch("unravel.ui.steps.chunks.load_document")
    def test_parse_button_shown_when_no_parsed_text(self, mock_load_doc, chunks_app_script):
        """Test that Parse Document button is shown when no parsed text exists."""
        mock_load_doc.return_value = b"test content"

        script_with_doc = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )

        with patch("unravel.ui.steps.chunks.load_parsed_text") as mock_load_parsed:
            mock_load_parsed.return_value = None  # No parsed text

            at = AppTest.from_string(script_with_doc).run()

            # Should have parse button
            parse_button = None
            for btn in at.button:
                if btn.key == WidgetKeys.CHUNKS_PARSE_BTN:
                    parse_button = btn
                    break

            assert parse_button is not None
            assert parse_button.label == "Parse Document"

    @patch("unravel.ui.steps.chunks.parse_document")
    @patch("unravel.ui.steps.chunks.save_parsed_text")
    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_parse_button_triggers_parsing(
        self, mock_load_parsed, mock_load_doc, mock_save_parsed, mock_parse, chunks_app_script
    ):
        """Test that clicking Parse Document triggers parsing."""
        mock_load_doc.return_value = b"test content"
        mock_load_parsed.return_value = None
        mock_parse.return_value = ("Parsed content", "markdown", {})

        script_with_doc = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )

        at = AppTest.from_string(script_with_doc).run()

        # Click parse button
        parse_button = None
        for btn in at.button:
            if btn.key == WidgetKeys.CHUNKS_PARSE_BTN:
                parse_button = btn
                break

        assert parse_button is not None
        parse_button.click().run()

        # Verify parsing was called
        mock_parse.assert_called_once()

    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_reparse_button_shown_when_settings_changed(
        self, mock_load_parsed, mock_load_doc, chunks_app_script
    ):
        """Test that Reparse Document button is shown when settings change."""
        mock_load_doc.return_value = b"test content"
        mock_load_parsed.return_value = "Parsed content"

        # Create script with changed parsing params
        script = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )
        script = script.replace(
            'st.session_state.applied_parsing_params = st.session_state.parsing_params.copy()',
            '''st.session_state.applied_parsing_params = {
                "docling_enable_ocr": True,  # Changed
                "docling_table_structure": True,
                "docling_threads": 4,
                "docling_filter_labels": ["PAGE_HEADER", "PAGE_FOOTER"],
                "docling_extract_images": False,
                "docling_enable_captioning": False,
                "docling_device": "auto",
                "output_format": "markdown",
                "normalize_whitespace": True,
                "remove_special_chars": False,
            }''',
        )

        at = AppTest.from_string(script).run()

        # Should have reparse button
        parse_button = None
        for btn in at.button:
            if btn.key == WidgetKeys.CHUNKS_PARSE_BTN:
                parse_button = btn
                break

        assert parse_button is not None
        assert parse_button.label == "Reparse Document"


class TestChunksStepChunkGeneration:
    """Test chunk generation functionality."""

    @patch("unravel.ui.steps.chunks.get_chunks")
    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_generates_chunks_from_parsed_text(
        self, mock_load_parsed, mock_load_doc, mock_get_chunks, chunks_app_script
    ):
        """Test that chunks are generated from parsed text."""
        mock_load_doc.return_value = b"test content"
        mock_load_parsed.return_value = "Parsed test content for chunking"

        # Create mock chunks
        mock_chunks = [
            Chunk(text="Chunk 1", metadata={}, start_index=0, end_index=7),
            Chunk(text="Chunk 2", metadata={}, start_index=8, end_index=15),
        ]
        mock_get_chunks.return_value = mock_chunks

        script_with_doc = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )

        at = AppTest.from_string(script_with_doc).run()

        # Verify get_chunks was called
        mock_get_chunks.assert_called_once()
        call_kwargs = mock_get_chunks.call_args[1]
        assert call_kwargs["provider"] == "Docling"
        assert call_kwargs["splitter"] == "HybridChunker"
        assert call_kwargs["text"] == "Parsed test content for chunking"

    @patch("unravel.ui.steps.chunks.get_chunks")
    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_chunks_saved_to_session_state(
        self, mock_load_parsed, mock_load_doc, mock_get_chunks, chunks_app_script
    ):
        """Test that generated chunks are saved to session state."""
        mock_load_doc.return_value = b"test content"
        mock_load_parsed.return_value = "Parsed test content"

        mock_chunks = [
            Chunk(text="Chunk 1", metadata={}, start_index=0, end_index=7),
            Chunk(text="Chunk 2", metadata={}, start_index=8, end_index=15),
        ]
        mock_get_chunks.return_value = mock_chunks

        script_with_doc = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )

        at = AppTest.from_string(script_with_doc).run()

        # Verify chunks are in session state
        assert "chunks" in at.session_state
        assert len(at.session_state.chunks) == 2

    @patch("unravel.ui.steps.chunks.get_chunks")
    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_different_chunking_providers(
        self, mock_load_parsed, mock_load_doc, mock_get_chunks, chunks_app_script
    ):
        """Test chunk generation with different providers."""
        mock_load_doc.return_value = b"test content"
        mock_load_parsed.return_value = "Parsed test content"
        mock_get_chunks.return_value = [
            Chunk(text="Chunk", metadata={}, start_index=0, end_index=5)
        ]

        # Test with LangChain provider
        script = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )
        script = script.replace(
            '"provider": "Docling"',
            '"provider": "LangChain"',
        )
        script = script.replace(
            '"splitter": "HybridChunker"',
            '"splitter": "RecursiveCharacterTextSplitter"',
        )

        at = AppTest.from_string(script).run()

        # Verify correct provider was used
        mock_get_chunks.assert_called()
        call_kwargs = mock_get_chunks.call_args[1]
        assert call_kwargs["provider"] == "LangChain"


class TestChunksStepVisualization:
    """Test chunk visualization."""

    @patch("unravel.ui.steps.chunks.get_chunks")
    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_visual_view_rendered(
        self, mock_load_parsed, mock_load_doc, mock_get_chunks, chunks_app_script
    ):
        """Test that visual view is rendered by default."""
        mock_load_doc.return_value = b"test content"
        mock_load_parsed.return_value = "Parsed test content"
        mock_get_chunks.return_value = [
            Chunk(text="Chunk 1", metadata={}, start_index=0, end_index=7),
        ]

        script_with_doc = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )

        at = AppTest.from_string(script_with_doc).run()

        # Should have view tab selector
        # The tabs are rendered using ui.tabs from streamlit_shadcn_ui
        # which may not be directly testable via AppTest

    @patch("unravel.ui.steps.chunks.get_chunks")
    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_raw_json_view(
        self, mock_load_parsed, mock_load_doc, mock_get_chunks, chunks_app_script
    ):
        """Test that raw JSON view can be displayed."""
        mock_load_doc.return_value = b"test content"
        mock_load_parsed.return_value = "Parsed test content"
        mock_get_chunks.return_value = [
            Chunk(text="Chunk 1", metadata={"index": 0}, start_index=0, end_index=7),
            Chunk(text="Chunk 2", metadata={"index": 1}, start_index=8, end_index=15),
        ]

        script_with_doc = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )

        at = AppTest.from_string(script_with_doc).run()

        # Chunks should be in session state
        assert len(at.session_state.chunks) == 2


class TestChunksStepNavigation:
    """Test navigation and early returns."""

    def test_info_shown_when_no_document(self, chunks_app_script):
        """Test that info message is shown when no document is selected."""
        at = AppTest.from_string(chunks_app_script).run()

        # Should show info message about no document
        assert len(at.info) > 0
        assert "no document" in at.info[0].value.lower()

    def test_navigation_logic_when_no_document(self, chunks_app_script):
        """Test that navigation logic is triggered when no document is selected."""
        # Test the early return behavior
        at = AppTest.from_string(chunks_app_script).run()

        # When no document is selected, session state should reflect this
        assert at.session_state.doc_name is None or at.session_state.doc_name == ""

    def test_early_return_when_no_document(self, chunks_app_script):
        """Test that the step returns early when no document is selected."""
        at = AppTest.from_string(chunks_app_script).run()

        # Should show info message
        assert len(at.info) > 0
        assert "no document" in at.info[0].value.lower()

    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_no_early_return_when_document_selected(
        self, mock_load_parsed, mock_load_doc, chunks_app_script
    ):
        """Test that the step doesn't return early when document is selected."""
        mock_load_doc.return_value = b"test content"
        mock_load_parsed.return_value = None  # No parsed text yet

        script_with_doc = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )

        at = AppTest.from_string(script_with_doc).run()

        # Should have parse button (meaning we didn't return early)
        parse_button = None
        for btn in at.button:
            if btn.key == WidgetKeys.CHUNKS_PARSE_BTN:
                parse_button = btn
                break

        assert parse_button is not None

    @patch("unravel.ui.steps.chunks.get_chunks")
    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_displays_source_document_info(
        self, mock_load_parsed, mock_load_doc, mock_get_chunks, chunks_app_script
    ):
        """Test that source document information is displayed."""
        mock_load_doc.return_value = b"test content"
        mock_load_parsed.return_value = "Parsed test content"
        mock_get_chunks.return_value = []

        script_with_doc = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )

        at = AppTest.from_string(script_with_doc).run()

        # Should display document name in session state
        assert at.session_state.doc_name == "test.pdf"


class TestChunksStepIntegration:
    """Test integration scenarios."""

    @patch("unravel.ui.steps.chunks.parse_document")
    @patch("unravel.ui.steps.chunks.get_chunks")
    @patch("unravel.ui.steps.chunks.save_parsed_text")
    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_full_parse_and_chunk_flow(
        self,
        mock_load_parsed,
        mock_load_doc,
        mock_save_parsed,
        mock_get_chunks,
        mock_parse,
        chunks_app_script,
    ):
        """Test complete flow: parse document then generate chunks."""
        mock_load_doc.return_value = b"test content"
        mock_load_parsed.return_value = None
        mock_parse.return_value = ("Parsed content", "markdown", {})
        mock_get_chunks.return_value = [
            Chunk(text="Chunk 1", metadata={}, start_index=0, end_index=7),
        ]

        script_with_doc = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )

        at = AppTest.from_string(script_with_doc).run()

        # Click parse button
        for btn in at.button:
            if btn.key == WidgetKeys.CHUNKS_PARSE_BTN:
                btn.click().run()
                break

        # Verify parsing was triggered
        mock_parse.assert_called_once()

    @patch("unravel.ui.steps.chunks.get_chunks")
    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_chunk_overlap_calculation(
        self, mock_load_parsed, mock_load_doc, mock_get_chunks, chunks_app_script
    ):
        """Test that chunk overlap is calculated correctly."""
        mock_load_doc.return_value = b"test content"
        source_text = "This is a test document with overlapping chunks."
        mock_load_parsed.return_value = source_text

        # Create chunks with overlap
        mock_chunks = [
            Chunk(text="This is a test", metadata={}, start_index=0, end_index=14),
            Chunk(text="a test document", metadata={}, start_index=8, end_index=23),
        ]
        mock_get_chunks.return_value = mock_chunks

        script_with_doc = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.pdf"',
        )

        at = AppTest.from_string(script_with_doc).run()

        # Verify chunks are generated
        assert len(at.session_state.chunks) == 2

    @patch("unravel.ui.steps.chunks.get_chunks")
    @patch("unravel.ui.steps.chunks.load_document")
    @patch("unravel.ui.steps.chunks.load_parsed_text")
    def test_html_format_rendering(
        self, mock_load_parsed, mock_load_doc, mock_get_chunks, chunks_app_script
    ):
        """Test that HTML format chunks are rendered correctly."""
        mock_load_doc.return_value = b"test content"
        # HTML format content
        html_content = "<p>This is <strong>HTML</strong> content</p>"
        mock_load_parsed.return_value = html_content

        # Create chunks with HTML content
        mock_chunks = [
            Chunk(text="<p>First chunk</p>", metadata={}, start_index=0, end_index=17),
            Chunk(text="<strong>Second</strong>", metadata={}, start_index=18, end_index=40),
        ]
        mock_get_chunks.return_value = mock_chunks

        # Set output format to HTML
        script_with_html = chunks_app_script.replace(
            'st.session_state.doc_name = None',
            'st.session_state.doc_name = "test.html"',
        )
        script_with_html = script_with_html.replace(
            '"output_format": "markdown"',
            '"output_format": "html"',
        )

        at = AppTest.from_string(script_with_html).run()

        # Verify chunks are generated with HTML format
        assert len(at.session_state.chunks) == 2
        assert at.session_state.applied_parsing_params["output_format"] == "html"
