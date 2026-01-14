"""Tests for sidebar text splitting features.

Feature 24: Changes Text Splitting Library
- User Action: Changes the Library dropdown
- Expected UI Change: Splitter strategy options update
- Session State Variables Affected: chunking_params.provider, sidebar_chunking_provider

Feature 25: Changes Text Splitting Strategy
- User Action: Changes the Strategy dropdown
- Expected UI Change: Expander info updates, dynamic params appear
- Session State Variables Affected: chunking_params.splitter, sidebar_splitter

Feature 27: Adjusts splitter parameters (max_tokens, overlap, etc.)
- User Action: Changes parameter inputs
- Expected UI Change: Number inputs/selects update
- Session State Variables Affected: chunking_params.<param_name>, sidebar_param_<param_name>

Fixtures used from conftest.py:
- mock_storage_dir: Creates temp directory with test documents
- sidebar_app_script: App script for rendering the sidebar
"""

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.conftest import element_exists, get_form_submit_button


class TestTextSplittingLibraryDisplay:
    """Tests for text splitting library dropdown display."""

    def test_chunking_provider_dropdown_exists(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify chunking provider dropdown is rendered."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            provider_selectbox = at.selectbox(key="sidebar_chunking_provider")
            assert provider_selectbox is not None

    def test_chunking_provider_has_docling(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify chunking provider has Docling option."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            provider_selectbox = at.selectbox(key="sidebar_chunking_provider")
            assert "Docling" in provider_selectbox.options

    def test_chunking_provider_defaults_to_docling(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify chunking provider defaults to Docling."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            provider_selectbox = at.selectbox(key="sidebar_chunking_provider")
            assert provider_selectbox.value == "Docling"


class TestTextSplittingLibrarySessionState:
    """Tests for text splitting library session state."""

    def test_chunking_params_has_provider(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify chunking_params contains provider key."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "chunking_params" in at.session_state
            assert "provider" in at.session_state["chunking_params"]
            assert at.session_state["chunking_params"]["provider"] == "Docling"

    def test_sidebar_chunking_provider_initialized(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify sidebar_chunking_provider session state is initialized."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "sidebar_chunking_provider" in at.session_state
            assert at.session_state["sidebar_chunking_provider"] == "Docling"


class TestTextSplittingStrategyDisplay:
    """Tests for text splitting strategy dropdown display."""

    def test_splitter_dropdown_exists(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify splitter strategy dropdown is rendered."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            splitter_selectbox = at.selectbox(key="sidebar_splitter")
            assert splitter_selectbox is not None

    def test_splitter_has_hybrid_option(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify splitter has Hybrid option."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            splitter_selectbox = at.selectbox(key="sidebar_splitter")
            has_hybrid = any("Hybrid" in opt for opt in splitter_selectbox.options)
            assert has_hybrid

    def test_splitter_has_multiple_options(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify splitter dropdown has multiple strategy options."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            splitter_selectbox = at.selectbox(key="sidebar_splitter")
            assert len(splitter_selectbox.options) >= 2


class TestTextSplittingStrategySessionState:
    """Tests for text splitting strategy session state."""

    def test_chunking_params_has_splitter(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify chunking_params contains splitter key."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "splitter" in at.session_state["chunking_params"]
            assert at.session_state["chunking_params"]["splitter"] == "HybridChunker"

    def test_sidebar_splitter_initialized(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify sidebar_splitter session state is initialized."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "sidebar_splitter" in at.session_state


class TestSplitterStrategyUserAction:
    """Tests for user interaction with splitter strategy dropdown."""

    def test_can_change_splitter_strategy(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify user can change splitter strategy."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            splitter_selectbox = at.selectbox(key="sidebar_splitter")
            initial_value = splitter_selectbox.value

            # Find a different option
            options = splitter_selectbox.options
            other_option = [opt for opt in options if opt != initial_value][0]

            splitter_selectbox.set_value(other_option).run()

            splitter_selectbox = at.selectbox(key="sidebar_splitter")
            assert splitter_selectbox.value == other_option


class TestSplitterParametersDisplay:
    """Tests for splitter parameter inputs display."""

    def test_max_tokens_input_exists(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify max_tokens number input is rendered for HybridChunker."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            max_tokens_input = at.number_input(key="sidebar_param_max_tokens")
            assert max_tokens_input is not None

    def test_chunk_overlap_input_exists(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify chunk_overlap number input is rendered for HybridChunker."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            chunk_overlap_input = at.number_input(key="sidebar_param_chunk_overlap")
            assert chunk_overlap_input is not None

    def test_max_tokens_defaults_to_512(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify max_tokens defaults to 512."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            max_tokens_input = at.number_input(key="sidebar_param_max_tokens")
            assert max_tokens_input.value == 512

    def test_chunk_overlap_defaults_to_50(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify chunk_overlap defaults to 50."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            chunk_overlap_input = at.number_input(key="sidebar_param_chunk_overlap")
            assert chunk_overlap_input.value == 50


class TestSplitterParametersSessionState:
    """Tests for splitter parameters session state."""

    def test_chunking_params_has_max_tokens(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify chunking_params contains max_tokens key."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "max_tokens" in at.session_state["chunking_params"]
            assert at.session_state["chunking_params"]["max_tokens"] == 512

    def test_chunking_params_has_chunk_overlap(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify chunking_params contains chunk_overlap key."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "chunk_overlap" in at.session_state["chunking_params"]
            assert at.session_state["chunking_params"]["chunk_overlap"] == 50


class TestSplitterParametersUserAction:
    """Tests for user interaction with splitter parameter inputs."""

    def test_can_change_max_tokens(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify user can change max_tokens value."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            max_tokens_input = at.number_input(key="sidebar_param_max_tokens")
            max_tokens_input.set_value(1024).run()

            max_tokens_input = at.number_input(key="sidebar_param_max_tokens")
            assert max_tokens_input.value == 1024

    def test_can_change_chunk_overlap(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify user can change chunk_overlap value."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            chunk_overlap_input = at.number_input(key="sidebar_param_chunk_overlap")
            chunk_overlap_input.set_value(100).run()

            chunk_overlap_input = at.number_input(key="sidebar_param_chunk_overlap")
            assert chunk_overlap_input.value == 100


class TestTextSplittingFormSubmission:
    """Tests for text splitting form submission updating session state."""

    def test_form_submit_updates_max_tokens(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission updates chunking_params.max_tokens."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Initial state
            assert at.session_state["chunking_params"]["max_tokens"] == 512

            # Change value
            at.number_input(key="sidebar_param_max_tokens").set_value(1024)

            # Submit the form
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify updated
            assert at.session_state["chunking_params"]["max_tokens"] == 1024

    def test_form_submit_updates_chunk_overlap(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission updates chunking_params.chunk_overlap."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Change value
            at.number_input(key="sidebar_param_chunk_overlap").set_value(75)

            # Submit the form
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify updated
            assert at.session_state["chunking_params"]["chunk_overlap"] == 75

    def test_form_submit_updates_applied_chunking_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission also updates applied_chunking_params."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            at.number_input(key="sidebar_param_max_tokens").set_value(768)
            at.number_input(key="sidebar_param_chunk_overlap").set_value(60)
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            assert at.session_state["chunking_params"]["max_tokens"] == 768
            assert at.session_state["chunking_params"]["chunk_overlap"] == 60
            assert at.session_state["applied_chunking_params"]["max_tokens"] == 768
            assert at.session_state["applied_chunking_params"]["chunk_overlap"] == 60

    def test_form_submit_updates_splitter(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission updates chunking_params.splitter."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Get available options
            splitter_selectbox = at.selectbox(key="sidebar_splitter")
            options = splitter_selectbox.options

            # Find Hierarchical splitter option
            hierarchical_option = next(
                (opt for opt in options if "Hierarchical" in opt),
                None
            )

            if hierarchical_option:
                at.selectbox(key="sidebar_splitter").set_value(hierarchical_option)
                submit_btn = get_form_submit_button(at)
                submit_btn.click().run()

                assert at.session_state["chunking_params"]["splitter"] == "HierarchicalChunker"


class TestTextSplittingCombined:
    """Tests for combined text splitting parameter changes."""

    def test_multiple_params_can_be_changed_together(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify multiple chunking params can be changed in one form submission."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Change multiple parameters
            at.number_input(key="sidebar_param_max_tokens").set_value(1024)
            at.number_input(key="sidebar_param_chunk_overlap").set_value(80)

            # Submit form
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify all updated
            assert at.session_state["chunking_params"]["max_tokens"] == 1024
            assert at.session_state["chunking_params"]["chunk_overlap"] == 80

    def test_chunking_params_preserve_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify changing chunking params preserves parsing params."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            initial_device = at.session_state["parsing_params"]["docling_device"]
            initial_output_format = at.session_state["parsing_params"]["output_format"]

            # Change chunking params
            at.number_input(key="sidebar_param_max_tokens").set_value(2048)
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify parsing params preserved
            assert at.session_state["parsing_params"]["docling_device"] == initial_device
            assert at.session_state["parsing_params"]["output_format"] == initial_output_format
