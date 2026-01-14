"""Tests for sidebar max characters to parse feature.

Feature 7: Changes "Max characters to parse"
- User Action: Changes the number input value
- Expected UI Change: Number input updates
- Session State Variables Affected: parsing_params.max_characters, sidebar_max_characters

Note: This input is inside the "Advanced Parsing Options" expander.

Fixtures used from conftest.py:
- mock_storage_dir: Creates temp directory with test documents
- sidebar_app_script: App script for rendering the sidebar
"""

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.conftest import get_form_submit_button


class TestMaxCharactersDisplay:
    """Tests for max characters number input display."""

    def test_max_characters_input_exists(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify max characters number input is rendered."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            max_chars_input = at.number_input(key="sidebar_max_characters")
            assert max_chars_input is not None

    def test_max_characters_defaults_to_40000(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify max characters defaults to 40000."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            max_chars_input = at.number_input(key="sidebar_max_characters")
            assert max_chars_input.value == 40000


class TestMaxCharactersSessionState:
    """Tests for max characters session state behavior."""

    def test_parsing_params_has_max_characters(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify parsing_params contains max_characters key."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "parsing_params" in at.session_state
            assert "max_characters" in at.session_state["parsing_params"]
            assert at.session_state["parsing_params"]["max_characters"] == 40000

    def test_preset_max_characters_reflected_in_input(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify pre-set max_characters value is reflected in number input."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script)
            parsing_params = {
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
                "max_characters": 100000,  # Custom value
            }
            at.session_state["parsing_params"] = parsing_params
            at.session_state["applied_parsing_params"] = parsing_params.copy()
            at.run()

            max_chars_input = at.number_input(key="sidebar_max_characters")
            assert max_chars_input.value == 100000


class TestMaxCharactersUserAction:
    """Tests for user interaction with max characters input."""

    def test_input_accepts_higher_value(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify number input accepts higher values."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            max_chars_input = at.number_input(key="sidebar_max_characters")
            max_chars_input.set_value(100000).run()

            max_chars_input = at.number_input(key="sidebar_max_characters")
            assert max_chars_input.value == 100000

    def test_input_accepts_lower_value(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify number input accepts lower values."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            max_chars_input = at.number_input(key="sidebar_max_characters")
            max_chars_input.set_value(10000).run()

            max_chars_input = at.number_input(key="sidebar_max_characters")
            assert max_chars_input.value == 10000

    def test_input_accepts_minimum_value(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify number input accepts minimum value of 1000."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            max_chars_input = at.number_input(key="sidebar_max_characters")
            max_chars_input.set_value(1000).run()

            max_chars_input = at.number_input(key="sidebar_max_characters")
            assert max_chars_input.value == 1000

    def test_input_accepts_maximum_value(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify number input accepts maximum value of 1000000."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            max_chars_input = at.number_input(key="sidebar_max_characters")
            max_chars_input.set_value(1000000).run()

            max_chars_input = at.number_input(key="sidebar_max_characters")
            assert max_chars_input.value == 1000000


class TestMaxCharactersFormSubmission:
    """Tests for max characters form submission updating session state."""

    def test_form_submit_updates_max_characters(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission updates parsing_params.max_characters."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Initial state should be 40000
            assert at.session_state["parsing_params"]["max_characters"] == 40000

            # Change value
            at.number_input(key="sidebar_max_characters").set_value(80000)

            # Submit the form
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify parsing_params was updated
            assert at.session_state["parsing_params"]["max_characters"] == 80000

    def test_form_submit_updates_applied_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission also updates applied_parsing_params."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            at.number_input(key="sidebar_max_characters").set_value(50000)
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            assert at.session_state["parsing_params"]["max_characters"] == 50000
            assert at.session_state["applied_parsing_params"]["max_characters"] == 50000

    def test_form_submit_preserves_other_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission preserves other parsing params when changing max chars."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            initial_output_format = at.session_state["parsing_params"]["output_format"]
            initial_normalize = at.session_state["parsing_params"]["normalize_whitespace"]
            initial_device = at.session_state["parsing_params"]["docling_device"]

            # Change only max_characters
            at.number_input(key="sidebar_max_characters").set_value(75000)
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify other params preserved
            assert at.session_state["parsing_params"]["output_format"] == initial_output_format
            assert at.session_state["parsing_params"]["normalize_whitespace"] == initial_normalize
            assert at.session_state["parsing_params"]["docling_device"] == initial_device
            assert at.session_state["parsing_params"]["max_characters"] == 75000


class TestMaxCharactersEdgeCases:
    """Tests for max characters edge cases and validation."""

    def test_max_characters_persists_across_output_format_changes(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify max_characters value persists when switching output formats."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Set a custom max_characters value
            at.number_input(key="sidebar_max_characters").set_value(60000)
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Change output format
            at.selectbox(key="sidebar_output_format").set_value("Plain Text").run()

            # Verify max_characters still has the custom value
            max_chars_input = at.number_input(key="sidebar_max_characters")
            assert max_chars_input.value == 60000

    def test_max_characters_is_integer(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify max_characters is stored as integer."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            at.number_input(key="sidebar_max_characters").set_value(55000)
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            max_chars = at.session_state["parsing_params"]["max_characters"]
            assert isinstance(max_chars, int)
            assert max_chars == 55000
