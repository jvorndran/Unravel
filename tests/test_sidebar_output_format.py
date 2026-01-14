"""Tests for sidebar output format selection feature.

Feature 3: Changes Output Format
- User Action: Changes Output Format selection
- Expected UI Change: Dropdown updates selection
- Session State Variables Affected: parsing_params.output_format, sidebar_output_format

Fixtures used from conftest.py:
- mock_storage_dir: Creates temp directory with test documents
- sidebar_app_script: App script for rendering the sidebar
- element_exists: Helper to check if element exists
"""

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.conftest import element_exists, get_form_submit_button


class TestOutputFormatDropdownDisplay:
    """Tests for output format dropdown display behavior."""

    def test_output_format_dropdown_exists(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify output format dropdown is rendered."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            output_format_selectbox = at.selectbox(key="sidebar_output_format")
            assert output_format_selectbox is not None

    def test_output_format_has_all_options(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify output format dropdown has all three format options."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            output_format_selectbox = at.selectbox(key="sidebar_output_format")
            options = output_format_selectbox.options

            assert "Markdown (Recommended)" in options
            assert "Original Format" in options
            assert "Plain Text" in options
            assert len(options) == 3

    def test_output_format_defaults_to_markdown(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify output format defaults to Markdown."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            output_format_selectbox = at.selectbox(key="sidebar_output_format")
            assert output_format_selectbox.value == "Markdown (Recommended)"


class TestOutputFormatSessionState:
    """Tests for output format session state behavior."""

    def test_sidebar_output_format_initialized(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify sidebar_output_format session state is initialized."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "sidebar_output_format" in at.session_state
            assert at.session_state["sidebar_output_format"] == "Markdown (Recommended)"

    def test_parsing_params_has_output_format(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify parsing_params contains output_format key."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "parsing_params" in at.session_state
            assert "output_format" in at.session_state["parsing_params"]
            assert at.session_state["parsing_params"]["output_format"] == "markdown"

    def test_preset_output_format_reflected_in_dropdown(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify pre-set output_format value is reflected in dropdown."""
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
                "output_format": "plain_text",
                "normalize_whitespace": True,
                "remove_special_chars": False,
                "max_characters": 40000,
            }
            at.session_state["parsing_params"] = parsing_params
            at.session_state["applied_parsing_params"] = parsing_params.copy()
            at.run()

            output_format_selectbox = at.selectbox(key="sidebar_output_format")
            assert output_format_selectbox.value == "Plain Text"


class TestOutputFormatUserAction:
    """Tests for user interaction with output format selection."""

    def test_selectbox_accepts_original_format_selection(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify selectbox accepts Original Format selection."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            output_format_selectbox = at.selectbox(key="sidebar_output_format")
            output_format_selectbox.set_value("Original Format").run()

            output_format_selectbox = at.selectbox(key="sidebar_output_format")
            assert output_format_selectbox.value == "Original Format"

    def test_selectbox_accepts_plain_text_selection(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify selectbox accepts Plain Text selection."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            output_format_selectbox = at.selectbox(key="sidebar_output_format")
            output_format_selectbox.set_value("Plain Text").run()

            output_format_selectbox = at.selectbox(key="sidebar_output_format")
            assert output_format_selectbox.value == "Plain Text"


class TestOutputFormatFormSubmission:
    """Tests for output format form submission updating session state."""

    def test_form_submit_updates_output_format_to_original(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission updates parsing_params.output_format to original."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Initial state should be markdown
            assert at.session_state["parsing_params"]["output_format"] == "markdown"

            # Change selectbox to Original Format
            at.selectbox(key="sidebar_output_format").set_value("Original Format")

            # Submit the form
            submit_btn = get_form_submit_button(at)
            assert submit_btn is not None, "Save & Apply button not found"
            submit_btn.click().run()

            # Verify parsing_params was updated
            assert at.session_state["parsing_params"]["output_format"] == "original"

    def test_form_submit_updates_output_format_to_plain_text(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission updates parsing_params.output_format to plain_text."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Change selectbox to Plain Text
            at.selectbox(key="sidebar_output_format").set_value("Plain Text")

            # Submit the form
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify parsing_params was updated
            assert at.session_state["parsing_params"]["output_format"] == "plain_text"

    def test_form_submit_updates_applied_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission also updates applied_parsing_params."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Change output format
            at.selectbox(key="sidebar_output_format").set_value("Plain Text")
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Both should be updated
            assert at.session_state["parsing_params"]["output_format"] == "plain_text"
            assert at.session_state["applied_parsing_params"]["output_format"] == "plain_text"

    def test_form_submit_preserves_other_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission preserves other parsing params when changing format."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Get initial values
            initial_device = at.session_state["parsing_params"]["docling_device"]
            initial_max_chars = at.session_state["parsing_params"]["max_characters"]

            # Change output format only
            at.selectbox(key="sidebar_output_format").set_value("Original Format")
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify other params are preserved
            assert at.session_state["parsing_params"]["docling_device"] == initial_device
            assert at.session_state["parsing_params"]["max_characters"] == initial_max_chars
            assert at.session_state["parsing_params"]["output_format"] == "original"


class TestOutputFormatCycling:
    """Tests for cycling through output format options."""

    def test_full_format_cycle_maintains_correct_state(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify cycling through all formats and submitting updates parsing_params correctly."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Start at markdown
            assert at.selectbox(key="sidebar_output_format").value == "Markdown (Recommended)"
            assert at.session_state["parsing_params"]["output_format"] == "markdown"

            # Change to Original Format and submit
            at.selectbox(key="sidebar_output_format").set_value("Original Format")
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()
            assert at.selectbox(key="sidebar_output_format").value == "Original Format"
            assert at.session_state["parsing_params"]["output_format"] == "original"

            # Change to Plain Text and submit
            at.selectbox(key="sidebar_output_format").set_value("Plain Text")
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()
            assert at.selectbox(key="sidebar_output_format").value == "Plain Text"
            assert at.session_state["parsing_params"]["output_format"] == "plain_text"

            # Change back to Markdown and submit
            at.selectbox(key="sidebar_output_format").set_value("Markdown (Recommended)")
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()
            assert at.selectbox(key="sidebar_output_format").value == "Markdown (Recommended)"
            assert at.session_state["parsing_params"]["output_format"] == "markdown"
