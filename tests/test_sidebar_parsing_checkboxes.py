"""Tests for sidebar parsing checkbox features.

Feature 4: Toggles "Normalize Whitespace"
- User Action: Toggles the checkbox
- Expected UI Change: Checkbox state changes
- Session State Variables Affected: parsing_params.normalize_whitespace, sidebar_normalize_whitespace

Feature 5: Toggles "Remove Special Characters"
- User Action: Toggles the checkbox
- Expected UI Change: Checkbox state changes
- Session State Variables Affected: parsing_params.remove_special_chars, sidebar_remove_special_chars

Fixtures used from conftest.py:
- mock_storage_dir: Creates temp directory with test documents
- sidebar_app_script: App script for rendering the sidebar
"""

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.conftest import get_form_submit_button


class TestNormalizeWhitespaceDisplay:
    """Tests for normalize whitespace checkbox display."""

    def test_normalize_whitespace_checkbox_exists(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify normalize whitespace checkbox is rendered."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            checkbox = at.checkbox(key="sidebar_normalize_whitespace")
            assert checkbox is not None

    def test_normalize_whitespace_defaults_to_true(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify normalize whitespace defaults to True."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            checkbox = at.checkbox(key="sidebar_normalize_whitespace")
            assert checkbox.value is True


class TestNormalizeWhitespaceSessionState:
    """Tests for normalize whitespace session state behavior."""

    def test_parsing_params_has_normalize_whitespace(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify parsing_params contains normalize_whitespace key."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "parsing_params" in at.session_state
            assert "normalize_whitespace" in at.session_state["parsing_params"]
            assert at.session_state["parsing_params"]["normalize_whitespace"] is True

    def test_preset_normalize_whitespace_reflected_in_checkbox(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify pre-set normalize_whitespace value is reflected in checkbox."""
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
                "normalize_whitespace": False,  # Set to False
                "remove_special_chars": False,
            }
            at.session_state["parsing_params"] = parsing_params
            at.session_state["applied_parsing_params"] = parsing_params.copy()
            at.run()

            checkbox = at.checkbox(key="sidebar_normalize_whitespace")
            assert checkbox.value is False


class TestNormalizeWhitespaceUserAction:
    """Tests for user interaction with normalize whitespace checkbox."""

    def test_checkbox_accepts_false_value(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify checkbox accepts False value (unchecking)."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            checkbox = at.checkbox(key="sidebar_normalize_whitespace")
            checkbox.set_value(False).run()

            checkbox = at.checkbox(key="sidebar_normalize_whitespace")
            assert checkbox.value is False

    def test_checkbox_can_be_toggled_back_to_true(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify checkbox can be toggled back to True."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Turn off
            at.checkbox(key="sidebar_normalize_whitespace").set_value(False).run()
            assert at.checkbox(key="sidebar_normalize_whitespace").value is False

            # Turn back on
            at.checkbox(key="sidebar_normalize_whitespace").set_value(True).run()
            assert at.checkbox(key="sidebar_normalize_whitespace").value is True


class TestNormalizeWhitespaceFormSubmission:
    """Tests for normalize whitespace form submission."""

    def test_form_submit_updates_normalize_whitespace_to_false(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission updates parsing_params.normalize_whitespace."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Initial state should be True
            assert at.session_state["parsing_params"]["normalize_whitespace"] is True

            # Uncheck the box
            at.checkbox(key="sidebar_normalize_whitespace").set_value(False)

            # Submit the form
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify parsing_params was updated
            assert at.session_state["parsing_params"]["normalize_whitespace"] is False

    def test_form_submit_updates_applied_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission also updates applied_parsing_params."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            at.checkbox(key="sidebar_normalize_whitespace").set_value(False)
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            assert at.session_state["parsing_params"]["normalize_whitespace"] is False
            assert at.session_state["applied_parsing_params"]["normalize_whitespace"] is False


class TestRemoveSpecialCharsDisplay:
    """Tests for remove special characters checkbox display."""

    def test_remove_special_chars_checkbox_exists(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify remove special characters checkbox is rendered."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            checkbox = at.checkbox(key="sidebar_remove_special_chars")
            assert checkbox is not None

    def test_remove_special_chars_defaults_to_false(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify remove special characters defaults to False."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            checkbox = at.checkbox(key="sidebar_remove_special_chars")
            assert checkbox.value is False


class TestRemoveSpecialCharsSessionState:
    """Tests for remove special characters session state behavior."""

    def test_parsing_params_has_remove_special_chars(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify parsing_params contains remove_special_chars key."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "parsing_params" in at.session_state
            assert "remove_special_chars" in at.session_state["parsing_params"]
            assert at.session_state["parsing_params"]["remove_special_chars"] is False

    def test_preset_remove_special_chars_reflected_in_checkbox(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify pre-set remove_special_chars value is reflected in checkbox."""
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
                "remove_special_chars": True,  # Set to True
            }
            at.session_state["parsing_params"] = parsing_params
            at.session_state["applied_parsing_params"] = parsing_params.copy()
            at.run()

            checkbox = at.checkbox(key="sidebar_remove_special_chars")
            assert checkbox.value is True


class TestRemoveSpecialCharsUserAction:
    """Tests for user interaction with remove special characters checkbox."""

    def test_checkbox_accepts_true_value(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify checkbox accepts True value (checking)."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            checkbox = at.checkbox(key="sidebar_remove_special_chars")
            checkbox.set_value(True).run()

            checkbox = at.checkbox(key="sidebar_remove_special_chars")
            assert checkbox.value is True

    def test_checkbox_can_be_toggled_back_to_false(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify checkbox can be toggled back to False."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Turn on
            at.checkbox(key="sidebar_remove_special_chars").set_value(True).run()
            assert at.checkbox(key="sidebar_remove_special_chars").value is True

            # Turn back off
            at.checkbox(key="sidebar_remove_special_chars").set_value(False).run()
            assert at.checkbox(key="sidebar_remove_special_chars").value is False


class TestRemoveSpecialCharsFormSubmission:
    """Tests for remove special characters form submission."""

    def test_form_submit_updates_remove_special_chars_to_true(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission updates parsing_params.remove_special_chars."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Initial state should be False
            assert at.session_state["parsing_params"]["remove_special_chars"] is False

            # Check the box
            at.checkbox(key="sidebar_remove_special_chars").set_value(True)

            # Submit the form
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify parsing_params was updated
            assert at.session_state["parsing_params"]["remove_special_chars"] is True

    def test_form_submit_updates_applied_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission also updates applied_parsing_params."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            at.checkbox(key="sidebar_remove_special_chars").set_value(True)
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            assert at.session_state["parsing_params"]["remove_special_chars"] is True
            assert at.session_state["applied_parsing_params"]["remove_special_chars"] is True


class TestParsingCheckboxesCombined:
    """Tests for combined parsing checkbox behavior."""

    def test_both_checkboxes_can_be_changed_together(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify both checkboxes can be changed in the same form submission."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Change both checkboxes
            at.checkbox(key="sidebar_normalize_whitespace").set_value(False)
            at.checkbox(key="sidebar_remove_special_chars").set_value(True)

            # Submit the form
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify both updated
            assert at.session_state["parsing_params"]["normalize_whitespace"] is False
            assert at.session_state["parsing_params"]["remove_special_chars"] is True

    def test_checkboxes_preserve_other_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify changing checkboxes preserves other parsing params."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            initial_device = at.session_state["parsing_params"]["docling_device"]
            initial_output_format = at.session_state["parsing_params"]["output_format"]
            initial_table_structure = at.session_state["parsing_params"]["docling_table_structure"]

            # Change checkboxes
            at.checkbox(key="sidebar_normalize_whitespace").set_value(False)
            at.checkbox(key="sidebar_remove_special_chars").set_value(True)
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify checkboxes actually updated
            assert at.session_state["parsing_params"]["normalize_whitespace"] is False
            assert at.session_state["parsing_params"]["remove_special_chars"] is True

            # Verify other params preserved
            assert at.session_state["parsing_params"]["docling_device"] == initial_device
            assert at.session_state["parsing_params"]["output_format"] == initial_output_format
            assert at.session_state["parsing_params"]["docling_table_structure"] == initial_table_structure
