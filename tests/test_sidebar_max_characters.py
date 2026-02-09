"""Tests for chunking config max pages to parse feature.

Feature: Changes "Max pages to parse"
- User Action: Changes the number input value
- Expected UI Change: Number input updates
- Session State Variables Affected: chunking_config_max_pages

Fixtures used from conftest.py:
- mock_storage_dir: Creates temp directory with test documents
- chunking_config_app_script: App script for rendering chunking config
"""

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

class TestMaxPagesDisplay:
    """Tests for max pages number input display."""

    def test_max_pages_input_exists(
        self, mock_storage_dir, chunking_config_app_script
    ):
        """Verify max pages number input is rendered."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(chunking_config_app_script).run()

            max_pages_input = at.number_input(key="chunking_config_max_pages")
            assert max_pages_input is not None

    def test_max_pages_defaults_to_50(
        self, mock_storage_dir, chunking_config_app_script
    ):
        """Verify max pages defaults to 50."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(chunking_config_app_script).run()

            max_pages_input = at.number_input(key="chunking_config_max_pages")
            assert max_pages_input.value == 50


class TestMaxPagesSessionState:
    """Tests for max pages session state behavior."""

    def test_preset_max_pages_reflected_in_input(
        self, mock_storage_dir, chunking_config_app_script
    ):
        """Verify pre-set max_pages value is reflected in number input."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(chunking_config_app_script)
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
                "max_pages": 12,  # Custom value
            }
            at.session_state["parsing_params"] = parsing_params
            at.session_state["applied_parsing_params"] = parsing_params.copy()
            at.run()

            max_pages_input = at.number_input(key="chunking_config_max_pages")
            assert max_pages_input.value == 12


class TestMaxPagesUserAction:
    """Tests for user interaction with max pages input."""

    def test_input_accepts_higher_value(
        self, mock_storage_dir, chunking_config_app_script
    ):
        """Verify number input accepts higher values."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(chunking_config_app_script).run()

            max_pages_input = at.number_input(key="chunking_config_max_pages")
            max_pages_input.set_value(200).run()

            max_pages_input = at.number_input(key="chunking_config_max_pages")
            assert max_pages_input.value == 200

    def test_input_accepts_lower_value(
        self, mock_storage_dir, chunking_config_app_script
    ):
        """Verify number input accepts lower values."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(chunking_config_app_script).run()

            max_pages_input = at.number_input(key="chunking_config_max_pages")
            max_pages_input.set_value(5).run()

            max_pages_input = at.number_input(key="chunking_config_max_pages")
            assert max_pages_input.value == 5

    def test_input_accepts_minimum_value(
        self, mock_storage_dir, chunking_config_app_script
    ):
        """Verify number input accepts minimum value of 1."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(chunking_config_app_script).run()

            max_pages_input = at.number_input(key="chunking_config_max_pages")
            max_pages_input.set_value(1).run()

            max_pages_input = at.number_input(key="chunking_config_max_pages")
            assert max_pages_input.value == 1

    def test_input_accepts_maximum_value(
        self, mock_storage_dir, chunking_config_app_script
    ):
        """Verify number input accepts maximum value of 1000."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(chunking_config_app_script).run()

            max_pages_input = at.number_input(key="chunking_config_max_pages")
            max_pages_input.set_value(1000).run()

            max_pages_input = at.number_input(key="chunking_config_max_pages")
            assert max_pages_input.value == 1000


class TestMaxPagesEdgeCases:
    """Tests for max pages edge cases and validation."""

    def test_max_pages_is_integer(
        self, mock_storage_dir, chunking_config_app_script
    ):
        """Verify max_pages is stored as integer."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(chunking_config_app_script).run()

            at.number_input(key="chunking_config_max_pages").set_value(37).run()

            max_pages = at.session_state["chunking_config_max_pages"]
            assert isinstance(max_pages, int)
            assert max_pages == 37
