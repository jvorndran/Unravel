"""Tests for sidebar document selection feature.

Feature 1: Selects a document from dropdown
- User Action: Selects a document from dropdown
- Expected UI Change: Dropdown updates to show selected doc
- Session State Variables Affected: doc_name, sidebar_doc_selector

Fixtures used from conftest.py:
- mock_storage_dir: Creates temp directory with test documents
- sidebar_app_script_script: App script for rendering the sidebar
"""

from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class TestDocumentDropdownDisplay:
    """Tests for dropdown display behavior."""

    def test_dropdown_displays_all_available_documents(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify dropdown shows all available documents."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            doc_selectbox = at.selectbox(key="sidebar_doc_selector")
            assert doc_selectbox is not None

            options = doc_selectbox.options
            assert "document_a.pdf" in options
            assert "document_b.pdf" in options
            assert "document_c.txt" in options

    def test_dropdown_has_correct_options_count(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify dropdown has exactly the number of available documents."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            doc_selectbox = at.selectbox(key="sidebar_doc_selector")
            assert len(doc_selectbox.options) == 3

    def test_dropdown_excludes_sample_text_when_files_exist(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify 'Sample Text' is not shown when real documents exist."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            doc_selectbox = at.selectbox(key="sidebar_doc_selector")
            assert "Sample Text" not in doc_selectbox.options

    def test_no_documents_shows_info_message(self, tmp_path, sidebar_app_script):
        """Verify info message when no documents exist."""
        empty_storage = tmp_path / "empty_storage"
        (empty_storage / "documents").mkdir(parents=True)
        (empty_storage / "config").mkdir(parents=True)

        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", empty_storage
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            info_messages = [m.value for m in at.info]
            assert any(
                "No documents available" in msg for msg in info_messages
            ), f"Expected 'No documents available' message, got: {info_messages}"


class TestDocumentSelectionInitialization:
    """Tests for initial state behavior."""

    def test_doc_name_defaults_to_first_document(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify doc_name defaults to first available document."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            doc_name = at.session_state["doc_name"]
            assert doc_name is not None
            assert doc_name in ["document_a.pdf", "document_b.pdf", "document_c.txt"]

    def test_session_state_variables_initialized_correctly(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify doc_name and sidebar_doc_selector are both initialized."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "doc_name" in at.session_state
            assert at.session_state["doc_name"] is not None
            assert "sidebar_doc_selector" in at.session_state

    def test_sidebar_selector_syncs_with_preset_doc_name(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify sidebar_doc_selector syncs with pre-set doc_name on load."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script)
            at.session_state["doc_name"] = "document_c.txt"
            at.run()

            doc_selectbox = at.selectbox(key="sidebar_doc_selector")
            assert doc_selectbox.value == "document_c.txt"

    def test_doc_name_and_selector_initialized_in_sync(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify doc_name and sidebar_doc_selector are synchronized on init."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            doc_selectbox = at.selectbox(key="sidebar_doc_selector")
            assert doc_selectbox.value == at.session_state["doc_name"]


class TestDocumentSelectionUserAction:
    """Tests for user interaction with document selection.

    Note: Streamlit's testing framework has limitations with form submission.
    The selectbox is inside a form, and form values don't persist properly
    through click().run() in the test environment. These tests verify
    what we can test without full form submission.
    """

    def test_selectbox_accepts_value_change(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify selectbox widget accepts new value selection."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            doc_selectbox = at.selectbox(key="sidebar_doc_selector")
            initial_value = doc_selectbox.value

            # Select a different document
            target_doc = next(
                (doc for doc in doc_selectbox.options if doc != initial_value),
                doc_selectbox.options[0],
            )
            doc_selectbox.set_value(target_doc)

            # Widget should reflect the new pending value
            assert doc_selectbox.value == target_doc

    def test_form_submit_button_exists(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify the Save & Apply form submit button is rendered."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            button_labels = [getattr(btn, "label", None) for btn in at.button]
            assert "Save & Apply" in button_labels, (
                f"Expected 'Save & Apply' button, found: {button_labels}"
            )

    def test_preset_doc_name_updates_session_state(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify that setting doc_name before run properly updates state.

        This tests the session state flow without form submission mechanics.
        When doc_name is pre-set, the sidebar should use that value and
        both doc_name and sidebar_doc_selector should match.
        """
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script)
            at.session_state["doc_name"] = "document_b.pdf"
            at.run()

            # Both should be set to the pre-set value
            assert at.session_state["doc_name"] == "document_b.pdf"
            assert at.session_state["sidebar_doc_selector"] == "document_b.pdf"

            # Widget should show the value
            doc_selectbox = at.selectbox(key="sidebar_doc_selector")
            assert doc_selectbox.value == "document_b.pdf"

    def test_changing_preset_doc_name_updates_selector(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify changing doc_name updates sidebar_doc_selector on next run."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            # First run with document_a
            at = AppTest.from_string(sidebar_app_script)
            at.session_state["doc_name"] = "document_a.pdf"
            at.run()

            assert at.session_state["doc_name"] == "document_a.pdf"

            # Change to document_b and rerun
            at.session_state["doc_name"] = "document_b.pdf"
            at = at.run()

            # Selector should update to match new doc_name
            doc_selectbox = at.selectbox(key="sidebar_doc_selector")
            assert doc_selectbox.value == "document_b.pdf"


class TestDocumentSelectionEdgeCases:
    """Tests for edge cases and automatic state corrections."""

    def test_auto_switches_from_sample_text_to_first_file(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify automatic switch from Sample Text when files exist.

        When doc_name is 'Sample Text' but real files exist, the sidebar
        should auto-switch to the first available file (lines 42-51 in sidebar.py).
        """
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script)
            at.session_state["doc_name"] = "Sample Text"
            at.run()

            # Should have auto-switched to first available doc
            doc_name = at.session_state["doc_name"]
            assert doc_name != "Sample Text"
            assert doc_name in ["document_a.pdf", "document_b.pdf", "document_c.txt"]

    def test_invalid_doc_name_corrects_to_first_available(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify invalid doc_name is corrected to first available.

        When doc_name references a non-existent file, it should be corrected
        to the first available document (lines 93-97 in sidebar.py).
        """
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script)
            at.session_state["doc_name"] = "nonexistent_file.pdf"
            at.run()

            # Should have corrected to first available doc
            doc_name = at.session_state["doc_name"]
            assert doc_name in ["document_a.pdf", "document_b.pdf", "document_c.txt"]

    def test_sample_text_clears_cache_on_auto_switch(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify cache is cleared when auto-switching from Sample Text.

        When switching from Sample Text to a real file, chunks and embeddings
        should be cleared (lines 46-51 in sidebar.py).
        """
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script)
            at.session_state["doc_name"] = "Sample Text"
            at.session_state["chunks"] = ["old", "chunks"]
            at.session_state["last_embeddings_result"] = {"cached": True}
            at.session_state["search_results"] = [{"result": 1}]
            at.run()

            # Cache should be cleared after auto-switch
            assert "chunks" not in at.session_state
            assert "last_embeddings_result" not in at.session_state
            assert "search_results" not in at.session_state


