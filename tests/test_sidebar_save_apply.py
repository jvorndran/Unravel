"""Tests for sidebar Save & Apply button feature.

Feature 29: Clicks "Save & Apply"
- User Action: Clicks the Save & Apply button
- Expected UI Change: Success toast appears, page reruns
- Session State Variables Affected:
  - Updates: doc_name, embedding_model_name, parsing_params, chunking_params,
             applied_parsing_params, applied_chunking_params
  - Deletes (conditionally): chunks, last_embeddings_result, search_results

Fixtures used from conftest.py:
- mock_storage_dir: Creates temp directory with test documents
- sidebar_app_script: App script for rendering the sidebar
"""

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.conftest import get_form_submit_button


class TestSaveApplyButtonDisplay:
    """Tests for Save & Apply button display."""

    def test_save_apply_button_exists(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify Save & Apply button is rendered."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            submit_btn = get_form_submit_button(at)
            assert submit_btn is not None

    def test_save_apply_button_has_correct_label(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify Save & Apply button has correct label."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            submit_btn = get_form_submit_button(at)
            assert submit_btn.label == "Save & Apply"


class TestSaveApplyBasicFunctionality:
    """Tests for Save & Apply basic functionality."""

    def test_save_apply_updates_session_state(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify Save & Apply updates session state."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Make a change
            at.number_input(key="sidebar_param_chunk_size").set_value(700)

            # Click Save & Apply
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify session state updated
            assert at.session_state["chunking_params"]["chunk_size"] == 700

    def test_save_apply_without_changes_still_works(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify Save & Apply works even without making changes."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            initial_chunk_size = at.session_state["chunking_params"]["chunk_size"]

            # Click Save & Apply without changes
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify state is maintained
            assert at.session_state["chunking_params"]["chunk_size"] == initial_chunk_size


class TestSaveApplyAppliedParamsSync:
    """Tests for Save & Apply syncing params with applied params."""

    def test_save_apply_syncs_parsing_params_to_applied(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify Save & Apply syncs parsing_params to applied_parsing_params."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Change output format
            at.selectbox(key="sidebar_output_format").set_value("Plain Text")

            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Both should be synced
            assert at.session_state["parsing_params"]["output_format"] == "plain_text"
            assert at.session_state["applied_parsing_params"]["output_format"] == "plain_text"

    def test_save_apply_syncs_chunking_params_to_applied(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify Save & Apply syncs chunking_params to applied_chunking_params."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Change chunk size
            at.number_input(key="sidebar_param_chunk_size").set_value(900)

            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Both should be synced
            assert at.session_state["chunking_params"]["chunk_size"] == 900
            assert at.session_state["applied_chunking_params"]["chunk_size"] == 900


class TestSaveApplyCacheInvalidation:
    """Tests for Save & Apply cache invalidation behavior."""

    def test_parsing_change_clears_chunks(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify parsing param change clears chunks cache."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Set up existing chunks
            at.session_state["chunks"] = ["chunk1", "chunk2"]

            # Change a parsing param
            at.selectbox(key="sidebar_output_format").set_value("Plain Text")

            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Chunks should be cleared
            assert "chunks" not in at.session_state

    def test_chunking_change_clears_chunks(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify chunking param change clears chunks cache."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Set up existing chunks
            at.session_state["chunks"] = ["chunk1", "chunk2"]

            # Change a chunking param
            at.number_input(key="sidebar_param_chunk_size").set_value(800)

            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Chunks should be cleared
            assert "chunks" not in at.session_state

    def test_parsing_change_clears_embeddings_result(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify parsing param change clears embeddings result."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Set up existing embeddings
            at.session_state["last_embeddings_result"] = {"data": "embeddings"}

            # Change a parsing param
            at.checkbox(key="sidebar_normalize_whitespace").set_value(False)

            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Embeddings should be cleared
            assert "last_embeddings_result" not in at.session_state

    def test_parsing_change_clears_search_results(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify parsing param change clears search results."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Set up existing search results
            at.session_state["search_results"] = [{"result": 1}]

            # Change a parsing param
            at.checkbox(key="sidebar_remove_special_chars").set_value(True)

            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Search results should be cleared
            assert "search_results" not in at.session_state


class TestSaveApplyDocumentChange:
    """Tests for Save & Apply document change behavior."""

    def test_document_change_clears_chunks(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify document change clears chunks cache."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Set up existing chunks
            at.session_state["chunks"] = ["chunk1", "chunk2"]

            # Get available documents
            doc_selectbox = at.selectbox(key="sidebar_doc_selector")
            options = doc_selectbox.options
            current_doc = doc_selectbox.value

            # Change to a different document
            other_docs = [d for d in options if d != current_doc]
            if other_docs:
                at.selectbox(key="sidebar_doc_selector").set_value(other_docs[0])

                submit_btn = get_form_submit_button(at)
                submit_btn.click().run()

                # Chunks should be cleared
                assert "chunks" not in at.session_state

    def test_document_change_clears_search_results(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify document change clears search results."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Set up existing search results
            at.session_state["search_results"] = [{"result": 1}]

            # Get available documents
            doc_selectbox = at.selectbox(key="sidebar_doc_selector")
            options = doc_selectbox.options
            current_doc = doc_selectbox.value

            # Change to a different document
            other_docs = [d for d in options if d != current_doc]
            if other_docs:
                at.selectbox(key="sidebar_doc_selector").set_value(other_docs[0])

                submit_btn = get_form_submit_button(at)
                submit_btn.click().run()

                # Search results should be cleared
                assert "search_results" not in at.session_state


class TestSaveApplyMultipleChanges:
    """Tests for Save & Apply with multiple simultaneous changes."""

    def test_multiple_parsing_changes(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify multiple parsing param changes are all applied."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Make multiple changes
            at.selectbox(key="sidebar_output_format").set_value("Original Format")
            at.checkbox(key="sidebar_normalize_whitespace").set_value(False)
            at.checkbox(key="sidebar_remove_special_chars").set_value(True)

            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify all changes applied
            assert at.session_state["parsing_params"]["output_format"] == "original"
            assert at.session_state["parsing_params"]["normalize_whitespace"] is False
            assert at.session_state["parsing_params"]["remove_special_chars"] is True

    def test_mixed_parsing_and_chunking_changes(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify mixed parsing and chunking changes are all applied."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Make parsing changes
            at.selectbox(key="sidebar_output_format").set_value("Plain Text")

            # Make chunking changes
            at.number_input(key="sidebar_param_chunk_size").set_value(750)
            at.number_input(key="sidebar_param_chunk_overlap").set_value(100)

            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify all changes applied
            assert at.session_state["parsing_params"]["output_format"] == "plain_text"
            assert at.session_state["chunking_params"]["chunk_size"] == 750
            assert at.session_state["chunking_params"]["chunk_overlap"] == 100


class TestSaveApplyStateConsistency:
    """Tests for session state consistency after Save & Apply."""

    def test_applied_params_match_current_params_after_save(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify applied params match current params after Save & Apply."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Make some changes
            at.number_input(key="sidebar_param_chunk_size").set_value(777)
            at.selectbox(key="sidebar_output_format").set_value("Original Format")

            # Submit
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify applied params match current params
            assert at.session_state["parsing_params"] == at.session_state["applied_parsing_params"]
            assert at.session_state["chunking_params"] == at.session_state["applied_chunking_params"]

    def test_all_session_state_keys_present_after_save(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify all required session state keys present after Save & Apply."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify all required keys exist
            assert "doc_name" in at.session_state
            assert "embedding_model_name" in at.session_state
            assert "parsing_params" in at.session_state
            assert "chunking_params" in at.session_state
            assert "applied_parsing_params" in at.session_state
            assert "applied_chunking_params" in at.session_state
