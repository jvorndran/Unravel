"""Tests for chunks.py navigation - 'Go to Upload Step' button functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class SessionState(dict):
    """Mock session state that supports both dict and attribute access like Streamlit."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class TestGoToUploadStepButton:
    """Tests for the 'Go to Upload Step' button when no document is selected."""

    @pytest.fixture
    def mock_streamlit(self):
        """Set up mocked Streamlit components."""
        with patch("rag_lens.ui.steps.chunks.st") as mock_st:
            mock_st.session_state = SessionState()
            mock_st.rerun = MagicMock()
            mock_st.info = MagicMock()
            yield mock_st

    @pytest.fixture
    def mock_ui(self):
        """Set up mocked UI components."""
        with patch("rag_lens.ui.steps.chunks.ui") as mock_ui:
            mock_ui.button = MagicMock(return_value=False)
            yield mock_ui

    def test_button_shown_when_doc_name_is_none(self, mock_streamlit, mock_ui):
        """Button should appear when doc_name is None."""
        from rag_lens.ui.steps.chunks import render_chunks_step

        mock_streamlit.session_state.update({"doc_name": None})

        render_chunks_step()

        # Verify info message shown
        mock_streamlit.info.assert_called_once()
        actual_message = mock_streamlit.info.call_args[0][0]
        assert "No document selected" in actual_message

        # Verify button rendered with correct key
        mock_ui.button.assert_called_once()
        call_kwargs = mock_ui.button.call_args[1]
        assert call_kwargs["key"] == "goto_upload_chunks"

    def test_button_shown_when_doc_name_is_empty_string(self, mock_streamlit, mock_ui):
        """Button should appear when doc_name is empty string."""
        from rag_lens.ui.steps.chunks import render_chunks_step

        mock_streamlit.session_state.update({"doc_name": ""})

        render_chunks_step()

        # Verify button was rendered with correct key
        mock_ui.button.assert_called_once()
        assert mock_ui.button.call_args[1]["key"] == "goto_upload_chunks"

    def test_button_shown_when_doc_name_not_in_session_state(self, mock_streamlit, mock_ui):
        """Button should appear when doc_name key doesn't exist in session state."""
        from rag_lens.ui.steps.chunks import render_chunks_step

        # Empty session state - doc_name not set
        render_chunks_step()

        # Verify button was rendered
        mock_ui.button.assert_called_once()
        assert mock_ui.button.call_args[1]["key"] == "goto_upload_chunks"

    def test_button_click_sets_current_step_to_upload(self, mock_streamlit, mock_ui):
        """Clicking button should set current_step to 'upload'."""
        from rag_lens.ui.steps.chunks import render_chunks_step

        mock_streamlit.session_state.update({"doc_name": None})
        mock_ui.button.return_value = True  # Simulate button click

        render_chunks_step()

        assert mock_streamlit.session_state["current_step"] == "upload"

    def test_button_click_triggers_rerun(self, mock_streamlit, mock_ui):
        """Clicking button should trigger st.rerun()."""
        from rag_lens.ui.steps.chunks import render_chunks_step

        mock_streamlit.session_state.update({"doc_name": None})
        mock_ui.button.return_value = True  # Simulate button click

        render_chunks_step()

        mock_streamlit.rerun.assert_called_once()

    def test_early_return_when_no_document(self, mock_streamlit, mock_ui):
        """Function should return early without rendering chunks when no document."""
        from rag_lens.ui.steps.chunks import render_chunks_step

        mock_streamlit.session_state.update({"doc_name": None})
        mock_ui.button.return_value = False  # Button not clicked

        # The function should return without calling chunk rendering functions
        with patch("rag_lens.ui.steps.chunks.get_chunks") as mock_get_chunks:
            render_chunks_step()
            mock_get_chunks.assert_not_called()

    def test_button_not_shown_when_document_selected(self, mock_streamlit, mock_ui):
        """Button should NOT appear when a document is selected."""
        from rag_lens.ui.steps.chunks import render_chunks_step

        mock_streamlit.session_state.update(
            {
                "doc_name": "test_document.pdf",
                "parsing_params": {"strategy": "auto"},
                "applied_parsing_params": {"strategy": "auto"},
                "chunking_params": {
                    "provider": "langchain",
                    "splitter": "recursive",
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                },
            }
        )

        # Mock Streamlit UI components
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=False)
        mock_streamlit.columns = MagicMock(return_value=[mock_col, mock_col, mock_col])
        mock_streamlit.container = MagicMock(return_value=mock_col)

        # Mock dependencies needed when document is selected
        with patch(
            "rag_lens.ui.steps.chunks.get_storage_dir",
            return_value=Path("/tmp/storage"),
        ):
            with patch("rag_lens.ui.steps.chunks.get_chunks", return_value=[]):
                render_chunks_step()

        # Verify the "Go to Upload Step" button key was never used
        call_keys = [
            call.kwargs.get("key") for call in mock_ui.button.call_args_list
        ]
        assert "goto_upload_chunks" not in call_keys


class TestNavigationStateManagement:
    """Tests for navigation-related session state management."""

    @pytest.fixture
    def mock_streamlit(self):
        """Set up mocked Streamlit components."""
        with patch("rag_lens.ui.steps.chunks.st") as mock_st:
            mock_st.session_state = SessionState()
            mock_st.rerun = MagicMock()
            mock_st.info = MagicMock()
            yield mock_st

    @pytest.fixture
    def mock_ui(self):
        """Set up mocked UI components."""
        with patch("rag_lens.ui.steps.chunks.ui") as mock_ui:
            mock_ui.button = MagicMock(return_value=True)
            yield mock_ui

    def test_current_step_preserved_before_click(self, mock_streamlit, mock_ui):
        """current_step should only change after button click."""
        from rag_lens.ui.steps.chunks import render_chunks_step

        mock_streamlit.session_state.update(
            {
                "doc_name": None,
                "current_step": "chunks",
            }
        )
        mock_ui.button.return_value = False  # No click

        render_chunks_step()

        assert mock_streamlit.session_state["current_step"] == "chunks"

    def test_navigation_from_chunks_to_upload(self, mock_streamlit, mock_ui):
        """Verify complete navigation flow from chunks to upload step."""
        from rag_lens.ui.steps.chunks import render_chunks_step

        mock_streamlit.session_state.update(
            {
                "doc_name": None,
                "current_step": "chunks",
            }
        )
        mock_ui.button.return_value = True  # Click

        render_chunks_step()

        # Verify state transition
        assert mock_streamlit.session_state["current_step"] == "upload"
        mock_streamlit.rerun.assert_called_once()

    def test_idempotent_navigation_to_upload(self, mock_streamlit, mock_ui):
        """Navigation should work even if already on upload step."""
        from rag_lens.ui.steps.chunks import render_chunks_step

        mock_streamlit.session_state.update(
            {
                "doc_name": None,
                "current_step": "upload",  # Already on upload
            }
        )
        mock_ui.button.return_value = True  # Click

        render_chunks_step()

        # Should still set to upload (idempotent)
        assert mock_streamlit.session_state["current_step"] == "upload"
        mock_streamlit.rerun.assert_called_once()
