"""Tests for sidebar embedding model selection feature.

Feature 28: Changes Embedding Model
- User Action: Changes the Embedding Model dropdown
- Expected UI Change: Dropdown updates selection
- Session State Variables Affected: embedding_model_name, sidebar_embedding_model

Fixtures used from conftest.py:
- mock_storage_dir: Creates temp directory with test documents
- sidebar_app_script: App script for rendering the sidebar
"""

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.conftest import get_form_submit_button


class TestEmbeddingModelDropdownDisplay:
    """Tests for embedding model dropdown display."""

    def test_embedding_model_dropdown_exists(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify embedding model dropdown is rendered."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
            assert embedding_selectbox is not None

    def test_embedding_model_has_options(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify embedding model dropdown has available options."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
            assert len(embedding_selectbox.options) > 0

    def test_embedding_model_has_default_model(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify embedding model dropdown includes the default model."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
            # The default model "all-MiniLM-L6-v2" should be in options
            assert "all-MiniLM-L6-v2" in embedding_selectbox.options

    def test_embedding_model_defaults_to_minilm(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify embedding model defaults to all-MiniLM-L6-v2."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
            assert embedding_selectbox.value == "all-MiniLM-L6-v2"


class TestEmbeddingModelSessionState:
    """Tests for embedding model session state behavior."""

    def test_embedding_model_name_initialized(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify embedding_model_name session state is initialized."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "embedding_model_name" in at.session_state
            assert at.session_state["embedding_model_name"] == "all-MiniLM-L6-v2"

    def test_sidebar_embedding_model_initialized(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify sidebar_embedding_model session state is initialized."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "sidebar_embedding_model" in at.session_state
            assert at.session_state["sidebar_embedding_model"] == "all-MiniLM-L6-v2"

    def test_preset_embedding_model_reflected_in_dropdown(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify pre-set embedding_model_name value is reflected in dropdown."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script)
            at.session_state["embedding_model_name"] = "all-mpnet-base-v2"
            at.run()

            embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
            # Should reflect the preset value if it exists in options
            if "all-mpnet-base-v2" in embedding_selectbox.options:
                assert embedding_selectbox.value == "all-mpnet-base-v2"


class TestEmbeddingModelUserAction:
    """Tests for user interaction with embedding model dropdown."""

    def test_can_change_embedding_model(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify user can change embedding model selection."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
            initial_value = embedding_selectbox.value
            options = embedding_selectbox.options

            # Find a different option
            other_options = [opt for opt in options if opt != initial_value]
            if other_options:
                other_option = other_options[0]
                embedding_selectbox.set_value(other_option).run()

                embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
                assert embedding_selectbox.value == other_option

    def test_model_selection_persists_after_rerun(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify model selection persists after app rerun."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
            options = embedding_selectbox.options
            initial_value = embedding_selectbox.value

            # Change to different model
            other_options = [opt for opt in options if opt != initial_value]
            if other_options:
                new_model = other_options[0]
                embedding_selectbox.set_value(new_model).run()

                # Rerun and check persistence
                at.run()
                embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
                assert embedding_selectbox.value == new_model


class TestEmbeddingModelFormSubmission:
    """Tests for embedding model form submission updating session state."""

    def test_form_submit_updates_embedding_model_name(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission updates embedding_model_name in session state."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Initial state
            assert at.session_state["embedding_model_name"] == "all-MiniLM-L6-v2"

            embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
            options = embedding_selectbox.options

            # Find a different model
            other_options = [opt for opt in options if opt != "all-MiniLM-L6-v2"]
            if other_options:
                new_model = other_options[0]
                at.selectbox(key="sidebar_embedding_model").set_value(new_model)

                # Submit the form
                submit_btn = get_form_submit_button(at)
                submit_btn.click().run()

                # Verify updated
                assert at.session_state["embedding_model_name"] == new_model

    def test_form_submit_preserves_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify changing embedding model preserves parsing params."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            initial_output_format = at.session_state["parsing_params"]["output_format"]
            initial_docling_device = at.session_state["parsing_params"]["docling_device"]

            embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
            options = embedding_selectbox.options
            other_options = [opt for opt in options if opt != "all-MiniLM-L6-v2"]

            if other_options:
                at.selectbox(key="sidebar_embedding_model").set_value(other_options[0])
                submit_btn = get_form_submit_button(at)
                submit_btn.click().run()

                # Verify parsing params preserved
                assert at.session_state["parsing_params"]["output_format"] == initial_output_format
                assert at.session_state["parsing_params"]["docling_device"] == initial_docling_device

    def test_form_submit_preserves_chunking_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify changing embedding model preserves chunking params."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            initial_chunk_size = at.session_state["chunking_params"]["chunk_size"]
            initial_chunk_overlap = at.session_state["chunking_params"]["chunk_overlap"]

            embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
            options = embedding_selectbox.options
            other_options = [opt for opt in options if opt != "all-MiniLM-L6-v2"]

            if other_options:
                at.selectbox(key="sidebar_embedding_model").set_value(other_options[0])
                submit_btn = get_form_submit_button(at)
                submit_btn.click().run()

                # Verify chunking params preserved
                assert at.session_state["chunking_params"]["chunk_size"] == initial_chunk_size
                assert at.session_state["chunking_params"]["chunk_overlap"] == initial_chunk_overlap


class TestEmbeddingModelCacheInvalidation:
    """Tests for cache invalidation when embedding model changes."""

    def test_model_change_clears_embeddings_result(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify changing embedding model clears last_embeddings_result."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Set up some existing embeddings result
            at.session_state["last_embeddings_result"] = {"dummy": "data"}

            embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
            options = embedding_selectbox.options
            other_options = [opt for opt in options if opt != "all-MiniLM-L6-v2"]

            if other_options:
                at.selectbox(key="sidebar_embedding_model").set_value(other_options[0])
                submit_btn = get_form_submit_button(at)
                submit_btn.click().run()

                # Verify embeddings result is cleared
                assert "last_embeddings_result" not in at.session_state

    def test_model_change_clears_search_results(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify changing embedding model clears search_results."""
        with patch(
            "unravel.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Set up some existing search results
            at.session_state["search_results"] = [{"dummy": "result"}]

            embedding_selectbox = at.selectbox(key="sidebar_embedding_model")
            options = embedding_selectbox.options
            other_options = [opt for opt in options if opt != "all-MiniLM-L6-v2"]

            if other_options:
                at.selectbox(key="sidebar_embedding_model").set_value(other_options[0])
                submit_btn = get_form_submit_button(at)
                submit_btn.click().run()

                # Verify search results are cleared
                assert "search_results" not in at.session_state
