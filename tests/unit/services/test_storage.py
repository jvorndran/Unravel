"""Unit tests for storage service.

Tests the storage functionality including:
- Document save/load
- JSON save/load
- Session state persistence
- Configuration persistence
- BM25 index persistence
- File sanitization
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from unravel.services.storage import (
    DEFAULT_STORAGE_DIR,
    clear_documents,
    clear_llm_config,
    clear_session_state,
    clear_storage,
    delete_document,
    ensure_storage_dir,
    get_chunks_dir,
    get_current_document,
    get_documents_dir,
    get_embeddings_dir,
    get_indices_dir,
    get_storage_dir,
    get_storage_stats,
    list_documents,
    load_document,
    load_json,
    load_llm_config,
    load_rag_config,
    load_session_state,
    save_document,
    save_json,
    save_llm_config,
    save_rag_config,
    save_session_state,
)


class TestStorageDirectories:
    """Test storage directory management."""

    def test_get_storage_dir_returns_path(self):
        """get_storage_dir returns Path object."""
        storage_dir = get_storage_dir()
        assert isinstance(storage_dir, Path)
        assert storage_dir == DEFAULT_STORAGE_DIR

    def test_ensure_storage_dir_creates_structure(self, tmp_path):
        """ensure_storage_dir creates required subdirectories."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            storage_dir = ensure_storage_dir()

            assert storage_dir.exists()
            assert (storage_dir / "documents").exists()
            assert (storage_dir / "chunks").exists()
            assert (storage_dir / "embeddings").exists()
            assert (storage_dir / "indices").exists()

    def test_ensure_storage_dir_creates_env_template(self, tmp_path):
        """ensure_storage_dir creates .env template if missing."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            env_file = tmp_path / ".env"
            assert env_file.exists()

            content = env_file.read_text()
            assert "OPENAI_API_KEY" in content
            assert "ANTHROPIC_API_KEY" in content
            assert "GEMINI_API_KEY" in content
            assert "OPENROUTER_API_KEY" in content

    def test_get_documents_dir(self, tmp_path):
        """get_documents_dir returns documents subdirectory."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            docs_dir = get_documents_dir()
            assert docs_dir == tmp_path / "documents"

    def test_get_chunks_dir(self, tmp_path):
        """get_chunks_dir returns chunks subdirectory."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            chunks_dir = get_chunks_dir()
            assert chunks_dir == tmp_path / "chunks"

    def test_get_embeddings_dir(self, tmp_path):
        """get_embeddings_dir returns embeddings subdirectory."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            emb_dir = get_embeddings_dir()
            assert emb_dir == tmp_path / "embeddings"

    def test_get_indices_dir(self, tmp_path):
        """get_indices_dir returns indices subdirectory."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            idx_dir = get_indices_dir()
            assert idx_dir == tmp_path / "indices"


class TestDocumentStorage:
    """Test document save/load/delete operations."""

    def test_save_document_creates_file(self, tmp_path):
        """save_document creates file in documents directory."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            filename = "test.pdf"
            content = b"PDF content here"

            path = save_document(filename, content)

            assert path.exists()
            assert path.name == filename
            assert path.read_bytes() == content

    def test_save_document_clears_existing(self, tmp_path):
        """save_document clears existing documents first."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            docs_dir = get_documents_dir()
            docs_dir.mkdir(parents=True, exist_ok=True)

            # Create existing document
            (docs_dir / "old.pdf").write_bytes(b"old content")

            # Save new document
            save_document("new.pdf", b"new content")

            # Old document should be deleted
            assert not (docs_dir / "old.pdf").exists()
            assert (docs_dir / "new.pdf").exists()

    def test_load_document_returns_content(self, tmp_path):
        """load_document returns file content."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            filename = "test.txt"
            content = b"Test content"

            save_document(filename, content)
            loaded_content = load_document(filename)

            assert loaded_content == content

    def test_load_document_nonexistent_returns_none(self, tmp_path):
        """load_document returns None for nonexistent file."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()
            content = load_document("nonexistent.pdf")
            assert content is None

    def test_delete_document_removes_file(self, tmp_path):
        """delete_document removes file and returns True."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            filename = "test.pdf"
            save_document(filename, b"content")

            result = delete_document(filename)

            assert result is True
            assert not (get_documents_dir() / filename).exists()

    def test_delete_document_nonexistent_returns_false(self, tmp_path):
        """delete_document returns False for nonexistent file."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()
            result = delete_document("nonexistent.pdf")
            assert result is False

    def test_list_documents_returns_filenames(self, tmp_path):
        """list_documents returns list of document filenames."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            save_document("doc1.pdf", b"content1")
            save_document("doc2.txt", b"content2")

            docs = list_documents()

            assert len(docs) == 1  # Only one due to clear on save
            assert docs[0] in ["doc1.pdf", "doc2.txt"]

    def test_list_documents_empty_directory(self, tmp_path):
        """list_documents returns empty list for empty directory."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()
            docs = list_documents()
            assert docs == []

    def test_get_current_document_returns_tuple(self, tmp_path):
        """get_current_document returns (filename, content) tuple."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            filename = "test.pdf"
            content = b"PDF content"

            save_document(filename, content)
            result = get_current_document()

            assert result is not None
            assert result[0] == filename
            assert result[1] == content

    def test_get_current_document_empty_returns_none(self, tmp_path):
        """get_current_document returns None when no documents."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()
            result = get_current_document()
            assert result is None

    def test_clear_documents_removes_all(self, tmp_path):
        """clear_documents removes all documents."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            docs_dir = get_documents_dir()
            docs_dir.mkdir(parents=True, exist_ok=True)

            (docs_dir / "file1.pdf").write_bytes(b"content1")
            (docs_dir / "file2.txt").write_bytes(b"content2")

            clear_documents()

            assert list(docs_dir.iterdir()) == []


class TestFileSanitization:
    """Test filename sanitization for security."""

    def test_sanitize_removes_path_traversal(self, tmp_path):
        """Filename sanitization prevents path traversal."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            # Try path traversal
            path = save_document("../../etc/passwd", b"malicious")

            # Should save in documents directory, not traverse
            assert path.parent == get_documents_dir()
            assert ".." not in str(path)

    def test_sanitize_removes_absolute_paths(self, tmp_path):
        """Filename sanitization removes absolute path components."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            # Try absolute path
            path = save_document("/etc/passwd", b"malicious")

            # Should save in documents directory
            assert path.parent == get_documents_dir()
            assert path.name == "passwd"

    def test_sanitize_handles_empty_filename(self, tmp_path):
        """Filename sanitization handles empty/whitespace names."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            path = save_document("   ", b"content")

            # Should use default name
            assert path.name != ""
            assert path.name == "unnamed_file"


class TestJSONStorage:
    """Test JSON save/load operations."""

    def test_save_json_creates_file(self, tmp_path):
        """save_json creates JSON file in subdirectory."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            data = {"key": "value", "number": 42}
            path = save_json("chunks", "test.json", data)

            assert path.exists()
            assert path == tmp_path / "chunks" / "test.json"

            loaded = json.loads(path.read_text())
            assert loaded == data

    def test_load_json_returns_data(self, tmp_path):
        """load_json returns deserialized JSON data."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            data = {"list": [1, 2, 3], "nested": {"a": "b"}}
            save_json("chunks", "test.json", data)

            loaded = load_json("chunks", "test.json")
            assert loaded == data

    def test_load_json_nonexistent_returns_none(self, tmp_path):
        """load_json returns None for nonexistent file."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()
            data = load_json("chunks", "nonexistent.json")
            assert data is None

    def test_save_json_handles_complex_types(self, tmp_path):
        """save_json handles complex data types via default serializer."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            # Use Path object which needs str() conversion
            data = {"path": str(Path("/tmp/test")), "value": 123}
            path = save_json("chunks", "complex.json", data)

            loaded = json.loads(path.read_text())
            assert "path" in loaded


class TestSessionStatePersistence:
    """Test session state save/load operations."""

    def test_save_session_state_basic_fields(self, tmp_path):
        """save_session_state saves basic session fields."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            state_data = {
                "doc_name": "test.pdf",
                "embedding_model_name": "all-MiniLM-L6-v2",
                "chunking_params": {"max_tokens": 512},
                "parsing_params": {"normalize_whitespace": True},
            }

            save_session_state(state_data)

            # Check file was created
            state_file = tmp_path / "session" / "session_state.json"
            assert state_file.exists()

            # Check content
            loaded = json.loads(state_file.read_text())
            assert loaded["doc_name"] == "test.pdf"
            assert loaded["embedding_model_name"] == "all-MiniLM-L6-v2"

    def test_load_session_state_basic_fields(self, tmp_path):
        """load_session_state restores basic session fields."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            state_data = {
                "doc_name": "document.pdf",
                "chunking_params": {"provider": "Docling"},
            }

            save_session_state(state_data)
            loaded = load_session_state()

            assert loaded is not None
            assert loaded["doc_name"] == "document.pdf"
            assert loaded["chunking_params"]["provider"] == "Docling"

    def test_load_session_state_nonexistent_returns_none(self, tmp_path):
        """load_session_state returns None when no saved state."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()
            loaded = load_session_state()
            assert loaded is None

    def test_clear_session_state_removes_directory(self, tmp_path):
        """clear_session_state removes session directory."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            save_session_state({"doc_name": "test.pdf"})
            assert (tmp_path / "session").exists()

            clear_session_state()
            assert not (tmp_path / "session").exists()


class TestLLMConfigPersistence:
    """Test LLM configuration save/load operations."""

    def test_save_llm_config_creates_file(self, tmp_path):
        """save_llm_config saves configuration to file."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            config = {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
            }

            save_llm_config(config)

            config_file = tmp_path / "config" / "llm_config.json"
            assert config_file.exists()

    def test_save_llm_config_removes_api_key(self, tmp_path):
        """save_llm_config removes API key for security."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            config = {
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "sk-secret123",  # Should be removed
            }

            save_llm_config(config)

            # Load and check API key was removed
            config_file = tmp_path / "config" / "llm_config.json"
            saved_config = json.loads(config_file.read_text())
            assert "api_key" not in saved_config

    def test_load_llm_config_returns_data(self, tmp_path):
        """load_llm_config returns saved configuration."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            config = {
                "provider": "anthropic",
                "model": "claude-3",
                "temperature": 0.5,
            }

            save_llm_config(config)
            loaded = load_llm_config()

            assert loaded is not None
            assert loaded["provider"] == "anthropic"
            assert loaded["model"] == "claude-3"

    def test_load_llm_config_nonexistent_returns_none(self, tmp_path):
        """load_llm_config returns None when no saved config."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()
            loaded = load_llm_config()
            assert loaded is None

    def test_clear_llm_config_removes_file(self, tmp_path):
        """clear_llm_config removes configuration file."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            save_llm_config({"provider": "openai"})
            config_file = tmp_path / "config" / "llm_config.json"
            assert config_file.exists()

            clear_llm_config()
            assert not config_file.exists()


class TestRAGConfigPersistence:
    """Test RAG configuration save/load operations."""

    def test_save_rag_config_creates_file(self, tmp_path):
        """save_rag_config saves configuration to file."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            config = {
                "doc_name": "document.pdf",
                "embedding_model_name": "all-mpnet-base-v2",
                "chunking_params": {"max_tokens": 512},
            }

            save_rag_config(config)

            config_file = tmp_path / "config" / "rag_config.json"
            assert config_file.exists()

    def test_load_rag_config_returns_data(self, tmp_path):
        """load_rag_config returns saved configuration."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            config = {
                "doc_name": "test.pdf",
                "chunking_params": {"provider": "Docling"},
            }

            save_rag_config(config)
            loaded = load_rag_config()

            assert loaded is not None
            assert loaded["doc_name"] == "test.pdf"

    def test_load_rag_config_nonexistent_returns_none(self, tmp_path):
        """load_rag_config returns None when no saved config."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()
            loaded = load_rag_config()
            assert loaded is None


class TestStorageStatistics:
    """Test storage statistics collection."""

    def test_get_storage_stats_empty(self, tmp_path):
        """get_storage_stats returns zeros for empty storage."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            stats = get_storage_stats()

            assert stats["documents"] == 0
            assert stats["chunks"] == 0
            assert stats["embeddings"] == 0
            assert stats["indices"] == 0

    def test_get_storage_stats_with_files(self, tmp_path):
        """get_storage_stats counts files in each directory."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            # Add files
            save_document("doc.pdf", b"content")
            save_json("chunks", "chunk1.json", {})
            save_json("chunks", "chunk2.json", {})
            save_json("embeddings", "emb.json", {})

            stats = get_storage_stats()

            assert stats["documents"] == 1
            assert stats["chunks"] == 2
            assert stats["embeddings"] == 1


class TestClearStorage:
    """Test clearing all storage."""

    def test_clear_storage_removes_everything(self, tmp_path):
        """clear_storage removes all data and recreates structure."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            # Add some data
            save_document("doc.pdf", b"content")
            save_json("chunks", "test.json", {})

            # Clear storage
            clear_storage()

            # Directory should be recreated but empty
            assert tmp_path.exists()
            assert (tmp_path / "documents").exists()

            # Files should be gone
            assert list((tmp_path / "documents").iterdir()) == []
            assert list((tmp_path / "chunks").iterdir()) == []


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_save_document_unicode_filename(self, tmp_path):
        """save_document handles Unicode filenames."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            filename = "文档.pdf"
            content = b"content"

            path = save_document(filename, content)
            assert path.exists()

    def test_save_json_empty_data(self, tmp_path):
        """save_json handles empty data structures."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            # Empty dict
            save_json("chunks", "empty.json", {})
            loaded = load_json("chunks", "empty.json")
            assert loaded == {}

            # Empty list
            save_json("chunks", "empty_list.json", [])
            loaded = load_json("chunks", "empty_list.json")
            assert loaded == []

    def test_load_json_corrupted_raises_error(self, tmp_path):
        """load_json raises JSONDecodeError for corrupted JSON."""
        with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
            ensure_storage_dir()

            chunks_dir = tmp_path / "chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)

            # Write corrupted JSON
            (chunks_dir / "corrupted.json").write_text("not valid json{{{")

            # Should raise JSONDecodeError
            with pytest.raises(json.JSONDecodeError):
                load_json("chunks", "corrupted.json")
