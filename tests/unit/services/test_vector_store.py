"""Unit tests for vector store service.

Tests the vector store functionality including:
- Vector store creation with different metrics
- Adding embeddings
- Search functionality
- Save/load operations
- Payload caching
- Edge cases
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from unravel.services.vector_store import SearchResult, VectorStore


class TestVectorStoreInitialization:
    """Test vector store initialization."""

    @patch("unravel.services.vector_store._get_client")
    def test_vector_store_basic_init(self, mock_get_client):
        """VectorStore initializes with basic parameters."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        assert vs.dimension == 384
        assert vs.metric == "cosine"
        assert vs.collection_name == "unravel_chunks"

    @patch("unravel.services.vector_store._get_client")
    def test_vector_store_custom_collection_name(self, mock_get_client):
        """VectorStore accepts custom collection name."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_get_client.return_value = mock_client

        vs = VectorStore(
            dimension=384,
            metric="cosine",
            collection_name="custom_collection",
        )

        assert vs.collection_name == "custom_collection"

    @patch("unravel.services.vector_store._get_client")
    def test_vector_store_different_metrics(self, mock_get_client):
        """VectorStore supports different distance metrics."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_get_client.return_value = mock_client

        for metric in ["cosine", "l2", "ip"]:
            vs = VectorStore(dimension=384, metric=metric)
            assert vs.metric == metric

    @patch("unravel.services.vector_store._get_client")
    def test_vector_store_invalid_metric_raises_error(self, mock_get_client):
        """VectorStore raises ValueError for invalid metric when creating collection."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_get_client.return_value = mock_client

        # Invalid metric should raise error when _distance() is called during init
        with pytest.raises(ValueError, match="Unknown metric"):
            vs = VectorStore(dimension=384, metric="invalid_metric")


class TestVectorStoreAdd:
    """Test adding embeddings to vector store."""

    @patch("unravel.services.vector_store._get_client")
    def test_add_single_embedding(self, mock_get_client):
        """Adding single embedding works correctly."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        # Add single embedding
        embedding = np.random.rand(384).astype(np.float32)
        texts = ["Test text"]
        metadata = [{"key": "value"}]

        vs.add(embedding, texts, metadata)

        # Verify upsert was called
        mock_client.upsert.assert_called_once()
        assert vs.size == 1

    @patch("unravel.services.vector_store._get_client")
    def test_add_multiple_embeddings(self, mock_get_client):
        """Adding multiple embeddings works correctly."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        # Add multiple embeddings
        embeddings = np.random.rand(5, 384).astype(np.float32)
        texts = [f"Text {i}" for i in range(5)]
        metadata = [{"index": i} for i in range(5)]

        vs.add(embeddings, texts, metadata)

        # Verify upsert was called
        mock_client.upsert.assert_called_once()
        assert vs.size == 5

    @patch("unravel.services.vector_store._get_client")
    def test_add_without_metadata(self, mock_get_client):
        """Adding embeddings without metadata uses empty dicts."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        embeddings = np.random.rand(3, 384).astype(np.float32)
        texts = ["Text 1", "Text 2", "Text 3"]

        vs.add(embeddings, texts, metadata=None)

        # Should not raise error
        assert vs.size == 3

    @patch("unravel.services.vector_store._get_client")
    def test_add_dimension_mismatch_raises_error(self, mock_get_client):
        """Adding embeddings with wrong dimension raises ValueError."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        # Wrong dimension
        embeddings = np.random.rand(2, 768).astype(np.float32)
        texts = ["Text 1", "Text 2"]

        with pytest.raises(ValueError, match="Embedding dimension"):
            vs.add(embeddings, texts)

    @patch("unravel.services.vector_store._get_client")
    def test_add_text_count_mismatch_raises_error(self, mock_get_client):
        """Adding embeddings with mismatched text count raises ValueError."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        embeddings = np.random.rand(3, 384).astype(np.float32)
        texts = ["Text 1", "Text 2"]  # Only 2 texts for 3 embeddings

        with pytest.raises(ValueError, match="Number of texts"):
            vs.add(embeddings, texts)

    @patch("unravel.services.vector_store._get_client")
    def test_add_normalizes_for_cosine(self, mock_get_client):
        """Embeddings are normalized for cosine similarity."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        # Unnormalized embedding
        embedding = np.random.rand(384).astype(np.float32) * 10  # Large magnitude
        texts = ["Test"]

        vs.add(embedding, texts)

        # Check that upsert was called with normalized vectors
        call_args = mock_client.upsert.call_args
        points = call_args[1]["points"]
        vector = np.array(points[0].vector)

        # Vector should be approximately unit length
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 0.01


class TestVectorStoreSearch:
    """Test vector similarity search."""

    @patch("unravel.services.vector_store._get_client")
    def test_search_basic(self, mock_get_client):
        """Basic search returns SearchResult objects."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_client.count.return_value.count = 5

        # Mock search results
        mock_result = MagicMock()
        mock_result.id = 0
        mock_result.score = 0.95
        mock_result.payload = {"text": "Test chunk", "metadata": {"key": "value"}}

        # Try query_points first (newer API)
        mock_client.query_points.return_value.points = [mock_result]

        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        # Perform search
        query_embedding = np.random.rand(384).astype(np.float32)
        results = vs.search(query_embedding, k=5)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    @patch("unravel.services.vector_store._get_client")
    def test_search_returns_correct_k(self, mock_get_client):
        """Search returns at most k results."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_client.count.return_value.count = 10

        # Create mock results
        mock_results = []
        for i in range(3):
            mock_result = MagicMock()
            mock_result.id = i
            mock_result.score = 0.9 - i * 0.1
            mock_result.payload = {"text": f"Text {i}", "metadata": {}}
            mock_results.append(mock_result)

        mock_client.query_points.return_value.points = mock_results
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        query_embedding = np.random.rand(384).astype(np.float32)
        results = vs.search(query_embedding, k=3)

        assert len(results) <= 3

    @patch("unravel.services.vector_store._get_client")
    def test_search_empty_store_returns_empty(self, mock_get_client):
        """Search on empty store returns empty list."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_client.count.return_value.count = 0
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        query_embedding = np.random.rand(384).astype(np.float32)
        results = vs.search(query_embedding, k=5)

        assert results == []

    @patch("unravel.services.vector_store._get_client")
    def test_search_result_structure(self, mock_get_client):
        """SearchResult has all required fields."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_client.count.return_value.count = 1

        mock_result = MagicMock()
        mock_result.id = 42
        mock_result.score = 0.88
        mock_result.payload = {"text": "Test", "metadata": {"foo": "bar"}}

        mock_client.query_points.return_value.points = [mock_result]
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        query_embedding = np.random.rand(384).astype(np.float32)
        results = vs.search(query_embedding, k=1)

        assert len(results) == 1
        result = results[0]
        assert result.index == 42
        assert result.score == 0.88
        assert result.text == "Test"
        assert result.metadata == {"foo": "bar"}


class TestVectorStoreGetters:
    """Test getter methods for texts, metadata, embeddings."""

    @patch("unravel.services.vector_store._get_client")
    def test_get_texts_empty_store(self, mock_get_client):
        """get_texts returns empty list for empty store."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        texts = vs.get_texts()
        assert texts == []

    @patch("unravel.services.vector_store._get_client")
    def test_get_metadata_empty_store(self, mock_get_client):
        """get_metadata returns empty list for empty store."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        metadata = vs.get_metadata()
        assert metadata == []

    @patch("unravel.services.vector_store._get_client")
    def test_get_all_embeddings_empty_store(self, mock_get_client):
        """get_all_embeddings returns empty array for empty store."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_client.count.return_value.count = 0
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        embeddings = vs.get_all_embeddings()
        assert embeddings.shape == (0, 384)


class TestVectorStoreSaveLoad:
    """Test saving and loading vector store."""

    @patch("unravel.services.vector_store._get_client")
    def test_save_creates_metadata(self, mock_get_client, tmp_path):
        """save() creates metadata.json file."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_client.count.return_value.count = 0
        mock_get_client.return_value = mock_client

        store_path = tmp_path / "store"
        store_path.mkdir(parents=True, exist_ok=True)
        vs = VectorStore(dimension=384, metric="cosine", storage_path=store_path)

        save_path = tmp_path / "saved_store"
        vs.save(save_path)

        # Check metadata file exists
        metadata_file = save_path / "metadata.json"
        assert metadata_file.exists()

        # Check metadata content
        import json

        metadata = json.loads(metadata_file.read_text())
        assert metadata["dimension"] == 384
        assert metadata["metric"] == "cosine"

    @patch("unravel.services.vector_store._get_client")
    def test_load_restores_metadata(self, mock_get_client, tmp_path):
        """load() restores vector store from saved metadata."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        # Create metadata file
        save_path = tmp_path / "saved_store"
        save_path.mkdir(parents=True)

        metadata = {
            "dimension": 768,
            "metric": "l2",
            "size": 10,
            "collection_name": "test_collection",
        }

        import json

        (save_path / "metadata.json").write_text(json.dumps(metadata))

        # Load
        vs = VectorStore.load(save_path)

        assert vs.dimension == 768
        assert vs.metric == "l2"
        assert vs.collection_name == "test_collection"

    @patch("unravel.services.vector_store._get_client")
    def test_load_missing_metadata_raises_error(self, mock_get_client, tmp_path):
        """load() raises FileNotFoundError if metadata.json missing."""
        save_path = tmp_path / "missing_store"
        save_path.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Missing metadata.json"):
            VectorStore.load(save_path)


class TestVectorStoreClear:
    """Test clearing vector store."""

    @patch("unravel.services.vector_store._get_client")
    def test_clear_resets_store(self, mock_get_client):
        """clear() resets the vector store to empty state."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        # Simulate adding data
        vs._texts = ["Text 1", "Text 2"]
        vs._metadata = [{"a": 1}, {"b": 2}]
        vs._size = 2

        # Clear
        vs.clear()

        # Verify reset
        assert vs._texts == []
        assert vs._metadata == []
        assert vs._size == 0
        assert vs._cache_loaded is True


class TestVectorStoreSize:
    """Test size property."""

    @patch("unravel.services.vector_store._get_client")
    def test_size_empty_store(self, mock_get_client):
        """size returns 0 for empty store."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_client.count.return_value.count = 0
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        assert vs.size == 0

    @patch("unravel.services.vector_store._get_client")
    def test_size_after_add(self, mock_get_client):
        """size reflects number of added embeddings."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        # Add embeddings
        embeddings = np.random.rand(3, 384).astype(np.float32)
        texts = ["A", "B", "C"]
        vs.add(embeddings, texts)

        assert vs.size == 3


class TestVectorStoreEdgeCases:
    """Test edge cases and error handling."""

    @patch("unravel.services.vector_store._get_client")
    def test_add_1d_embedding_reshapes(self, mock_get_client):
        """1D embedding is automatically reshaped to 2D."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        # 1D embedding
        embedding = np.random.rand(384).astype(np.float32)
        texts = ["Test"]

        vs.add(embedding, texts)

        # Should not crash
        assert vs.size == 1

    @patch("unravel.services.vector_store._get_client")
    def test_search_1d_query_reshapes(self, mock_get_client):
        """1D query embedding is automatically reshaped."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_client.count.return_value.count = 1

        mock_result = MagicMock()
        mock_result.id = 0
        mock_result.score = 0.9
        mock_result.payload = {"text": "Test", "metadata": {}}

        mock_client.query_points.return_value.points = [mock_result]
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        # 1D query
        query = np.random.rand(384).astype(np.float32)
        results = vs.search(query, k=1)

        # Should not crash
        assert len(results) >= 0

    @patch("unravel.services.vector_store._get_client")
    def test_search_k_exceeds_size(self, mock_get_client):
        """Search with k > size returns all available results."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_client.count.return_value.count = 2

        mock_results = [
            MagicMock(id=0, score=0.9, payload={"text": "A", "metadata": {}}),
            MagicMock(id=1, score=0.8, payload={"text": "B", "metadata": {}}),
        ]

        mock_client.query_points.return_value.points = mock_results
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        query = np.random.rand(384).astype(np.float32)
        results = vs.search(query, k=100)  # k much larger than size

        # Should return all available
        assert len(results) == 2

    @patch("unravel.services.vector_store._get_client")
    def test_multiple_adds_increment_size(self, mock_get_client):
        """Multiple add operations correctly increment size."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        vs = VectorStore(dimension=384, metric="cosine")

        # First add
        vs.add(np.random.rand(2, 384).astype(np.float32), ["A", "B"])
        assert vs.size == 2

        # Second add
        vs.add(np.random.rand(3, 384).astype(np.float32), ["C", "D", "E"])
        assert vs.size == 5
