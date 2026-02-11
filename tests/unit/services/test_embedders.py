"""Unit tests for embedding service.

Tests the embedding functionality including:
- Model initialization and loading
- Batch embedding generation
- Query embedding
- Dimension validation
- Normalization
- Backend adapter pattern
"""

from unittest.mock import MagicMock, patch

import numpy as np

from unravel.services.embedders import (
    DEFAULT_MODEL,
    EMBEDDING_MODELS,
    Embedder,
    get_embedder,
    list_available_models,
)


class TestEmbedderInitialization:
    """Test embedder initialization and configuration."""

    def test_embedder_default_initialization(self):
        """Embedder initializes with default model."""
        embedder = Embedder()
        assert embedder.model_name == DEFAULT_MODEL
        assert embedder._model is None  # Lazy loading

    def test_embedder_custom_model_initialization(self):
        """Embedder initializes with custom model name."""
        model_name = "all-mpnet-base-v2"
        embedder = Embedder(model_name=model_name)
        assert embedder.model_name == model_name

    def test_embedder_backend_detection_sentence_transformers(self):
        """Backend is correctly detected for sentence-transformers models."""
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        assert embedder.backend_name == "sentence-transformers"

    def test_embedder_backend_detection_flagembedding(self):
        """Backend is correctly detected for FlagEmbedding models."""
        embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
        assert embedder.backend_name == "flagembedding"

    def test_embedder_unknown_model_defaults_to_sentence_transformers(self):
        """Unknown models default to sentence-transformers backend."""
        embedder = Embedder(model_name="unknown/model")
        assert embedder.backend_name == "sentence-transformers"

    def test_embedder_device_setting(self):
        """Device setting is stored correctly."""
        embedder = Embedder(device="cpu")
        assert embedder._device == "cpu"


class TestEmbedderDimension:
    """Test embedding dimension handling."""

    def test_dimension_from_registry(self):
        """Dimension is retrieved from model registry."""
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        assert embedder.dimension == 384

    def test_dimension_for_different_models(self):
        """Different models have correct dimensions."""
        test_cases = [
            ("all-MiniLM-L6-v2", 384),
            ("all-mpnet-base-v2", 768),
            ("BAAI/bge-large-en-v1.5", 1024),
        ]

        for model_name, expected_dim in test_cases:
            embedder = Embedder(model_name=model_name)
            assert embedder.dimension == expected_dim

    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.load_model")
    def test_dimension_fallback_for_unknown_model(self, mock_load):
        """Dimension fallback works for unknown models."""
        # Mock model with dimension method
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 512
        mock_load.return_value = mock_model

        embedder = Embedder(model_name="unknown/custom-model")
        embedder._model = mock_model
        assert embedder.dimension == 512


class TestEmbedderModelLoading:
    """Test lazy model loading."""

    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.load_model")
    def test_model_lazy_loading(self, mock_load):
        """Model is loaded only when accessed."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        assert embedder._model is None  # Not loaded yet

        # Access model property triggers loading
        _ = embedder.model
        mock_load.assert_called_once()
        assert embedder._model is not None

    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.load_model")
    def test_model_loaded_only_once(self, mock_load):
        """Model is loaded only once (cached)."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        _ = embedder.model  # First access
        _ = embedder.model  # Second access

        # Should only be called once
        mock_load.assert_called_once()


class TestEmbedTexts:
    """Test batch text embedding generation."""

    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.load_model")
    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.embed_texts")
    def test_embed_texts_basic(self, mock_embed, mock_load):
        """embed_texts generates embeddings for multiple texts."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        # Mock embeddings: 3 texts, 384 dimensions
        expected_embeddings = np.random.rand(3, 384).astype(np.float32)
        mock_embed.return_value = expected_embeddings

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        texts = ["First text", "Second text", "Third text"]
        embeddings = embedder.embed_texts(texts)

        # Check shape
        assert embeddings.shape == (3, 384)
        assert np.array_equal(embeddings, expected_embeddings)

    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.load_model")
    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.embed_texts")
    def test_embed_texts_empty_list(self, mock_embed, mock_load):
        """embed_texts with empty list returns empty array."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        embeddings = embedder.embed_texts([])

        # Should return empty array with correct shape
        assert embeddings.shape == (0, 384)
        # Should not call backend embed
        mock_embed.assert_not_called()

    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.load_model")
    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.embed_texts")
    def test_embed_texts_with_batch_size(self, mock_embed, mock_load):
        """embed_texts respects batch_size parameter."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        expected_embeddings = np.random.rand(5, 384).astype(np.float32)
        mock_embed.return_value = expected_embeddings

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        texts = ["Text"] * 5
        embeddings = embedder.embed_texts(texts, batch_size=2)

        # Verify batch_size is passed to backend
        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args[1]
        assert call_kwargs["batch_size"] == 2

    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.load_model")
    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.embed_texts")
    def test_embed_texts_normalization(self, mock_embed, mock_load):
        """embed_texts normalization parameter is passed correctly."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        expected_embeddings = np.random.rand(2, 384).astype(np.float32)
        mock_embed.return_value = expected_embeddings

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        texts = ["Text 1", "Text 2"]

        # Test with normalize=True (default)
        embedder.embed_texts(texts, normalize=True)
        call_kwargs = mock_embed.call_args[1]
        assert call_kwargs["normalize"] is True

        # Test with normalize=False
        embedder.embed_texts(texts, normalize=False)
        call_kwargs = mock_embed.call_args[1]
        assert call_kwargs["normalize"] is False


class TestEmbedQuery:
    """Test single query embedding generation."""

    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.load_model")
    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.embed_query")
    def test_embed_query_basic(self, mock_embed, mock_load):
        """embed_query generates embedding for single query."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        # Mock query embedding
        expected_embedding = np.random.rand(384).astype(np.float32)
        mock_embed.return_value = expected_embedding

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        query = "What is the meaning of life?"
        embedding = embedder.embed_query(query)

        # Check shape (should be 1D)
        assert embedding.shape == (384,)
        assert np.array_equal(embedding, expected_embedding)

    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.load_model")
    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.embed_query")
    def test_embed_query_normalization(self, mock_embed, mock_load):
        """embed_query normalization parameter is passed correctly."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        expected_embedding = np.random.rand(384).astype(np.float32)
        mock_embed.return_value = expected_embedding

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        query = "Test query"

        # Test with normalize=True (default)
        embedder.embed_query(query, normalize=True)
        call_kwargs = mock_embed.call_args[1]
        assert call_kwargs["normalize"] is True

        # Test with normalize=False
        embedder.embed_query(query, normalize=False)
        call_kwargs = mock_embed.call_args[1]
        assert call_kwargs["normalize"] is False


class TestGetEmbedder:
    """Test embedder factory function."""

    @patch("streamlit.cache_resource")
    def test_get_embedder_returns_embedder(self, mock_cache):
        """get_embedder returns Embedder instance."""
        # Mock cache decorator to pass through
        mock_cache.side_effect = lambda func: func

        embedder = get_embedder()
        assert isinstance(embedder, Embedder)
        assert embedder.model_name == DEFAULT_MODEL

    @patch("streamlit.cache_resource")
    def test_get_embedder_custom_model(self, mock_cache):
        """get_embedder accepts custom model name."""
        mock_cache.side_effect = lambda func: func

        model_name = "all-mpnet-base-v2"
        embedder = get_embedder(model_name=model_name)
        assert embedder.model_name == model_name


class TestListAvailableModels:
    """Test model listing functionality."""

    def test_list_available_models_returns_list(self):
        """list_available_models returns non-empty list."""
        models = list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_list_available_models_structure(self):
        """Each model entry has required fields."""
        models = list_available_models()

        for model in models:
            assert "name" in model
            assert "dimension" in model
            assert "description" in model
            assert isinstance(model["name"], str)
            assert isinstance(model["dimension"], int)
            assert isinstance(model["description"], str)

    def test_list_available_models_includes_default(self):
        """Default model is included in available models."""
        models = list_available_models()
        model_names = [m["name"] for m in models]
        assert DEFAULT_MODEL in model_names

    def test_list_available_models_includes_various_backends(self):
        """Available models include different backends."""
        models = list_available_models()

        # Check for sentence-transformers models
        st_models = [m for m in models if m.get("backend") == "sentence-transformers"]
        assert len(st_models) > 0

        # Check for FlagEmbedding models
        flag_models = [m for m in models if m.get("backend") == "flagembedding"]
        assert len(flag_models) > 0


class TestEmbeddingModelsRegistry:
    """Test EMBEDDING_MODELS registry structure."""

    def test_registry_has_models(self):
        """EMBEDDING_MODELS registry is not empty."""
        assert len(EMBEDDING_MODELS) > 0

    def test_default_model_in_registry(self):
        """Default model is in registry."""
        assert DEFAULT_MODEL in EMBEDDING_MODELS

    def test_registry_model_structure(self):
        """Each model in registry has required fields."""
        required_fields = ["dimension", "description", "backend"]

        for model_name, model_info in EMBEDDING_MODELS.items():
            for field in required_fields:
                assert field in model_info, f"Model {model_name} missing {field}"

    def test_registry_dimension_types(self):
        """All dimensions are positive integers."""
        for model_name, model_info in EMBEDDING_MODELS.items():
            dim = model_info["dimension"]
            assert isinstance(dim, int), f"Model {model_name} dimension is not int"
            assert dim > 0, f"Model {model_name} dimension is not positive"

    def test_registry_backend_values(self):
        """All backends are valid values."""
        valid_backends = {"sentence-transformers", "flagembedding"}

        for model_name, model_info in EMBEDDING_MODELS.items():
            backend = model_info["backend"]
            assert backend in valid_backends, f"Model {model_name} has invalid backend"


class TestEmbedderEdgeCases:
    """Test edge cases and error handling."""

    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.load_model")
    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.embed_texts")
    def test_embed_texts_single_text(self, mock_embed, mock_load):
        """embed_texts works with single text."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        expected_embeddings = np.random.rand(1, 384).astype(np.float32)
        mock_embed.return_value = expected_embeddings

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        embeddings = embedder.embed_texts(["Single text"])

        assert embeddings.shape == (1, 384)

    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.load_model")
    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.embed_texts")
    def test_embed_texts_unicode_handling(self, mock_embed, mock_load):
        """embed_texts handles Unicode text."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        expected_embeddings = np.random.rand(2, 384).astype(np.float32)
        mock_embed.return_value = expected_embeddings

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        texts = ["Hello 世界", "Café ☕"]
        embeddings = embedder.embed_texts(texts)

        # Should not crash and return correct shape
        assert embeddings.shape == (2, 384)

    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.load_model")
    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.embed_query")
    def test_embed_query_empty_string(self, mock_embed, mock_load):
        """embed_query handles empty string."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        expected_embedding = np.random.rand(384).astype(np.float32)
        mock_embed.return_value = expected_embedding

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        embedding = embedder.embed_query("")

        # Should not crash
        assert embedding.shape == (384,)

    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.load_model")
    @patch("unravel.services.embedding_backends.SentenceTransformersAdapter.embed_texts")
    def test_embed_texts_very_long_text(self, mock_embed, mock_load):
        """embed_texts handles very long texts."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        expected_embeddings = np.random.rand(1, 384).astype(np.float32)
        mock_embed.return_value = expected_embeddings

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        long_text = "word " * 10000  # Very long text
        embeddings = embedder.embed_texts([long_text])

        # Should handle long text
        assert embeddings.shape == (1, 384)
