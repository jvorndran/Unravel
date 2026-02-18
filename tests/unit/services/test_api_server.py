"""Unit tests for API server."""

import json
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from unravel.services.api_server import (
    PipelineState,
    QueryRequest,
    create_app,
    pipeline_state,
    update_pipeline_state,
)
from unravel.services.llm import LLMConfig


@pytest.fixture
def reset_pipeline_state() -> None:
    """Reset pipeline state before each test."""
    pipeline_state.vector_store = None
    pipeline_state.embedder = None
    pipeline_state.llm_config = None
    pipeline_state.system_prompt = ""
    pipeline_state.retrieval_config = {}
    pipeline_state.reranking_config = {}
    pipeline_state.bm25_index_data = None
    pipeline_state.top_k = 5
    pipeline_state.threshold = 0.3


@pytest.fixture
def test_client(reset_pipeline_state: None) -> TestClient:
    """Create test client with FastAPI app."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_vector_store() -> Mock:
    """Create mock vector store."""
    mock = Mock()
    mock.size = 100
    return mock


@pytest.fixture
def mock_embedder() -> Mock:
    """Create mock embedder."""
    mock = Mock()
    mock.model_name = "test-embedder"
    return mock


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    """Create mock LLM config."""
    return LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="test-key",
        temperature=0.7,
    )


@pytest.mark.unit
class TestPipelineState:
    """Tests for PipelineState class."""

    def test_initialization(self) -> None:
        """Test PipelineState initializes with correct defaults."""
        state = PipelineState()

        assert state.vector_store is None
        assert state.embedder is None
        assert state.llm_config is None
        assert state.system_prompt == ""
        assert state.retrieval_config == {}
        assert state.reranking_config == {}
        assert state.bm25_index_data is None
        assert state.top_k == 5
        assert state.threshold == 0.3


@pytest.mark.unit
class TestQueryRequest:
    """Tests for QueryRequest model."""

    def test_valid_request(self) -> None:
        """Test QueryRequest with valid data."""
        request = QueryRequest(query="What is RAG?")

        assert request.query == "What is RAG?"

    def test_empty_query(self) -> None:
        """Test QueryRequest with empty query."""
        # Pydantic allows empty string, validation happens in endpoint
        request = QueryRequest(query="")
        assert request.query == ""


@pytest.mark.unit
class TestUpdatePipelineState:
    """Tests for update_pipeline_state function."""

    def test_update_all_fields(
        self,
        mock_vector_store: Mock,
        mock_embedder: Mock,
        mock_llm_config: LLMConfig,
        reset_pipeline_state: None,
    ) -> None:
        """Test updating all pipeline state fields."""
        retrieval_config = {"strategy": "DenseRetriever", "params": {}}
        reranking_config = {"enabled": True, "model": "ms-marco-MiniLM-L-12-v2"}
        bm25_data = {"index": "test"}

        update_pipeline_state(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm_config=mock_llm_config,
            system_prompt="Test prompt",
            retrieval_config=retrieval_config,
            reranking_config=reranking_config,
            bm25_index_data=bm25_data,
            top_k=10,
            threshold=0.5,
        )

        assert pipeline_state.vector_store == mock_vector_store
        assert pipeline_state.embedder == mock_embedder
        assert pipeline_state.llm_config == mock_llm_config
        assert pipeline_state.system_prompt == "Test prompt"
        assert pipeline_state.retrieval_config == retrieval_config
        assert pipeline_state.reranking_config == reranking_config
        assert pipeline_state.bm25_index_data == bm25_data
        assert pipeline_state.top_k == 10
        assert pipeline_state.threshold == 0.5

    def test_update_partial_fields(
        self, mock_vector_store: Mock, reset_pipeline_state: None
    ) -> None:
        """Test updating only some fields."""
        update_pipeline_state(vector_store=mock_vector_store, top_k=7)

        assert pipeline_state.vector_store == mock_vector_store
        assert pipeline_state.top_k == 7
        # Other fields remain at defaults
        assert pipeline_state.embedder is None
        assert pipeline_state.threshold == 0.3


@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, test_client: TestClient) -> None:
        """Test health check returns healthy status."""
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


@pytest.mark.unit
class TestStatusEndpoint:
    """Tests for /status endpoint."""

    def test_status_not_configured(self, test_client: TestClient) -> None:
        """Test status when pipeline is not configured."""
        response = test_client.get("/status")

        assert response.status_code == 200
        data = response.json()

        assert data["pipeline_ready"] is False
        assert data["vector_store_size"] == 0
        assert data["embedder_model"] is None
        assert data["llm_provider"] is None
        assert data["llm_model"] is None

    def test_status_configured(
        self,
        test_client: TestClient,
        mock_vector_store: Mock,
        mock_embedder: Mock,
        mock_llm_config: LLMConfig,
    ) -> None:
        """Test status when pipeline is configured."""
        retrieval_config = {"strategy": "DenseRetriever"}

        update_pipeline_state(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm_config=mock_llm_config,
            retrieval_config=retrieval_config,
        )

        response = test_client.get("/status")

        assert response.status_code == 200
        data = response.json()

        assert data["pipeline_ready"] is True
        assert data["vector_store_size"] == 100
        assert data["embedder_model"] == "test-embedder"
        assert data["llm_provider"] == "openai"
        assert data["llm_model"] == "gpt-4"
        assert data["retrieval_strategy"] == "DenseRetriever"


@pytest.mark.unit
class TestQueryEndpoint:
    """Tests for /query endpoint."""

    def test_query_pipeline_not_initialized(self, test_client: TestClient) -> None:
        """Test query endpoint when pipeline is not initialized."""
        response = test_client.post(
            "/query",
            json={"query": "What is RAG?"},
        )

        assert response.status_code == 503
        assert "Pipeline not initialized" in response.json()["detail"]

    def test_query_llm_not_configured(
        self, test_client: TestClient, mock_vector_store: Mock, mock_embedder: Mock
    ) -> None:
        """Test query endpoint when LLM is not configured."""
        update_pipeline_state(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        response = test_client.post(
            "/query",
            json={"query": "What is RAG?"},
        )

        assert response.status_code == 503
        assert "LLM not configured" in response.json()["detail"]

    def test_query_empty_query(
        self,
        test_client: TestClient,
        mock_vector_store: Mock,
        mock_embedder: Mock,
        mock_llm_config: LLMConfig,
    ) -> None:
        """Test query endpoint with empty query."""
        update_pipeline_state(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm_config=mock_llm_config,
        )

        response = test_client.post(
            "/query",
            json={"query": "   "},
        )

        assert response.status_code == 400
        assert "Query cannot be empty" in response.json()["detail"]

    @patch("unravel.services.api_server.retrieve")
    @patch("unravel.services.api_server.get_model")
    def test_query_successful_stream(
        self,
        mock_get_model: MagicMock,
        mock_retrieve: MagicMock,
        test_client: TestClient,
        mock_vector_store: Mock,
        mock_embedder: Mock,
        mock_llm_config: LLMConfig,
    ) -> None:
        """Test successful query with streaming response."""
        # Setup mocks
        mock_result = Mock()
        mock_result.text = "Sample chunk text"
        mock_result.score = 0.85
        mock_result.metadata = {"source": "test.pdf"}
        mock_retrieve.return_value = [mock_result]

        mock_model = Mock()
        mock_model.stream.return_value = iter(["Hello", " world"])
        mock_get_model.return_value = mock_model

        # Configure pipeline
        update_pipeline_state(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm_config=mock_llm_config,
            system_prompt="Test prompt",
            retrieval_config={"strategy": "DenseRetriever", "params": {}},
            top_k=5,
            threshold=0.3,
        )

        # Make request
        response = test_client.post(
            "/query",
            json={"query": "What is RAG?"},
        )

        assert response.status_code == 200

        # Parse SSE events
        events = []
        for line in response.text.strip().split("\n"):
            if line.startswith("data: "):
                event_data = json.loads(line[6:])
                events.append(event_data)

        # Verify event sequence
        event_types = [e["type"] for e in events]
        assert "status" in event_types  # "Retrieving context..."
        assert "chunks" in event_types
        assert "text" in event_types
        assert "done" in event_types

        # Verify chunks data
        chunks_event = next(e for e in events if e["type"] == "chunks")
        assert len(chunks_event["data"]) == 1
        assert chunks_event["data"][0]["text"] == "Sample chunk text"
        assert chunks_event["data"][0]["score"] == 0.85

    @patch("unravel.services.api_server.retrieve")
    def test_query_no_results_above_threshold(
        self,
        mock_retrieve: MagicMock,
        test_client: TestClient,
        mock_vector_store: Mock,
        mock_embedder: Mock,
        mock_llm_config: LLMConfig,
    ) -> None:
        """Test query when no results are above threshold."""
        # Return result below threshold
        mock_result = Mock()
        mock_result.text = "Sample chunk"
        mock_result.score = 0.1  # Below default threshold of 0.3
        mock_result.metadata = {}
        mock_retrieve.return_value = [mock_result]

        update_pipeline_state(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm_config=mock_llm_config,
            threshold=0.3,
        )

        response = test_client.post(
            "/query",
            json={"query": "What is RAG?"},
        )

        # Parse events
        events = []
        for line in response.text.strip().split("\n"):
            if line.startswith("data: "):
                event_data = json.loads(line[6:])
                events.append(event_data)

        # Should have error about no relevant chunks
        error_event = next((e for e in events if e["type"] == "error"), None)
        assert error_event is not None
        assert "No relevant chunks found" in error_event["message"]

    @patch("unravel.services.api_server.retrieve")
    @patch("unravel.services.api_server.get_model")
    def test_query_uses_pipeline_top_k_threshold(
        self,
        mock_get_model: MagicMock,
        mock_retrieve: MagicMock,
        test_client: TestClient,
        mock_vector_store: Mock,
        mock_embedder: Mock,
        mock_llm_config: LLMConfig,
    ) -> None:
        """Test that query uses top_k and threshold from pipeline state."""
        # Setup mocks
        mock_results = [Mock(text=f"Chunk {i}", score=0.9 - i * 0.1, metadata={}) for i in range(10)]
        mock_retrieve.return_value = mock_results

        mock_model = Mock()
        mock_model.stream.return_value = iter(["Response"])
        mock_get_model.return_value = mock_model

        # Configure with custom top_k and threshold
        update_pipeline_state(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm_config=mock_llm_config,
            retrieval_config={"strategy": "DenseRetriever", "params": {}},
            top_k=3,
            threshold=0.7,
        )

        response = test_client.post(
            "/query",
            json={"query": "Test query"},
        )

        # Verify retrieve was called with top_k=3
        mock_retrieve.assert_called_once()
        assert mock_retrieve.call_args[1]["k"] == 3

        # Parse chunks from response
        events = []
        for line in response.text.strip().split("\n"):
            if line.startswith("data: "):
                event_data = json.loads(line[6:])
                events.append(event_data)

        chunks_event = next(e for e in events if e["type"] == "chunks")
        # Should only have chunks with score >= 0.7
        assert all(chunk["score"] >= 0.7 for chunk in chunks_event["data"])
        # Should have at most 3 chunks (top_k)
        assert len(chunks_event["data"]) <= 3
