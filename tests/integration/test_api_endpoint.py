"""Integration tests for API endpoint feature."""

import json
import socket
import time
from typing import Any
from unittest.mock import Mock, patch

import pytest
import requests

from unravel.services.api_server import update_pipeline_state
from unravel.services.llm import LLMConfig
from unravel.utils.server_manager import ServerManager


@pytest.fixture
def integration_server_manager() -> ServerManager:
    """Create ServerManager for integration tests."""
    return ServerManager(host="127.0.0.1", port=8003)


@pytest.fixture
def mock_vector_store() -> Mock:
    """Create mock vector store with test data."""
    mock = Mock()
    mock.size = 50
    return mock


@pytest.fixture
def mock_embedder() -> Mock:
    """Create mock embedder."""
    mock = Mock()
    mock.model_name = "integration-test-embedder"
    return mock


@pytest.fixture
def test_llm_config() -> LLMConfig:
    """Create test LLM configuration."""
    return LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="test-key-integration",
        temperature=0.5,
    )


def is_port_available(port: int) -> bool:
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


@pytest.mark.integration
class TestAPIEndpointFullPipeline:
    """Integration tests for full API endpoint pipeline."""

    def test_server_startup_and_health_check(
        self, integration_server_manager: ServerManager
    ) -> None:
        """Test server starts up and responds to health checks."""
        integration_server_manager.start()

        try:
            time.sleep(1)

            # Verify port is in use
            assert not is_port_available(integration_server_manager.port)

            # Health check
            response = requests.get(
                f"{integration_server_manager.get_base_url()}/health",
                timeout=5,
            )

            assert response.status_code == 200
            assert response.json() == {"status": "healthy"}
        finally:
            integration_server_manager.stop()
            time.sleep(1)

    def test_server_restart(
        self, integration_server_manager: ServerManager
    ) -> None:
        """Test server can be stopped and restarted."""
        # Start server
        integration_server_manager.start()
        time.sleep(1)

        response = requests.get(
            f"{integration_server_manager.get_base_url()}/health",
            timeout=5,
        )
        assert response.status_code == 200

        # Stop server
        integration_server_manager.stop()
        time.sleep(1)

        # Verify stopped
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.get(
                f"{integration_server_manager.get_base_url()}/health",
                timeout=2,
            )

        # Restart server
        integration_server_manager.start()
        time.sleep(1)

        response = requests.get(
            f"{integration_server_manager.get_base_url()}/health",
            timeout=5,
        )
        assert response.status_code == 200

        integration_server_manager.stop()

    def test_status_endpoint_reflects_pipeline_state(
        self,
        integration_server_manager: ServerManager,
        mock_vector_store: Mock,
        mock_embedder: Mock,
        test_llm_config: LLMConfig,
    ) -> None:
        """Test /status endpoint reflects current pipeline configuration."""
        integration_server_manager.start()

        try:
            time.sleep(1)

            # Initially not configured
            response = requests.get(
                f"{integration_server_manager.get_base_url()}/status",
                timeout=5,
            )
            status = response.json()
            assert status["pipeline_ready"] is False

            # Configure pipeline
            update_pipeline_state(
                vector_store=mock_vector_store,
                embedder=mock_embedder,
                llm_config=test_llm_config,
                system_prompt="Integration test prompt",
                retrieval_config={"strategy": "DenseRetriever"},
                top_k=8,
                threshold=0.4,
            )

            # Check status again
            response = requests.get(
                f"{integration_server_manager.get_base_url()}/status",
                timeout=5,
            )
            status = response.json()

            assert status["pipeline_ready"] is True
            assert status["vector_store_size"] == 50
            assert status["embedder_model"] == "integration-test-embedder"
            assert status["llm_provider"] == "openai"
            assert status["llm_model"] == "gpt-4"
            assert status["retrieval_strategy"] == "DenseRetriever"
        finally:
            integration_server_manager.stop()

    @patch("unravel.services.api_server.retrieve")
    @patch("unravel.services.api_server.get_model")
    def test_query_endpoint_full_flow(
        self,
        mock_get_model: Mock,
        mock_retrieve: Mock,
        integration_server_manager: ServerManager,
        mock_vector_store: Mock,
        mock_embedder: Mock,
        test_llm_config: LLMConfig,
    ) -> None:
        """Test full query flow: retrieval -> generation -> streaming."""
        # Setup mocks
        mock_results = [
            Mock(
                text=f"Test chunk {i}",
                score=0.9 - i * 0.1,
                metadata={"source": f"doc{i}.pdf"},
            )
            for i in range(5)
        ]
        mock_retrieve.return_value = mock_results

        mock_model = Mock()
        mock_model.stream.return_value = iter(
            ["This", " is", " a", " test", " response"]
        )
        mock_get_model.return_value = mock_model

        # Start server and configure pipeline
        integration_server_manager.start()

        try:
            time.sleep(1)

            update_pipeline_state(
                vector_store=mock_vector_store,
                embedder=mock_embedder,
                llm_config=test_llm_config,
                system_prompt="Answer concisely",
                retrieval_config={"strategy": "DenseRetriever", "params": {}},
                top_k=5,
                threshold=0.5,
            )

            # Make query request
            response = requests.post(
                f"{integration_server_manager.get_base_url()}/query",
                json={"query": "What is RAG?"},
                stream=True,
                timeout=10,
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

            # Parse SSE events
            events: list[dict[str, Any]] = []
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    event_data = json.loads(line[6:])
                    events.append(event_data)

            # Verify event sequence
            event_types = [e["type"] for e in events]
            assert "status" in event_types
            assert "chunks" in event_types
            assert "text" in event_types
            assert "done" in event_types

            # Verify chunks event
            chunks_event = next(e for e in events if e["type"] == "chunks")
            # Should filter by threshold (0.5)
            assert all(chunk["score"] >= 0.5 for chunk in chunks_event["data"])
            # Should respect top_k (5)
            assert len(chunks_event["data"]) <= 5

            # Verify text chunks are streamed
            text_events = [e for e in events if e["type"] == "text"]
            assert len(text_events) == 5
            text_chunks = [e["chunk"] for e in text_events]
            assert text_chunks == ["This", " is", " a", " test", " response"]

            # Verify done event
            done_event = next(e for e in events if e["type"] == "done")
            assert done_event is not None
        finally:
            integration_server_manager.stop()

    def test_query_endpoint_respects_ui_settings(
        self,
        integration_server_manager: ServerManager,
        mock_vector_store: Mock,
        mock_embedder: Mock,
        test_llm_config: LLMConfig,
    ) -> None:
        """Test query endpoint uses settings from UI (top_k, threshold from pipeline)."""
        integration_server_manager.start()

        try:
            time.sleep(1)

            with patch("unravel.services.api_server.retrieve") as mock_retrieve, patch(
                "unravel.services.api_server.get_model"
            ) as mock_get_model:
                # Setup mocks
                mock_results = [
                    Mock(text=f"Chunk {i}", score=0.95 - i * 0.05, metadata={})
                    for i in range(20)
                ]
                mock_retrieve.return_value = mock_results

                mock_model = Mock()
                mock_model.stream.return_value = iter(["Response"])
                mock_get_model.return_value = mock_model

                # Configure with specific settings
                update_pipeline_state(
                    vector_store=mock_vector_store,
                    embedder=mock_embedder,
                    llm_config=test_llm_config,
                    retrieval_config={"strategy": "DenseRetriever", "params": {}},
                    top_k=3,  # UI setting
                    threshold=0.8,  # UI setting
                )

                # Query (note: no top_k or threshold in request body)
                response = requests.post(
                    f"{integration_server_manager.get_base_url()}/query",
                    json={"query": "Test query"},
                    stream=True,
                    timeout=10,
                )

                # Verify retrieve was called with top_k from pipeline
                mock_retrieve.assert_called_once()
                assert mock_retrieve.call_args[1]["k"] == 3

                # Parse response
                events = []
                for line in response.iter_lines(decode_unicode=True):
                    if line.startswith("data: "):
                        event_data = json.loads(line[6:])
                        events.append(event_data)

                # Verify chunks respect threshold and top_k
                chunks_event = next(e for e in events if e["type"] == "chunks")
                assert all(chunk["score"] >= 0.8 for chunk in chunks_event["data"])
                assert len(chunks_event["data"]) <= 3
        finally:
            integration_server_manager.stop()

    def test_multiple_queries_in_sequence(
        self,
        integration_server_manager: ServerManager,
        mock_vector_store: Mock,
        mock_embedder: Mock,
        test_llm_config: LLMConfig,
    ) -> None:
        """Test multiple sequential queries work correctly."""
        integration_server_manager.start()

        try:
            time.sleep(1)

            with patch("unravel.services.api_server.retrieve") as mock_retrieve, patch(
                "unravel.services.api_server.get_model"
            ) as mock_get_model:
                # Setup mocks
                mock_result = Mock(text="Test", score=0.9, metadata={})
                mock_retrieve.return_value = [mock_result]

                mock_model = Mock()
                mock_model.stream.return_value = iter(["Response"])
                mock_get_model.return_value = mock_model

                update_pipeline_state(
                    vector_store=mock_vector_store,
                    embedder=mock_embedder,
                    llm_config=test_llm_config,
                )

                # Make multiple queries
                queries = ["Query 1", "Query 2", "Query 3"]
                for query_text in queries:
                    response = requests.post(
                        f"{integration_server_manager.get_base_url()}/query",
                        json={"query": query_text},
                        stream=True,
                        timeout=10,
                    )

                    assert response.status_code == 200

                    # Verify we get done event
                    events = []
                    for line in response.iter_lines(decode_unicode=True):
                        if line.startswith("data: "):
                            event_data = json.loads(line[6:])
                            events.append(event_data)

                    assert any(e["type"] == "done" for e in events)
        finally:
            integration_server_manager.stop()


@pytest.mark.integration
class TestAPIEndpointErrorHandling:
    """Integration tests for error handling."""

    def test_query_before_pipeline_configured(
        self, integration_server_manager: ServerManager
    ) -> None:
        """Test querying before pipeline is configured returns error."""
        integration_server_manager.start()

        try:
            time.sleep(1)

            response = requests.post(
                f"{integration_server_manager.get_base_url()}/query",
                json={"query": "Test"},
                timeout=5,
            )

            assert response.status_code == 503
            assert "Pipeline not initialized" in response.json()["detail"]
        finally:
            integration_server_manager.stop()

    def test_query_with_empty_query(
        self,
        integration_server_manager: ServerManager,
        mock_vector_store: Mock,
        mock_embedder: Mock,
        test_llm_config: LLMConfig,
    ) -> None:
        """Test empty query returns validation error."""
        integration_server_manager.start()

        try:
            time.sleep(1)

            update_pipeline_state(
                vector_store=mock_vector_store,
                embedder=mock_embedder,
                llm_config=test_llm_config,
            )

            response = requests.post(
                f"{integration_server_manager.get_base_url()}/query",
                json={"query": "   "},
                timeout=5,
            )

            assert response.status_code == 400
            assert "Query cannot be empty" in response.json()["detail"]
        finally:
            integration_server_manager.stop()


@pytest.mark.integration
@pytest.mark.slow
class TestAPIEndpointPerformance:
    """Integration tests for performance characteristics."""

    def test_server_startup_time(
        self, integration_server_manager: ServerManager
    ) -> None:
        """Test server starts up within reasonable time."""
        start_time = time.time()
        integration_server_manager.start()

        try:
            # Wait for health check to succeed
            max_wait = 5
            while time.time() - start_time < max_wait:
                try:
                    response = requests.get(
                        f"{integration_server_manager.get_base_url()}/health",
                        timeout=1,
                    )
                    if response.status_code == 200:
                        break
                except requests.exceptions.RequestException:
                    time.sleep(0.1)

            startup_time = time.time() - start_time

            # Server should start within 5 seconds
            assert startup_time < 5
        finally:
            integration_server_manager.stop()
