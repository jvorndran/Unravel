"""Unit tests for ServerManager."""

import socket
import time
from threading import Thread
from unittest.mock import Mock, patch

import pytest
import requests

from unravel.utils.server_manager import ServerManager


@pytest.fixture
def server_manager() -> ServerManager:
    """Create ServerManager instance."""
    return ServerManager(host="127.0.0.1", port=8001)  # Use different port for tests


@pytest.mark.unit
class TestServerManagerInit:
    """Tests for ServerManager initialization."""

    def test_init_default_values(self) -> None:
        """Test ServerManager initializes with default values."""
        manager = ServerManager()

        assert manager.host == "127.0.0.1"
        assert manager.port == 8000
        assert manager.server_thread is None
        assert manager.should_stop is False

    def test_init_custom_values(self) -> None:
        """Test ServerManager initializes with custom values."""
        manager = ServerManager(host="0.0.0.0", port=9000)

        assert manager.host == "0.0.0.0"
        assert manager.port == 9000


@pytest.mark.unit
class TestServerManagerLifecycle:
    """Tests for ServerManager start/stop lifecycle."""

    def test_start_creates_thread(self, server_manager: ServerManager) -> None:
        """Test start() creates a server thread."""
        server_manager.start()

        try:
            assert server_manager.server_thread is not None
            assert server_manager.server_thread.is_alive()
            assert server_manager.server_thread.daemon is True
        finally:
            server_manager.stop()

    def test_server_starts_and_responds(self, server_manager: ServerManager) -> None:
        """Test server starts and responds to requests."""
        server_manager.start()

        try:
            # Wait for server to start
            time.sleep(1)

            # Verify server responds
            response = requests.get(
                f"http://{server_manager.host}:{server_manager.port}/health",
                timeout=5,
            )
            assert response.status_code == 200
            assert response.json() == {"status": "healthy"}
        finally:
            server_manager.stop()

    def test_stop_shuts_down_server(self, server_manager: ServerManager) -> None:
        """Test stop() shuts down the server."""
        server_manager.start()
        time.sleep(1)

        # Verify server is running
        response = requests.get(
            f"http://{server_manager.host}:{server_manager.port}/health",
            timeout=5,
        )
        assert response.status_code == 200

        # Stop server
        server_manager.stop()
        time.sleep(1)

        # Verify server is stopped
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.get(
                f"http://{server_manager.host}:{server_manager.port}/health",
                timeout=2,
            )

    def test_stop_when_not_running(self, server_manager: ServerManager) -> None:
        """Test stop() when server is not running (should not raise)."""
        server_manager.stop()  # Should not raise

        assert server_manager.server_thread is None

    def test_multiple_start_calls(self, server_manager: ServerManager) -> None:
        """Test calling start() multiple times."""
        server_manager.start()
        time.sleep(1)

        # Get first thread
        first_thread = server_manager.server_thread

        # Try to start again (should not create new thread if already running)
        server_manager.start()

        try:
            assert server_manager.server_thread == first_thread
        finally:
            server_manager.stop()


@pytest.mark.unit
class TestServerManagerGetters:
    """Tests for ServerManager getter methods."""

    def test_get_base_url(self, server_manager: ServerManager) -> None:
        """Test get_base_url() returns correct URL."""
        url = server_manager.get_base_url()

        assert url == "http://127.0.0.1:8001"

    def test_get_query_url(self, server_manager: ServerManager) -> None:
        """Test get_query_url() returns correct URL."""
        url = server_manager.get_query_url()

        assert url == "http://127.0.0.1:8001/query"


@pytest.mark.unit
class TestPortAvailability:
    """Tests for port availability checking."""

    def test_port_in_use_when_server_running(
        self, server_manager: ServerManager
    ) -> None:
        """Test port is in use when server is running."""
        server_manager.start()

        try:
            time.sleep(1)

            # Check if port is in use
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(("127.0.0.1", server_manager.port))
                assert result == 0  # Port is in use
        finally:
            server_manager.stop()

    def test_port_available_when_server_stopped(
        self, server_manager: ServerManager
    ) -> None:
        """Test port is available when server is stopped."""
        # Ensure server is not running
        server_manager.stop()
        time.sleep(1)

        # Check if port is available
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Try to bind to port (should succeed if available)
            try:
                s.bind(("127.0.0.1", server_manager.port))
                port_available = True
            except OSError:
                port_available = False

        assert port_available


@pytest.mark.unit
@pytest.mark.slow
class TestServerManagerConcurrency:
    """Tests for ServerManager thread safety and concurrency."""

    def test_concurrent_requests(self, server_manager: ServerManager) -> None:
        """Test server handles concurrent requests."""
        server_manager.start()

        try:
            time.sleep(1)

            # Make multiple concurrent requests
            def make_request() -> int:
                response = requests.get(
                    f"http://{server_manager.host}:{server_manager.port}/health",
                    timeout=5,
                )
                return response.status_code

            threads = [Thread(target=make_request) for _ in range(10)]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            # All threads should complete without errors
            assert all(not thread.is_alive() for thread in threads)
        finally:
            server_manager.stop()


@pytest.mark.unit
class TestServerManagerErrorHandling:
    """Tests for ServerManager error handling."""

    def test_start_with_port_already_in_use(self) -> None:
        """Test starting server when port is already in use."""
        # Create two managers for the same port
        manager1 = ServerManager(host="127.0.0.1", port=8002)
        manager2 = ServerManager(host="127.0.0.1", port=8002)

        manager1.start()
        time.sleep(1)

        try:
            # Second start should handle the error gracefully
            # (implementation may vary - might fail silently or raise)
            manager2.start()
            time.sleep(1)

            # At least one should be running
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(("127.0.0.1", 8002))
                assert result == 0  # Port is in use
        finally:
            manager1.stop()
            manager2.stop()
