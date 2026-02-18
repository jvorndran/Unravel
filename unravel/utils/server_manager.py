"""Server manager for running FastAPI server in background."""

import threading
import time
from typing import Any

import uvicorn


class ServerManager:
    """Manages the FastAPI server lifecycle in a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Initialize server manager.

        Args:
            host: Host address to bind to
            port: Port number to listen on
        """
        self.host = host
        self.port = port
        self.server: uvicorn.Server | None = None
        self.thread: threading.Thread | None = None
        self.is_running = False

    def start(self) -> None:
        """Start the FastAPI server in a background thread."""
        if self.is_running:
            return

        from unravel.services.api_server import create_app

        app = create_app()

        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False,
        )

        self.server = uvicorn.Server(config)

        def run_server() -> None:
            """Run server in thread."""
            self.server.run()

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        self.is_running = True

        # Wait a moment for server to start
        time.sleep(1)

    def stop(self) -> None:
        """Stop the FastAPI server."""
        if not self.is_running or not self.server:
            return

        self.server.should_exit = True
        self.is_running = False

        # Wait for thread to finish (with timeout)
        if self.thread:
            self.thread.join(timeout=2)

    def get_base_url(self) -> str:
        """Get the base server URL.

        Returns:
            Full base URL of the running server
        """
        return f"http://{self.host}:{self.port}"

    def get_query_url(self) -> str:
        """Get the query endpoint URL.

        Returns:
            Full URL of the query endpoint
        """
        return f"{self.get_base_url()}/query"
