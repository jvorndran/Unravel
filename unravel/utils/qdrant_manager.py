"""Qdrant server lifecycle helpers for the local app."""

from __future__ import annotations

import http.client
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request

import streamlit as st

from unravel.services.storage import ensure_storage_dir

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
QDRANT_CONTAINER_NAME = "unravel-qdrant"
QDRANT_VOLUME_NAME = "unravel-qdrant-data"


def get_qdrant_config() -> tuple[str, str | None]:
    """Get Qdrant URL and API key from environment or use defaults.

    Returns:
        Tuple of (url, api_key). API key is None for local Docker instance.
    """
    # Check for cloud Qdrant URL (for demo deployments)
    cloud_url = os.getenv("QDRANT_URL")
    if cloud_url:
        api_key = os.getenv("QDRANT_API_KEY")
        return (cloud_url, api_key)

    # Default to local Docker
    return (QDRANT_URL, None)


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _container_status() -> str | None:
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name={QDRANT_CONTAINER_NAME}",
                "--format",
                "{{.Status}}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        status = result.stdout.strip()
        return status or None
    except subprocess.TimeoutExpired:
        return None


def _start_or_create_container() -> None:
    status = _container_status()
    if status:
        if status.lower().startswith("up"):
            return
        subprocess.run(
            ["docker", "start", QDRANT_CONTAINER_NAME],
            check=False,
            capture_output=True,
            text=True,
        )
        return

    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            QDRANT_CONTAINER_NAME,
            "-p",
            f"{QDRANT_PORT}:{QDRANT_PORT}",
            "-v",
            f"{QDRANT_VOLUME_NAME}:/qdrant/storage",
            "qdrant/qdrant",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _is_healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/healthz", timeout=1.0) as response:
            return response.status == 200
    except (
        urllib.error.URLError,
        http.client.RemoteDisconnected,
        TimeoutError,
        ValueError,
    ):
        return False


def _inspect_mount_type() -> str | None:
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{range .Mounts}}{{.Type}}{{end}}",
                QDRANT_CONTAINER_NAME,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        mount_type = result.stdout.strip()
        return mount_type or None
    except subprocess.TimeoutExpired:
        return None


def get_qdrant_status() -> dict[str, str | bool | None]:
    status = _container_status()
    running = bool(status and status.lower().startswith("up"))
    return {
        "running": running,
        "status": status,
        "url": QDRANT_URL if _is_port_open(QDRANT_HOST, QDRANT_PORT) else None,
        "docker_available": _docker_available(),
        "mount_type": _inspect_mount_type(),
        "error": st.session_state.get("qdrant_start_error"),
    }


def restart_qdrant_server() -> tuple[str, str | None]:
    """Restart the Qdrant server container.

    Returns:
        Tuple of (url, api_key) after restart

    Raises:
        RuntimeError: If Docker is unavailable or restart fails
    """
    if not _docker_available():
        error_msg = (
            "Docker is not available. Install Docker Desktop and make sure "
            "it is running to start the local Qdrant server."
        )
        st.session_state["qdrant_start_error"] = error_msg
        raise RuntimeError(error_msg)

    subprocess.run(
        ["docker", "restart", QDRANT_CONTAINER_NAME],
        check=False,
        capture_output=True,
        text=True,
    )
    ensure_qdrant_server.clear()
    return ensure_qdrant_server()


@st.cache_resource(show_spinner=False)
def ensure_qdrant_server() -> tuple[str, str | None]:
    """Ensure Qdrant server is running and return its URL and API key.

    Checks for cloud Qdrant first (via QDRANT_URL env var), then falls back
    to local Docker instance.

    Returns:
        Tuple of (url, api_key). API key is None for local Docker instance.

    Raises:
        RuntimeError: If Docker is unavailable or server fails to start (local only)
    """
    # Check for cloud Qdrant URL (for demo deployments)
    cloud_url = os.getenv("QDRANT_URL")
    if cloud_url:
        api_key = os.getenv("QDRANT_API_KEY")
        st.session_state.pop("qdrant_start_error", None)
        st.session_state["qdrant_startup_status"] = "cloud"
        st.session_state["qdrant_api_key"] = api_key
        return (cloud_url, api_key)

    # Local Docker mode
    # Check if already running
    if _is_port_open(QDRANT_HOST, QDRANT_PORT):
        st.session_state.pop("qdrant_start_error", None)
        st.session_state["qdrant_startup_status"] = "running"
        st.session_state["qdrant_api_key"] = None
        return (QDRANT_URL, None)

    # Check Docker availability
    if not _docker_available():
        error_msg = (
            "Docker is not available. Install Docker Desktop and make sure "
            "the Docker daemon is running to use the Qdrant server."
        )
        st.session_state["qdrant_start_error"] = error_msg
        st.session_state["qdrant_startup_status"] = "docker_unavailable"
        raise RuntimeError(error_msg)

    # Try to start/create container
    ensure_storage_dir()
    st.session_state["qdrant_startup_status"] = "starting"

    try:
        _start_or_create_container()
    except Exception as e:
        error_msg = f"Failed to start Qdrant container: {e}"
        st.session_state["qdrant_start_error"] = error_msg
        st.session_state["qdrant_startup_status"] = "start_failed"
        raise RuntimeError(error_msg) from e

    # Wait for server to become healthy
    for i in range(20):
        if _is_port_open(QDRANT_HOST, QDRANT_PORT) and _is_healthy(QDRANT_URL):
            st.session_state.pop("qdrant_start_error", None)
            st.session_state["qdrant_startup_status"] = "running"
            st.session_state["qdrant_api_key"] = None
            return (QDRANT_URL, None)
        time.sleep(0.5)

    # Timeout - server didn't become healthy
    error_msg = (
        "Qdrant container started but did not become healthy within 10 seconds. "
        "Ensure Docker is running properly and port 6333 is free, then reload."
    )
    st.session_state["qdrant_start_error"] = error_msg
    st.session_state["qdrant_startup_status"] = "unhealthy"

    # If port is at least open, return URL as it may still be usable
    if _is_port_open(QDRANT_HOST, QDRANT_PORT):
        st.session_state["qdrant_api_key"] = None
        return (QDRANT_URL, None)
    raise RuntimeError(error_msg)
