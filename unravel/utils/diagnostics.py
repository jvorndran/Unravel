"""Diagnostic utilities for troubleshooting Unravel issues.

Run this module directly to diagnose and fix common issues:
    python -m unravel.utils.diagnostics
"""

import sys

from unravel.services.storage import get_storage_dir
from unravel.utils.qdrant_manager import (
    QDRANT_HOST,
    QDRANT_PORT,
    _docker_available,
    _is_port_open,
    get_qdrant_status,
)

# ASCII-safe status indicators for Windows compatibility
OK = "[OK]"
FAIL = "[X]"
WARN = "[!]"


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_storage() -> None:
    """Check storage directory status."""
    print_section("Storage Status")

    storage_dir = get_storage_dir()
    print(f"Storage directory: {storage_dir}")
    print(f"Exists: {storage_dir.exists()}")

    if storage_dir.exists():
        subdirs = ["documents", "chunks", "embeddings", "indices", "session"]
        for subdir in subdirs:
            path = storage_dir / subdir
            exists = path.exists()
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if exists else 0
            file_count = len(list(path.rglob("*"))) if exists else 0
            status = OK if exists else FAIL
            print(f"  {subdir}/: {status} ({file_count} files, {size:,} bytes)")


def check_docker_requirement() -> None:
    """Check if Docker is available for embeddings."""
    print_section("Docker Requirement")

    docker_ok = _docker_available()
    if docker_ok:
        print(f"{OK} Docker is available - embeddings will work")
    else:
        print(f"{FAIL} Docker is NOT available")
        print("  Embeddings and Query features require Docker Desktop to be installed and running")
        print("  Other features (Upload, Chunks, Export) will still work")


def check_qdrant() -> None:
    """Check Qdrant server status."""
    print_section("Qdrant Server Status")

    # Check Docker
    docker_ok = _docker_available()
    docker_status = OK + " Available" if docker_ok else FAIL + " Not available"
    print(f"Docker: {docker_status}")

    # Check port
    port_open = _is_port_open(QDRANT_HOST, QDRANT_PORT)
    port_status = OK + " Open" if port_open else FAIL + " Closed"
    print(f"Port {QDRANT_PORT}: {port_status}")

    # Check container status
    qdrant_status = get_qdrant_status()
    running = qdrant_status.get("running", False)
    container_status = OK + " Running" if running else FAIL + " Stopped"
    print(f"Container: {container_status}")

    if running:
        url = qdrant_status.get("url")
        print(f"  URL: {url}")
        mount_type = qdrant_status.get("mount_type")
        if mount_type:
            print(f"  Mount type: {mount_type}")

    error = qdrant_status.get("error")
    if error:
        print(f"  Error: {error}")


def check_processes() -> None:
    """Check for running Python/Streamlit processes."""
    print_section("Running Processes")

    import subprocess

    try:
        if sys.platform == "win32":
            # Windows: use tasklist
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                capture_output=True,
                text=True,
                check=False,
            )
            lines = result.stdout.strip().split("\n")
            processes = [line for line in lines if "python.exe" in line.lower()]

            if len(processes) > 1:  # More than just the header
                print(f"Found {len(processes) - 1} Python processes:")
                for line in processes[1:]:  # Skip header
                    parts = line.split('","')
                    if len(parts) >= 2:
                        pid = parts[1].replace('"', "")
                        print(f"  PID: {pid}")
            else:
                print("No Python processes found (besides this one)")
        else:
            # Unix-like: use ps
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                check=False,
            )
            lines = [
                line
                for line in result.stdout.split("\n")
                if "python" in line.lower() or "streamlit" in line.lower()
            ]
            if lines:
                print(f"Found {len(lines)} Python/Streamlit processes:")
                for line in lines[:10]:  # Limit to 10
                    print(f"  {line.strip()}")
            else:
                print("No Python/Streamlit processes found")
    except Exception as e:
        print(f"⚠ Could not check processes: {e}")


def run_diagnostics() -> None:
    """Run all diagnostic checks."""
    print("\n" + "=" * 60)
    print("  Unravel Diagnostics")
    print("=" * 60)

    check_storage()
    check_docker_requirement()
    check_qdrant()
    check_processes()

    print("\n" + "=" * 60)
    print("  Diagnostic Report Complete")
    print("=" * 60 + "\n")


def interactive_fix() -> None:
    """Interactive menu for fixing common issues."""
    while True:
        print("\n" + "=" * 60)
        print("  Fix Issues")
        print("=" * 60)
        print("\n1. Re-run diagnostics")
        print("2. Exit")

        choice = input("\nSelect an option (1-2): ").strip()

        if choice == "1":
            run_diagnostics()

        elif choice == "2":
            print("\nExiting...")
            break

        else:
            print("Invalid choice. Please select 1-2.")


def main() -> None:
    """Main entry point for diagnostic CLI."""
    run_diagnostics()

    # Check if Docker is available
    docker_ok = _docker_available()
    if not docker_ok:
        print(f"\n{WARN} Docker is not available. Embeddings features will not work.")
        print("To fix: Install Docker Desktop and ensure it is running.")
    else:
        print(f"\n{OK} System is ready!")


if __name__ == "__main__":
    main()
