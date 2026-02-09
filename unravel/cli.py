"""CLI entry point for Unravel."""

import sys
from pathlib import Path

import click


@click.command()
@click.option(
    "--port",
    "-p",
    default=8501,
    type=int,
    help="Port to run the Streamlit app on.",
)
@click.option(
    "--host",
    "-h",
    default="localhost",
    type=str,
    help="Host to bind the Streamlit app to.",
)
@click.version_option(package_name="unravel")
def main(port: int, host: str) -> None:
    """Launch the Unravel Streamlit application.

    A visual sandbox for experimenting with RAG configurations.
    Upload documents, configure chunking strategies, visualize embeddings,
    and test queries interactively.
    """
    # Import here to avoid slow startup for --help
    from streamlit.web import cli as stcli

    from unravel.services.storage import ensure_storage_dir

    # Ensure local storage directory exists
    storage_path = ensure_storage_dir()
    click.echo(f"Storage directory: {storage_path}")

    # Get the path to the main Streamlit app
    app_path = Path(__file__).parent / "app.py"

    click.echo(f"Starting Unravel on http://{host}:{port}")
    click.echo("Press Ctrl+C to stop the server.\n")

    # Build streamlit arguments
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--browser.gatherUsageStats",
        "false",
    ]

    # Launch Streamlit
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()

