#!/usr/bin/env python3
"""Cleanup script for demo mode Qdrant collections.

Removes old demo collections to prevent storage exhaustion.
Run this manually or schedule it with cron/Task Scheduler.

Usage:
    python scripts/cleanup_demo_collections.py [options]

Examples:
    # Dry run (see what would be deleted)
    python scripts/cleanup_demo_collections.py --dry-run

    # Delete empty collections
    python scripts/cleanup_demo_collections.py

    # Keep only 20 most recent collections
    python scripts/cleanup_demo_collections.py --keep-count 20

    # Use environment variables for credentials
    export QDRANT_URL="https://your-cluster.qdrant.io"
    export QDRANT_API_KEY="your-api-key"
    python scripts/cleanup_demo_collections.py
"""

import argparse
import os
import sys

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    print("Error: qdrant-client is not installed")
    print("Install it with: pip install qdrant-client")
    sys.exit(1)


def cleanup_collections(
    url: str,
    api_key: str | None,
    keep_count: int = 20,
    dry_run: bool = False,
) -> dict:
    """Clean up old demo collections from Qdrant.

    Args:
        url: Qdrant server URL
        api_key: Qdrant API key (optional for local)
        keep_count: Number of most recent collections to keep
        dry_run: If True, show what would be deleted without deleting

    Returns:
        Dictionary with cleanup statistics
    """
    client = QdrantClient(url=url, api_key=api_key)

    stats = {
        "total": 0,
        "deleted": 0,
        "kept": 0,
        "errors": [],
        "deleted_names": [],
        "kept_names": [],
    }

    try:
        # Get all collections
        collections = client.get_collections()

        # Filter demo collections
        demo_collections = [
            col for col in collections.collections if col.name.startswith("unravel_demo_")
        ]

        stats["total"] = len(demo_collections)

        if stats["total"] == 0:
            return stats

        # Sort by name (chronological-ish due to session IDs)
        sorted_collections = sorted(demo_collections, key=lambda c: c.name)

        if stats["total"] <= keep_count:
            # Keep all
            stats["kept"] = stats["total"]
            stats["kept_names"] = [col.name for col in sorted_collections]
        else:
            # Keep the most recent N, delete the rest
            to_delete = sorted_collections[: -keep_count]
            to_keep = sorted_collections[-keep_count :]

            stats["kept"] = len(to_keep)
            stats["kept_names"] = [col.name for col in to_keep]

            for collection in to_delete:
                try:
                    if dry_run:
                        stats["deleted_names"].append(collection.name)
                        stats["deleted"] += 1
                    else:
                        client.delete_collection(collection.name)
                        stats["deleted_names"].append(collection.name)
                        stats["deleted"] += 1
                except (UnexpectedResponse, Exception) as e:
                    stats["errors"].append(f"{collection.name}: {str(e)}")

    except (UnexpectedResponse, Exception) as e:
        stats["errors"].append(f"Connection error: {str(e)}")

    return stats


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Clean up old Unravel demo collections from Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--url",
        type=str,
        help="Qdrant URL (default: QDRANT_URL env var)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="Qdrant API key (default: QDRANT_API_KEY env var)",
    )

    parser.add_argument(
        "--keep-count",
        type=int,
        default=20,
        help="Number of most recent collections to keep (default: 20)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    args = parser.parse_args()

    # Get credentials
    url = args.url or os.getenv("QDRANT_URL")
    api_key = args.api_key or os.getenv("QDRANT_API_KEY")

    if not url:
        print("Error: Qdrant URL required")
        print("Provide via --url or QDRANT_URL environment variable")
        sys.exit(1)

    print(f"Connecting to Qdrant: {url}")
    print(f"Keep count: {args.keep_count}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    # Run cleanup
    stats = cleanup_collections(
        url=url,
        api_key=api_key,
        keep_count=args.keep_count,
        dry_run=args.dry_run,
    )

    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total demo collections: {stats['total']}")
    print(f"Deleted: {stats['deleted']}")
    print(f"Kept: {stats['kept']}")
    print()

    if stats["deleted_names"]:
        print(f"Deleted {len(stats['deleted_names'])} collections:")
        for name in stats["deleted_names"][:10]:  # Show first 10
            print(f"  ✗ {name}")
        if len(stats["deleted_names"]) > 10:
            print(f"  ... and {len(stats['deleted_names']) - 10} more")
        print()

    if stats["kept_names"] and len(stats["kept_names"]) <= 10:
        print(f"Kept {len(stats['kept_names'])} collections:")
        for name in stats["kept_names"]:
            print(f"  ✓ {name}")
        print()

    if stats["errors"]:
        print("Errors:")
        for error in stats["errors"]:
            print(f"  ! {error}")
        print()

    if args.dry_run:
        print("🔍 DRY RUN - No changes made")
        print("Run without --dry-run to actually delete collections")
    else:
        print("✅ Cleanup complete!")


if __name__ == "__main__":
    main()
