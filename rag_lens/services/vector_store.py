"""
Qdrant vector store for RAG-Visualizer.

Provides a wrapper around Qdrant for storing and searching embeddings.
"""

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.http import models


_PATH_CLIENTS: dict[Path, QdrantClient] = {}
_URL_CLIENTS: dict[str, QdrantClient] = {}


def _get_client(storage_path: Path | None = None, url: str | None = None) -> QdrantClient:
    if url:
        client = _URL_CLIENTS.get(url)
        if client is None:
            client = QdrantClient(url=url)
            _URL_CLIENTS[url] = client
        return client

    if storage_path is None:
        raise ValueError("storage_path is required when url is not provided.")

    storage_path = storage_path.resolve()
    client = _PATH_CLIENTS.get(storage_path)
    if client is None:
        client = QdrantClient(path=str(storage_path))
        _PATH_CLIENTS[storage_path] = client
    return client


@dataclass
class SearchResult:
    """Result from a vector similarity search."""
    index: int
    score: float
    text: str
    metadata: dict[str, Any]


class VectorStore:
    """
    Qdrant-based vector store for embedding storage and retrieval.
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        storage_path: Path | None = None,
        collection_name: str = "rag_lens_chunks",
        url: str | None = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            dimension: Embedding dimension
            metric: Distance metric ('cosine', 'l2', or 'ip' for inner product)
            storage_path: Optional path for Qdrant local storage
            collection_name: Qdrant collection name
        """
        self.dimension = dimension
        self.metric = metric
        self.collection_name = collection_name
        self._url = url
        self._storage_path = (
            Path(storage_path)
            if storage_path is not None
            else Path(tempfile.mkdtemp(prefix="rag_lens_qdrant_"))
        )
        self._client = _get_client(storage_path=self._storage_path, url=url)
        self._ensure_collection()

        self._texts: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._cache_loaded = False
        self._next_id = 0
        self._size: int | None = None

    def _distance(self) -> models.Distance:
        if self.metric == "cosine":
            return models.Distance.COSINE
        if self.metric == "l2":
            return models.Distance.EUCLID
        if self.metric == "ip":
            return models.Distance.DOT
        raise ValueError(
            f"Unknown metric: {self.metric}. Use 'cosine', 'l2', or 'ip'."
        )

    def _ensure_collection(self) -> None:
        collections = self._client.get_collections().collections
        if any(c.name == self.collection_name for c in collections):
            return
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimension,
                distance=self._distance(),
            ),
        )

    def _ensure_cache_loaded(self) -> None:
        if self._cache_loaded:
            return
        self._load_payload_cache()

    def _load_payload_cache(self) -> None:
        points: list[tuple[int, dict[str, Any]]] = []
        next_offset: int | None = None
        while True:
            batch, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                offset=next_offset,
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            for point in batch:
                payload = point.payload or {}
                point_id = int(point.id)
                points.append((point_id, payload))
            if next_offset is None:
                break

        points.sort(key=lambda item: item[0])
        self._texts = [p[1].get("text", "") for p in points]
        self._metadata = [
            cast(dict[str, Any], p[1].get("metadata", {})) for p in points
        ]
        self._next_id = (points[-1][0] + 1) if points else 0
        self._size = len(points)
        self._cache_loaded = True

    @property
    def size(self) -> int:
        """Return the number of vectors in the store."""
        if self._size is None:
            count = self._client.count(
                collection_name=self.collection_name, exact=True
            )
            self._size = int(count.count)
        return self._size

    @property
    def storage_path(self) -> Path:
        return self._storage_path

    @property
    def url(self) -> str | None:
        return self._url

    def add(
        self,
        embeddings: NDArray[Any],
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add embeddings to the vector store.

        Args:
            embeddings: numpy array of shape (n, dimension)
            texts: List of corresponding text strings
            metadata: Optional list of metadata dicts for each embedding
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"store dimension {self.dimension}"
            )

        if len(texts) != embeddings.shape[0]:
            raise ValueError(
                f"Number of texts ({len(texts)}) doesn't match "
                f"number of embeddings ({embeddings.shape[0]})"
            )

        # Ensure embeddings are float32 for consistent storage
        embeddings = embeddings.astype(np.float32)

        # For cosine similarity, ensure embeddings are L2-normalized
        # This is defensive - embedders should already normalize, but we enforce it here
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero for zero vectors
            embeddings = embeddings / (norms + 1e-10)

        self._ensure_cache_loaded()
        payloads = metadata if metadata is not None else [{} for _ in texts]

        points: list[models.PointStruct] = []
        for idx, (vector, text, payload) in enumerate(
            zip(embeddings, texts, payloads, strict=True)
        ):
            point_id = self._next_id + idx
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={"text": text, "metadata": payload},
                )
            )

        self._client.upsert(collection_name=self.collection_name, points=points)
        self._next_id += len(points)
        self._size = (self._size or 0) + len(points)
        self._texts.extend(texts)
        self._metadata.extend(payloads)

    def search(
        self,
        query_embedding: NDArray[Any],
        k: int = 5,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            k: Number of results to return

        Returns:
            List of SearchResult objects sorted by similarity (highest first)
        """
        if self.size == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Ensure float32
        query_embedding = query_embedding.astype(np.float32)

        # For cosine similarity, ensure query embedding is L2-normalized
        if self.metric == "cosine":
            norm = np.linalg.norm(query_embedding)
            if norm > 1e-10:  # Avoid division by zero
                query_embedding = query_embedding / norm

        k = min(k, self.size)

        results: list[SearchResult] = []
        if hasattr(self._client, "query_points"):
            points = self._client.query_points(
                collection_name=self.collection_name,
                query=query_embedding[0].tolist(),
                limit=k,
                with_payload=True,
            ).points
        elif hasattr(self._client, "search"):
            points = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding[0].tolist(),
                limit=k,
                with_payload=True,
            )
        else:
            raise AttributeError("Qdrant client has no supported search method.")
        for point in points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    index=int(point.id),
                    score=float(point.score),
                    text=cast(str, payload.get("text", "")),
                    metadata=cast(dict[str, Any], payload.get("metadata", {})),
                )
            )

        return results

    def get_all_embeddings(self) -> NDArray[np.float32]:
        """Retrieve all embeddings from the store.

        Returns:
            numpy array of shape (n, dimension)
        """
        if self.size == 0:
            return np.array([]).reshape(0, self.dimension)

        vectors: list[tuple[int, list[float]]] = []
        next_offset: int | None = None
        while True:
            batch, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                offset=next_offset,
                limit=256,
                with_payload=False,
                with_vectors=True,
            )
            for point in batch:
                vector = cast(list[float], point.vector)
                vectors.append((int(point.id), vector))
            if next_offset is None:
                break

        vectors.sort(key=lambda item: item[0])
        data = np.array([v for _, v in vectors], dtype=np.float32)
        return cast(NDArray[np.float32], data)

    def get_texts(self) -> list[str]:
        """Get all stored texts."""
        self._ensure_cache_loaded()
        return self._texts.copy()

    def get_metadata(self) -> list[dict[str, Any]]:
        """Get all stored metadata."""
        self._ensure_cache_loaded()
        return self._metadata.copy()

    def clear(self) -> None:
        """Clear all data from the store."""
        self._client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
        self._texts = []
        self._metadata = []
        self._cache_loaded = True
        self._next_id = 0
        self._size = 0

    def save(self, path: Path) -> None:
        """Save the vector store to disk.

        Args:
            path: Directory path to save to (will be created if needed)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if path.resolve() != self._storage_path.resolve():
            shutil.copytree(self._storage_path, path, dirs_exist_ok=True)

        metadata = {
            "dimension": self.dimension,
            "metric": self.metric,
            "size": self.size,
            "collection_name": self.collection_name,
        }
        with (path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path, url: str | None = None) -> "VectorStore":
        """Load a vector store from disk.

        Args:
            path: Directory path to load from

        Returns:
            Loaded VectorStore instance
        """
        path = Path(path)

        if not (path / "metadata.json").exists():
            raise FileNotFoundError(f"Missing metadata.json in {path}")
        with (path / "metadata.json").open(encoding="utf-8") as f:
            metadata = cast(dict[str, Any], json.load(f))  # noqa: S301

        store = cls(
            dimension=metadata["dimension"],
            metric=metadata["metric"],
            storage_path=path,
            collection_name=metadata.get("collection_name", "rag_lens_chunks"),
            url=url,
        )
        store._load_payload_cache()
        return store


def _resolve_qdrant_url(url: str | None) -> str | None:
    if url:
        return url
    try:
        import streamlit as st
    except ModuleNotFoundError:
        return None
    return st.session_state.get("qdrant_url")


def create_vector_store(
    dimension: int,
    metric: str = "cosine",
    storage_path: Path | None = None,
    url: str | None = None,
) -> VectorStore:
    """Factory function to create a vector store.

    Args:
        dimension: Embedding dimension
        metric: Distance metric ('cosine', 'l2', or 'ip')
        storage_path: Optional path for Qdrant local storage
        url: Optional Qdrant server URL

    Returns:
        New VectorStore instance

    Raises:
        RuntimeError: If Qdrant server is not available
    """
    resolved_url = _resolve_qdrant_url(url)
    if not resolved_url:
        raise RuntimeError(
            "Qdrant server is not available. Docker must be running to use "
            "embeddings functionality. Please start Docker Desktop and restart the app."
        )

    from rag_lens.services.storage import ensure_storage_dir, get_indices_dir

    ensure_storage_dir()
    storage_path = get_indices_dir()

    return VectorStore(
        dimension=dimension,
        metric=metric,
        storage_path=storage_path,
        url=resolved_url,
    )

