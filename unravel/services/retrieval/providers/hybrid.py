"""Hybrid retriever combining dense and sparse search."""

from typing import TYPE_CHECKING, Any

from .base import RetrieverInfo, RetrieverProvider
from .dense import DenseRetriever
from .sparse import SparseRetriever

if TYPE_CHECKING:
    from unravel.services.embedders import Embedder
    from unravel.services.vector_store import SearchResult, VectorStore


class HybridRetriever(RetrieverProvider):
    """Hybrid retrieval combining dense vector search and sparse BM25."""

    @property
    def name(self) -> str:
        return "HybridRetriever"

    def get_available_retrievers(self) -> list[RetrieverInfo]:
        return [
            RetrieverInfo(
                name="HybridRetriever",
                display_name="Hybrid (Dense + BM25)",
                description="Combines vector similarity and keyword search",
                category="Retrieval",
            )
        ]

    def search(
        self,
        retriever_name: str,
        query: str,
        k: int,
        vector_store: "VectorStore",
        embedder: "Embedder",
        **params: Any,
    ) -> list["SearchResult"]:
        """Search using both dense and sparse retrieval, then fuse results."""
        from unravel.services.vector_store import SearchResult

        dense_weight = params.get("dense_weight", 0.7)
        fusion_method = params.get("fusion_method", "weighted_sum")

        # Get dense results (retrieve more for fusion)
        dense_provider = DenseRetriever()
        dense_results = dense_provider.search(
            "DenseRetriever", query, k * 2, vector_store, embedder
        )

        # Get sparse results
        sparse_provider = SparseRetriever()
        sparse_results = sparse_provider.search(
            "SparseRetriever", query, k * 2, vector_store, embedder, **params
        )

        # Normalize and combine scores
        if fusion_method == "weighted_sum":
            return self._weighted_sum_fusion(
                dense_results, sparse_results, dense_weight, k
            )
        else:  # RRF
            return self._rrf_fusion(dense_results, sparse_results, k)

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """MinMax normalize scores to 0-1 range.

        When all scores are equal (no variation), returns neutral 0.5 instead
        of misleading 1.0. This prevents giving maximum weight to retrieval
        methods that found no variation (e.g., all-zero BM25 scores when no
        keywords match).
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range < 1e-10:
            # All scores equal - return neutral 0.5 instead of misleading 1.0
            # This prevents giving maximum weight to methods that found no variation
            return [0.5] * len(scores)

        return [(s - min_score) / score_range for s in scores]

    def _weighted_sum_fusion(
        self,
        dense_results: list["SearchResult"],
        sparse_results: list["SearchResult"],
        dense_weight: float,
        k: int,
    ) -> list["SearchResult"]:
        """Combine results using weighted sum of normalized scores."""
        from unravel.services.vector_store import SearchResult

        sparse_weight = 1.0 - dense_weight

        # Create score dictionaries by index
        dense_scores = {r.index: r.score for r in dense_results}
        sparse_scores = {r.index: r.score for r in sparse_results}

        # Normalize scores
        if dense_scores:
            dense_score_list = list(dense_scores.values())
            norm_dense_scores = self._normalize_scores(dense_score_list)
            dense_scores = dict(zip(dense_scores.keys(), norm_dense_scores, strict=True))

        if sparse_scores:
            sparse_score_list = list(sparse_scores.values())
            norm_sparse_scores = self._normalize_scores(sparse_score_list)
            sparse_scores = dict(
                zip(sparse_scores.keys(), norm_sparse_scores, strict=True)
            )

        # Combine scores
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined_scores: dict[int, float] = {}

        for idx in all_indices:
            d_score = dense_scores.get(idx, 0.0)
            s_score = sparse_scores.get(idx, 0.0)
            combined_scores[idx] = dense_weight * d_score + sparse_weight * s_score

        # Sort by combined score and get top k
        sorted_indices = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]

        # Build results (get text from either dense or sparse results)
        result_map = {r.index: r for r in dense_results + sparse_results}
        results = []

        for idx, score in sorted_indices:
            original = result_map[idx]
            d_score = dense_scores.get(idx, 0.0)
            s_score = sparse_scores.get(idx, 0.0)
            
            results.append(
                SearchResult(
                    index=idx,
                    score=score,
                    text=original.text,
                    metadata={
                        **original.metadata,
                        "fusion_method": "weighted_sum",
                        "dense_weight": dense_weight,
                        "dense_score": d_score,
                        "sparse_score": s_score,
                        "dense_weighted_score": dense_weight * d_score,
                        "sparse_weighted_score": sparse_weight * s_score,
                    },
                )
            )

        return results

    def _rrf_fusion(
        self,
        dense_results: list["SearchResult"],
        sparse_results: list["SearchResult"],
        k: int,
    ) -> list["SearchResult"]:
        """Combine results using Reciprocal Rank Fusion."""
        from unravel.services.vector_store import SearchResult

        rrf_k = 60  # Standard RRF constant

        # Build rank dictionaries
        dense_ranks = {r.index: i + 1 for i, r in enumerate(dense_results)}
        sparse_ranks = {r.index: i + 1 for i, r in enumerate(sparse_results)}

        # Calculate RRF scores
        all_indices = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        rrf_scores: dict[int, float] = {}

        for idx in all_indices:
            score = 0.0
            if idx in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[idx])
            if idx in sparse_ranks:
                score += 1.0 / (rrf_k + sparse_ranks[idx])
            rrf_scores[idx] = score

        # Normalize RRF scores to 0-1 range for comparability with cosine similarity
        if rrf_scores:
            rrf_score_list = list(rrf_scores.values())
            normalized_scores = self._normalize_scores(rrf_score_list)
            # Create a map of raw score -> normalized score for metadata if needed
            # But here we just need to update the scores in rrf_scores
            # However, we lost the raw score if we overwrite. 
            # Let's keep a separate map for normalized scores.
            normalized_rrf_scores = dict(zip(rrf_scores.keys(), normalized_scores, strict=True))
        else:
            normalized_rrf_scores = {}

        # Sort by RRF score (normalized or raw, order is same) and get top k
        sorted_indices = sorted(normalized_rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            :k
        ]

        # Build results
        result_map = {r.index: r for r in dense_results + sparse_results}
        results = []

        for idx, score in sorted_indices:
            original = result_map[idx]
            
            # Calculate contribution scores for visualization
            dense_rank = dense_ranks.get(idx)
            sparse_rank = sparse_ranks.get(idx)
            
            dense_rrf_score = (1.0 / (rrf_k + dense_rank)) if dense_rank else 0.0
            sparse_rrf_score = (1.0 / (rrf_k + sparse_rank)) if sparse_rank else 0.0
            raw_total = dense_rrf_score + sparse_rrf_score
            dense_rrf_share = dense_rrf_score / raw_total if raw_total > 0 else 0.0
            sparse_rrf_share = sparse_rrf_score / raw_total if raw_total > 0 else 0.0
            dense_rrf_contribution = score * dense_rrf_share
            sparse_rrf_contribution = score * sparse_rrf_share
            
            results.append(
                SearchResult(
                    index=idx,
                    score=score,
                    text=original.text,
                    metadata={
                        **original.metadata,
                        "fusion_method": "rrf",
                        "rrf_k": rrf_k,
                        "dense_rank": dense_rank,
                        "sparse_rank": sparse_rank,
                        "dense_rrf_score": dense_rrf_score,
                        "sparse_rrf_score": sparse_rrf_score,
                        "dense_rrf_contribution": dense_rrf_contribution,
                        "sparse_rrf_contribution": sparse_rrf_contribution,
                    },
                )
            )

        return results

    def preprocess(
        self,
        retriever_name: str,
        vector_store: "VectorStore",
        **params: Any,
    ) -> dict[str, Any]:
        """Build BM25 index needed for hybrid retrieval."""
        sparse_provider = SparseRetriever()
        return sparse_provider.preprocess("SparseRetriever", vector_store, **params)
