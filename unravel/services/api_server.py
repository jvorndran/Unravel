"""FastAPI server for exposing RAG pipeline as an API endpoint."""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from unravel.services.embedders import get_embedder
from unravel.services.llm import LLMConfig, RAGContext, get_model
from unravel.services.retrieval import retrieve


class QueryRequest(BaseModel):
    """Request model for RAG query."""

    query: str


class PipelineState:
    """Shared state for the pipeline components."""

    def __init__(self) -> None:
        self.vector_store: Any = None
        self.embedder: Any = None
        self.llm_config: LLMConfig | None = None
        self.system_prompt: str = ""
        self.retrieval_config: dict[str, Any] = {}
        self.reranking_config: dict[str, Any] = {}
        self.bm25_index_data: Any = None
        self.top_k: int = 5
        self.threshold: float = 0.3


# Global state instance
pipeline_state = PipelineState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context for FastAPI app."""
    # Startup
    yield
    # Shutdown cleanup if needed


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Unravel RAG API",
        description="API endpoint for RAG pipeline query and retrieval",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/status")
    async def get_status() -> dict[str, Any]:
        """Get pipeline status and configuration."""
        return {
            "pipeline_ready": pipeline_state.vector_store is not None,
            "vector_store_size": (
                pipeline_state.vector_store.size if pipeline_state.vector_store else 0
            ),
            "embedder_model": (
                pipeline_state.embedder.model_name if pipeline_state.embedder else None
            ),
            "llm_provider": (
                pipeline_state.llm_config.provider if pipeline_state.llm_config else None
            ),
            "llm_model": pipeline_state.llm_config.model if pipeline_state.llm_config else None,
            "retrieval_strategy": pipeline_state.retrieval_config.get("strategy", "Unknown"),
        }

    @app.post("/query")
    async def query_pipeline(request: QueryRequest) -> StreamingResponse:
        """Execute RAG pipeline and stream response."""
        # Validate pipeline is configured
        if not pipeline_state.vector_store or not pipeline_state.embedder:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        if not pipeline_state.llm_config:
            raise HTTPException(status_code=503, detail="LLM not configured")

        query_text = request.query.strip()
        if not query_text:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        async def generate_stream() -> AsyncIterator[str]:
            """Generate Server-Sent Events stream."""
            try:
                # Step 1: Retrieve chunks
                yield f"data: {json.dumps({'type': 'status', 'message': 'Retrieving context...'})}\n\n"

                # Build retrieval params
                params = pipeline_state.retrieval_config.get("params", {}).copy()
                retrieval_strategy = pipeline_state.retrieval_config.get(
                    "strategy", "DenseRetriever"
                )

                # Add BM25 data if needed for sparse/hybrid retrieval
                if retrieval_strategy in ["SparseRetriever", "HybridRetriever"]:
                    if not pipeline_state.bm25_index_data:
                        # Build BM25 index on-demand
                        try:
                            from unravel.services.retrieval import preprocess_retriever

                            yield f"data: {json.dumps({'type': 'status', 'message': 'Building BM25 index...'})}\n\n"
                            bm25_data = preprocess_retriever(
                                "SparseRetriever",
                                pipeline_state.vector_store,
                                **params,
                            )
                            pipeline_state.bm25_index_data = bm25_data
                            params["bm25_index_data"] = bm25_data
                        except Exception as e:
                            # Fall back to dense retrieval if BM25 fails
                            yield f"data: {json.dumps({'type': 'status', 'message': 'BM25 index failed, using dense retrieval'})}\n\n"
                            retrieval_strategy = "DenseRetriever"
                            params = {}
                    else:
                        params["bm25_index_data"] = pipeline_state.bm25_index_data

                # Retrieve (use top_k from pipeline state, not request)
                results = retrieve(
                    query=query_text,
                    vector_store=pipeline_state.vector_store,
                    embedder=pipeline_state.embedder,
                    retriever_name=retrieval_strategy,
                    k=pipeline_state.top_k,
                    **params,
                )

                # Apply threshold (use threshold from pipeline state, not request)
                results = [r for r in results if r.score >= pipeline_state.threshold]
                results = results[: pipeline_state.top_k]

                if not results:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant chunks found'})}\n\n"
                    return

                # Apply reranking if enabled
                if pipeline_state.reranking_config.get("enabled", False) and results:
                    try:
                        from unravel.services.retrieval.reranking import (
                            RerankerConfig,
                            rerank_results,
                        )

                        yield f"data: {json.dumps({'type': 'status', 'message': 'Reranking results...'})}\n\n"

                        rerank_cfg = RerankerConfig(
                            enabled=True,
                            model=pipeline_state.reranking_config.get(
                                "model", "ms-marco-MiniLM-L-12-v2"
                            ),
                            top_n=pipeline_state.reranking_config.get("top_n", 5),
                        )
                        results = rerank_results(query_text, results, rerank_cfg)
                    except ImportError:
                        yield f"data: {json.dumps({'type': 'status', 'message': 'FlashRank not installed, skipping reranking'})}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'status', 'message': f'Reranking failed: {str(e)}'})}\n\n"

                # Send retrieval results
                chunks_data = [
                    {
                        "text": r.text,
                        "score": r.score,
                        "metadata": r.metadata,
                    }
                    for r in results
                ]
                yield f"data: {json.dumps({'type': 'chunks', 'data': chunks_data})}\n\n"

                # Step 2: Generate response
                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"

                context = RAGContext(
                    query=query_text,
                    chunks=[r.text for r in results],
                    scores=[r.score for r in results],
                )

                model = get_model(
                    provider=pipeline_state.llm_config.provider,
                    model=pipeline_state.llm_config.model,
                    api_key=pipeline_state.llm_config.api_key,
                    base_url=pipeline_state.llm_config.base_url,
                    temperature=pipeline_state.llm_config.temperature,
                )

                # Stream LLM response
                for chunk in model.stream(context, pipeline_state.system_prompt):
                    yield f"data: {json.dumps({'type': 'text', 'chunk': chunk})}\n\n"
                    # Allow event loop to process other tasks
                    await asyncio.sleep(0)

                # Send completion signal
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                error_msg = str(e)
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def update_pipeline_state(
    vector_store: Any = None,
    embedder: Any = None,
    llm_config: LLMConfig | None = None,
    system_prompt: str = "",
    retrieval_config: dict[str, Any] | None = None,
    reranking_config: dict[str, Any] | None = None,
    bm25_index_data: Any = None,
    top_k: int | None = None,
    threshold: float | None = None,
) -> None:
    """Update the pipeline state with new components.

    Args:
        vector_store: Vector store instance
        embedder: Embedder instance
        llm_config: LLM configuration
        system_prompt: System prompt for the LLM
        retrieval_config: Retrieval strategy configuration
        reranking_config: Reranking configuration
        bm25_index_data: BM25 index data for sparse retrieval
        top_k: Number of chunks to retrieve
        threshold: Minimum similarity score threshold
    """
    if vector_store is not None:
        pipeline_state.vector_store = vector_store
    if embedder is not None:
        pipeline_state.embedder = embedder
    if llm_config is not None:
        pipeline_state.llm_config = llm_config
    if system_prompt:
        pipeline_state.system_prompt = system_prompt
    if retrieval_config is not None:
        pipeline_state.retrieval_config = retrieval_config
    if reranking_config is not None:
        pipeline_state.reranking_config = reranking_config
    if bm25_index_data is not None:
        pipeline_state.bm25_index_data = bm25_index_data
    if top_k is not None:
        pipeline_state.top_k = top_k
    if threshold is not None:
        pipeline_state.threshold = threshold
