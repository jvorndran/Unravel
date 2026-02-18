"""API Endpoint UI component for Query step."""

from typing import Any

import streamlit as st
import streamlit_shadcn_ui as ui

from unravel.ui.constants import WidgetKeys


def render_api_endpoint_expander(
    vector_store: Any,
    embedder: Any,
    query_system_prompt: str,
) -> None:
    """Render API endpoint configuration as an expander.

    Args:
        vector_store: Vector store instance
        embedder: Embedder instance
        query_system_prompt: System prompt for queries
    """
    with st.expander("API Endpoint", expanded=False):
        # Always show toggle and URL
        col_toggle, col_url = st.columns([1, 3])

        with col_toggle:
            api_enabled = ui.switch(
                label="Enable Server",
                default_checked=st.session_state.api_endpoint_enabled,
                key=WidgetKeys.QUERY_API_ENDPOINT_TOGGLE,
            )

        with col_url:
            if st.session_state.api_endpoint_enabled:
                # Show URL even if manager is None (e.g., after refresh)
                server_url = "http://127.0.0.1:8000/query"
                st.code(server_url, language="")
            else:
                st.caption("Server stopped")

        # Handle server start/stop
        if api_enabled != st.session_state.api_endpoint_enabled:
            st.session_state.api_endpoint_enabled = api_enabled

            if api_enabled:
                # Start server
                from unravel.utils.server_manager import ServerManager

                if not st.session_state.api_server_manager:
                    st.session_state.api_server_manager = ServerManager(
                        host="127.0.0.1", port=8000
                    )

                with st.spinner("Starting API server..."):
                    try:
                        st.session_state.api_server_manager.start()

                        # Update pipeline state
                        from unravel.services.api_server import update_pipeline_state
                        from unravel.ui.steps.query import _get_llm_config_from_sidebar

                        llm_config, _ = _get_llm_config_from_sidebar()
                        retrieval_config = st.session_state.get(
                            "retrieval_config",
                            {"strategy": "DenseRetriever", "params": {}},
                        )
                        reranking_config = st.session_state.get(
                            "reranking_config",
                            {"enabled": False},
                        )
                        bm25_data = st.session_state.get("bm25_index_data")

                        # Build BM25 index if needed for sparse/hybrid retrieval
                        retrieval_strategy = retrieval_config.get(
                            "strategy", "DenseRetriever"
                        )
                        if retrieval_strategy in [
                            "SparseRetriever",
                            "HybridRetriever",
                        ] and not bm25_data:
                            with st.spinner("Building BM25 index for API..."):
                                try:
                                    from unravel.services.retrieval import (
                                        preprocess_retriever,
                                    )

                                    params = retrieval_config.get("params", {})
                                    bm25_data = preprocess_retriever(
                                        "SparseRetriever",
                                        vector_store,
                                        **params,
                                    )
                                    st.session_state["bm25_index_data"] = bm25_data
                                except Exception as e:
                                    st.warning(
                                        f"Could not build BM25 index: {e}. API will build it on first request."
                                    )

                        update_pipeline_state(
                            vector_store=vector_store,
                            embedder=embedder,
                            llm_config=llm_config,
                            system_prompt=query_system_prompt,
                            retrieval_config=retrieval_config,
                            reranking_config=reranking_config,
                            bm25_index_data=bm25_data,
                            top_k=st.session_state.get("query_top_k", 5),
                            threshold=st.session_state.get("query_threshold", 0.3),
                        )

                        st.success("API server started!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to start server: {e}")
                        st.session_state.api_endpoint_enabled = False
            else:
                # Stop server
                if st.session_state.api_server_manager:
                    st.session_state.api_server_manager.stop()
                    st.session_state.api_server_manager = None
                st.success("API server stopped!")
                st.rerun()
                
        st.caption("The API uses all settings from this page (retrieval strategy, system prompt, reranking, etc.)")

        with st.expander("API Documentation", expanded=False):
            st.markdown("#### Request")
            st.code(
                """POST http://127.0.0.1:8000/query
Content-Type: application/json

{
  "query": "What is retrieval-augmented generation?"
}""",
                language="http",
            )

            st.markdown("**Parameters**")
            st.markdown("• `query` — Your question (required)")

            st.markdown("**Returns**")
            st.markdown("Server-Sent Events stream with events: `status`, `chunks`, `text`, `done`")

            st.markdown("#### Response")
            st.code(
                """data: {"type": "status", "message": "Retrieving context..."}

data: {"type": "chunks", "data": [
  {"text": "...", "score": 0.85, "metadata": {...}},
  {"text": "...", "score": 0.72, "metadata": {...}}
]}

data: {"type": "status", "message": "Generating response..."}

data: {"type": "text", "chunk": "Retrieval"}
data: {"type": "text", "chunk": "-augmented"}
data: {"type": "text", "chunk": " generation"}
...

data: {"type": "done"}""",
                language="",
            )

            st.markdown("**Event Types**")
            st.markdown("• `status` — Progress updates")
            st.markdown("• `chunks` — Retrieved context with scores")
            st.markdown("• `text` — Streaming LLM response (word-by-word)")
            st.markdown("• `done` — Query complete")
            st.markdown("• `error` — Error message")
