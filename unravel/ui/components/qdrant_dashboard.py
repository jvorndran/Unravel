"""Qdrant dashboard UI component."""

from __future__ import annotations

import streamlit as st


def render_qdrant_dashboard() -> None:
    """Render the Qdrant dashboard iframe."""
    st.write("")
    st.markdown("#### Qdrant Database Viewer")
    st.caption("Inspect collections, points, and payloads directly in Qdrant.")

    qdrant_url = st.session_state.get("qdrant_url")
    if not qdrant_url:
        st.info("Start the Qdrant server in the status panel above to view the dashboard.")
        return

    dashboard_url = f"{qdrant_url}/dashboard"
    st.components.v1.iframe(dashboard_url, height=850, scrolling=True)
    st.markdown(f"[Open dashboard in a new tab]({dashboard_url})")
