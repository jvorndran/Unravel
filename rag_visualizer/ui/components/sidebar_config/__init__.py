"""Streamlit custom component for sidebar configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_COMPONENT_NAME = "sidebar_config"
_DEV_URL = "http://localhost:3001"


def _get_component() -> Any:
    if os.getenv("STREAMLIT_COMPONENT_DEV"):
        return components.declare_component(_COMPONENT_NAME, url=_DEV_URL)

    build_dir = Path(__file__).parent / "frontend" / "build"
    return components.declare_component(_COMPONENT_NAME, path=str(build_dir))


def render_sidebar_config(
    *,
    docs: list[str],
    current_doc: str | None,
    retrieval_config: dict[str, Any],
    reranking_config: dict[str, Any],
    model_names: list[str],
    current_model: str,
) -> dict[str, Any] | None:
    component = _get_component()
    return component(
        docs=docs,
        current_doc=current_doc,
        retrieval_config=retrieval_config,
        reranking_config=reranking_config,
        model_names=model_names,
        current_model=current_model,
        default=None,
        key="sidebar_config_component",
    )
