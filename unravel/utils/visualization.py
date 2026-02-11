from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import umap
from numpy.typing import NDArray
from sklearn.cluster import KMeans


@st.cache_resource
def _get_umap_reducer(n_components: int = 2, random_state: int | None = None) -> umap.UMAP:
    """Get or create a cached UMAP reducer."""
    kwargs: dict[str, Any] = {"n_components": n_components}
    if random_state is not None:
        kwargs["random_state"] = random_state
    return umap.UMAP(**kwargs)


@st.cache_data
def reduce_dimensions(
    embeddings: NDArray[Any], n_components: int = 2, random_state: int | None = None
) -> tuple[NDArray[Any], umap.UMAP | None]:
    """Reduce dimensionality of embeddings using UMAP.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        n_components: The dimension of the space to embed into
        random_state: Optional random state for reproducibility

    Returns:
        Tuple containing:
        - numpy array of shape (n_samples, n_components)
        - Fitted UMAP reducer object (or None if fallback used)
    """
    # Handle empty or small inputs
    if embeddings.shape[0] == 0:
        return np.zeros((0, n_components)), None

    if embeddings.shape[0] < 5:
        # Use PCA for small datasets instead of raw embedding dimensions
        # Raw dimensions are essentially random and don't represent semantic relationships
        from sklearn.decomposition import PCA

        n_comp = min(n_components, embeddings.shape[0], embeddings.shape[1])
        pca_kwargs: dict[str, Any] = {"n_components": n_comp}
        if random_state is not None:
            pca_kwargs["random_state"] = random_state
        pca = PCA(**pca_kwargs)
        result = pca.fit_transform(embeddings)
        # Pad with zeros if needed
        if result.shape[1] < n_components:
            padded = np.zeros((result.shape[0], n_components))
            padded[:, : result.shape[1]] = result
            result = padded
        return cast(NDArray[Any], result), None

    reducer = _get_umap_reducer(n_components, random_state)
    embedding_2d = reducer.fit_transform(embeddings)
    return cast(NDArray[Any], embedding_2d), reducer


@st.cache_data
def cluster_embeddings(embeddings: NDArray[Any], n_clusters: int = 5) -> NDArray[Any]:
    """Cluster embeddings using KMeans.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        n_clusters: Number of clusters to find

    Returns:
        numpy array of cluster labels
    """
    if embeddings.shape[0] < n_clusters:
        n_clusters = max(1, embeddings.shape[0])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels


def create_embedding_plot(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str | None = None,
    color_col: str | None = None,
    hover_data: list[str] | None = None,
    title: str = "Embedding Visualization",
    query_point: dict[str, Any] | None = None,
    neighbors_indices: list[int] | None = None,
) -> go.Figure:
    """Create a 2D or 3D scatter plot of embeddings.

    Args:
        df: DataFrame containing the data
        x_col: Name of column for x-axis
        y_col: Name of column for y-axis
        z_col: Name of column for z-axis (optional, triggers 3D plot)
        color_col: Name of column to color by
        hover_data: List of columns to show on hover
        title: Plot title
        query_point: Optional dict with 'x', 'y' (and 'z' if 3D) coordinates for query point
        neighbors_indices: Optional list of indices in df that are neighbors

    Returns:
        Plotly Figure object
    """
    is_3d = z_col is not None
    fig = go.Figure()

    # Marker settings
    marker_settings = dict(
        size=5 if is_3d else 8,
        opacity=0.8,
        line=dict(width=1, color="#ffffff"),  # White outline for better visibility with colors
    )

    # Handle color
    if color_col and color_col in df.columns:
        marker_settings["color"] = df[color_col]
        marker_settings["colorscale"] = "Viridis"  # Good default categorical/sequential map
        marker_settings["showscale"] = False
    else:
        marker_settings["color"] = "#7dd3fc"  # Default Light blue/teal
        marker_settings["line"] = dict(width=1, color="#0ea5e9")

    # 1. Add chunks trace
    if is_3d:
        fig.add_trace(
            go.Scatter3d(
                x=df[x_col],
                y=df[y_col],
                z=df[z_col],
                mode="markers",
                marker=marker_settings,
                text=df["text_preview"] if "text_preview" in df.columns else None,
                customdata=df[hover_data].values if hover_data else None,
                hovertemplate="<b>Chunk</b><br>%{text}<extra></extra>",
                name="Chunks",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=marker_settings,
                text=df["text_preview"] if "text_preview" in df.columns else None,
                customdata=df[hover_data].values if hover_data else None,
                hovertemplate="<b>Chunk</b><br>%{text}<extra></extra>",
                name="Chunks",
            )
        )

    # 2. Add query point if present
    if query_point:
        query_marker = dict(
            size=8 if is_3d else 12,
            opacity=1,
            color="#f43f5e",  # Pink/Red
            symbol="circle",
            line=dict(width=2, color="white"),
        )

        if is_3d and "z" in query_point:
            fig.add_trace(
                go.Scatter3d(
                    x=[query_point["x"]],
                    y=[query_point["y"]],
                    z=[query_point["z"]],
                    mode="markers",
                    marker=query_marker,
                    name="Question",
                    hovertemplate="<b>Question</b><br>Your query<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[query_point["x"]],
                    y=[query_point["y"]],
                    mode="markers",
                    marker=query_marker,
                    name="Question",
                    hovertemplate="<b>Question</b><br>Your query<extra></extra>",
                )
            )

        # 3. Add connecting lines to neighbors
        if neighbors_indices:
            # For 3D lines, we need to add a single trace with None separators
            # or multiple traces. Multiple traces is easier but heavier.
            # Let's use individual traces for now as N is small (typically 5).

            line_style = dict(
                color="#94a3b8",
                width=2,
                dash=(
                    "dash" if not is_3d else None
                ),  # Dash not supported in 3D lines by default same way
            )

            for idx in neighbors_indices:
                neighbor_row = df.iloc[idx]

                if is_3d and "z" in query_point:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[query_point["x"], neighbor_row[x_col]],
                            y=[query_point["y"], neighbor_row[y_col]],
                            z=[query_point["z"], neighbor_row[z_col]],
                            mode="lines",
                            line=line_style,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=[query_point["x"], neighbor_row[x_col]],
                            y=[query_point["y"], neighbor_row[y_col]],
                            mode="lines",
                            line=line_style,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

    # Layout configuration
    layout_args = dict(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
        margin=dict(l=0, r=0, t=40, b=0),  # Tighter margins
        height=600 if is_3d else 500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="sans-serif"),
    )

    if is_3d:
        layout_args["scene"] = dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", backgroundcolor="white"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", backgroundcolor="white"),
            zaxis=dict(showgrid=True, gridcolor="#f1f5f9", backgroundcolor="white"),
            bgcolor="white",
        )
    else:
        layout_args["xaxis_title"] = "X Axis"
        layout_args["yaxis_title"] = "Y Axis"

    fig.update_layout(**layout_args)

    if not is_3d:
        # Add grid for 2D
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9", zeroline=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9", zeroline=False)

    return fig


def calculate_similarity_matrix(embeddings: NDArray[Any]) -> NDArray[Any]:
    """Calculate pairwise cosine similarities between embeddings.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)

    Returns:
        numpy array of shape (n_samples, n_samples) with similarity scores
    """
    if embeddings.shape[0] == 0:
        return cast(NDArray[Any], np.array([]))

    # Ensure embeddings are L2-normalized for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-10)

    # For normalized vectors, cosine similarity = dot product
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    return cast(NDArray[Any], similarity_matrix)


def create_similarity_histogram(
    similarity_matrix: NDArray[Any],
    title: str = "Pairwise Similarity Distribution",
) -> go.Figure:
    """Create a histogram of pairwise similarity scores.

    Args:
        similarity_matrix: Square matrix of pairwise similarities
        title: Plot title

    Returns:
        Plotly Figure object
    """
    # Extract upper triangle (excluding diagonal) to avoid duplicates and self-similarity
    n = similarity_matrix.shape[0]
    if n < 2:
        # Not enough data points for meaningful distribution
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 chunks for similarity analysis",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#666"),
        )
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="white",
        )
        return fig

    # Get upper triangle indices (excluding diagonal)
    upper_triangle_indices = np.triu_indices(n, k=1)
    similarities = similarity_matrix[upper_triangle_indices]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=similarities,
            nbinsx=30,
            marker=dict(color="#7dd3fc", line=dict(width=1, color="#0ea5e9")),
            name="Similarity Scores",
            hovertemplate="Similarity: %{x:.3f}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
        xaxis_title="Cosine Similarity",
        yaxis_title="Frequency",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="sans-serif"),
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9", zeroline=False)

    return fig


def find_outliers(
    embeddings: np.ndarray,
    chunks: list[Any],
    n_outliers: int = 5,
    preview_length: int = 250,
) -> list[dict[str, Any]]:
    """Identify chunks that are semantically distant from others.

    An outlier is a chunk with low average similarity to all other chunks.
    This can indicate unique concepts, noise, or headers/footers.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        chunks: List of chunk objects corresponding to embeddings
        n_outliers: Number of outliers to return
        preview_length: Maximum length of text preview

    Returns:
        List of dicts with keys: 'index', 'chunk', 'avg_similarity', 'text_preview'
    """
    if embeddings.shape[0] < 2:
        return []

    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(embeddings)

    # Calculate average similarity for each chunk (excluding self-similarity)
    n = similarity_matrix.shape[0]
    avg_similarities = np.zeros(n)

    for i in range(n):
        # Exclude diagonal (self-similarity = 1.0)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        avg_similarities[i] = similarity_matrix[i, mask].mean()

    # Find indices of chunks with lowest average similarity
    outlier_indices = np.argsort(avg_similarities)[:n_outliers]

    outliers = []
    for idx in outlier_indices:
        chunk = chunks[idx]
        outliers.append(
            {
                "index": int(idx),
                "chunk": chunk,
                "avg_similarity": float(avg_similarities[idx]),
                "text_preview": (
                    chunk.text[:preview_length] + "..."
                    if len(chunk.text) > preview_length
                    else chunk.text
                ),
            }
        )

    return outliers
