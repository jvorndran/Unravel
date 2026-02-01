import streamlit as st
import streamlit_shadcn_ui as ui


def apply_custom_styles() -> None:
    """Apply global custom styles to the Streamlit app."""
    # Reduced to minimal font imports and essential container tweaks
    st.markdown(
        """<style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Ensure streamlit-shadcn-ui components take full width */
        div[data-testid="stIFrame"] {
            width: 100% !important;
        }
        div[data-testid="stIFrame"] iframe {
            width: 100% !important;
        }

        /* Clean up main container padding */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 1300px !important;
        }

        /* Chunk highlighting specific styles (retained as it's custom HTML) */
        .chunk-highlight {
            background-color: #fee2e2;
            border-radius: 2px;
            padding: 0 2px;
        }
        
        /* Stats Bar (legacy support until fully refactored) */
        .stats-bar {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        /* Custom Primary Button Style (Black) */
        div.stButton > button[kind="primary"] {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #000000 !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #333333 !important;
            border-color: #333333 !important;
            color: #ffffff !important;
        }
        div.stButton > button[kind="primary"]:active {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        div.stButton > button[kind="primary"]:focus {
            box-shadow: none !important;
            outline: none !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )


def render_step_nav(active_step: str = "chunks") -> None:
    """Render the step navigation bar using shadcn tabs."""
    
    # Map internal IDs to Display Labels
    step_map = {
        "upload": "Upload",
        "chunks": "Text Splitting",
        "embeddings": "Vector Embedding",
        "query": "Response Generation",
        "export": "Export Code",
    }
    
    # Reverse map for lookup
    label_map = {v: k for k, v in step_map.items()}
    
    # Get current label
    default_value = step_map.get(active_step, "Text Splitting")
    
    # Render Tabs
    st.write("") # Spacer
    selected_label = ui.tabs(
        options=list(step_map.values()),
        default_value=default_value,
        key="main_navigation_tabs",
    )
    
    # Handle Navigation
    selected_id = label_map.get(selected_label)
    
    if selected_id and selected_id != active_step:
        st.session_state.current_step = selected_id
        st.rerun()
