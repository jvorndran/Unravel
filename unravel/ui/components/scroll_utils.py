"""Utility functions for scroll behavior in the UI."""

import streamlit.components.v1 as components


def scroll_to_element(element_id: str) -> None:
    """
    Scroll to element using JavaScript injection.

    Args:
        element_id: The ID of the element to scroll to
    """
    components.html(
        f"""
        <script>
            setTimeout(function() {{
                const element = window.parent.document.getElementById('{element_id}');
                if (element) {{
                    element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }}, 100);
        </script>
        """,
        height=0,
    )
