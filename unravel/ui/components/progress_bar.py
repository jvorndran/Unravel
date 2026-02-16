import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, Tuple

import streamlit as st


def timed_progress_bar(
    task: Callable[..., Any],
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    label: str = "Processing...",
    estimated_time: float = 2.0,
) -> Any:
    """
    Runs a task in a separate thread while displaying a progress bar that advances based on estimated time.

    Args:
        task: The function to execute.
        args: Positional arguments for the task.
        kwargs: Keyword arguments for the task.
        label: The label to display on the progress bar.
        estimated_time: The estimated time in seconds for the task to complete.

    Returns:
        The result of the task function.
    """
    if kwargs is None:
        kwargs = {}

    progress_bar = st.progress(0, text=label)
    time.sleep(0.05)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task, *args, **kwargs)

        while not future.done():
            elapsed = time.time() - start_time
            # Calculate progress (0.0 to 1.0)
            # We cap it at 0.95 so it doesn't finish before the task
            progress = min(0.95, elapsed / estimated_time)

            progress_bar.progress(progress, text=label)
            time.sleep(0.1)

        # Task is done
        progress_bar.progress(1.0, text="Done!")
        time.sleep(0.5)  # Show 100% for a moment
        progress_bar.empty()

        return future.result()
