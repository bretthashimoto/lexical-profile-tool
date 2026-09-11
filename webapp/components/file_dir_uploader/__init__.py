"""A minimal custom Streamlit component: one control with two buttons,
"Choose files..." and "Choose folder...", backed by real browser file
inputs -- the folder button sets the `webkitdirectory` attribute, which
prompts a native OS folder picker in Chromium/Firefox/Safari, so a user
can pick either loose files or a whole directory from their own machine
without zipping anything first. This works the same whether the app is
running locally or hosted, since the picker is entirely client-side; only
the resulting file contents get sent to the Streamlit backend.

No JS build step: the frontend hand-implements Streamlit's small
postMessage-based component protocol (componentReady / setComponentValue
/ setFrameHeight) directly, rather than using the React component
template, since this component has no need for a build pipeline.
"""
from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).parent / "frontend"

_component_func = components.declare_component(
    "file_dir_uploader", path=str(_FRONTEND_DIR),
)


def file_dir_uploader(label: str = "Choose files...", key: str | None = None):
    """Renders the picker and returns a list of {"name", "content_b64"}
    dicts for every selected .txt/.docx/.pdf file -- raw bytes, base64-
    encoded in the browser -- or None until the user has picked something.
    Use `text_extract.extract_text(name, base64.b64decode(content_b64))`
    to get plain text back out."""
    return _component_func(label=label, key=key, default=None)
