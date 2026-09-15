"""Shared helper for turning Flask file uploads into extracted text."""

from __future__ import annotations

from . import text_extract


def collect_uploaded_texts(file_storages) -> tuple[dict[str, str], list[str]]:
    """Extract text from a list of werkzeug FileStorage objects.

    Returns (texts, warnings): `texts` is {display_name: text} for every
    file that extracted successfully; `warnings` holds one message per
    file that failed (unsupported type, corrupt file) -- mirroring the
    Streamlit app's "skip with a warning" behavior instead of failing the
    whole upload.
    """
    texts: dict[str, str] = {}
    warnings: list[str] = []
    for f in file_storages:
        if not f or not f.filename:
            continue
        # Folder-picker uploads (webkitdirectory) carry the relative path
        # in .filename (e.g. "mycorpus/sub/a.txt") -- kept as-is for the
        # display name since it's still unique and informative.
        name = f.filename
        try:
            texts[name] = text_extract.extract_text(name, f.read())
        except ValueError as e:
            warnings.append(str(e))
    return texts, warnings
