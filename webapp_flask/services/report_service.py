"""Temp-file round trip for lexical_profiler.report's export_* functions,
which all write directly to a filesystem path (no in-memory/bytes
variant). Ported from webapp/app.py's export_to_bytes helper.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def export_to_bytes(export_fn, results, suffix: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        export_fn(results, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
