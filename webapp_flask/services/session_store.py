"""Per-session file store: directory layout under SESSION_DIR and meta.json
read/write.

Each session gets `<SESSION_DIR>/<session_id>/`:
  reference.json  -- written by Reference.save(), reloaded via Reference.load()
  meta.json        -- small JSON: reference-build params, ignore config, target texts
  uploads/           -- scratch space for uploads that need a real filesystem
                        path (word lists, saved references)

Nothing here holds a built Reference or ProfileResult in memory across
requests -- see profiling_service.get_results() for why results are always
recomputed fresh instead of cached.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

DEFAULT_META = {
    "reference_build": None,
    "ignore_config": {"exclude_proper_nouns": True, "exclude_digits": True, "ignore_words": []},
    "target_texts": {},
}


def session_dir(sessions_root: Path, session_id: str) -> Path:
    return Path(sessions_root) / session_id


def ensure_session_dir(sessions_root: Path, session_id: str) -> Path:
    d = session_dir(sessions_root, session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def uploads_dir(sessions_root: Path, session_id: str) -> Path:
    d = session_dir(sessions_root, session_id) / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reference_path(sessions_root: Path, session_id: str) -> Path:
    return session_dir(sessions_root, session_id) / "reference.json"


def meta_path(sessions_root: Path, session_id: str) -> Path:
    return session_dir(sessions_root, session_id) / "meta.json"


def has_reference(sessions_root: Path, session_id: str) -> bool:
    return reference_path(sessions_root, session_id).exists()


def read_meta(sessions_root: Path, session_id: str) -> dict:
    path = meta_path(sessions_root, session_id)
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_META))  # cheap deep copy
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    merged = json.loads(json.dumps(DEFAULT_META))
    merged.update(data)
    return merged


def write_meta(sessions_root: Path, session_id: str, meta: dict) -> None:
    ensure_session_dir(sessions_root, session_id)
    with open(meta_path(sessions_root, session_id), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def update_meta(sessions_root: Path, session_id: str, **updates) -> dict:
    """Merge `updates` into the session's meta.json and persist it."""
    meta = read_meta(sessions_root, session_id)
    meta.update(updates)
    write_meta(sessions_root, session_id, meta)
    return meta


def touch(sessions_root: Path, session_id: str) -> None:
    """Mark a session as recently active, so the reaper doesn't sweep it."""
    marker = ensure_session_dir(sessions_root, session_id) / ".touch"
    marker.touch()


def start_reaper(sessions_root: Path, ttl: int) -> None:
    """Start a daemon thread that deletes session directories whose
    `.touch` marker (or, if absent, mtime) is older than `ttl`. Started
    once per app instance from the app factory."""
    interval = max(ttl // 6, 60)

    def sweep() -> None:
        while True:
            time.sleep(interval)
            cutoff = time.time() - ttl
            root = Path(sessions_root)
            if not root.exists():
                continue
            for d in root.iterdir():
                if not d.is_dir():
                    continue
                marker = d / ".touch"
                mtime = marker.stat().st_mtime if marker.exists() else d.stat().st_mtime
                if mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)

    threading.Thread(target=sweep, daemon=True).start()
