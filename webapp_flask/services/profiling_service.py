"""Wraps LexicalProfiler construction and results re-computation.

Crux design decision (see the plan file / PR description): ProfileResult
is never cached across requests -- ProfileResult.to_dict() drops
word_counts/band_ranges/num_bands, so a cached dict can't reconstruct a
fully working ProfileResult anyway. Instead every route that needs
results calls get_results(), which reloads the (cheap) Reference JSON and
re-runs profiling against the (small) target texts fresh each time.
"""

from __future__ import annotations

from pathlib import Path

from lexical_profiler import LexicalProfiler, ProfileResult, Reference

from . import session_store


def read_word_list(raw: str) -> list[str]:
    """Parse a word-list textarea/file: one word per line, '#' comments
    and blank lines ignored. Ported from webapp/app.py's read_word_list."""
    return [w.strip() for w in raw.splitlines() if w.strip() and not w.strip().startswith("#")]


def build_profiler(reference: Reference, ignore_config: dict) -> LexicalProfiler:
    return LexicalProfiler(
        reference,
        ignore_words=ignore_config.get("ignore_words", []),
        exclude_proper_nouns=ignore_config.get("exclude_proper_nouns", True),
        exclude_digits=ignore_config.get("exclude_digits", True),
    )


def get_profiler(sessions_root: Path, session_id: str) -> LexicalProfiler:
    """Reload the session's Reference and rebuild a LexicalProfiler from
    its ignore config. Cheap (just a JSON read), safe to call per request."""
    meta = session_store.read_meta(sessions_root, session_id)
    reference = Reference.load(str(session_store.reference_path(sessions_root, session_id)))
    return build_profiler(reference, meta["ignore_config"])


def get_results(
    sessions_root: Path, session_id: str, progress_callback=None,
) -> dict[str, ProfileResult]:
    meta = session_store.read_meta(sessions_root, session_id)
    profiler = get_profiler(sessions_root, session_id)
    return profiler.profile_texts(meta["target_texts"], progress_callback=progress_callback)
