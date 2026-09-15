"""Thin wrapper around building/loading a lexical_profiler.Reference from
Flask form data.

Milestone 1 only wires up the builtin-word-list path (synchronous, fast
enough to run inline in a request); corpus/word-list/saved-reference
building are added in later milestones.
"""

from __future__ import annotations

from lexical_profiler import Reference
from lexical_profiler.reference import BUILTIN_WORD_LISTS

BAND_PARAM_FIELDS = (
    "band_size", "language", "lemmatize",
    "fine_band_size", "fine_grained_until",
    "coarse_band_size", "coarse_grained_from",
)


def parse_band_params(form) -> dict:
    """Pull the shared reference-build parameters (band size, language,
    lemmatize, fine/coarse band overrides) out of a submitted form."""

    def _int_or_none(key: str) -> int | None:
        raw = (form.get(key) or "").strip()
        return int(raw) if raw else None

    return {
        "band_size": int(form.get("band_size") or 1000),
        "language": form.get("language", "en"),
        "lemmatize": form.get("lemmatize") == "on",
        "fine_band_size": _int_or_none("fine_band_size"),
        "fine_grained_until": _int_or_none("fine_grained_until"),
        "coarse_band_size": _int_or_none("coarse_band_size"),
        "coarse_grained_from": _int_or_none("coarse_grained_from"),
    }


def build_from_builtin(name: str, **band_params) -> Reference:
    return Reference.from_builtin(name, **band_params)


def builtin_choices() -> dict:
    return BUILTIN_WORD_LISTS
