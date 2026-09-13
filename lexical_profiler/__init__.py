"""
lexical_profiler
=================

A small toolkit for lexical frequency profiling of texts.

A few ways to build a *reference* frequency model:
  1. From a corpus of texts (word frequencies are derived automatically).
  2. From an existing word list (e.g. a published frequency list such as
     the GSL, NGSL, or a COCA-derived list), one word per line, ordered
     from most to least frequent (optionally "word<TAB>frequency").
  3. From a published word list bundled with this package (AVL, NGSL,
     NAWL, or COCA) via `Reference.from_builtin(...)`, with nothing to
     download or format yourself.

That reference is then used to profile one or more *target* texts,
reporting how much of the target text's vocabulary falls into each
frequency band (e.g. the most frequent 1000 words, the next 1000, etc.),
plus type/token ratio, coverage statistics, and a list of "off-list"
(unknown / rare) words.

Typical usage:

    from lexical_profiler import Reference, LexicalProfiler

    ref = Reference.from_corpus(["corpus/a.txt", "corpus/b.txt"], band_size=1000)
    profiler = LexicalProfiler(ref)
    result = profiler.profile_text(open("essay.txt").read())
    print(result.summary())
"""

from .profiler import HighlightedToken, LexicalProfiler, ProfileResult
from .reference import Reference
from .tokenizer import download_model, lemmatizer_available, tokenize

__all__ = [
    "Reference", "LexicalProfiler", "ProfileResult", "HighlightedToken",
    "tokenize", "lemmatizer_available", "download_model",
]
__version__ = "0.2.0"
