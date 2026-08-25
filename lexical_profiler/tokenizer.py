"""
Tokenization utilities, powered by spaCy.

spaCy is used for both tokenization and (optional) lemmatization, and
supports many languages via its language codes / pipeline packages
(https://spacy.io/usage/models#languages), e.g. 'en', 'es', 'de', 'fr',
'zh', 'ja', 'ru', etc.

Two levels of support per language:

  1. A full installed pipeline (e.g. `en_core_web_sm`), which gives
     proper part-of-speech-aware lemmatization.
  2. A "blank" pipeline (`spacy.blank(lang)`), which spaCy can construct
     for essentially any supported language code without any download.
     This gives good tokenization but no lemmatizer.

If a full pipeline isn't installed for a requested language,
`tokenize(..., lemmatize=True)` will automatically and silently fall
back to tokenization-only (surface word forms) using the blank pipeline,
rather than raising an error. Use `lemmatizer_available(language)` to
check ahead of time, or `download_model(language)` to fetch one.
"""

from __future__ import annotations

import re
import subprocess
import sys
from functools import lru_cache

import spacy
from spacy.language import Language

# Common ISO 639-1 language codes mapped to spaCy's small pipeline package
# names. This covers spaCy's officially supported languages with trained
# pipelines as of spaCy 3.x. See https://spacy.io/models for the full,
# up-to-date list. Pass a full model name directly (e.g. 'en_core_web_lg')
# to bypass this map entirely.
LANGUAGE_MODEL_MAP = {
    "ca": "ca_core_news_sm",
    "zh": "zh_core_web_sm",
    "hr": "hr_core_news_sm",
    "da": "da_core_news_sm",
    "nl": "nl_core_news_sm",
    "en": "en_core_web_sm",
    "fi": "fi_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "el": "el_core_news_sm",
    "it": "it_core_news_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "lt": "lt_core_news_sm",
    "mk": "mk_core_news_sm",
    "nb": "nb_core_news_sm",
    "pl": "pl_core_news_sm",
    "pt": "pt_core_news_sm",
    "ro": "ro_core_news_sm",
    "ru": "ru_core_news_sm",
    "sl": "sl_core_news_sm",
    "es": "es_core_news_sm",
    "sv": "sv_core_news_sm",
    "uk": "uk_core_news_sm",
}

# Fallback used when nothing better is known/installed for a language and
# no blank tokenizer works either. spaCy's multi-language tokenizer is a
# reasonable generic default.
_MULTI_LANGUAGE_FALLBACK = "xx"

_MODEL_NAME_RE = re.compile(r"^[a-z]{2,3}_[a-z]+_(sm|md|lg|trf)$")

_pipeline_cache: dict = {}


def _is_full_model_name(language: str) -> bool:
    return bool(_MODEL_NAME_RE.match(language))


def _resolve_model_name(language: str) -> str | None:
    if _is_full_model_name(language):
        return language
    return LANGUAGE_MODEL_MAP.get(language.lower())


def download_model(language: str) -> bool:
    """Attempt to download/install a spaCy pipeline for `language`
    (either an ISO code like 'de' or a full model name like
    'de_core_news_sm'). Returns True on success.

    Requires network access. Safe to call even if already installed
    (spaCy/pip will just no-op).
    """
    model_name = _resolve_model_name(language)
    if model_name is None:
        return False
    try:
        # Shell out to `python -m spacy download ...`. This is exactly
        # what spaCy's own CLI does, and reusing it means we don't have to
        # duplicate its model-resolution/installation logic here.
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True, capture_output=True,
        )
        # A newly downloaded model wouldn't be picked up by pipelines we've
        # already cached, so clear both caches to force a fresh load next
        # time this language is requested.
        _pipeline_cache.clear()
        _cached_pipeline.cache_clear()
        return True
    except subprocess.CalledProcessError:
        return False


def _load_pipeline(language: str) -> tuple[Language, bool]:
    """Return (nlp, has_lemmatizer) for `language`, caching by language.

    Tries a full trained pipeline first; falls back to a blank
    (tokenizer-only) pipeline for the same language if unavailable;
    falls back further to spaCy's generic multi-language tokenizer if
    even that fails (e.g. unrecognized code).
    """
    if language in _pipeline_cache:
        return _pipeline_cache[language]

    nlp = None
    has_lemmatizer = False

    # Attempt 1: a full trained pipeline (e.g. en_core_web_sm). We only
    # need its tokenizer + lemmatizer, so the parser and NER components are
    # excluded to keep loading fast and memory light.
    model_name = _resolve_model_name(language)
    if model_name:
        try:
            nlp = spacy.load(model_name, exclude=["parser", "ner"])
            has_lemmatizer = "lemmatizer" in nlp.pipe_names or "morphologizer" in nlp.pipe_names
        except OSError:
            # Model name is known but not installed locally; fall through
            # to the blank-pipeline attempt below.
            nlp = None

    # Attempt 2: no trained pipeline available, so fall back to a "blank"
    # pipeline for the language, which still gives correct tokenization
    # (just no lemmatizer) and needs no download.
    if nlp is None:
        blank_code = language if not _is_full_model_name(language) else language.split("_")[0]
        try:
            nlp = spacy.blank(blank_code)
        except Exception:
            # Attempt 3: language code spaCy doesn't recognize at all --
            # fall back to spaCy's generic multi-language tokenizer so
            # tokenization never hard-fails just because of an odd/unknown
            # language code.
            nlp = spacy.blank(_MULTI_LANGUAGE_FALLBACK)

    nlp.max_length = 5_000_000  # allow reasonably large corpora/documents
    result = (nlp, has_lemmatizer)
    _pipeline_cache[language] = result
    return result


def lemmatizer_available(language: str = "en") -> bool:
    """Whether a full pipeline with lemmatization is available (installed)
    for `language` in this environment."""
    _, has_lemmatizer = _load_pipeline(language)
    return has_lemmatizer


@lru_cache(maxsize=32)
def _cached_pipeline(language: str) -> tuple[Language, bool]:
    return _load_pipeline(language)


def tokenize(text: str, language: str = "en", lowercase: bool = True,
             lemmatize: bool = False, min_length: int = 1):
    """Tokenize raw text into a list of word tokens using spaCy.

    Args:
        text: input string.
        language: ISO 639-1 language code (e.g. 'en', 'es', 'de', 'fr',
            'zh', 'ja', 'ru', ...) or a full spaCy model name (e.g.
            'en_core_web_sm'). If a trained pipeline for the language
            isn't installed, tokenization still works via spaCy's blank
            (rule-based) tokenizer for that language.
        lowercase: lowercase tokens (recommended for frequency profiling).
        lemmatize: reduce tokens to a base dictionary form. Requires a
            trained pipeline for `language` to be installed; silently
            falls back to surface word forms otherwise (check ahead of
            time with `lemmatizer_available(language)`, or fetch one
            with `download_model(language)`).
        min_length: drop tokens shorter than this (after lowercasing,
            before lemmatizing). Set to 1 to keep single-letter words.

    Returns:
        List of token strings, in order of appearance.
    """
    nlp, has_lemmatizer = _cached_pipeline(language)
    doc = nlp(text)

    # Only actually lemmatize if the caller asked for it *and* we have a
    # pipeline capable of it; otherwise silently fall back to surface
    # forms rather than erroring (see module docstring for rationale).
    do_lemmatize = lemmatize and has_lemmatizer

    tokens = []
    for tok in doc:
        surface = tok.text
        # Keep only tokens containing at least one alphabetic character
        # (drops pure punctuation, numbers, whitespace, symbols) while
        # still allowing internal apostrophes/hyphens, e.g. "don't".
        if not any(ch.isalpha() for ch in surface):
            continue
        # tok.lemma_ can come back empty for some tokens even when a
        # lemmatizer is active, so fall back to the surface form in that
        # case rather than emitting an empty string.
        word = tok.lemma_ if (do_lemmatize and tok.lemma_) else surface
        if lowercase:
            word = word.lower()
        if len(word) >= min_length:
            tokens.append(word)
    return tokens
