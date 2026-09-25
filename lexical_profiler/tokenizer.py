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

import os
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

# Human-readable names for LANGUAGE_MODEL_MAP's codes, for UIs (e.g. the web
# app's language dropdown) that want a friendly label instead of a bare code.
LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "nb": "Norwegian Bokmål",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "uk": "Ukrainian",
}

# Maps spaCy's fine-grained Penn-Treebank-style tag (`token.tag_`) to a
# small set of single-letter part-of-speech codes, used to build composite
# "lemma_code" keys for POS-aware reference matching (see Reference.
# pos_tagged). Deliberately coarser than tag_ and NOT the same as spaCy's
# own Universal POS (`token.pos_`): tag_ tags verb *forms* morphologically
# (e.g. "is"/"was"/"be" are all VBZ/VBD/VB regardless of auxiliary-vs-main-
# verb use), which is what lets "be"/"have"/"do" land under the plain verb
# code "v" here -- routing through the coarser Universal POS instead would
# tag those as AUX, which silently fails to match a "v"-tagged reference
# entry. Anything not in this map (numbers, punctuation, symbols, proper
# nouns, foreign words, ...) has no POS code and can't take part in
# POS-aware matching. Codes are lowercase because word-list files get
# lowercased on load (Reference.from_word_list's `lowercase` default) --
# using uppercase codes here would silently mismatch target-text tokens
# (which always get a lowercase code) against a loaded reference file.
_POS_TAG_MAP: dict[str, str] = {
    "NN": "n", "NNS": "n", "NNP": "n", "NNPS": "n",
    "VB": "v", "VBD": "v", "VBG": "v", "VBN": "v", "VBP": "v", "VBZ": "v",
    "MD": "m",
    "JJ": "j", "JJR": "j", "JJS": "j",
    "RB": "r", "RBR": "r", "RBS": "r", "RP": "r",
    "IN": "i",
    "PRP": "p", "PRP$": "p",
    "DT": "d", "PDT": "d",
    "CC": "c",
    "UH": "u",
    "WP": "w", "WP$": "w", "WDT": "w", "WRB": "w",
    # "to" (infinitive marker AND preposition use) is always tagged TO by
    # spaCy regardless of function -- it gets its own code rather than
    # folding into "i" (preposition), since it's one of the most frequent
    # words in English and deserves to be trackable on its own.
    "TO": "t",
    # Existential "there" ("there is/are a...") vs. ordinary locative/
    # adverbial "there" ("over there") -- EX is a distinct, reliable tag,
    # not folded into "r" (adverb).
    "EX": "e",
}

# Human-readable names for _POS_TAG_MAP's codes, for UIs that want a
# friendly label instead of a bare code (e.g. the web app's off-list word
# tables). Same pattern as LANGUAGE_DISPLAY_NAMES.
POS_DISPLAY_NAMES: dict[str, str] = {
    "n": "noun",
    "v": "verb",
    "m": "modal verb",
    "j": "adjective",
    "r": "adverb",
    "i": "preposition",
    "p": "pronoun",
    "d": "determiner",
    "c": "conjunction",
    "u": "interjection",
    "w": "wh-word",
    "t": "to-marker",
    "e": "existential there",
}


def pos_suffix(token) -> str | None:
    """The POS code for one spaCy token (see _POS_TAG_MAP), or None if its
    tag isn't one we classify -- such a token can't take part in POS-aware
    matching (it just won't get a "_CODE" suffix)."""
    return _POS_TAG_MAP.get(token.tag_)


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


def pipeline_for(language: str = "en") -> tuple[Language, bool]:
    """Public accessor for the cached spaCy pipeline for `language`,
    returning (nlp, has_lemmatizer). Useful for callers that need the raw
    spaCy tokens themselves (surface text + trailing whitespace), rather
    than just the flat token list `tokenize()` returns -- e.g. rendering
    original text with a per-token annotation without losing spacing."""
    return _cached_pipeline(language)


def _classify_doc(doc, *, language: str, lowercase: bool, min_length: int,
                   do_lemmatize: bool, do_pos_tag: bool,
                   has_lemmatizer: bool) -> list[tuple[str, str]]:
    """Shared per-document token-classification loop used by both
    `classify_tokens()` (one spaCy Doc at a time) and `classify_texts()`
    (many Docs from a single `nlp.pipe()` batch) -- kept as one function
    so the two call paths can never silently drift apart."""
    tokens: list[tuple[str, str]] = []
    for tok in doc:
        surface = tok.text
        # A "digit" token is one spaCy's `like_num` recognizes as numeric
        # *and* that's actually written with digit characters (e.g. "42",
        # "3.14", "3rd") -- the `like_num` check on its own would also
        # catch spelled-out number words ("twelve"), which should stay
        # ordinary vocabulary, and the digit check on its own would wrongly
        # catch alphanumeric words like "word2" that aren't numbers at all.
        # Works even with a blank (no trained pipeline) tokenizer.
        is_digit = tok.like_num and any(ch.isdigit() for ch in surface)
        # Keep only tokens containing at least one alphabetic character
        # (drops pure punctuation, whitespace, symbols) while still
        # allowing internal apostrophes/hyphens (e.g. "don't") -- unless
        # it's a digit token, which is kept regardless.
        if not is_digit and not any(ch.isalpha() for ch in surface):
            continue
        # tok.lemma_ can come back empty for some tokens even when a
        # lemmatizer is active, so fall back to the surface form in that
        # case rather than emitting an empty string.
        word = tok.lemma_ if (do_lemmatize and tok.lemma_) else surface
        if lowercase:
            word = word.lower()
        # spaCy treats "a"/"an" as distinct lemmas -- the alternation is
        # purely phonological (triggered by a following vowel sound), not
        # inflectional like "cats"/"cat" -- so lemmatizing alone never
        # collapses "an" into "a". Do it here so reference word lists only
        # need to list "a" and English "an" isn't wrongly flagged off-list.
        if do_lemmatize and language.startswith("en") and word.lower() == "an":
            word = "a" if word.islower() else "A"
        if len(word) < min_length:
            continue
        if do_pos_tag:
            code = pos_suffix(tok)
            if code:
                word = f"{word}_{code}"

        if is_digit:
            category = "digit"
        elif has_lemmatizer and tok.pos_ == "PROPN":
            category = "proper_noun"
        else:
            category = "word"
        tokens.append((word, category))
    return tokens


def classify_tokens(text: str, language: str = "en", lowercase: bool = True,
                     lemmatize: bool = False, min_length: int = 1,
                     pos_tag: bool = False) -> list[tuple[str, str]]:
    """Like `tokenize()`, but returns (word, category) pairs instead of a
    flat word list, where category is "word", "proper_noun", or "digit".
    This is what lets a caller (see LexicalProfiler's
    exclude_proper_nouns/exclude_digits) report proper nouns and
    digit tokens separately from ordinary running text instead of profiling
    them like any other word.

    Two differences from `tokenize()`:
      - A digit token (e.g. "42", "3.14") is kept and categorized
        "digit" instead of being dropped outright for having no
        alphabetic character -- `tokenize()` itself still drops these
        (it's built on top of this function, filtering "digit" out),
        so existing callers see no change.
      - Every kept token is tagged "word", "proper_noun", or "digit".

    "proper_noun" classification uses the language's POS tagger, so (like
    `lemmatize`/`pos_tag`) it requires a trained pipeline for `language`
    to be installed; with none installed, no token is ever classified
    "proper_noun" -- everything that isn't a digit token is just "word".

    Note this only catches tokens actually written with digit characters
    (e.g. "42", "3.14", "3rd"), not spelled-out number words like "twelve"
    or "forty-two" -- those stay ordinary vocabulary ("word"), unlike
    spaCy's broader `like_num`. An alphanumeric word that merely contains a
    digit (e.g. "word2") isn't a "digit" token either, since it isn't a
    number at all.

    Processing many texts? Use `classify_texts()` instead -- it feeds
    them through spaCy as a single batch (`nlp.pipe()`), which is
    substantially faster than calling this function once per text in a
    Python loop.

    Args: same as `tokenize()`.
    """
    nlp, has_lemmatizer = _cached_pipeline(language)
    doc = nlp(text)

    # Only actually lemmatize/tag if the caller asked for it *and* we have
    # a pipeline capable of it; otherwise silently fall back to surface
    # forms rather than erroring (see module docstring for rationale).
    do_lemmatize = lemmatize and has_lemmatizer
    do_pos_tag = pos_tag and has_lemmatizer

    return _classify_doc(doc, language=language, lowercase=lowercase, min_length=min_length,
                          do_lemmatize=do_lemmatize, do_pos_tag=do_pos_tag,
                          has_lemmatizer=has_lemmatizer)


def classify_texts(texts: list[str], language: str = "en", lowercase: bool = True,
                    lemmatize: bool = False, min_length: int = 1, pos_tag: bool = False,
                    n_process: int = 1, batch_size: int = 1000,
                    ) -> list[list[tuple[str, str]]]:
    """Like `classify_tokens()`, but for many texts at once -- feeds them
    through spaCy as a single batch via `nlp.pipe()` instead of the caller
    looping and calling `classify_tokens()` once per text.

    `nlp.pipe()` amortizes per-call overhead across the batch and (unlike
    repeated single-text calls) can spread the work across processes, so
    this is the right tool for "many independent texts, order doesn't
    depend on each other" workloads -- building a reference from a corpus,
    or profiling a folder of target texts.

    Returns a list of per-text (word, category) lists, in the same order
    as `texts`.

    Args:
        texts: the texts to process, in order.
        n_process: worker processes to use (default 1, i.e. no
            multiprocessing -- batching alone already gives most of the
            speedup for typical corpus/target-text sizes; consider raising
            this only for a large number of documents or a slow pipeline,
            since each extra process pays a real startup cost -- reloading
            the spaCy pipeline in each worker takes real time and memory).
            -1 uses all available CPU cores. On Windows (or anywhere
            multiprocessing uses "spawn" rather than "fork"), a value > 1
            requires the calling code to run under
            `if __name__ == "__main__":`, same as spaCy's own multi-
            processing docs -- otherwise spawning re-imports and re-runs
            the entry script in each worker.
        batch_size: texts per internal batch sent to a worker. spaCy's own
            default (1000) is fine for typical text lengths; lower it if
            individual texts are unusually large.
        Other args: same as `classify_tokens()`.
    """
    nlp, has_lemmatizer = _cached_pipeline(language)
    do_lemmatize = lemmatize and has_lemmatizer
    do_pos_tag = pos_tag and has_lemmatizer

    return [
        _classify_doc(doc, language=language, lowercase=lowercase, min_length=min_length,
                      do_lemmatize=do_lemmatize, do_pos_tag=do_pos_tag,
                      has_lemmatizer=has_lemmatizer)
        for doc in nlp.pipe(texts, n_process=n_process, batch_size=batch_size)
    ]


def tokenize(text: str, language: str = "en", lowercase: bool = True,
             lemmatize: bool = False, min_length: int = 1, pos_tag: bool = False):
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
        pos_tag: append a part-of-speech code to each token, e.g.
            "record_v" vs "record_n" (see `_POS_TAG_MAP`), for matching
            against a POS-tagged Reference. Requires the same trained
            pipeline as `lemmatize`; silently produces plain tokens
            otherwise. A token whose tag isn't one of the codes we
            classify keeps no suffix at all.

    Returns:
        List of token strings, in order of appearance. Pure digit tokens
        (e.g. "42") are dropped, same as pure punctuation -- use
        `classify_tokens()` if you need to see them.
    """
    classified = classify_tokens(
        text, language=language, lowercase=lowercase, lemmatize=lemmatize,
        min_length=min_length, pos_tag=pos_tag,
    )
    return [word for word, category in classified if category != "digit"]


def tokenize_texts(texts: list[str], language: str = "en", lowercase: bool = True,
                    lemmatize: bool = False, min_length: int = 1, pos_tag: bool = False,
                    n_process: int = 1, batch_size: int = 1000) -> list[list[str]]:
    """Like `tokenize()`, but for many texts at once -- see
    `classify_texts()` for why/when to prefer this over calling
    `tokenize()` once per text in a loop.

    Returns a list of per-text token-string lists, in the same order as
    `texts`. Pure digit tokens are dropped, same as `tokenize()`.
    """
    classified = classify_texts(
        texts, language=language, lowercase=lowercase, lemmatize=lemmatize,
        min_length=min_length, pos_tag=pos_tag, n_process=n_process, batch_size=batch_size,
    )
    return [[word for word, category in doc_tokens if category != "digit"]
            for doc_tokens in classified]


def auto_n_process(n_texts: int, texts_per_process: int = 6000) -> int:
    """Pick a reasonable `n_process` for `classify_texts()`/`tokenize_texts()`
    (and the `n_process` args on `Reference.from_corpus`/`add_texts` and
    `LexicalProfiler.profile_texts`/`profile_corpus`) based on how many
    texts there are, so callers don't have to hand-tune it themselves.

    Below `texts_per_process` texts this returns 1 (no multiprocessing).
    Above that, it scales up roughly one process per `texts_per_process`
    texts, capped at the number of CPU cores available (`os.cpu_count()`,
    or 1 if that can't be determined).

    The default threshold is high on purpose, not a typo: each extra
    worker process has to reload the *entire* spaCy pipeline (tokenizer +
    tagger + lemmatizer) from scratch before it can do anything, and that
    fixed reload cost turned out to be substantial in practice -- on the
    machine this was benchmarked on (Windows, `en_core_web_sm`), 2
    processes were still slower than 1 at 3,000 real documents, and only
    projected to break even somewhere around 6,000-8,000. That reload cost
    is independent of corpus size, so there's no threshold that's both
    "low enough to matter for typical corpus sizes" and "high enough not
    to backfire" -- multiprocessing here only pays off for genuinely large
    corpora. This will vary by machine, OS (process startup is pricier on
    Windows than Linux), and spaCy pipeline size, so benchmark your own
    deployment (`n_process=1` vs a higher value, at your real corpus
    sizes) before assuming a lower threshold would help.

    This is opt-in -- nothing calls it automatically, since spinning up
    real worker processes changes execution semantics in a way a library
    shouldn't do silently (see `classify_texts`'s `n_process` docs for the
    Windows/"spawn" caveat, which callers using this need to be set up
    for). Callers that know their own execution context is safe for it
    (e.g. a background job thread, not a bare top-level script) should
    call this themselves and pass the result as `n_process`.
    """
    if n_texts < texts_per_process:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, n_texts // texts_per_process))
