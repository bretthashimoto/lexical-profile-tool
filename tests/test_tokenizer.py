import pytest

from lexical_profiler.tokenizer import classify_tokens, lemmatizer_available, tokenize


def test_tokenize_lowercases_by_default():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_preserves_case_when_disabled():
    assert tokenize("Hello World", lowercase=False) == ["Hello", "World"]


def test_tokenize_drops_pure_punctuation_and_numbers():
    tokens = tokenize("Hello, world! 123 -- 456.")
    assert tokens == ["hello", "world"]


def test_tokenize_keeps_tokens_with_internal_punctuation():
    # spaCy's tokenizer splits "don't" into "do" / "n't"; both pieces
    # contain alphabetic characters, so neither should be dropped by the
    # any(ch.isalpha()) filter.
    tokens = tokenize("don't stop")
    assert tokens == ["do", "n't", "stop"]


def test_tokenize_min_length_filters_short_tokens():
    tokens = tokenize("a bb ccc dddd", min_length=3)
    assert tokens == ["ccc", "dddd"]


def test_tokenize_empty_string_returns_empty_list():
    assert tokenize("") == []


def test_tokenize_unknown_language_code_falls_back_without_crashing():
    # 'xx' / nonsense codes shouldn't raise; should fall back to the
    # generic multi-language tokenizer instead.
    tokens = tokenize("hello world", language="zznotalang")
    assert tokens == ["hello", "world"]


def test_lemmatize_falls_back_silently_without_installed_pipeline():
    # No trained pipeline is installed in the test environment, so
    # lemmatize=True should silently fall back to surface forms rather
    # than raising.
    assert lemmatizer_available("en") is False
    tokens = tokenize("running dogs", lemmatize=True)
    assert tokens == ["running", "dogs"]


def test_pos_tag_falls_back_silently_without_installed_pipeline():
    # Same fallback convention as lemmatize=True: pos_tag=True with no
    # trained pipeline available should silently produce plain tokens
    # (no "_code" suffix) rather than raising.
    assert lemmatizer_available("en") is False
    tokens = tokenize("running dogs", lemmatize=True, pos_tag=True)
    assert tokens == ["running", "dogs"]


@pytest.mark.skipif(
    not lemmatizer_available("en"),
    reason="requires a trained en pipeline (python -m spacy download en_core_web_sm)",
)
def test_pos_tag_distinguishes_noun_and_verb_usage():
    # "record" used as a verb vs. a noun should get different POS codes,
    # since that's the whole point of pos_tag -- see tokenizer.
    # POS_DISPLAY_NAMES for what each code means.
    tokens = tokenize(
        "He records the record.", lemmatize=True, pos_tag=True,
    )
    assert tokens == ["he_p", "record_v", "the_d", "record_n"]


# ---------- classify_tokens ----------

def test_classify_tokens_tags_plain_words():
    assert classify_tokens("Hello world") == [("hello", "word"), ("world", "word")]


def test_classify_tokens_tags_digits_and_keeps_pure_digits():
    # Unlike tokenize(), a pure-digit token isn't dropped -- it's kept and
    # categorized "digit" instead.
    tokens = classify_tokens("I have 42 apples and 3.14 pies")
    assert ("42", "digit") in tokens
    assert ("3.14", "digit") in tokens
    assert ("apples", "word") in tokens


def test_tokenize_still_drops_digits_via_classify_tokens():
    # tokenize() is built on classify_tokens() but filters "digit" out, so
    # its documented behavior (pure digit tokens are dropped) is unchanged.
    assert tokenize("I have 42 apples") == ["i", "have", "apples"]


def test_classify_tokens_number_word_is_not_a_digit():
    # Spelled-out number words aren't digit tokens (unlike spaCy's
    # like_num) -- they stay ordinary vocabulary.
    tokens = classify_tokens("twelve apples")
    assert ("twelve", "word") in tokens
    assert ("apples", "word") in tokens


def test_classify_tokens_no_pipeline_never_tags_proper_noun():
    # Without a trained pipeline, proper-noun detection can't run (same
    # silent-fallback convention as lemmatize/pos_tag), so even an
    # obviously-capitalized name is just "word".
    tokens = classify_tokens("Everest is tall", language="zznotalang")
    assert ("everest", "word") in tokens


@pytest.mark.skipif(
    not lemmatizer_available("en"),
    reason="requires a trained en pipeline (python -m spacy download en_core_web_sm)",
)
def test_classify_tokens_detects_proper_nouns_with_trained_pipeline():
    tokens = classify_tokens("Brett visited Paris")
    by_word = dict(tokens)
    assert by_word["brett"] == "proper_noun"
    assert by_word["paris"] == "proper_noun"
    assert by_word["visited"] == "word"
