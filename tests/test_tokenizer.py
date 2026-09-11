import pytest

from lexical_profiler.tokenizer import lemmatizer_available, tokenize


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
