from lexical_profiler.tokenizer import tokenize, lemmatizer_available


def test_tokenize_lowercases_by_default():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_preserves_case_when_disabled():
    assert tokenize("Hello World", lowercase=False) == ["Hello", "World"]


def test_tokenize_drops_pure_punctuation_and_numbers():
    tokens = tokenize("Hello, world! 123 -- 456.")
    assert tokens == ["hello", "world"]


def test_tokenize_keeps_tokens_with_internal_punctuation():
    # spaCy's tokenizer splits "don't" into "do" / "n't" -- both pieces
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
    # 'xx' / nonsense codes shouldn't raise -- should fall back to the
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
