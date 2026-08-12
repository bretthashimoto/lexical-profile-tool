import pytest

from lexical_profiler.profiler import LexicalProfiler
from lexical_profiler.reference import Reference


@pytest.fixture
def small_reference():
    # band_size=1 puts each of the 3 words in its own band: the=1, cat=2, dog=3.
    return Reference.from_corpus(["the the the cat cat dog"], band_size=1)


def test_profile_text_classifies_bands_and_off_list(small_reference):
    profiler = LexicalProfiler(small_reference)
    result = profiler.profile_text("the cat mouse")

    assert result.total_tokens == 3
    assert result.total_types == 3
    assert result.band_token_counts[1] == 1  # "the"
    assert result.band_token_counts[2] == 1  # "cat"
    assert result.band_token_counts[3] == 0
    assert result.off_list_tokens == 1
    assert result.off_list_words == ["mouse"]
    assert result.ignored_tokens == 0


def test_profile_text_with_ignore_words(small_reference):
    profiler = LexicalProfiler(small_reference, ignore_words=["mouse"])
    result = profiler.profile_text("the cat mouse")

    assert result.off_list_tokens == 0
    assert result.off_list_words == []
    assert result.ignored_tokens == 1
    assert result.ignored_words == ["mouse"]
    # Ignored words still count toward totals.
    assert result.total_tokens == 3


def test_profile_text_repeated_words_counted_correctly(small_reference):
    profiler = LexicalProfiler(small_reference)
    result = profiler.profile_text("mouse mouse mouse cat")

    assert result.total_tokens == 4
    assert result.total_types == 2
    assert result.off_list_tokens == 3
    assert result.off_list_types == 1
    assert result.band_token_counts[2] == 1


def test_profile_texts_are_independent(small_reference):
    profiler = LexicalProfiler(small_reference)
    results = profiler.profile_texts({
        "a": "the the cat",
        "b": "dog mouse",
    })
    assert set(results.keys()) == {"a", "b"}
    assert results["a"].total_tokens == 3
    assert results["b"].total_tokens == 2
    assert results["b"].off_list_words == ["mouse"]


def test_profile_document(tmp_path, small_reference):
    target = tmp_path / "essay.txt"
    target.write_text("the cat mouse", encoding="utf-8")
    profiler = LexicalProfiler(small_reference)
    result = profiler.profile_document(str(target))
    assert result.total_tokens == 3


def test_profile_document_rejects_non_txt(tmp_path, small_reference):
    target = tmp_path / "essay.pdf"
    target.write_text("the cat mouse", encoding="utf-8")
    profiler = LexicalProfiler(small_reference)
    with pytest.raises(ValueError):
        profiler.profile_document(str(target))


def test_profile_corpus(tmp_path, small_reference):
    (tmp_path / "one.txt").write_text("the cat", encoding="utf-8")
    (tmp_path / "two.txt").write_text("dog mouse", encoding="utf-8")
    profiler = LexicalProfiler(small_reference)
    results = profiler.profile_corpus(str(tmp_path))
    assert set(results.keys()) == {"one.txt", "two.txt"}
    assert results["two.txt"].off_list_words == ["mouse"]


def test_profile_corpus_missing_directory_raises(small_reference):
    profiler = LexicalProfiler(small_reference)
    with pytest.raises(ValueError):
        profiler.profile_corpus("does/not/exist")


def test_profile_corpus_empty_directory_raises(tmp_path, small_reference):
    profiler = LexicalProfiler(small_reference)
    with pytest.raises(ValueError):
        profiler.profile_corpus(str(tmp_path))


def test_summary_and_to_dict(small_reference):
    profiler = LexicalProfiler(small_reference)
    result = profiler.profile_text("the cat mouse")

    summary = result.summary()
    assert "Tokens: 3" in summary
    assert "mouse" in summary

    as_dict = result.to_dict()
    assert as_dict["total_tokens"] == 3
    assert as_dict["off_list_words"] == ["mouse"]
