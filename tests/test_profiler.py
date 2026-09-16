import pytest

from lexical_profiler.profiler import LexicalProfiler
from lexical_profiler.reference import Reference
from lexical_profiler.tokenizer import lemmatizer_available


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
    # Ignored words are excluded from totals.
    assert result.total_tokens == 2
    assert result.ignored_pct_tokens == pytest.approx(100 / 3)


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


def test_profile_texts_reports_progress(small_reference):
    profiler = LexicalProfiler(small_reference)
    calls = []
    results = profiler.profile_texts(
        {"a": "the cat", "b": "dog mouse"},
        progress_callback=lambda current, total, message: calls.append(
            (current, total, message)
        ),
    )
    assert set(results.keys()) == {"a", "b"}
    # One initial call at 0, then one call per text as it finishes, in order.
    assert [c[0] for c in calls] == [0, 1, 2]
    assert all(c[1] == 2 for c in calls)
    assert "a" in calls[1][2]
    assert "b" in calls[2][2]


def test_profile_texts_without_progress_callback_still_works(small_reference):
    # progress_callback is optional; omitting it must not raise.
    profiler = LexicalProfiler(small_reference)
    results = profiler.profile_texts({"a": "the cat"})
    assert results["a"].total_tokens == 2


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


def test_cumulative_token_pct_runs_across_bands(small_reference):
    profiler = LexicalProfiler(small_reference)
    result = profiler.profile_text("the cat mouse")

    # band1="the" (1/3 tokens), band2="cat" (1/3 tokens), band3=none.
    assert result.cumulative_token_pct[1] == pytest.approx(100 / 3)
    assert result.cumulative_token_pct[2] == pytest.approx(200 / 3)
    assert result.cumulative_token_pct[3] == pytest.approx(200 / 3)


def test_summary_and_to_dict(small_reference):
    profiler = LexicalProfiler(small_reference)
    result = profiler.profile_text("the cat mouse")

    summary = result.summary()
    assert "Tokens: 3" in summary
    assert "mouse" in summary
    assert "Cum %" in summary

    as_dict = result.to_dict()
    assert as_dict["total_tokens"] == 3
    assert as_dict["off_list_words"] == ["mouse"]
    assert as_dict["cumulative_token_pct"]["1-1"] == pytest.approx(100 / 3)


def test_highlight_classifies_and_preserves_original_text(small_reference):
    profiler = LexicalProfiler(small_reference, ignore_words=["mouse"])
    text = "The cat, mouse!"
    tokens = profiler.highlight(text)

    # Reconstructing text + whitespace for every token must give back the
    # original string exactly, so rendering never drops/garbles anything.
    assert "".join(t.text + t.whitespace for t in tokens) == text

    by_text = {t.text: t for t in tokens}
    assert by_text["The"].status == "band"
    assert by_text["The"].band == 1  # "the" is band 1 in small_reference
    assert by_text["cat"].status == "band"
    assert by_text["cat"].band == 2
    assert by_text["mouse"].status == "ignored"
    assert by_text[","].status == "skipped"
    assert by_text["!"].status == "skipped"


def test_highlight_off_list_word(small_reference):
    profiler = LexicalProfiler(small_reference)
    tokens = profiler.highlight("the zebra")
    by_text = {t.text: t for t in tokens}
    assert by_text["zebra"].status == "off_list"
    assert by_text["zebra"].band is None


def test_highlight_digit_included_by_default(small_reference):
    profiler = LexicalProfiler(small_reference)
    tokens = profiler.highlight("the 42")
    by_text = {t.text: t for t in tokens}
    # Not excluded, so it's just profiled normally -- lands off-list since
    # small_reference has no digit vocabulary.
    assert by_text["42"].status == "off_list"


def test_highlight_digit_excluded(small_reference):
    profiler = LexicalProfiler(small_reference, exclude_digits=True)
    tokens = profiler.highlight("the 42")
    by_text = {t.text: t for t in tokens}
    assert by_text["42"].status == "digit"


# ---------- proper nouns / digits ----------

def test_digits_included_by_default_and_counted_off_list(small_reference):
    # exclude_digits defaults to False: digit tokens are profiled like any
    # other word -- and since no reference lists them, they land off-list
    # rather than vanishing the way they used to (pre-classify_tokens).
    profiler = LexicalProfiler(small_reference)
    result = profiler.profile_text("the cat 42")

    assert result.total_tokens == 3
    assert result.digit_tokens == 0
    assert result.digit_words == []
    assert "42" in result.off_list_words
    assert result.digit_pct_tokens == 0.0


def test_exclude_digits_reports_them_separately(small_reference):
    profiler = LexicalProfiler(small_reference, exclude_digits=True)
    result = profiler.profile_text("the cat 42 42 100")

    assert result.digit_tokens == 3
    assert result.digit_types == 2
    assert set(result.digit_words) == {"42", "100"}
    assert "42" not in result.off_list_words
    # Digits are excluded from totals, same convention as ignored_words.
    assert result.total_tokens == 2
    assert result.off_list_tokens == 0
    assert result.digit_pct_tokens == pytest.approx(60.0)


def test_exclude_proper_nouns_bypassed_without_trained_pipeline(small_reference):
    # _profile_tokens is fed pre-classified tokens directly here, so this
    # exercises the bucketing logic without needing spaCy's tagger.
    profiler = LexicalProfiler(small_reference, exclude_proper_nouns=True)
    result = profiler._profile_tokens([
        ("the", "word"), ("brett", "proper_noun"), ("zebra", "word"),
    ])

    assert result.proper_noun_tokens == 1
    assert result.proper_noun_words == ["brett"]
    assert "brett" not in result.off_list_words
    assert result.off_list_words == ["zebra"]
    assert result.total_tokens == 2
    assert result.proper_noun_pct_tokens == pytest.approx(100 / 3)


def test_ignore_words_takes_priority_over_exclude_categories(small_reference):
    # A word matching both ignore_words and an auto-detected category
    # should be reported as ignored, not proper_noun/digit.
    profiler = LexicalProfiler(
        small_reference, ignore_words=["brett"], exclude_proper_nouns=True,
    )
    result = profiler._profile_tokens([("brett", "proper_noun")])

    assert result.ignored_words == ["brett"]
    assert result.proper_noun_words == []


# ---------- pos_tagged matching ----------

@pytest.fixture
def pos_tagged_reference(tmp_path):
    # band_size=1 so record_n and record_v land in different bands,
    # independent of any real spaCy tagging -- exercises the pure
    # dict-matching side of pos_tagged, not tokenization itself.
    path = tmp_path / "pos_words.txt"
    path.write_text("record_n\t100\nrecord_v\t50\nbanana_n\t10\n", encoding="utf-8")
    return Reference.from_word_list(str(path), band_size=1, pos_tagged=True)


def test_profile_tokens_matches_by_lemma_and_pos(pos_tagged_reference):
    profiler = LexicalProfiler(pos_tagged_reference)
    # Bypass _classify (which needs a real spaCy pipeline to actually tag
    # POS) and feed already-tagged (word, category) pairs directly, to
    # test the matching logic on its own: record_n (band 1) and record_v
    # (band 2) must be scored as distinct words, not merged.
    result = profiler._profile_tokens([
        ("record_n", "word"), ("record_v", "word"),
        ("record_n", "word"), ("apple_n", "word"),
    ])

    assert result.band_token_counts[1] == 2  # record_n x2
    assert result.band_token_counts[2] == 1  # record_v x1
    assert result.off_list_words == ["apple_n"]
    assert result.word_counts["record_n"] == 2
    assert result.word_counts["record_v"] == 1


@pytest.mark.skipif(
    not lemmatizer_available("en"),
    reason="requires a trained en pipeline (python -m spacy download en_core_web_sm)",
)
def test_profile_text_end_to_end_distinguishes_noun_and_verb(pos_tagged_reference):
    profiler = LexicalProfiler(pos_tagged_reference)
    result = profiler.profile_text("He records the record.")
    assert result.word_counts["record_v"] == 1  # "records" (verb)
    assert result.word_counts["record_n"] == 1  # "record" (noun)


@pytest.mark.skipif(
    not lemmatizer_available("en"),
    reason="requires a trained en pipeline (python -m spacy download en_core_web_sm)",
)
def test_highlight_end_to_end_distinguishes_noun_and_verb(pos_tagged_reference):
    profiler = LexicalProfiler(pos_tagged_reference)
    text = "He records the record."
    tokens = profiler.highlight(text)
    assert "".join(t.text + t.whitespace for t in tokens) == text

    by_text = {t.text: t for t in tokens}
    assert by_text["records"].status == "band"
    assert by_text["record"].status == "band"
    assert by_text["records"].band != by_text["record"].band
