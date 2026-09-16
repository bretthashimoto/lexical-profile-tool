import pytest

from lexical_profiler.reference import (
    Reference,
    compute_band_assignment,
    require_txt_extension,
)

# ---------- compute_band_assignment ----------

def test_compute_band_assignment_uniform_bands():
    band_of_position, band_ranges = compute_band_assignment(2500, band_size=1000)
    assert band_ranges == {1: (1, 1000), 2: (1001, 2000), 3: (2001, 2500)}
    assert band_of_position[0] == 1
    assert band_of_position[999] == 1
    assert band_of_position[1000] == 2
    assert band_of_position[-1] == 3


def test_compute_band_assignment_fine_grained_then_uniform():
    band_of_position, band_ranges = compute_band_assignment(
        2200, band_size=1000, fine_band_size=100, fine_grained_until=2000,
    )
    # 20 fine bands of 100 covering ranks 1-2000, then one more band for 2001-2200.
    assert len(band_ranges) == 21
    assert band_ranges[1] == (1, 100)
    assert band_ranges[20] == (1901, 2000)
    assert band_ranges[21] == (2001, 2200)
    assert band_of_position[0] == 1
    assert band_of_position[2000] == 21


def test_compute_band_assignment_coarse_grained_tail():
    band_of_position, band_ranges = compute_band_assignment(
        12000, band_size=1000, coarse_band_size=5000, coarse_grained_from=8000,
    )
    # Normal 1000-word bands for ranks 1-7999 (the last one truncated to
    # end right where the coarse section starts), then one 4001-wide
    # coarse band covering the rest (8000-12000) instead of several more
    # normal bands -- coarse_band_size is a ceiling, not a fixed width.
    assert band_ranges[8] == (7001, 7999)
    assert band_ranges[9] == (8000, 12000)
    assert len(band_ranges) == 9
    assert band_of_position[7999] == 9  # rank 8000
    assert band_of_position[-1] == 9  # rank 12000


def test_compute_band_assignment_fine_and_coarse_together():
    band_of_position, band_ranges = compute_band_assignment(
        10000, band_size=1000,
        fine_band_size=100, fine_grained_until=1000,
        coarse_band_size=5000, coarse_grained_from=8000,
    )
    # 10 fine bands (1-1000), normal bands for 1001-7999, one coarse band
    # for the rest.
    assert band_ranges[1] == (1, 100)
    assert band_ranges[10] == (901, 1000)
    assert band_ranges[11] == (1001, 2000)
    last_band = max(band_ranges)
    assert band_ranges[last_band] == (8000, 10000)
    assert band_of_position[0] == 1
    assert band_of_position[-1] == last_band


def test_compute_band_assignment_coarse_grained_from_overlapping_fine_section():
    # coarse_grained_from falls inside the fine-grained section -- the
    # coarse section should start right after the fine one ends instead
    # of overlapping it.
    band_of_position, band_ranges = compute_band_assignment(
        5000, band_size=1000,
        fine_band_size=500, fine_grained_until=3000,
        coarse_band_size=777, coarse_grained_from=1000,
    )
    assert band_ranges[6] == (2501, 3000)  # last fine-grained band
    assert band_ranges[7] == (3001, 3777)  # coarse band starts right after
    assert band_of_position[3000] == 7  # rank 3001 (0-indexed 3000)


def test_require_txt_extension_rejects_non_txt():
    with pytest.raises(ValueError):
        require_txt_extension("notes.pdf")
    require_txt_extension("notes.txt")  # doesn't raise


# ---------- Reference.from_corpus ----------

def test_from_corpus_ranks_by_frequency():
    ref = Reference.from_corpus(["the the the cat cat dog"], band_size=1000)
    assert ref.rank_of("the") == 1
    assert ref.rank_of("cat") == 2
    assert ref.rank_of("dog") == 3
    assert ref.band_of("the") == 1
    assert len(ref) == 3


def test_from_corpus_empty_source_raises():
    with pytest.raises(ValueError):
        Reference.from_corpus([])


def test_from_corpus_no_usable_words_raises():
    with pytest.raises(ValueError):
        Reference.from_corpus(["123 456 !!!"])


def test_from_corpus_reads_directory(tmp_path):
    (tmp_path / "a.txt").write_text("cat cat dog", encoding="utf-8")
    (tmp_path / "b.txt").write_text("dog bird", encoding="utf-8")
    ref = Reference.from_corpus(str(tmp_path), band_size=1000)
    assert ref.rank_of("cat") == 1  # cat=2, dog=2; tie broken by Counter order
    assert "bird" in ref


def test_from_corpus_rejects_non_txt_file(tmp_path):
    bad = tmp_path / "notes.pdf"
    bad.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        Reference.from_corpus(str(bad))


# ---------- Reference.from_word_list ----------

def test_from_word_list_plain_words(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("the\nbe\nto\n", encoding="utf-8")
    ref = Reference.from_word_list(str(path), band_size=1000)
    assert ref.rank_of("the") == 1
    assert ref.rank_of("be") == 2
    assert ref.rank_of("to") == 3
    assert ref.counts == {}  # rank-only, no frequency data


def test_from_word_list_with_frequencies_sorts_by_freq(tmp_path):
    path = tmp_path / "freqs.txt"
    # Deliberately out of order; should be re-sorted by frequency desc.
    path.write_text("be\t50\nthe\t100\nto\t10\n", encoding="utf-8")
    ref = Reference.from_word_list(str(path), band_size=1000)
    assert ref.rank_of("the") == 1
    assert ref.rank_of("be") == 2
    assert ref.rank_of("to") == 3
    assert ref.counts["the"] == 100


def test_from_word_list_coarse_grained_tail(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("\n".join(f"word{i}" for i in range(30)), encoding="utf-8")
    ref = Reference.from_word_list(
        str(path), band_size=5, coarse_band_size=10, coarse_grained_from=20,
    )
    assert ref.coarse_band_size == 10
    assert ref.coarse_grained_from == 20
    # Normal 5-word bands for ranks 1-19 (word0-word18), then a 10-word
    # coarse band for ranks 20-29 (word19-word28), then a final 1-word
    # band for the last leftover rank (word29).
    assert ref.band_of("word18") == 4  # rank 19, last of the normal section
    assert ref.band_of("word19") == 5  # rank 20, first of the coarse section
    assert ref.band_of("word28") == 5  # rank 29, still in the coarse band
    assert ref.band_of("word29") == 6  # rank 30, leftover past the coarse band
    assert ref.num_bands == 6


def test_from_word_list_comma_delimiter_autodetected(tmp_path):
    path = tmp_path / "freqs.txt"
    path.write_text("the,100\nbe,50\n", encoding="utf-8")
    ref = Reference.from_word_list(str(path), band_size=1000)
    assert ref.rank_of("the") == 1
    assert ref.counts["the"] == 100


def test_from_word_list_missing_file_raises():
    with pytest.raises(ValueError):
        Reference.from_word_list("does/not/exist.txt")


def test_from_word_list_empty_file_raises(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        Reference.from_word_list(str(path))


# ---------- Reference.from_builtin ----------

def test_from_builtin_loads_avl():
    ref = Reference.from_builtin("avl", band_size=100)
    assert "study_n" in ref  # a high-frequency AVL word (lemma + POS)
    assert ref.rank_of("study_n") == 1
    assert len(ref) > 2000
    assert "avl" in ref.source_description.lower()


def test_from_builtin_is_case_insensitive():
    ref = Reference.from_builtin("AVL", band_size=100)
    assert "study_n" in ref


def test_from_builtin_avl_is_pos_tagged():
    ref = Reference.from_builtin("avl", band_size=1000)
    assert ref.pos_tagged is True
    assert ref.lemmatize is True  # pos_tagged forces this on
    # "group" as a noun and as a verb are distinct entries, not merged.
    assert "group_n" in ref
    assert "group_v" in ref
    assert ref.rank_of("group_n") != ref.rank_of("group_v")


def test_from_builtin_unknown_name_raises():
    with pytest.raises(ValueError):
        Reference.from_builtin("not-a-real-list")


def test_from_builtin_coca_is_pos_tagged():
    ref = Reference.from_builtin("coca", band_size=1000)
    assert ref.pos_tagged is True
    assert ref.lemmatize is True  # pos_tagged forces this on
    # "record" as a noun and as a verb are distinct entries, not merged.
    assert "record_n" in ref
    assert "record_v" in ref
    assert ref.rank_of("record_n") != ref.rank_of("record_v")


# ---------- pos_tagged ----------

def test_from_word_list_pos_tagged_forces_lemmatize(tmp_path):
    path = tmp_path / "pos_words.txt"
    path.write_text("record_n\t100\nrecord_v\t50\n", encoding="utf-8")
    ref = Reference.from_word_list(str(path), band_size=1000, pos_tagged=True)
    assert ref.pos_tagged is True
    assert ref.lemmatize is True
    assert ref.rank_of("record_n") == 1
    assert ref.rank_of("record_v") == 2


def test_from_corpus_pos_tagged_forces_lemmatize():
    # No trained pipeline is installed, so pos_tag silently degrades to
    # plain tokens -- this just checks the flag/forcing behavior, not
    # real tagging (see test_tokenizer.py for that, gated on a real
    # pipeline being available).
    ref = Reference.from_corpus(["cat dog"], band_size=1000, pos_tagged=True)
    assert ref.pos_tagged is True
    assert ref.lemmatize is True


# ---------- save / load ----------

def test_save_and_load_round_trip(tmp_path):
    ref = Reference.from_corpus(["the the the cat cat dog"], band_size=1000)
    save_path = tmp_path / "reference.json"
    ref.save(str(save_path))

    loaded = Reference.load(str(save_path))
    assert loaded.rank_of("the") == ref.rank_of("the")
    assert loaded.band_of("cat") == ref.band_of("cat")
    assert loaded.num_bands == ref.num_bands
    assert loaded.language == ref.language


def test_save_and_load_round_trip_preserves_pos_tagged(tmp_path):
    ref = Reference.from_corpus(["cat dog"], band_size=1000, pos_tagged=True)
    save_path = tmp_path / "reference.json"
    ref.save(str(save_path))

    loaded = Reference.load(str(save_path))
    assert loaded.pos_tagged is True
    assert loaded.lemmatize == ref.lemmatize


def test_save_and_load_round_trip_preserves_coarse_bands(tmp_path):
    # 20 words with strictly decreasing frequency, so rank order is stable.
    text = " ".join(f"word{i}" for i in range(20) for _ in range(20 - i))
    ref = Reference.from_corpus(
        [text], band_size=5, coarse_band_size=10, coarse_grained_from=15,
    )
    save_path = tmp_path / "reference.json"
    ref.save(str(save_path))

    loaded = Reference.load(str(save_path))
    assert loaded.coarse_band_size == 10
    assert loaded.coarse_grained_from == 15
    assert loaded.band_ranges == ref.band_ranges


def test_load_missing_file_raises():
    with pytest.raises(ValueError):
        Reference.load("does/not/exist.json")


def test_load_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not valid json{{{", encoding="utf-8")
    with pytest.raises(ValueError):
        Reference.load(str(path))
