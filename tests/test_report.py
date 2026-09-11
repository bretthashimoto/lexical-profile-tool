import csv
import json

import pytest

from lexical_profiler import report as report_mod
from lexical_profiler.profiler import LexicalProfiler
from lexical_profiler.reference import Reference


@pytest.fixture
def results():
    reference = Reference.from_corpus(["the the the cat cat dog"], band_size=1)
    profiler = LexicalProfiler(
        reference, ignore_words=["ignoreme"], exclude_numerals=True,
    )
    return profiler.profile_texts({
        "essay.txt": "the cat mouse ignoreme 42",
    })


def test_export_json(tmp_path, results):
    path = tmp_path / "out.json"
    report_mod.export_json(results, str(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "essay.txt" in payload
    assert payload["essay.txt"]["total_tokens"] == 5
    assert payload["essay.txt"]["off_list_words"] == ["mouse"]
    assert payload["essay.txt"]["numeral_words"] == ["42"]


def test_export_csv(tmp_path, results):
    path = tmp_path / "out.csv"
    report_mod.export_csv(results, str(path))

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == [
        "text", "band", "tokens", "pct_tokens", "cumulative_pct_tokens",
        "types", "pct_types",
    ]
    body_texts = {row[0] for row in rows[1:]}
    body_categories = {row[1] for row in rows[1:]}
    assert body_texts == {"essay.txt"}
    assert "off_list" in body_categories
    assert "ignored" in body_categories  # ignore_words matched something
    assert "numeral" in body_categories  # exclude_numerals matched something


def test_export_off_list_csv(tmp_path, results):
    path = tmp_path / "off_list.csv"
    report_mod.export_off_list_csv(results, str(path))

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["text", "word", "count"]
    assert rows[1] == ["essay.txt", "mouse", "1"]


def test_export_ignored_csv(tmp_path, results):
    path = tmp_path / "ignored.csv"
    report_mod.export_ignored_csv(results, str(path))

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["text", "word", "count"]
    assert rows[1] == ["essay.txt", "ignoreme", "1"]


def test_export_numerals_csv(tmp_path, results):
    path = tmp_path / "numerals.csv"
    report_mod.export_numerals_csv(results, str(path))

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["text", "word", "count"]
    assert rows[1] == ["essay.txt", "42", "1"]


def test_export_proper_nouns_csv_empty_when_none_excluded(tmp_path, results):
    # The fixture profiler doesn't set exclude_proper_nouns, so this
    # export is just a header with no rows -- no proper nouns were ever
    # broken out.
    path = tmp_path / "proper_nouns.csv"
    report_mod.export_proper_nouns_csv(results, str(path))

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows == [["text", "word", "count"]]


def test_export_json_bad_directory_raises(results):
    with pytest.raises(ValueError):
        report_mod.export_json(results, "does/not/exist/out.json")
