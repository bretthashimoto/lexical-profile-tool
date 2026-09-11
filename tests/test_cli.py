import json

import pytest

from lexical_profiler.cli import main


@pytest.fixture
def wordlist_path(tmp_path):
    path = tmp_path / "wordlist.txt"
    path.write_text("the\nbe\nto\ncat\ndog\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def target_path(tmp_path):
    path = tmp_path / "essay.txt"
    path.write_text("the cat mouse", encoding="utf-8")
    return str(path)


def test_cli_runs_and_prints_summary(wordlist_path, target_path, capsys):
    main([
        "--reference-wordlist", wordlist_path,
        "--target", target_path,
    ])
    captured = capsys.readouterr()
    assert "essay.txt" in captured.out
    assert "Tokens: 3" in captured.out


def test_cli_writes_json_report(wordlist_path, target_path, tmp_path):
    out_path = tmp_path / "report.json"
    main([
        "--reference-wordlist", wordlist_path,
        "--target", target_path,
        "--out-json", str(out_path),
    ])
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "essay.txt" in payload


def test_cli_requires_a_target(wordlist_path):
    with pytest.raises(SystemExit):
        main(["--reference-wordlist", wordlist_path])


def test_cli_requires_exactly_one_reference_source(wordlist_path, target_path):
    with pytest.raises(SystemExit):
        main([
            "--reference-wordlist", wordlist_path,
            "--reference-corpus", wordlist_path,
            "--target", target_path,
        ])


def test_cli_reference_builtin(target_path, capsys):
    main([
        "--reference-builtin", "avl",
        "--target", target_path,
    ])
    captured = capsys.readouterr()
    assert "essay.txt" in captured.out


def test_cli_rejects_non_txt_target(wordlist_path, tmp_path):
    bad_target = tmp_path / "essay.pdf"
    bad_target.write_text("the cat mouse", encoding="utf-8")
    with pytest.raises(SystemExit):
        main([
            "--reference-wordlist", wordlist_path,
            "--target", str(bad_target),
        ])
