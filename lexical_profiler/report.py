"""
Export helpers for turning ProfileResult objects into CSV / JSON reports.
"""

from __future__ import annotations

import csv
import json

from .profiler import ProfileResult


def _open_output_file(path: str, newline: str | None = None):
    """Open `path` for writing, turning common failures (missing parent
    folder, no write permission, path is a folder) into a plain-language
    ValueError instead of a raw OS-level traceback.
    """
    try:
        return open(path, "w", encoding="utf-8", newline=newline)
    except FileNotFoundError:
        raise ValueError(
            f"Can't write to '{path}': the folder it's supposed to go "
            f"in doesn't exist. Create that folder first, or choose an "
            f"output path in a folder that already exists."
        ) from None
    except PermissionError:
        raise ValueError(
            f"Don't have permission to write to '{path}'. Check that the "
            f"file isn't open in another program and that you have write "
            f"access to that folder."
        ) from None
    except IsADirectoryError:
        trimmed = path.rstrip("/\\")
        raise ValueError(
            f"'{path}' is a folder, not a file path. Include a filename "
            f"in the path, e.g. '{trimmed}/results.csv'."
        ) from None


def export_json(results: dict[str, ProfileResult], path: str) -> None:
    """Write one or more named results to a JSON file.

    `results` maps a label (e.g. filename) to a ProfileResult.
    """
    payload = {name: r.to_dict() for name, r in results.items()}
    with _open_output_file(path) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_csv(results: dict[str, ProfileResult], path: str) -> None:
    """Write a tidy CSV: one row per (text, band), plus an off-list row.

    Columns: text, band, tokens, pct_tokens, cumulative_pct_tokens, types, pct_types
    """
    with _open_output_file(path, newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "text", "band", "tokens", "pct_tokens", "cumulative_pct_tokens",
            "types", "pct_types",
        ])
        for name, r in results.items():
            # One row per numbered frequency band, in order (band 1 first).
            for band in sorted(r.band_token_counts.keys()):
                writer.writerow([
                    name,
                    r.band_label(band),
                    r.band_token_counts[band],
                    f"{r.band_token_pct[band]:.4f}",
                    f"{r.cumulative_token_pct[band]:.4f}",
                    r.band_type_counts.get(band, 0),
                    f"{r.band_type_pct.get(band, 0):.4f}",
                ])
            # Off-list words aren't in any band, so they get one extra
            # summary row instead of being folded into the band rows above.
            # cumulative_pct_tokens is left blank here (as for "ignored"
            # below) since cumulative coverage is only defined over the
            # band 1..N progression, not these out-of-band buckets.
            writer.writerow([
                name, "off_list", r.off_list_tokens,
                f"{r.off_list_pct_tokens:.4f}", "", r.off_list_types, "",
            ])
            # Only add the "ignored" row if an ignore list was actually
            # used and matched something; keeps the CSV unchanged for
            # callers who never pass ignore_words.
            if r.ignored_tokens or r.ignored_words:
                writer.writerow([
                    name, "ignored", r.ignored_tokens,
                    f"{r.ignored_pct_tokens:.4f}", "", r.ignored_types, "",
                ])


def export_off_list_csv(results: dict[str, ProfileResult], path: str) -> None:
    """Write a CSV of off-list (unknown) words per text, with counts."""
    with _open_output_file(path, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "word", "count"])
        for name, r in results.items():
            for word in r.off_list_words:
                writer.writerow([name, word, r.word_counts[word]])


def export_ignored_csv(results: dict[str, ProfileResult], path: str) -> None:
    """Write a CSV of ignored words per text, with counts."""
    with _open_output_file(path, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "word", "count"])
        for name, r in results.items():
            for word in r.ignored_words:
                writer.writerow([name, word, r.word_counts[word]])
