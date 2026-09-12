"""
Command-line interface.

Examples:

    # Build reference from a corpus directory, profile one target file
    python -m lexical_profiler \\
        --reference-corpus corpus/ \\
        --target essay.txt

    # Use an existing word-frequency list as the reference instead
    python -m lexical_profiler \\
        --reference-wordlist wordlists/coca_20k.txt \\
        --target essay.txt --band-size 1000

    # Or use a word list bundled with this package (e.g. the Academic
    # Vocabulary List) instead of supplying your own file
    python -m lexical_profiler \\
        --reference-builtin avl \\
        --target essay.txt

    # Profile every file in a directory, export CSV + JSON
    python -m lexical_profiler \\
        --reference-corpus corpus/ \\
        --target-dir essays/ \\
        --out-csv report.csv --out-json report.json
"""

from __future__ import annotations

import argparse
import os
import sys

from . import report as report_mod
from .profiler import LexicalProfiler
from .reference import BUILTIN_WORD_LISTS, Reference, open_text_file, require_txt_extension


def _collect_ignore_words(args) -> list:
    """Mecrge --ignore-words (given directly on the command line) with
    --ignore-list (one word per line in a file), if both were passed."""
    words = list(args.ignore_words) if args.ignore_words else []
    if args.ignore_list:
        require_txt_extension(args.ignore_list)
        with open_text_file(args.ignore_list, args.encoding) as f:
            words.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
    return words


def _collect_targets(args) -> dict:
    """Read every target text file requested via --target and/or
    --target-dir into a {filename: content} dict for profiling.

    Only .txt files are accepted; anything else raises ValueError.
    """
    targets = {}
    if args.target:
        for path in args.target:
            require_txt_extension(path)
            with open_text_file(path, args.encoding) as f:
                targets[os.path.basename(path)] = f.read()
    if args.target_dir:
        # os.listdir() on a path that doesn't exist raises a fairly
        # low-level FileNotFoundError, so check up front that a typo'd
        # --target-dir gives a clearer message instead.
        if not os.path.isdir(args.target_dir):
            raise ValueError(
                f"Can't find the folder '{args.target_dir}' given to "
                f"--target-dir. Check the path is correct, and that it's "
                f"a folder, not a file (use --target for a single file)."
            )
        for fname in sorted(os.listdir(args.target_dir)):
            fpath = os.path.join(args.target_dir, fname)
            if os.path.isfile(fpath):
                require_txt_extension(fpath)
                with open_text_file(fpath, args.encoding) as f:
                    targets[fname] = f.read()
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="lexical_profiler",
        description="Lexical frequency profiling of texts against a corpus- or "
                     "list-derived reference frequency model.",
    )
    # Exactly one way of getting a reference model must be chosen --
    # argparse enforces that here so we don't have to check it by hand.
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument("--reference-corpus", metavar="PATH",
                            help="File or directory of texts to derive reference frequencies from.")
    ref_group.add_argument("--reference-wordlist", metavar="PATH",
                            help="Existing word list file (one word per line, or 'word,freq').")
    ref_group.add_argument(
        "--reference-builtin", metavar="NAME", choices=sorted(BUILTIN_WORD_LISTS),
        help="Use a word list bundled with this package instead of your own file. "
             "Available: " + ", ".join(
                 f"{name} ({entry['description']})"
                 for name, entry in sorted(BUILTIN_WORD_LISTS.items())
             ),
    )
    ref_group.add_argument("--reference-saved", metavar="PATH",
                            help="A previously saved reference (from --save-reference) to reuse.")

    parser.add_argument("--save-reference", metavar="PATH",
                         help="Save the built reference model to this JSON path for reuse.")

    target_group = parser.add_argument_group("target text(s) to profile")
    target_group.add_argument("--target", nargs="*", metavar="FILE",
                               help="One or more target text files to profile.")
    target_group.add_argument("--target-dir", metavar="DIR",
                               help="Directory of target text files to profile "
                                    "(each profiled independently).")

    parser.add_argument("--band-size", type=int, default=1000,
                         help="Words per frequency band (default: 1000).")
    parser.add_argument("--fine-band-size", type=int, default=None,
                         help="Optional narrower band width for the most frequent words "
                              "(e.g. 100), used up through --fine-until. Must be combined "
                              "with --fine-until.")
    parser.add_argument("--fine-until", type=int, default=None,
                         help="Rank up to which --fine-band-size applies (e.g. 2000 gives "
                              "100-word bands through rank 2000, then normal --band-size "
                              "bands after that).")
    parser.add_argument("--coarse-band-size", type=int, default=None,
                         help="Optional wider band width for the least frequent words "
                              "(e.g. 10000), used from --coarse-from through the end of "
                              "the list. Must be combined with --coarse-from.")
    parser.add_argument("--coarse-from", type=int, default=None,
                         help="Rank from which --coarse-band-size applies (e.g. 50000 "
                              "collapses the long tail past rank 50000 into a handful "
                              "of wide bands instead of many normal --band-size ones).")
    parser.add_argument("--language", default="en",
                         help="ISO 639-1 language code (e.g. en, es, de, fr, zh, ja, ru) "
                              "or a full spaCy model name (e.g. en_core_web_sm). "
                              "Default: en. Tokenization works for any spaCy-supported "
                              "language even without a downloaded model; lemmatization "
                              "requires one (see --download-model).")
    parser.add_argument("--download-model", action="store_true",
                         help="Download/install the spaCy pipeline for --language before "
                              "running (requires network access), then proceed.")
    parser.add_argument("--lemmatize", action="store_true",
                         help="Lemmatize tokens using spaCy. Requires a trained pipeline "
                              "for --language to be installed; falls back to surface forms "
                              "silently if unavailable.")
    parser.add_argument("--min-length", type=int, default=1,
                         help="Minimum token length to include (default: 1).")
    parser.add_argument("--encoding", default="utf-8",
                         help="Text encoding used to read reference and target files "
                              "(default: utf-8). Bytes that don't decode are dropped.")
    parser.add_argument("--ignore-words", nargs="*", metavar="WORD",
                         help="Words to exclude from band/off-list classification. "
                              "They still count toward total tokens/types, but are "
                              "reported separately under an 'Ignored' category.")
    parser.add_argument("--ignore-list", metavar="PATH",
                         help="File of words to ignore, one per line (merged with "
                              "--ignore-words if both are given).")
    parser.add_argument("--exclude-proper-nouns", action="store_true",
                         help="Report proper nouns (via the language's POS tagger) "
                              "separately instead of profiling them like any other "
                              "word. They still count toward total tokens/types. "
                              "Requires a trained pipeline for --language; silently "
                              "has no effect without one.")
    parser.add_argument("--exclude-digits", action="store_true",
                         help="Report digit tokens (e.g. '42', '3.14') separately "
                              "instead of profiling them like any other word. Spelled-"
                              "out number words (e.g. 'twelve') are unaffected. They "
                              "still count toward total tokens/types.")

    parser.add_argument("--out-json", metavar="PATH", help="Write full results as JSON.")
    parser.add_argument("--out-csv", metavar="PATH", help="Write a tidy band-coverage CSV.")
    parser.add_argument("--out-off-list-csv", metavar="PATH",
                         help="Write a CSV of off-list (unknown) words per text.")
    parser.add_argument("--out-ignored-csv", metavar="PATH",
                         help="Write a CSV of ignored words per text.")
    parser.add_argument("--out-proper-nouns-csv", metavar="PATH",
                         help="Write a CSV of proper nouns per text (populated only "
                              "when --exclude-proper-nouns is set).")
    parser.add_argument("--out-digits-csv", metavar="PATH",
                         help="Write a CSV of digit tokens per text (populated only "
                              "when --exclude-digits is set).")
    parser.add_argument("--max-off-list-shown", type=int, default=20,
                         help="How many off-list words to show in the console summary.")

    args = parser.parse_args(argv)

    # Load target texts before doing any of the (potentially slow)
    # reference-building work below, so a typo in --target fails fast.
    # ValueError here means a non-.txt file was passed, so report it as a
    # clean CLI usage error instead of an uncaught traceback.
    try:
        targets = _collect_targets(args)
    except ValueError as e:
        parser.error(str(e))
    if not targets:
        parser.error(
            "No target text to profile was given. Pass one or more files "
            "with --target essay.txt [more.txt ...], and/or a folder of "
            ".txt files with --target-dir some_folder/."
        )

    if args.download_model:
        # Imported lazily so a plain `--help` invocation doesn't need to
        # import the tokenizer module (and therefore spaCy) at all.
        from . import tokenizer as tokenizer_mod
        print(f"Downloading spaCy model for language={args.language} ...", file=sys.stderr)
        ok = tokenizer_mod.download_model(args.language)
        print("Download succeeded." if ok else "Download failed or unavailable "
              "for this language; continuing with fallback tokenizer.", file=sys.stderr)

    if bool(args.fine_band_size) != bool(args.fine_until):
        parser.error(
            "Fine-grained bands need both options set together: "
            "--fine-band-size (how wide, e.g. 100) and --fine-until (how "
            "far, e.g. 2000); you only gave one of the two. Either add "
            "the missing one, or drop the one you gave to use uniform "
            "--band-size bands throughout."
        )
    if bool(args.coarse_band_size) != bool(args.coarse_from):
        parser.error(
            "Coarse-grained bands need both options set together: "
            "--coarse-band-size (how wide, e.g. 10000) and --coarse-from "
            "(from which rank, e.g. 50000); you only gave one of the two. "
            "Either add the missing one, or drop the one you gave to use "
            "uniform --band-size bands throughout."
        )

    # Exactly one of these three is set, enforced by the mutually
    # exclusive --reference-* argument group above. Wrapped in try/except
    # so a non-.txt file (ValueError from require_txt_extension) becomes a
    # clean CLI error instead of a raw traceback.
    try:
        if args.reference_saved:
            reference = Reference.load(args.reference_saved)
        elif args.reference_corpus:
            reference = Reference.from_corpus(
                args.reference_corpus, band_size=args.band_size, lemmatize=args.lemmatize,
                min_length=args.min_length, language=args.language,
                fine_band_size=args.fine_band_size, fine_grained_until=args.fine_until,
                coarse_band_size=args.coarse_band_size, coarse_grained_from=args.coarse_from,
                encoding=args.encoding,
            )
        elif args.reference_builtin:
            reference = Reference.from_builtin(
                args.reference_builtin, band_size=args.band_size, language=args.language,
                fine_band_size=args.fine_band_size, fine_grained_until=args.fine_until,
                coarse_band_size=args.coarse_band_size, coarse_grained_from=args.coarse_from,
            )
        else:
            reference = Reference.from_word_list(
                args.reference_wordlist, band_size=args.band_size, language=args.language,
                fine_band_size=args.fine_band_size, fine_grained_until=args.fine_until,
                coarse_band_size=args.coarse_band_size, coarse_grained_from=args.coarse_from,
                encoding=args.encoding,
            )
    except ValueError as e:
        parser.error(str(e))

    print(f"Reference: {reference.source_description}  "
          f"({reference.num_bands} bands of ~{reference.band_size} words)", file=sys.stderr)

    if args.save_reference:
        try:
            reference.save(args.save_reference)
        except ValueError as e:
            parser.error(str(e))
        print(f"Saved reference to {args.save_reference}", file=sys.stderr)

    # Same clean-error treatment for a non-.txt --ignore-list file.
    try:
        ignore_words = _collect_ignore_words(args)
    except ValueError as e:
        parser.error(str(e))
    profiler = LexicalProfiler(
        reference, min_length=args.min_length, ignore_words=ignore_words,
        exclude_proper_nouns=args.exclude_proper_nouns,
        exclude_digits=args.exclude_digits,
    )
    results = profiler.profile_texts(targets)

    for name, result in results.items():
        print(f"\n=== {name} ===")
        print(result.summary(max_off_list_shown=args.max_off_list_shown))

    # Each export can fail for the same reasons a save-reference can (bad
    # output path, no write permission, etc.); same clean-error handling.
    try:
        if args.out_json:
            report_mod.export_json(results, args.out_json)
            print(f"\nWrote JSON report to {args.out_json}", file=sys.stderr)
        if args.out_csv:
            report_mod.export_csv(results, args.out_csv)
            print(f"Wrote CSV report to {args.out_csv}", file=sys.stderr)
        if args.out_off_list_csv:
            report_mod.export_off_list_csv(results, args.out_off_list_csv)
            print(f"Wrote off-list CSV to {args.out_off_list_csv}", file=sys.stderr)
        if args.out_ignored_csv:
            report_mod.export_ignored_csv(results, args.out_ignored_csv)
            print(f"Wrote ignored-words CSV to {args.out_ignored_csv}", file=sys.stderr)
        if args.out_proper_nouns_csv:
            report_mod.export_proper_nouns_csv(results, args.out_proper_nouns_csv)
            print(f"Wrote proper-nouns CSV to {args.out_proper_nouns_csv}", file=sys.stderr)
        if args.out_digits_csv:
            report_mod.export_digits_csv(results, args.out_digits_csv)
            print(f"Wrote digits CSV to {args.out_digits_csv}", file=sys.stderr)
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
