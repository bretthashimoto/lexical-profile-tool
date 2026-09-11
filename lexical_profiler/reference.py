"""
Reference frequency models.

A Reference maps each known word to:
  - a rank (1 = most frequent)
  - a frequency band (1 = the band containing the most frequent words)

It can be built two ways:
  * Reference.from_corpus(...):     derive frequencies from a corpus of texts
  * Reference.from_word_list(...):  load an existing ordered/frequency word list

By default, bands are a uniform width (`band_size`, 1000 words). You can
optionally request finer-grained bands for the most frequent words via
`fine_band_size` / `fine_grained_until`, e.g. 100-word bands up through
rank 2000, then normal 1000-word bands after that.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .tokenizer import tokenize

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Reference word lists bundled with the package, keyed by the name passed to
# Reference.from_builtin(...). Add an entry here (plus the corresponding file
# under lexical_profiler/data/) to make a new published list available.
BUILTIN_WORD_LISTS: dict[str, dict[str, str | bool]] = {
    "avl": {
        "file": "avl_academic.txt",
        "label": "Academic Vocabulary List (AVL): Gardner & Davies (2013)",
        "description": (
            "Academic Vocabulary List (AVL) -- Gardner & Davies (2013), "
            "~2,900 core academic word lemmas from the COCA Academic sub-corpus"
        ),
        "citation": (
            "Gardner, D., & Davies, M. (2014). A new academic vocabulary list. "
            "Applied Linguistics, 35(3), 305-327. http://www.academicwords.info"
        ),
    },
    "ngsl": {
        "file": "ngsl.txt",
        "label": "New General Service List (NGSL): Browne & Culligan (2013)",
        "description": (
            "New General Service List (NGSL) -- Browne & Culligan (2013), "
            "~2,800 core word families for everyday (non-academic) English"
        ),
        "citation": (
            "Browne, C., Culligan, B., & Phillips, J. (2013). The New General "
            "Service List. http://www.newgeneralservicelist.org"
        ),
    },
    "nawl": {
        "file": "nawl.txt",
        "label": "New Academic Word List (NAWL): Browne, Culligan & Phillips (2013)",
        "description": (
            "New Academic Word List (NAWL) -- Browne, Culligan & Phillips (2013), "
            "963 academic word families that complement the NGSL; alphabetical, "
            "not frequency-ranked"
        ),
        "citation": (
            "Browne, C., Culligan, B., & Phillips, J. (2013). The New Academic Word "
            "List. http://www.newgeneralservicelist.org/nawl-new-academic-word-list"
        ),
    },
    "coca": {
        "file": "coca.txt",
        "label": "COCA top 100,000 lemma+POS entries (Davies, 2008-)",
        "pos_tagged": True,
        "description": (
            "Corpus of Contemporary American English (COCA) -- Davies (2008-), "
            "top 100,000 lemma+part-of-speech entries; matches by lemma and part "
            "of speech, so e.g. 'record' as a verb is scored separately from "
            "'record' as a noun"
        ),
        "citation": (
            "Davies, M. (2008-). The Corpus of Contemporary American English (COCA). "
            "https://www.english-corpora.org/coca/"
        ),
    },
}


def require_txt_extension(path: str) -> None:
    """Raise ValueError unless `path` has a .txt extension.

    Every place this package reads a file from disk (corpus files, target
    texts, word lists) is restricted to plain .txt files, so this check is
    shared across all of them to give one consistent, clear error message.
    """
    if os.path.splitext(path)[1].lower() != ".txt":
        raise ValueError(
            f"Can't use '{path}': only plain .txt files are accepted "
            f"(this file doesn't end in .txt). If it's really a text file, "
            f"try renaming it with a .txt extension; otherwise, save/export "
            f"it as plain text first."
        )


def open_text_file(path: str, encoding: str = "utf-8"):
    """Open `path` for reading as text, turning the common failure modes
    (missing file, no permission, path is actually a folder, wrong
    encoding) into a plain-language ValueError instead of a raw
    OS-level traceback. Returns a file object usable as a context manager,
    just like the builtin `open()`.
    """
    try:
        return open(path, encoding=encoding, errors="ignore")
    except FileNotFoundError:
        raise ValueError(
            f"Can't find the file '{path}'. Double-check the spelling and "
            f"that the path is relative to the folder you're running the "
            f"command from (or use a full/absolute path)."
        ) from None
    except IsADirectoryError:
        raise ValueError(
            f"'{path}' is a folder, not a file. Point at a specific .txt "
            f"file inside it, or use the directory/corpus option instead "
            f"if you meant to include every file in that folder."
        ) from None
    except PermissionError:
        raise ValueError(
            f"Don't have permission to open '{path}'. Check that the file "
            f"isn't open in another program and that you have permission "
            f"to read it."
        ) from None
    except LookupError:
        raise ValueError(
            f"'{encoding}' isn't a valid text encoding. Common values are "
            f"'utf-8', 'latin-1', or 'utf-16'; double-check the spelling."
        ) from None


def _expand_sources(
    source: str | Iterable[str] | dict[str, str],
) -> list[tuple[str, str, str | None, bool]]:
    """Expand a flexible `source` into a flat list of items to read, without
    reading any file content yet -- just enough work (checking file/
    directory existence, walking directories) to know the total document
    count up front, so callers can drive a progress indicator before the
    expensive part (tokenizing) starts.

    Accepts:
      - a path to a single .txt file
      - a path to a directory (all .txt files inside, recursive)
      - a list of file paths
      - a list of raw text strings (if they don't look like existing paths)
      - a dict of {display name: raw text}

    A bare string `source` is always treated as a path (never raw text);
    if it doesn't exist, that's an error. The "not a path -> treat as raw
    text" fallback only applies to items inside a list, since that's the
    documented way to pass literal text content directly.

    Returns a list of (kind, value, name, catch_open_errors) tuples, where
    `kind` is "path" or "text", `value` is the file path or the raw text
    itself, `name` is a display name if one is known (a filename, or a
    dict key) or None, and `catch_open_errors` says whether a read failure
    for this item should be silently skipped (True for files discovered by
    walking a directory, since one bad file -- permissions, a broken
    symlink -- shouldn't abort the whole corpus build) or should propagate
    (False for files named explicitly, where a failure is more likely a
    mistake worth surfacing).
    """
    if isinstance(source, dict):
        return [("text", text, name, False) for name, text in source.items()]

    # Remember whether the caller passed one bare string (as opposed to a
    # list), so we know below whether a non-existent path should raise
    # (bare string, almost certainly a typo'd path) or fall back to raw
    # text (list item, deliberately supported for literal text content).
    single_path = isinstance(source, str)
    if single_path:
        source = [source]

    items: list[tuple[str, str, str | None, bool]] = []
    for item in source:
        if isinstance(item, str) and os.path.isdir(item):
            # Directory: recurse into it and read every .txt file we find.
            # Sorting filenames keeps corpus processing order deterministic
            # (mainly so results/debugging are reproducible run to run).
            for root, _dirs, files in os.walk(item):
                for fname in sorted(files):
                    fpath = os.path.join(root, fname)
                    require_txt_extension(fpath)
                    items.append(("path", fpath, fname, True))
        elif isinstance(item, str) and os.path.isfile(item):
            require_txt_extension(item)
            items.append(("path", item, os.path.basename(item), False))
        elif isinstance(item, str) and single_path:
            # A bare string source that isn't an existing file or folder is
            # almost certainly a typo'd path, not intentional raw text --
            # raise instead of silently tokenizing the path string itself.
            raise ValueError(
                f"Can't find '{item}' as a file or folder. Check the path "
                f"is correct (relative to the folder you're running from, "
                f"or use a full/absolute path). If you meant to pass raw "
                f"text directly instead of a path, wrap it in a list: "
                f"Reference.from_corpus([your_text_here])."
            )
        elif isinstance(item, str):
            # Not a path on disk -> treat as raw text content (only
            # reachable for items inside a list, per the docstring above).
            items.append(("text", item, None, False))
        else:
            raise TypeError(
                f"Don't know how to read a corpus item of type "
                f"{type(item).__name__} ({item!r}). Each item should be a "
                f"file path, a directory path, or a raw text string."
            )
    return items


def _read_source_item(kind: str, value: str, catch_open_errors: bool,
                       encoding: str) -> str | None:
    """Read one item produced by `_expand_sources` into raw text, or None
    if it was an unreadable file that should be silently skipped."""
    if kind == "text":
        return value
    if catch_open_errors:
        try:
            with open_text_file(value, encoding) as f:
                return f.read()
        except ValueError:
            return None
    with open_text_file(value, encoding) as f:
        return f.read()


def _tokenize_corpus(
    source: str | Iterable[str] | dict[str, str], *, language: str, lowercase: bool,
    lemmatize: bool, min_length: int, encoding: str, pos_tagged: bool,
    progress_callback: Callable[[int, int, str], None] | None,
    counter: Counter,
) -> int:
    """Expand `source`, tokenize every document into `counter` (updated in
    place), and return the number of documents successfully read.

    Drives `progress_callback(current, total, message)`, if given, through
    each real stage of the work: an initial "loading" call, one call per
    document as it's tokenized (and lemmatized, if requested), and a final
    call once every document is done and frequency bands are about to be
    computed -- so a caller like the web UI can show real, specific status
    text (not just a generic spinner) for however long this takes.
    """
    items = _expand_sources(source)
    total = len(items)
    n_docs = 0

    if progress_callback and total:
        progress_callback(0, total, "Loading documents...")

    stage_label = "Lemmatizing" if lemmatize else "Tokenizing"
    for idx, (kind, value, name, catch_open_errors) in enumerate(items, start=1):
        text = _read_source_item(kind, value, catch_open_errors, encoding)
        if text is None:
            continue
        n_docs += 1
        tokens = tokenize(text, language=language, lowercase=lowercase,
                           lemmatize=lemmatize, min_length=min_length, pos_tag=pos_tagged)
        counter.update(tokens)
        if progress_callback:
            suffix = f" {name}" if name else ""
            progress_callback(idx, total, f"{stage_label}{suffix} ({idx} of {total})")

    if progress_callback and total:
        progress_callback(total, total, "Computing frequency bands...")

    return n_docs


def compute_band_assignment(
    n_words: int, band_size: int,
    fine_band_size: int | None = None, fine_grained_until: int | None = None,
) -> tuple[list[int], dict[int, tuple[int, int]]]:
    """Assign each of `n_words` ranked words (rank 1 = most frequent) to a band.

    Args:
        n_words: total number of ranked words.
        band_size: width of each band once past the fine-grained section
            (or for the whole list, if fine-grained args are omitted).
        fine_band_size: if given (along with `fine_grained_until`), use
            this narrower band width for ranks 1..fine_grained_until,
            e.g. 100-word bands for the first 2000 words.
        fine_grained_until: the rank up to which `fine_band_size` applies.
            Bands from there onward use `band_size` as usual.

    Returns:
        (band_of_position, band_ranges) where:
          - band_of_position[i] is the band number (1-indexed) for the
            word at 0-indexed position i (rank i+1).
          - band_ranges maps band number -> (start_rank, end_rank), both
            inclusive, so labels can be built directly from real data
            rather than recomputed via a formula.
    """
    band_of_position: list[int] = []
    band_ranges: dict[int, tuple[int, int]] = {}
    band_num = 1
    rank = 1

    # First, walk through the fine-grained section (if requested), chopping
    # it into narrower bands up to `fine_grained_until`. We cap at n_words
    # in case the word list is shorter than the requested fine-grained range.
    if fine_band_size and fine_grained_until:
        fine_cutoff = min(fine_grained_until, n_words)
        while rank <= fine_cutoff:
            start = rank
            end = min(rank + fine_band_size - 1, fine_cutoff)
            width = end - start + 1
            band_of_position.extend([band_num] * width)
            band_ranges[band_num] = (start, end)
            band_num += 1
            rank = end + 1

    # Then cover everything else (or the whole list, if no fine-grained
    # section was requested) using the normal band width. `rank` picks up
    # right where the fine-grained loop left off, so band numbering is
    # continuous across the two sections.
    while rank <= n_words:
        start = rank
        end = min(rank + band_size - 1, n_words)
        width = end - start + 1
        band_of_position.extend([band_num] * width)
        band_ranges[band_num] = (start, end)
        band_num += 1
        rank = end + 1

    return band_of_position, band_ranges


@dataclass
class Reference:
    """A frequency reference model used to profile target texts.

    Attributes:
        word_to_rank: word -> rank (1-indexed, 1 = most frequent)
        word_to_band: word -> band number (1-indexed)
        band_ranges: band number -> (start_rank, end_rank), both inclusive.
            Used to label bands as e.g. "1-999", "1000-1999", or, with a
            fine-grained schedule, "1-99", "100-199", ..., "2000-2999".
        band_size: the (coarse) band width used to build this reference.
        fine_band_size / fine_grained_until: the optional fine-grained
            banding parameters used to build this reference, if any.
        num_bands: total number of bands
        counts: word -> raw frequency count, if known (may be empty for
            plain word lists with no frequency data, e.g. rank-only lists)
        lowercase: whether reference words are lowercased
        lemmatize: whether reference words were lemmatized when built
        pos_tagged: whether known words are keyed as "lemma_CODE" (e.g.
            "record_V" vs "record_N") rather than plain lemmas -- see
            tokenizer.POS_DISPLAY_NAMES for what each code means. When
            True, target texts are tokenized the same way before lookup,
            so e.g. "record" used as a verb only matches a "record_V"
            entry, not "record_N". Implies lemmatize=True (POS-aware
            matching against a lemma-keyed reference is meaningless
            against inflected surface forms).
    """

    word_to_rank: dict[str, int]
    word_to_band: dict[str, int]
    band_ranges: dict[int, tuple[int, int]]
    band_size: int
    num_bands: int
    fine_band_size: int | None = None
    fine_grained_until: int | None = None
    counts: dict[str, int] = field(default_factory=dict)
    lowercase: bool = True
    lemmatize: bool = False
    pos_tagged: bool = False
    language: str = "en"
    source_description: str = ""

    # ---------- constructors ----------

    @classmethod
    def from_corpus(cls, source: str | Iterable[str] | dict[str, str], band_size: int = 1000,
                     lowercase: bool = True, lemmatize: bool = False,
                     min_length: int = 1, language: str = "en",
                     fine_band_size: int | None = None,
                     fine_grained_until: int | None = None,
                     encoding: str = "utf-8",
                     progress_callback: Callable[[int, int, str], None] | None = None,
                     pos_tagged: bool = False,
                     ) -> Reference:
        """Build a reference frequency model from a corpus of texts.

        Args:
            source: a .txt file path, directory path (all .txt files inside,
                searched recursively), list of file paths, list of raw text
                strings, or a dict of {display name: raw text}. Any file
                path that isn't a .txt file raises ValueError.
            band_size: number of words per frequency band (default 1000,
                giving classic "1-999, 1000-1999, ..." bands as used in
                lexical frequency profiling research).
            lowercase: lowercase all tokens.
            lemmatize: lemmatize tokens using spaCy. Requires a trained
                spaCy pipeline for `language` to be installed; falls back
                to surface word forms silently if unavailable.
            min_length: minimum token length to include.
            language: ISO 639-1 language code (e.g. 'en', 'es', 'de', 'fr',
                'zh', 'ja', 'ru', ...) or a full spaCy model name (e.g.
                'en_core_web_sm'). Tokenization works even without an
                installed pipeline for the language (spaCy falls back to
                a rule-based tokenizer); lemmatization requires one.
            fine_band_size: optionally use a narrower band width for the
                most frequent words (e.g. 100), up through rank
                `fine_grained_until` (e.g. 2000), so the first 2000
                words are split into 20 bands of 100 instead of 2 bands
                of 1000, and `band_size` applies as usual after that.
            fine_grained_until: the rank up to which `fine_band_size`
                applies. Required if `fine_band_size` is given.
            encoding: text encoding used to read corpus files (default
                'utf-8'). Bytes that don't decode are dropped rather than
                raising (errors='ignore').
            progress_callback: optional `callback(current, total, message)`
                invoked as the corpus is processed -- once before reading
                starts, once per document as it's tokenized/lemmatized, and
                once more before frequency bands are computed. Useful for
                driving a progress bar for a large corpus.
            pos_tagged: key known words as "lemma_CODE" (e.g. "record_V")
                instead of a plain lemma, so profiling only matches a word
                used with the same part of speech (see tokenizer.
                POS_DISPLAY_NAMES for what each code means). Requires the
                same trained spaCy pipeline as lemmatize; silently produces
                a plain (non-POS) reference otherwise. Forces lemmatize=True
                regardless of what was passed, since POS-aware matching
                against surface forms is meaningless.
        """
        if pos_tagged:
            lemmatize = True

        # Tokenize every document in the corpus and tally raw word counts.
        # Frequency (not the source text's order) is what determines rank.
        counter: Counter = Counter()
        n_docs = _tokenize_corpus(
            source, language=language, lowercase=lowercase, lemmatize=lemmatize,
            min_length=min_length, encoding=encoding, pos_tagged=pos_tagged,
            progress_callback=progress_callback, counter=counter,
        )

        if not counter:
            if n_docs == 0:
                raise ValueError(
                    "No documents were found to build the reference from. "
                    "If you passed a folder, check that it actually "
                    "contains .txt files; if you passed a list, make sure "
                    "it isn't empty."
                )
            raise ValueError(
                f"Found {n_docs} document(s), but none of them contained "
                f"any usable words after tokenizing. They may be empty, "
                f"contain only numbers/punctuation, or be in a different "
                f"language than expected (try setting `language` to match "
                f"the text)."
            )

        # most_common() sorts most-frequent-first, so its position in this
        # list directly gives us the rank (1 = most frequent).
        ranked = [w for w, _ in counter.most_common()]
        band_of_position, band_ranges = compute_band_assignment(
            len(ranked), band_size, fine_band_size, fine_grained_until,
        )
        word_to_rank = {w: i + 1 for i, w in enumerate(ranked)}
        word_to_band = {w: band_of_position[i] for i, w in enumerate(ranked)}
        num_bands = len(band_ranges)

        return cls(
            word_to_rank=word_to_rank,
            word_to_band=word_to_band,
            band_ranges=band_ranges,
            band_size=band_size,
            num_bands=num_bands,
            fine_band_size=fine_band_size,
            fine_grained_until=fine_grained_until,
            counts=dict(counter),
            lowercase=lowercase,
            lemmatize=lemmatize,
            pos_tagged=pos_tagged,
            language=language,
            source_description=f"corpus ({n_docs} document(s), {len(ranked)} unique words, "
                                f"language={language})",
        )

    @classmethod
    def from_word_list(cls, path: str, band_size: int = 1000,
                        lowercase: bool = True, has_frequencies: bool | None = None,
                        delimiter: str | None = None, language: str = "en",
                        fine_band_size: int | None = None,
                        fine_grained_until: int | None = None,
                        encoding: str = "utf-8", lemmatize: bool = False,
                        pos_tagged: bool = False) -> Reference:
        """Load an existing frequency/rank word list from a file.

        Accepted formats (auto-detected unless overridden):
          - one word per line, already ordered most-to-least frequent:
                the
                be
                to
                ...
          - "word<delim>frequency" per line (delimiter auto-detected among
            tab, comma, or whitespace), in which case the list is sorted
            by frequency descending regardless of file order:
                the     22038615
                be      12545825
                to      12145630

        Args:
            path: path to the word list file. Must be a .txt file; raises
                ValueError otherwise.
            band_size: words per frequency band for downstream profiling
                (default 1000). See `fine_band_size` for finer-grained
                bands among the most frequent words.
            lowercase: lowercase all words on load.
            has_frequencies: force interpretation as "word, freq" pairs
                (True), single-word-per-line (False), or auto-detect
                (None, default).
            delimiter: force a specific delimiter; auto-detected if None.
            language: ISO 639-1 language code (or full spaCy model name)
                to use later when tokenizing *target* texts profiled
                against this reference. This does not affect how the
                word list itself is read.
            fine_band_size: optionally use a narrower band width for the
                most frequent words (e.g. 100), up through rank
                `fine_grained_until` (e.g. 2000).
            fine_grained_until: the rank up to which `fine_band_size`
                applies. Required if `fine_band_size` is given.
            encoding: text encoding used to read the word list file
                (default 'utf-8'). Bytes that don't decode are dropped
                rather than raising (errors='ignore').
            lemmatize: whether *target* texts profiled against this
                reference should be lemmatized before comparison (the
                word list's own entries are used as-is either way --
                this doesn't re-process them). Requires a trained spaCy
                pipeline for `language`; falls back to surface forms
                silently otherwise.
            pos_tagged: whether this word list's entries are already in
                "lemma_CODE" form (e.g. "record_V") rather than plain
                lemmas -- see tokenizer.POS_DISPLAY_NAMES for what each
                code means. This doesn't change how the file is read; it
                only tells profiling to tokenize target text the same way
                before matching. Forces lemmatize=True regardless of what
                was passed, since POS-aware matching against surface forms
                is meaningless.
        """
        if pos_tagged:
            lemmatize = True

        # Read the file into memory first (word lists are small enough that
        # this is fine) so we can peek at the first row to auto-detect the
        # format before deciding how to parse the rest.
        require_txt_extension(path)
        rows: list[str] = []
        with open_text_file(path, encoding) as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                rows.append(line)

        if not rows:
            raise ValueError(
                f"The word list at '{path}' doesn't have any usable lines "
                f"in it. It might be empty, or every line might be blank "
                f"or a '#' comment; add at least one real word to the "
                f"file and try again."
            )

        # Auto-detect "word<delim>freq" layout from the first non-empty row.
        def split_row(line: str):
            if delimiter is not None:
                parts = [p for p in line.split(delimiter) if p != ""]
            else:
                # No explicit delimiter given: normalize commas to tabs and
                # let str.split() handle any run of whitespace/tabs, so
                # "word,123", "word\t123" and "word   123" all work.
                parts = line.replace(",", "\t").split()
            return parts

        # If the second column of the first row looks like a number, assume
        # every row is "word, frequency" unless the caller overrode this.
        sample_parts = split_row(rows[0])
        auto_has_freq = len(sample_parts) >= 2 and sample_parts[1].replace(".", "", 1).isdigit()
        use_freq = auto_has_freq if has_frequencies is None else has_frequencies

        counts: dict[str, int] = {}
        ordered_words: list[str] = []

        if use_freq:
            # Frequencies in the file may not already be sorted, so collect
            # every (word, freq) pair first and then sort by frequency
            # descending; rank is derived from that sorted order below.
            pairs = []
            for line in rows:
                parts = split_row(line)
                if len(parts) < 2:
                    continue
                word, freq_str = parts[0], parts[1]
                try:
                    freq = float(freq_str)
                except ValueError:
                    continue
                word = word.lower() if lowercase else word
                pairs.append((word, freq))
            pairs.sort(key=lambda p: p[1], reverse=True)
            for word, freq in pairs:
                if word not in counts:  # keep first (highest-freq) occurrence
                    ordered_words.append(word)
                    counts[word] = int(freq)
        else:
            # Plain word list: rank comes directly from line order, and we
            # have no real frequency numbers, so counts are left at 0
            # (meaning "known rank, unknown magnitude").
            for line in rows:
                word = split_row(line)[0]
                word = word.lower() if lowercase else word
                if word not in counts:
                    ordered_words.append(word)
                    counts[word] = 0  # unknown magnitude, rank-only

        band_of_position, band_ranges = compute_band_assignment(
            len(ordered_words), band_size, fine_band_size, fine_grained_until,
        )
        word_to_rank = {w: i + 1 for i, w in enumerate(ordered_words)}
        word_to_band = {w: band_of_position[i] for i, w in enumerate(ordered_words)}
        num_bands = len(band_ranges)

        return cls(
            word_to_rank=word_to_rank,
            word_to_band=word_to_band,
            band_ranges=band_ranges,
            band_size=band_size,
            num_bands=num_bands,
            fine_band_size=fine_band_size,
            fine_grained_until=fine_grained_until,
            counts=counts if use_freq else {},
            lowercase=lowercase,
            lemmatize=lemmatize,
            pos_tagged=pos_tagged,
            language=language,
            source_description=f"word list '{os.path.basename(path)}' "
                                f"({len(ordered_words)} words, "
                                f"{'with' if use_freq else 'without'} frequencies, "
                                f"language={language})",
        )

    def add_texts(self, source: str | Iterable[str] | dict[str, str], encoding: str = "utf-8",
                  progress_callback: Callable[[int, int, str], None] | None = None,
                  ) -> Reference:
        """Add more documents to this reference, recomputing ranks and bands
        from the combined word counts.

        Reuses this reference's existing lowercase/lemmatize/language/
        band settings, and accepts the same `source` shapes as
        `from_corpus` (a path, directory, list of paths, list of raw text
        strings, or a dict of {display name: raw text}). Returns a new
        Reference; this one is left as-is.

        `progress_callback` behaves as in `from_corpus`.

        Raises ValueError if this reference has no raw word counts to
        merge with (e.g. a rank-only word list), since there'd be nothing
        meaningful to combine the new counts with.
        """
        if not self.counts:
            raise ValueError(
                "Can't add texts to this reference: it doesn't have raw "
                "word counts to merge with (rank-only word lists and "
                "built-in lists don't carry frequency data). Build a new "
                "reference with Reference.from_corpus(...) instead."
            )

        counter: Counter = Counter(self.counts)
        n_docs = _tokenize_corpus(
            source, language=self.language, lowercase=self.lowercase,
            lemmatize=self.lemmatize, min_length=1, encoding=encoding,
            pos_tagged=self.pos_tagged,
            progress_callback=progress_callback, counter=counter,
        )

        if n_docs == 0:
            raise ValueError(
                "No documents were found to add. If you passed a folder, "
                "check that it actually contains .txt files; if you "
                "passed a list, make sure it isn't empty."
            )

        ranked = [w for w, _ in counter.most_common()]
        band_of_position, band_ranges = compute_band_assignment(
            len(ranked), self.band_size, self.fine_band_size, self.fine_grained_until,
        )
        word_to_rank = {w: i + 1 for i, w in enumerate(ranked)}
        word_to_band = {w: band_of_position[i] for i, w in enumerate(ranked)}

        return type(self)(
            word_to_rank=word_to_rank,
            word_to_band=word_to_band,
            band_ranges=band_ranges,
            band_size=self.band_size,
            num_bands=len(band_ranges),
            fine_band_size=self.fine_band_size,
            fine_grained_until=self.fine_grained_until,
            counts=dict(counter),
            lowercase=self.lowercase,
            lemmatize=self.lemmatize,
            pos_tagged=self.pos_tagged,
            language=self.language,
            source_description=f"{self.source_description} + {n_docs} more document(s) "
                                f"added ({len(ranked)} unique words total)",
        )

    @classmethod
    def from_builtin(cls, name: str, band_size: int = 1000, lowercase: bool = True,
                      language: str = "en", fine_band_size: int | None = None,
                      fine_grained_until: int | None = None,
                      lemmatize: bool = False) -> Reference:
        """Load one of the reference word lists bundled with this package
        (see BUILTIN_WORD_LISTS for the available names, e.g. "avl" for the
        Academic Vocabulary List).

        Args:
            name: key into BUILTIN_WORD_LISTS, e.g. "avl". Case-insensitive.
            band_size, lowercase, language, fine_band_size, fine_grained_until,
            lemmatize: same as from_word_list().

        Whether the list is POS-tagged (e.g. "coca") is a property of the
        file itself, read from BUILTIN_WORD_LISTS -- not a caller choice.
        """
        key = name.strip().lower()
        entry = BUILTIN_WORD_LISTS.get(key)
        if entry is None:
            available = ", ".join(sorted(BUILTIN_WORD_LISTS))
            raise ValueError(
                f"'{name}' isn't a built-in word list this package ships with. "
                f"Available options: {available}."
            )
        path = os.path.join(_DATA_DIR, entry["file"])
        reference = cls.from_word_list(
            path, band_size=band_size, lowercase=lowercase, language=language,
            fine_band_size=fine_band_size, fine_grained_until=fine_grained_until,
            lemmatize=lemmatize, pos_tagged=entry.get("pos_tagged", False),
        )
        reference.source_description = f"built-in word list '{key}' ({entry['description']})"
        return reference

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        """Save this reference to a JSON file for reuse without rebuilding."""
        payload = {
            "word_to_rank": self.word_to_rank,
            "band_ranges": {str(b): list(r) for b, r in self.band_ranges.items()},
            "band_size": self.band_size,
            "num_bands": self.num_bands,
            "fine_band_size": self.fine_band_size,
            "fine_grained_until": self.fine_grained_until,
            "counts": self.counts,
            "lowercase": self.lowercase,
            "lemmatize": self.lemmatize,
            "pos_tagged": self.pos_tagged,
            "language": self.language,
            "source_description": self.source_description,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except FileNotFoundError:
            raise ValueError(
                f"Can't save to '{path}': the folder it's supposed to go "
                f"in doesn't exist. Create that folder first, or save "
                f"somewhere that already exists."
            ) from None
        except PermissionError:
            raise ValueError(
                f"Don't have permission to save to '{path}'. Check that "
                f"the file isn't open/read-only, and that you have write "
                f"access to that folder."
            ) from None
        except IsADirectoryError:
            raise ValueError(
                f"'{path}' is a folder, not a file path. Include a "
                f"filename, e.g. '{os.path.join(path, 'reference.json')}'."
            ) from None

    @classmethod
    def load(cls, path: str) -> Reference:
        """Load a reference previously saved with `.save(path)`.

        Raises ValueError (with a plain-language explanation) if the file
        is missing, unreadable, not valid JSON, or doesn't look like a
        reference that was actually produced by `.save(...)`.
        """
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            raise ValueError(
                f"Can't find a saved reference at '{path}'. Check the "
                f"path, or build a fresh reference with "
                f"Reference.from_corpus(...) or Reference.from_word_list(...) "
                f"instead."
            ) from None
        except PermissionError:
            raise ValueError(
                f"Don't have permission to read '{path}'. Check the file's "
                f"permissions and that it isn't locked by another program."
            ) from None
        except json.JSONDecodeError:
            raise ValueError(
                f"'{path}' doesn't look like a valid saved reference file "
                f"(it isn't valid JSON). Make sure it was created with "
                f"Reference.save(...) and hasn't been edited or corrupted."
            ) from None

        # Everything below just reads expected keys out of `payload`. Any
        # of them being missing/malformed means the file isn't actually a
        # reference saved by this package, so they're all treated as the
        # same friendly error rather than a raw KeyError/TypeError.
        try:
            word_to_rank = payload["word_to_rank"]

            # word_to_band isn't saved to disk; it's cheap to recompute
            # from word_to_rank + band_ranges, and doing so avoids ever
            # trusting a stale/duplicated copy of the same information.
            if "band_ranges" in payload:
                band_ranges = {int(b): tuple(r) for b, r in payload["band_ranges"].items()}
                # Rebuild word_to_band from rank + band_ranges rather than
                # trusting a stale formula, so this stays correct even for
                # fine-grained schedules.
                def band_for_rank(rank: int) -> int:
                    for b, (start, end) in band_ranges.items():
                        if start <= rank <= end:
                            return b
                    return max(band_ranges) if band_ranges else 1
                word_to_band = {w: band_for_rank(r) for w, r in word_to_rank.items()}
            else:
                # Backward compatibility with references saved before
                # band_ranges existed (uniform band_size only).
                band_size = payload["band_size"]
                word_to_band = {w: ((r - 1) // band_size) + 1 for w, r in word_to_rank.items()}
                n_words = len(word_to_rank)
                _, band_ranges = compute_band_assignment(n_words, band_size)

            reference = cls(
                word_to_rank=word_to_rank,
                word_to_band=word_to_band,
                band_ranges=band_ranges,
                band_size=payload["band_size"],
                num_bands=payload["num_bands"],
                fine_band_size=payload.get("fine_band_size"),
                fine_grained_until=payload.get("fine_grained_until"),
                counts=payload.get("counts", {}),
                lowercase=payload.get("lowercase", True),
                lemmatize=payload.get("lemmatize", False),
                pos_tagged=payload.get("pos_tagged", False),
                language=payload.get("language", "en"),
                source_description=payload.get("source_description", ""),
            )
        except (TypeError, KeyError, AttributeError, ValueError):
            raise ValueError(
                f"'{path}' doesn't look like a reference file saved by "
                f"this package (missing or malformed data). Make sure it "
                f"was created with Reference.save(...) rather than "
                f"hand-written or produced by something else."
            ) from None

        return reference

    # ---------- convenience ----------

    def __contains__(self, word: str) -> bool:
        return word in self.word_to_rank

    def __len__(self) -> int:
        return len(self.word_to_rank)

    def band_of(self, word: str) -> int | None:
        return self.word_to_band.get(word)

    def rank_of(self, word: str) -> int | None:
        return self.word_to_rank.get(word)

    def band_label(self, band: int) -> str:
        """Human-readable label for a band, e.g. "1-999", "1000-1999"."""
        r = self.band_ranges.get(band)
        return f"{r[0]}-{r[1]}" if r else f"Band {band}"
