"""
Core lexical frequency profiling.

Given a Reference frequency model and a target text (or texts), computes:
  - token- and type-level coverage of each frequency band
  - "off-list" words (not found anywhere in the reference)
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from .reference import Reference, open_text_file, require_txt_extension
from .tokenizer import classify_tokens, pipeline_for, pos_suffix


@dataclass
class ProfileResult:
    """Result of profiling a single text against a Reference."""

    total_tokens: int
    total_types: int
    band_token_counts: dict[int, int]      # band -> token count
    band_type_counts: dict[int, int]       # band -> type (unique word) count
    band_token_pct: dict[int, float]       # band -> % of all tokens
    band_type_pct: dict[int, float]        # band -> % of all types
    cumulative_token_pct: dict[int, float] # band -> % of tokens in bands 1..band
    off_list_tokens: int
    off_list_types: int
    off_list_pct_tokens: float
    off_list_words: list[str]              # unique off-list words, most frequent first
    ignored_tokens: int = 0
    ignored_types: int = 0
    ignored_pct_tokens: float = 0.0
    # unique ignored words, most frequent first
    ignored_words: list[str] = field(default_factory=list)
    # Proper nouns and numerals are only broken out into their own bucket
    # when LexicalProfiler was told to exclude them (exclude_proper_nouns/
    # exclude_numerals); otherwise they're just profiled like any other
    # word and these stay at zero/empty. Same accounting convention as
    # ignored_*: still counted toward total_tokens/total_types.
    proper_noun_tokens: int = 0
    proper_noun_types: int = 0
    proper_noun_pct_tokens: float = 0.0
    proper_noun_words: list[str] = field(default_factory=list)
    numeral_tokens: int = 0
    numeral_types: int = 0
    numeral_pct_tokens: float = 0.0
    numeral_words: list[str] = field(default_factory=list)
    word_counts: Counter = field(repr=False, default_factory=Counter)
    num_bands: int = 0
    band_ranges: dict[int, tuple[int, int]] = field(default_factory=dict)

    def band_label(self, band: int) -> str:
        """Human-readable label for a band, e.g. "1-999", "1000-1999",
        or (with a fine-grained schedule) "1-99", "100-199", ...,
        "2000-2999"."""
        r = self.band_ranges.get(band)
        return f"{r[0]}-{r[1]}" if r else f"Band {band}"

    def summary(self, max_bands_shown: int | None = None,
                max_off_list_shown: int = 20) -> str:
        """Human-readable text summary, similar in spirit to classic
        Lexical Frequency Profile reports (Laufer & Nation style)."""
        lines = []
        lines.append(f"Tokens: {self.total_tokens}   Types: {self.total_types}")
        lines.append("")
        # Size the "Band" column to fit the longest band label (e.g.
        # "10000-10999") so the table stays aligned regardless of how many
        # bands there are or how wide their labels get.
        label_width = max(
            10, max((len(self.band_label(b)) + 1 for b in self.band_token_counts), default=10)
        )
        lines.append(
            f"{'Band':<{label_width}}{'Tokens':>10}{'% Tokens':>12}{'Cum %':>10}"
            f"{'Types':>10}{'% Types':>12}"
        )
        bands = sorted(self.band_token_counts.keys())
        if max_bands_shown:
            bands = bands[:max_bands_shown]
        for b in bands:
            lines.append(
                f"{self.band_label(b):<{label_width}}{self.band_token_counts[b]:>10}"
                f"{self.band_token_pct[b]:>11.2f}%"
                f"{self.cumulative_token_pct[b]:>9.2f}%"
                f"{self.band_type_counts.get(b, 0):>10}"
                f"{self.band_type_pct.get(b, 0):>11.2f}%"
            )
        # Off-list/Ignored aren't part of the band 1..N progression, so
        # "coverage through this row" isn't a meaningful number for them --
        # leave the Cum % cell blank rather than showing a stale/misleading
        # value.
        lines.append(
            f"{'Off-list':<{label_width}}{self.off_list_tokens:>10}"
            f"{self.off_list_pct_tokens:>11.2f}%"
            f"{'':>10}"
            f"{self.off_list_types:>10}"
            f"{'':>12}"
        )
        if self.ignored_tokens or self.ignored_words:
            lines.append(
                f"{'Ignored':<{label_width}}{self.ignored_tokens:>10}"
                f"{self.ignored_pct_tokens:>11.2f}%"
                f"{'':>10}"
                f"{self.ignored_types:>10}"
                f"{'':>12}"
            )
        if self.proper_noun_tokens or self.proper_noun_words:
            lines.append(
                f"{'Proper nouns':<{label_width}}{self.proper_noun_tokens:>10}"
                f"{self.proper_noun_pct_tokens:>11.2f}%"
                f"{'':>10}"
                f"{self.proper_noun_types:>10}"
                f"{'':>12}"
            )
        if self.numeral_tokens or self.numeral_words:
            lines.append(
                f"{'Numerals':<{label_width}}{self.numeral_tokens:>10}"
                f"{self.numeral_pct_tokens:>11.2f}%"
                f"{'':>10}"
                f"{self.numeral_types:>10}"
                f"{'':>12}"
            )
        if self.off_list_words:
            shown = self.off_list_words[:max_off_list_shown]
            more = len(self.off_list_words) - len(shown)
            lines.append("")
            lines.append(f"Sample off-list words: {', '.join(shown)}"
                          + (f"  (+{more} more)" if more > 0 else ""))
        if self.ignored_words:
            shown = self.ignored_words[:max_off_list_shown]
            more = len(self.ignored_words) - len(shown)
            lines.append("")
            lines.append(f"Sample ignored words: {', '.join(shown)}"
                          + (f"  (+{more} more)" if more > 0 else ""))
        if self.proper_noun_words:
            shown = self.proper_noun_words[:max_off_list_shown]
            more = len(self.proper_noun_words) - len(shown)
            lines.append("")
            lines.append(f"Sample proper nouns: {', '.join(shown)}"
                          + (f"  (+{more} more)" if more > 0 else ""))
        if self.numeral_words:
            shown = self.numeral_words[:max_off_list_shown]
            more = len(self.numeral_words) - len(shown)
            lines.append("")
            lines.append(f"Sample numerals: {', '.join(shown)}"
                          + (f"  (+{more} more)" if more > 0 else ""))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "total_types": self.total_types,
            "band_token_counts": {self.band_label(b): n for b, n in self.band_token_counts.items()},
            "band_type_counts": {self.band_label(b): n for b, n in self.band_type_counts.items()},
            "band_token_pct": {self.band_label(b): n for b, n in self.band_token_pct.items()},
            "band_type_pct": {self.band_label(b): n for b, n in self.band_type_pct.items()},
            "cumulative_token_pct": {
                self.band_label(b): n for b, n in self.cumulative_token_pct.items()
            },
            "off_list_tokens": self.off_list_tokens,
            "off_list_types": self.off_list_types,
            "off_list_pct_tokens": self.off_list_pct_tokens,
            "off_list_words": self.off_list_words,
            "ignored_tokens": self.ignored_tokens,
            "ignored_types": self.ignored_types,
            "ignored_pct_tokens": self.ignored_pct_tokens,
            "ignored_words": self.ignored_words,
            "proper_noun_tokens": self.proper_noun_tokens,
            "proper_noun_types": self.proper_noun_types,
            "proper_noun_pct_tokens": self.proper_noun_pct_tokens,
            "proper_noun_words": self.proper_noun_words,
            "numeral_tokens": self.numeral_tokens,
            "numeral_types": self.numeral_types,
            "numeral_pct_tokens": self.numeral_pct_tokens,
            "numeral_words": self.numeral_words,
        }


@dataclass
class HighlightedToken:
    """One token of a text annotated for rendering, e.g. as per-word
    colored HTML (in the style of LexTutor VocabProfile / AntWordProfiler).

    Preserves the original surface form and trailing whitespace/punctuation
    so the full original text can be reconstructed by concatenating
    `text + whitespace` for every token in order.
    """

    text: str               # original surface form, as it appeared in the text
    whitespace: str         # whitespace/nothing following this token in the original text
    # "band", "off_list", "ignored", "proper_noun", "numeral", or
    # "skipped" (non-word, non-numeral token, e.g. punctuation).
    # "proper_noun"/"numeral" only appear when the profiler was told to
    # exclude that category -- otherwise such tokens are classified
    # "band"/"off_list" like any other word.
    status: str
    band: int | None = None  # set only when status == "band"


class LexicalProfiler:
    """Profiles target text(s) against a Reference frequency model."""

    def __init__(self, reference: Reference, min_length: int = 1,
                 ignore_words: Iterable[str] | None = None,
                 exclude_proper_nouns: bool = False,
                 exclude_numerals: bool = False):
        """
        Args:
            reference: the Reference frequency model to profile against.
            min_length: minimum token length to include.
            ignore_words: optional words to exclude from band/off-list
                classification. Matching tokens are still counted toward
                total_tokens/total_types, but reported separately under
                ProfileResult.ignored_* instead of a frequency band or
                off-list. Matched case-insensitively if the reference
                lowercases tokens (the default). Takes priority over
                exclude_proper_nouns/exclude_numerals below if a word
                happens to match both.
            exclude_proper_nouns: if True, words tagged as proper nouns
                (via the language's POS tagger) are pulled out of band/
                off-list classification and reported separately under
                ProfileResult.proper_noun_* instead, the same way
                ignore_words works. If False (default), proper nouns are
                profiled like any other word. Requires a trained pipeline
                for the reference's language; silently has no effect
                without one (every word is just classified normally).
            exclude_numerals: if True, numeral tokens (e.g. "42",
                "twelve") are pulled out of band/off-list classification
                and reported separately under ProfileResult.numeral_*
                instead. If False (default), numerals are profiled like
                any other word (in practice usually landing off-list,
                since reference word lists/corpora don't carry numerals
                as vocabulary entries).
        """
        self.reference = reference
        self.min_length = min_length
        # Normalize the ignore list the same way tokens get normalized
        # (lowercased, if the reference does that) so lookups below are a
        # simple set membership check instead of a case-insensitive one.
        if ignore_words:
            self.ignore_words = {
                (w.lower() if reference.lowercase else w) for w in ignore_words
            }
        else:
            self.ignore_words = set()
        self.exclude_proper_nouns = exclude_proper_nouns
        self.exclude_numerals = exclude_numerals

    def _classify(self, text: str) -> list[tuple[str, str]]:
        return classify_tokens(
            text,
            language=self.reference.language,
            lowercase=self.reference.lowercase,
            lemmatize=self.reference.lemmatize,
            min_length=self.min_length,
            pos_tag=self.reference.pos_tagged,
        )

    def profile_text(self, text: str) -> ProfileResult:
        """Profile a single text string against the reference."""
        classified = self._classify(text)
        return self._profile_tokens(classified)

    def highlight(self, text: str) -> list[HighlightedToken]:
        """Classify every token of `text` the same way `profile_text` does,
        but keep each token's original surface form and trailing
        whitespace/punctuation instead of collapsing it to a flat word
        list. Meant for rendering the original text back out with a
        per-word annotation (e.g. color-coded by frequency band), the way
        LexTutor's VocabProfile or AntWordProfiler display a marked-up
        text.

        Returns a list of HighlightedToken, in original order. Joining
        `tok.text + tok.whitespace` for every token reconstructs the
        original text exactly.
        """
        nlp, has_lemmatizer = pipeline_for(self.reference.language)
        do_lemmatize = self.reference.lemmatize and has_lemmatizer
        do_pos_tag = self.reference.pos_tagged and has_lemmatizer

        tokens: list[HighlightedToken] = []
        for tok in nlp(text):
            surface = tok.text
            # Mirror classify_tokens()'s rules exactly: a numeral is kept
            # even with no alphabetic character; anything else with none
            # is dropped as "skipped", as is anything shorter than
            # min_length (checked post-lowercase, pre-lemmatize).
            is_numeral = tok.like_num
            if not is_numeral and not any(ch.isalpha() for ch in surface):
                tokens.append(HighlightedToken(surface, tok.whitespace_, "skipped"))
                continue

            word = tok.lemma_ if (do_lemmatize and tok.lemma_) else surface
            if self.reference.lowercase:
                word = word.lower()
            if len(word) < self.min_length:
                tokens.append(HighlightedToken(surface, tok.whitespace_, "skipped"))
                continue
            if do_pos_tag:
                code = pos_suffix(tok)
                if code:
                    word = f"{word}_{code}"

            if word in self.ignore_words:
                tokens.append(HighlightedToken(surface, tok.whitespace_, "ignored"))
                continue
            if is_numeral and self.exclude_numerals:
                tokens.append(HighlightedToken(surface, tok.whitespace_, "numeral"))
                continue
            if not is_numeral and has_lemmatizer and tok.pos_ == "PROPN" and self.exclude_proper_nouns:
                tokens.append(HighlightedToken(surface, tok.whitespace_, "proper_noun"))
                continue

            band = self.reference.band_of(word)
            if band is None:
                tokens.append(HighlightedToken(surface, tok.whitespace_, "off_list"))
            else:
                tokens.append(HighlightedToken(surface, tok.whitespace_, "band", band=band))
        return tokens

    def profile_texts(self, texts: dict[str, str]) -> dict[str, ProfileResult]:
        """Profile multiple named texts (e.g. {filename: content, ...}).

        Returns a dict of filename -> ProfileResult, run independently
        per text (each text's own token/type counts, not pooled).
        """
        return {name: self.profile_text(content) for name, content in texts.items()}

    def profile_document(self, path: str, encoding: str = "utf-8") -> ProfileResult:
        """Profile a single target document, given a path to a .txt file.

        Args:
            path: path to a .txt file. Any other extension raises
                ValueError.
            encoding: text encoding used to read the file (default
                'utf-8'). Bytes that don't decode are dropped rather than
                raising (errors='ignore').
        """
        require_txt_extension(path)
        with open_text_file(path, encoding) as f:
            text = f.read()
        return self.profile_text(text)

    def profile_corpus(self, path: str, encoding: str = "utf-8") -> dict[str, ProfileResult]:
        """Profile every .txt file found in a directory, each independently.

        Searches `path` recursively, mirroring how Reference.from_corpus
        builds a reference from a directory. Any non-.txt file found along
        the way raises ValueError.

        Args:
            path: path to a directory of target .txt files.
            encoding: text encoding used to read each file (default
                'utf-8'). Bytes that don't decode are dropped rather than
                raising (errors='ignore').

        Returns:
            A dict mapping each file's path (relative to `path`) to its
            ProfileResult; relative paths (rather than bare filenames)
            avoid collisions between same-named files in different
            subdirectories.
        """
        # os.walk() on a path that doesn't exist just silently yields
        # nothing, which would otherwise look like "an empty folder" --
        # check up front so a typo'd path gives a clear error instead.
        if not os.path.isdir(path):
            raise ValueError(
                f"Can't find the folder '{path}'. Check the path is "
                f"correct (relative to the folder you're running from, or "
                f"use a full/absolute path), and that it's a folder, not a "
                f"file (use profile_document(...) for a single file)."
            )

        # os.walk descends into subdirectories on its own, so this picks up
        # every .txt file no matter how deeply nested. Sorting filenames
        # keeps the iteration order (and therefore dict order) deterministic.
        texts = {}
        for root, _dirs, files in os.walk(path):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                require_txt_extension(fpath)
                # Key by path relative to the corpus root (e.g.
                # "subfolder/essay.txt") rather than just the filename, so
                # two files named the same in different subfolders don't
                # overwrite each other in the results dict.
                rel_name = os.path.relpath(fpath, path)
                with open_text_file(fpath, encoding) as f:
                    texts[rel_name] = f.read()

        if not texts:
            raise ValueError(
                f"The folder '{path}' exists, but doesn't contain any "
                f".txt files (checked all subfolders too). Add some .txt "
                f"files to it, or double-check this is the folder you meant."
            )
        return self.profile_texts(texts)

    def _profile_tokens(self, classified_tokens: list[tuple[str, str]]) -> ProfileResult:
        """Classify pre-tokenized (word, category) pairs -- see
        `classify_tokens()` for what "category" means ("word",
        "proper_noun", or "numeral")."""
        # Counting by unique word up front (rather than scanning the raw
        # token list) means each word only needs one reference lookup no
        # matter how many times it appears in the text.
        word_counts: Counter = Counter()
        # A given surface word should always resolve to the same category
        # (it's a property of the token, not the occurrence), so the first
        # sighting is as good as any.
        word_category: dict[str, str] = {}
        for word, category in classified_tokens:
            word_counts[word] += 1
            word_category.setdefault(word, category)

        total_tokens = len(classified_tokens)
        total_types = len(word_counts)

        band_token_counts: dict[int, int] = {b: 0 for b in range(1, self.reference.num_bands + 1)}
        band_type_counts: dict[int, int] = {b: 0 for b in range(1, self.reference.num_bands + 1)}

        off_list_word_counts: Counter = Counter()
        ignored_word_counts: Counter = Counter()
        proper_noun_word_counts: Counter = Counter()
        numeral_word_counts: Counter = Counter()

        # Classify every unique word into exactly one bucket: ignored, a
        # proper noun / numeral (only if the profiler was told to exclude
        # that category), a frequency band, or off-list. Ignored words are
        # checked first (an explicit, user-supplied override), then the
        # automatic proper-noun/numeral categories, then the reference
        # itself. All of these still count toward the totals above; they
        # just don't land in a band or in off-list.
        for word, count in word_counts.items():
            category = word_category[word]
            if word in self.ignore_words:
                ignored_word_counts[word] = count
                continue
            if category == "proper_noun" and self.exclude_proper_nouns:
                proper_noun_word_counts[word] = count
                continue
            if category == "numeral" and self.exclude_numerals:
                numeral_word_counts[word] = count
                continue
            band = self.reference.band_of(word)
            if band is None:
                off_list_word_counts[word] = count
            else:
                band_token_counts[band] += count
                band_type_counts[band] += 1

        def pct(n, d):
            return (n / d * 100.0) if d else 0.0

        band_token_pct = {b: pct(n, total_tokens) for b, n in band_token_counts.items()}
        band_type_pct = {b: pct(n, total_types) for b, n in band_type_counts.items()}

        # Running total of token % across bands 1..b, in band order: "how
        # much of the text is covered by the N most frequent bands," the
        # classic Lexical Frequency Profile coverage curve.
        cumulative_token_pct: dict[int, float] = {}
        running = 0.0
        for b in sorted(band_token_pct.keys()):
            running += band_token_pct[b]
            cumulative_token_pct[b] = running

        # most_common() with no argument returns every entry sorted by
        # count descending, which is exactly the "most frequent first"
        # ordering these word lists are documented to have.
        off_list_tokens = sum(off_list_word_counts.values())
        ignored_tokens = sum(ignored_word_counts.values())
        proper_noun_tokens = sum(proper_noun_word_counts.values())
        numeral_tokens = sum(numeral_word_counts.values())

        return ProfileResult(
            total_tokens=total_tokens,
            total_types=total_types,
            band_token_counts=band_token_counts,
            band_type_counts=band_type_counts,
            band_token_pct=band_token_pct,
            band_type_pct=band_type_pct,
            cumulative_token_pct=cumulative_token_pct,
            off_list_tokens=off_list_tokens,
            off_list_types=len(off_list_word_counts),
            off_list_pct_tokens=pct(off_list_tokens, total_tokens),
            off_list_words=[w for w, _ in off_list_word_counts.most_common()],
            ignored_tokens=ignored_tokens,
            ignored_types=len(ignored_word_counts),
            ignored_pct_tokens=pct(ignored_tokens, total_tokens),
            ignored_words=[w for w, _ in ignored_word_counts.most_common()],
            proper_noun_tokens=proper_noun_tokens,
            proper_noun_types=len(proper_noun_word_counts),
            proper_noun_pct_tokens=pct(proper_noun_tokens, total_tokens),
            proper_noun_words=[w for w, _ in proper_noun_word_counts.most_common()],
            numeral_tokens=numeral_tokens,
            numeral_types=len(numeral_word_counts),
            numeral_pct_tokens=pct(numeral_tokens, total_tokens),
            numeral_words=[w for w, _ in numeral_word_counts.most_common()],
            word_counts=word_counts,
            num_bands=self.reference.num_bands,
            band_ranges=self.reference.band_ranges,
        )
