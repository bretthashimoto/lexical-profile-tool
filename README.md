# lexical_profiler

[![Tests](https://github.com/bretthashimoto/lexical-profile-tool/actions/workflows/tests.yml/badge.svg)](https://github.com/bretthashimoto/lexical-profile-tool/actions/workflows/tests.yml)

A small Python toolkit for **lexical frequency profiling**: measuring how
much of a text's vocabulary falls into common vs. rare/unknown frequency
bands, relative to a reference frequency model.

This is the kind of analysis used in vocabulary and readability research
(in the spirit of Laufer & Nation's Lexical Frequency Profile), where a
text is scored by what % of its words fall in the most frequent 1000
words (K1), the next 1000 (K2), and so on, with everything else marked
"off-list."

> **Trying this out as a tester?** See [TESTING.md](TESTING.md) for a
> guided setup + things to try, and how to send back feedback.

## Three ways to build the reference

1. **From a corpus of texts**: frequencies are counted directly from
   whatever documents you supply.
2. **From an existing word list**: a plain word list (one per line,
   ordered by frequency) or a `word,frequency` / `word<TAB>frequency`
   file, such as a published list (GSL, NGSL, COCA-derived lists, etc).
3. **From a word list bundled with this package**: the
   [Academic Vocabulary List (AVL)](http://www.academicwords.info)
   (Gardner & Davies, 2013), New General Service List (NGSL), New
   Academic Word List (NAWL), or COCA lemma+POS list, so you don't have
   to track down or format the file yourself -- just pass
   `--reference-builtin avl` (or `ngsl`/`nawl`/`coca`).

All three produce the same kind of `Reference` object, so the rest of the
pipeline (profiling, reporting) doesn't care which one you used.

## Install

```bash
pip install -r requirements.txt   # installs spaCy
```

Tokenization works out of the box for any spaCy-supported language, even
without downloading a language model (spaCy falls back to a rule-based
tokenizer for that language). **Lemmatization** requires an installed
spaCy pipeline for the language you're using:

```bash
python -m spacy download en_core_web_sm   # English
python -m spacy download es_core_news_sm  # Spanish
python -m spacy download de_core_news_sm  # German
python -m spacy download fr_core_news_sm  # French
# ...etc. See https://spacy.io/models for all supported languages.
```

If you request lemmatization for a language whose pipeline isn't
installed, the tool silently falls back to using surface word forms
instead of raising an error. Check ahead of time with
`lexical_profiler.lemmatizer_available("es")`, or fetch a model
programmatically with `lexical_profiler.download_model("es")` (or via
`--download-model` on the CLI).

## What your input files should look like

**Every file this package reads has to be a plain `.txt` file.** Word
documents, PDFs, CSVs with a `.csv` extension, etc. are rejected with a
clear error telling you to convert/rename them first. This keeps
tokenization predictable, since "plain text" is the one format every
tool can agree on.

There's a working set of examples checked into [`examples/`](examples/)
that the tutorial below uses directly, laid out like this:

```
examples/
  corpus/                        # a reference corpus: just plain text
    animals.txt
    weather.txt
    economy.txt
  targets/                       # texts you want to *profile*
    student_essay_1.txt
    student_essay_2.txt
  wordlists/
    simple_wordlist.txt          # one word per line, no frequencies
    frequency_wordlist.txt       # "word,frequency" pairs
  ignore_list.txt                # names/brand words to exclude from scoring
```

A **corpus file** (used to *build* a reference) is nothing fancier than
some representative prose, e.g. `examples/corpus/animals.txt`:

```text
The fox trotted through the quiet forest while the owl watched from a
high branch. Foxes are clever hunters, and owls are silent fliers who
rule the night. ...
```

A **target file** (used to *profile*) looks exactly the same: any
plain-text document you want scored against the reference, e.g.
`examples/targets/student_essay_1.txt` -- two short public-domain fables
from Aesop's Fables: A New Translation (V. S. Vernon Jones, 1912), via
[the Internet Archive](https://archive.org/stream/aesopsfablesanew11339gut/11339.txt).

Point at a whole **folder** instead of a single file and every `.txt`
file inside it (subfolders included) is picked up automatically. That
works for both corpus-building and target-profiling.

A **plain word list** (`examples/wordlists/simple_wordlist.txt`) is one
word per line, already sorted most-to-least frequent, with `#` comments
and blank lines allowed:

```text
# A plain word list: one word per line, ordered most to least frequent.
the
be
to
of
and
...
```

A **frequency word list** (`examples/wordlists/frequency_wordlist.txt`)
adds a count per word: the format is auto-detected, and rows are
re-sorted by frequency regardless of what order they're in the file:

```text
the,22038615
be,12545825
to,12145630
...
fox,1450
owl,980
```

An **ignore list** (`examples/ignore_list.txt`) is the same one-word-per-
line format, used to keep specific words (usually proper nouns, brand
names, or made-up terms) out of the band/off-list scoring entirely --
handy when a target text has names like `Zocharias` or brand terms like
`Fintastic` that aren't really "vocabulary difficulty." The bundled
example targets don't happen to need any, so the file ships empty
(just the explanatory comment) by default.

## Tutorial: profiling two Aesop's Fables passages

Let's actually use the files above. We want to know how much of two
short passages of Aesop's Fables (two fables apiece) falls outside
everyday vocabulary, relative to a small reference built from three
sample articles.

### Step 1: Build a reference

A `Reference` is just a lookup table: word → rank → frequency band. Build
one from the sample corpus folder:

```python
from lexical_profiler import Reference, LexicalProfiler

reference = Reference.from_corpus("examples/corpus", band_size=20)
print(reference.source_description)
# corpus (3 document(s), 168 total words, 129 unique words, language=English)
```

`band_size=20` means "put the 20 most frequent words in band 1, the next
20 in band 2," and so on. A real project profiling English essays would
use the standard `band_size=1000` (the default) against a much bigger
corpus, or a published list (see Step 6). Small numbers are just easier
to read in a tutorial.

### Step 2: Set up the profiler (and tell it what to ignore)

```python
ignore_words = [
    line.strip()
    for line in open("examples/ignore_list.txt")
    if line.strip() and not line.startswith("#")
]

profiler = LexicalProfiler(reference, ignore_words=ignore_words)
```

`ignore_words` keeps any names/brand terms you list from showing up as
"off-list" (i.e. unknown/rare) vocabulary -- they're not vocabulary
difficulty, and it'd be misleading to score them that way. The bundled
example targets don't have any such words, so `ignore_words` is just an
empty list here; it still works exactly the same way if you add some.

### Step 3: Profile a single text

```python
result = profiler.profile_document("examples/targets/student_essay_1.txt")
print(result.summary())
```

```text
Tokens: 306   Types: 155

Band          Tokens    % Tokens     Cum %     Types     % Types
1-20              61      19.93%    19.93%         7       4.52%
21-40              6       1.96%    21.90%         4       2.58%
41-60              7       2.29%    24.18%         3       1.94%
61-80              2       0.65%    24.84%         2       1.29%
81-100             8       2.61%    27.45%         2       1.29%
101-120            1       0.33%    27.78%         1       0.65%
121-129            7       2.29%    30.07%         1       0.65%
Off-list         214      69.93%                 135

Sample off-list words: wolf, he, at, so, his, hare, tortoise, for, i,
race, 's, but, as, had, boy, sheep, fun, slow, 'll, you  (+115 more)
```

Reading this: 306 tokens total, 155 of them unique types. About 20% of the
text's tokens are in band 1 (the most common words in our tiny
reference), and a big chunk (69.9%) is "off-list": words our 3-article
reference has just never seen, like `wolf` or `tortoise` -- unsurprising,
since a 129-word reference vocabulary barely covers ordinary English, let
alone a Victorian-era fable translation. **Cum %** is the running total
of `% Tokens` through that band: "how much of the text is covered by
the N most frequent bands," the number to watch against the standard
95%/98% reading-comprehension coverage thresholds once you're using a
real-sized reference instead of this tiny tutorial one.

### Step 4: Profile a whole folder of texts at once

Got a stack of texts to score instead of just one? Point
`profile_corpus` at the folder and every `.txt` file inside gets scored
independently:

```python
results = profiler.profile_corpus("examples/targets")
for name, result in results.items():
    print(name, "->", result.total_tokens, "tokens,",
          f"{result.off_list_pct_tokens:.1f}% off-list")
```

```text
student_essay_1.txt -> 306 tokens, 69.9% off-list
student_essay_2.txt -> 232 tokens, 69.4% off-list
```

(`profile_corpus` searches subfolders too, and keys its results by path
relative to the folder you gave it, e.g. `"unit1/essay3.txt"`, so files
with the same name in different subfolders don't collide.)

### Step 5: Export a report

```python
from lexical_profiler import report

report.export_csv(results, "band_coverage.csv")           # one row per band per essay
report.export_json(results, "full_report.json")            # everything, structured
report.export_off_list_csv(results, "unknown_words.csv")   # every off-list word, with counts
report.export_ignored_csv(results, "ignored_words.csv")    # every ignored word, with counts
```

### Step 6: Save your reference so you don't have to rebuild it

Building a reference from a big corpus can be slow; save it once and
reload it instantly next time:

```python
reference.save("nature_economy_reference.json")
reference = Reference.load("nature_economy_reference.json")
```

### Step 7: Or use a published word list instead of a corpus

If you don't have (or don't want to build) a corpus, load an existing
frequency list instead; same `Reference` object comes out either way:

```python
reference = Reference.from_word_list(
    "examples/wordlists/frequency_wordlist.txt", band_size=5,
)
```

Or, to profile against an *academic* vocabulary rather than a general
frequency list, use one of the bundled word lists directly by name -- no
file to download or format:

```python
reference = Reference.from_builtin("avl", band_size=500)
# also available: "ngsl", "nawl", "coca"
```

### Step 8: The same tutorial, from the command line

Everything above has a command-line equivalent, handy for scripting or
for people who'd rather not write Python:

```bash
# Build a reference from the sample corpus, profile one essay
python -m lexical_profiler \
    --reference-corpus examples/corpus --band-size 20 \
    --target examples/targets/student_essay_1.txt \
    --ignore-list examples/ignore_list.txt

# Profile every essay in the folder, ignore the same names, export reports
python -m lexical_profiler \
    --reference-corpus examples/corpus --band-size 20 \
    --target-dir examples/targets \
    --ignore-list examples/ignore_list.txt \
    --out-csv band_coverage.csv --out-json full_report.json \
    --out-off-list-csv unknown_words.csv --out-ignored-csv ignored_words.csv

# Build the reference from a word list instead of a corpus
python -m lexical_profiler \
    --reference-wordlist examples/wordlists/frequency_wordlist.txt \
    --band-size 5 \
    --target examples/targets/student_essay_2.txt

# Or profile against a bundled word list (avl, ngsl, nawl, or coca)
python -m lexical_profiler \
    --reference-builtin avl --band-size 500 \
    --target examples/targets/student_essay_2.txt
```

Run `python -m lexical_profiler --help` to see every flag at once.

## Web app

Prefer clicking buttons over writing Python? There's a Flask web UI (LEAH)
that wraps the same library: upload your own corpus (or a word list),
profile target texts, and explore the results interactively, including a
color-coded text view in the style of LexTutor VocabProfile / AntWordProfiler
(each word shaded by which frequency band it falls in; off-list words in
red).

```bash
pip install -r webapp_flask/requirements.txt
flask --app webapp_flask.app run --debug --no-reload
```

This opens in your browser at `http://localhost:5000`. Load the bundled
example data for an instant demo using the same `examples/` files as the
tutorial above, or upload your own `.txt`/`.docx`/`.pdf` corpus files / word
list and target texts. Band coverage charts, the highlighted text view, and
CSV/JSON export are all available from the results section.

If you turn on **Lemmatize** for a language that doesn't have a spaCy
pipeline installed yet, the UI offers a **Download spaCy model** action
that fetches one on the spot (wrapping `lexical_profiler.download_model`);
if it can't (e.g. the host doesn't allow installing packages at runtime),
it falls back to surface forms, same as the library does everywhere else.

## Parameter reference

### `Reference.from_corpus(...)`

| Parameter | Default | What it does |
|---|---|---|
| `source` | *required* | A `.txt` file path, a directory of `.txt` files (searched recursively), a list of file paths, or a list of raw text strings. A bare string is always treated as a path; if it doesn't exist, you get a clear error rather than it silently being tokenized as if it were the text itself. |
| `band_size` | `1000` | Words per frequency band (band 1 = the `band_size` most frequent words, etc). |
| `lowercase` | `True` | Lowercase every token before counting/matching. |
| `lemmatize` | `False` | Reduce words to a base dictionary form (`running` → `run`) using spaCy. Silently falls back to surface forms if no trained pipeline is installed for `language`; see [`lemmatizer_available`](#other-handy-functions). |
| `min_length` | `1` | Drop tokens shorter than this. |
| `language` | `"en"` | ISO 639-1 code (`"es"`, `"de"`, ...) or a full spaCy model name (`"en_core_web_sm"`). Tokenization works for any language spaCy knows about, with or without a downloaded model. |
| `fine_band_size` | `None` | Use a narrower band width for the most frequent words (e.g. `100`), up through `fine_grained_until`. Must be paired with it. |
| `fine_grained_until` | `None` | The rank up to which `fine_band_size` applies (e.g. `2000`). |
| `encoding` | `"utf-8"` | Text encoding used to read corpus files. Bytes that don't decode are dropped rather than raising. |

### `Reference.from_word_list(...)`

| Parameter | Default | What it does |
|---|---|---|
| `path` | *required* | Path to a `.txt` word list file (plain list or `word,frequency` pairs, auto-detected). |
| `band_size` | `1000` | Same as above. |
| `lowercase` | `True` | Lowercase every word on load. |
| `has_frequencies` | `None` (auto) | Force `True`/`False` instead of auto-detecting whether each line has a frequency column. |
| `delimiter` | `None` (auto) | Force a specific delimiter between word and frequency; auto-detects tab/comma/whitespace otherwise. |
| `language` | `"en"` | Used later when tokenizing *target* texts profiled against this reference; doesn't affect how the list itself is parsed. |
| `fine_band_size` / `fine_grained_until` | `None` | Same as above. |
| `encoding` | `"utf-8"` | Same as above. |

### `Reference.from_builtin(...)`

| Parameter | Default | What it does |
|---|---|---|
| `name` | *required* | Key of a word list bundled with this package (case-insensitive): `"avl"` (Academic Vocabulary List, lemma+POS), `"ngsl"` (New General Service List), `"nawl"` (New Academic Word List), or `"coca"` (COCA lemma+POS list). |
| `band_size` | `1000` | Same as above. |
| `lowercase` | `True` | Same as above. |
| `language` | `"en"` | Same as above. |
| `fine_band_size` / `fine_grained_until` | `None` | Same as above. |

### `Reference.save(path)` / `Reference.load(path)`

Save/reload a built reference as JSON, so you don't have to re-tokenize a
big corpus every run. `load` raises a clear error if the file is
missing, unreadable, not valid JSON, or wasn't actually produced by
`.save(...)`.

### `LexicalProfiler(reference, min_length=1, ignore_words=None, exclude_proper_nouns=False, exclude_digits=False)`

| Parameter | Default | What it does |
|---|---|---|
| `reference` | *required* | The `Reference` to profile target texts against. |
| `min_length` | `1` | Minimum token length to include when tokenizing target texts. |
| `ignore_words` | `None` | Specific words to pull out of band/off-list scoring entirely. They're also excluded from the text's total token/type counts, and reported separately as `ProfileResult.ignored_*`. Matched case-insensitively if the reference lowercases tokens. Takes priority over `exclude_proper_nouns`/`exclude_digits` if a word matches both. |
| `exclude_proper_nouns` | `False` | Pull proper nouns (detected via the language's POS tagger) out of band/off-list scoring, reported separately as `ProfileResult.proper_noun_*` -- same mechanism as `ignore_words`, just automatic. Requires a trained spaCy pipeline for the reference's language; silently has no effect without one. If `False` (default), proper nouns are profiled like any other word. |
| `exclude_digits` | `False` | Same, for tokens containing a digit character (e.g. `"42"`, `"3.14"`) -> `ProfileResult.digit_*`. Spelled-out number words (e.g. `"twelve"`) are unaffected and stay profiled as ordinary vocabulary. Works regardless of language pipeline availability. If `False` (default), digit tokens are profiled like any other word -- in practice usually landing off-list, since reference word lists/corpora don't carry digits as vocabulary. |

Profiling methods on `LexicalProfiler`:

| Method | Use it for |
|---|---|
| `.profile_text(text)` | A raw string you already have in memory. |
| `.profile_texts({name: text, ...})` | Multiple raw strings at once → `{name: ProfileResult}`. |
| `.profile_document(path, encoding="utf-8")` | A single `.txt` file on disk. |
| `.profile_corpus(path, encoding="utf-8")` | Every `.txt` file in a folder (recursive) → `{relative_path: ProfileResult}`. |

### `ProfileResult`

The return value of every profiling call above. Key fields/methods:

| Field / method | What it is |
|---|---|
| `total_tokens` / `total_types` | Total word count / unique word count for the text, **excluding** any `ignored_*`/`proper_noun_*`/`digit_*` tokens below -- only words that landed in a band or off-list count here, so `band_token_pct` + `off_list_pct_tokens` sum to 100%. |
| `band_token_counts`, `band_token_pct`, `band_type_counts`, `band_type_pct` | Per-band coverage, by dict of band number → value. |
| `cumulative_token_pct` | Running total of `band_token_pct` through each band (dict of band number → value): "how much of the text is covered by the N most frequent bands," the classic Lexical Frequency Profile coverage curve. Compare against the standard 95%/98% reading-comprehension coverage thresholds. |
| `off_list_tokens`, `off_list_types`, `off_list_pct_tokens`, `off_list_words` | Words absent from the reference entirely; `off_list_words` is the **full** list, most-frequent-first (not just a sample). |
| `ignored_tokens`, `ignored_types`, `ignored_pct_tokens`, `ignored_words` | Same, for words matched by `ignore_words`. `ignored_pct_tokens` is a % of the *whole* text (not of `total_tokens`, since these are excluded from it). |
| `proper_noun_tokens`, `proper_noun_types`, `proper_noun_pct_tokens`, `proper_noun_words` | Same, for proper nouns, when `exclude_proper_nouns=True`. Empty/zero otherwise. |
| `digit_tokens`, `digit_types`, `digit_pct_tokens`, `digit_words` | Same, for digit tokens, when `exclude_digits=True`. Empty/zero otherwise. |
| `.summary(max_bands_shown=None, max_off_list_shown=20)` | Human-readable report string. The `max_*_shown` params only limit *this printed view*; the underlying `off_list_words`/`ignored_words` fields always have everything. |
| `.to_dict()` | JSON/API-friendly dict of everything above (also unabridged). |
| `.band_label(band)` | Human-readable band label, e.g. `"1-999"`. |

### `report` module

| Function | Writes |
|---|---|
| `export_json(results, path)` | Every result's full `.to_dict()`, keyed by name. |
| `export_csv(results, path)` | One row per band per text, plus an off-list row and (if any) an ignored row. |
| `export_off_list_csv(results, path)` | One row per off-list word per text, with counts. |
| `export_ignored_csv(results, path)` | One row per ignored word per text, with counts. |
| `export_proper_nouns_csv(results, path)` | One row per proper noun per text, with counts (populated only if `exclude_proper_nouns=True`). |
| `export_digits_csv(results, path)` | One row per digit token per text, with counts (populated only if `exclude_digits=True`). |

All six raise a clear error if the output path's folder doesn't exist or
isn't writable, instead of a raw OS traceback.

### Other handy functions

| Function | What it does |
|---|---|
| `lexical_profiler.lemmatizer_available(language)` | Whether a trained spaCy pipeline (capable of lemmatization) is installed for `language`. |
| `lexical_profiler.download_model(language)` | Download/install the spaCy pipeline for `language`. Returns `True`/`False`. |
| `lexical_profiler.tokenize(text, language="en", lowercase=True, lemmatize=False, min_length=1)` | The raw tokenizer, if you want tokens without profiling anything. Drops pure digit tokens (like punctuation). |
| `lexical_profiler.tokenizer.classify_tokens(text, ...)` | Like `tokenize`, but returns `(word, category)` pairs -- `category` is `"word"`, `"proper_noun"`, or `"digit"` -- and keeps digit tokens instead of dropping them. What `LexicalProfiler` uses internally to support `exclude_proper_nouns`/`exclude_digits`. |

### Command-line flags

| Flag | What it does |
|---|---|
| `--reference-corpus PATH` | Build the reference from a `.txt` file or folder. |
| `--reference-wordlist PATH` | Build the reference from a word list file instead. |
| `--reference-builtin NAME` | Build the reference from a word list bundled with this package (`avl`, `ngsl`, `nawl`, or `coca`). |
| `--reference-saved PATH` | Reload a reference previously written with `--save-reference`. |
| `--save-reference PATH` | Save the built reference to this path for reuse. |
| `--target FILE [FILE ...]` | One or more target files to profile. |
| `--target-dir DIR` | A folder of target `.txt` files to profile independently. |
| `--band-size N` | Words per band (default `1000`). |
| `--fine-band-size N` / `--fine-until N` | Fine-grained bands for the most frequent words; must be given together. |
| `--language CODE` | ISO code or full spaCy model name (default `en`). |
| `--download-model` | Download the spaCy pipeline for `--language` before running. |
| `--lemmatize` | Lemmatize tokens instead of using surface forms. |
| `--min-length N` | Minimum token length to include (default `1`). |
| `--encoding NAME` | Text encoding for reading files (default `utf-8`). |
| `--ignore-words WORD [WORD ...]` | Words to exclude from band/off-list scoring. |
| `--ignore-list PATH` | Same, from a file (merged with `--ignore-words` if both given). |
| `--exclude-proper-nouns` | Report proper nouns separately instead of scoring them like any other word. Requires a trained pipeline for `--language`; silently has no effect without one. |
| `--exclude-digits` | Report digit tokens (e.g. `42`, `3.14`) separately instead of scoring them like any other word. Spelled-out number words (e.g. `twelve`) are unaffected. |
| `--out-json / --out-csv / --out-off-list-csv / --out-ignored-csv / --out-proper-nouns-csv / --out-digits-csv PATH` | Export reports. |
| `--max-off-list-shown N` | How many off-list words to print in the console summary (default `20`). |

## What gets measured

For each text, `ProfileResult` reports, per frequency band and overall:

- **Token counts / percentages**: coverage by running word count (how
  much of the text, word-for-word, is "easy"/common vocabulary).
- **Type counts / percentages**: coverage by unique word (vocabulary
  breadth per band, independent of repetition).
- **Type/Token counts**: both overall and per band, so you can see
  vocabulary breadth (unique words) alongside raw word usage.
- **Cumulative token coverage**: the running total of token % through
  each band (band 1, then bands 1-2, then bands 1-3, ...), i.e. "what
  share of the text is accounted for by the N most frequent bands." This
  is the classic Lexical Frequency Profile coverage curve, and the number
  to compare against reading-comprehension research: roughly 95% coverage
  is the threshold for "adequate" comprehension of a text, 98% for
  "comfortable" comprehension without needing to guess unknown words from
  context (Laufer & Ravenhorst-Kalovski, 2010).
- **Off-list words**: words absent from the reference entirely (not in
  any band), useful for spotting rare, technical, or misspelled
  vocabulary, or vocabulary specific to a domain not covered by the
  reference.
- **Ignored words**: specific words you deliberately excluded from
  scoring (via `ignore_words`), excluded from the totals and broken out
  separately instead of polluting the off-list.
- **Proper nouns / digits**: names, places, and digit tokens (e.g. `"42"`)
  are profiled like any other word by default (digits in practice usually
  land off-list, since reference word lists/corpora don't carry them as
  vocabulary). Spelled-out number words (e.g. `"twelve"`) are unaffected
  either way. Pass `exclude_proper_nouns=True` and/or
  `exclude_digits=True` to break either out into its own category
  instead, the same way `ignore_words` works.

## Notes & design choices

- Only plain `.txt` files are accepted anywhere this package reads from
  disk (corpus files, target texts, word lists, ignore lists). Anything
  else raises a clear error explaining what to do about it, rather than
  silently mis-reading a binary file as text.
- Errors throughout the package are written for the person running the
  tool, not just the person who wrote it. A bad path, a missing folder,
  or a corrupted saved reference all explain what's wrong and what to try
  next, instead of a raw Python traceback.
- Tokenization uses spaCy, so word-boundary rules are language-aware
  (much better than a generic regex tokenizer for many languages, e.g.
  Chinese/Japanese segmentation, German compounds, contractions).
  Tokens containing no alphabetic character (pure punctuation/numbers)
  are dropped. Adjust `min_length` to filter very short tokens.
- `language` accepts an ISO 639-1 code (`en`, `es`, `de`, `fr`, `zh`,
  `ja`, `ru`, ...) or a full spaCy model name (`en_core_web_sm`).
  Tokenization works even if no pipeline is installed for that language
  (spaCy provides a rule-based tokenizer for essentially any supported
  language via `spacy.blank(lang)`).
- Lemmatization (`--lemmatize` / `lemmatize=True`) requires a trained
  spaCy pipeline for the chosen language to be installed. If it isn't,
  the tool silently falls back to surface word forms rather than
  erroring. Check availability with `lemmatizer_available(language)`, or
  fetch a model with `download_model(language)` / `--download-model`.
- `Reference.from_word_list` auto-detects whether the file has
  `word,frequency` pairs or is just an ordered word list; you can force
  either interpretation via `has_frequencies=True/False`. Its `language`
  parameter only affects how *target* texts are later tokenized when
  profiled against that reference; not how the list itself is read.
- Bands are labeled as real rank ranges (e.g. "1-999", "1000-1999",
  "2000-2999", ...), not the "K1/K2/K3..." shorthand, so it's immediately
  clear which word ranks each band covers. Default band width is 1000
  words; pass `fine_band_size` + `fine_grained_until` to use a narrower
  width for the most frequent words (e.g. `fine_band_size=100,
  fine_grained_until=2000` gives 100-word bands through rank 2000, then
  normal 1000-word bands after that).
- A bare directory/corpus path that doesn't exist raises an error rather
  than being silently treated as literal text. If you actually want to
  profile a literal string, wrap it in a list:
  `Reference.from_corpus([your_text])`.

## Files

```
lexical_profiler/
  __init__.py      # public API
  tokenizer.py      # spaCy tokenizer/lemmatizer, multi-language support
  reference.py       # Reference: build from corpus, word list, or bundled list; save/load
  profiler.py         # LexicalProfiler + ProfileResult (the core analysis)
  report.py             # CSV/JSON export helpers
  cli.py                  # command-line interface
  __main__.py                # `python -m lexical_profiler` entry point
  data/                          # word lists bundled with the package (AVL, NGSL, NAWL, COCA)
webapp_flask/
  app.py                            # Flask web UI (`flask --app webapp_flask.app run`)
  blueprints/                            # route handlers (pages, profile, reference, jobs, export, chart)
  services/                                 # session storage, job management, etc.
  templates/                                   # Jinja templates
  static/                                         # CSS/JS/images
  requirements.txt                                   # deploy-time deps
examples/
  corpus/                       # sample reference corpus used in the tutorial
  targets/                      # sample target texts used in the tutorial
  wordlists/                    # sample plain + frequency word lists
  ignore_list.txt               # sample ignore list
example.py                        # end-to-end runnable demo
requirements.txt
```
