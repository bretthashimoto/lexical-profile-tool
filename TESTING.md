# Tester's Guide

Thanks for taking a look at `lexical_profiler`! This doc walks you through
getting it running and gives you concrete things to try. You don't need to
know Python well — most of this is command-line usage — but a little
comfort with a terminal will help.

If anything below doesn't work the way it's described, **that's useful
feedback in itself** — see [What to tell me](#what-to-tell-me) at the
bottom.

## 0. Get access

This repo is currently private. If you can't see
https://github.com/bretthashimoto/lexical-profile-tool, ask Brett to add
you as a collaborator first.

## 1. Set up (5-10 minutes)

You'll need **Python 3.10 or newer** and **git**.

```bash
# Check your Python version -- must be 3.10+
python --version

# Clone the repo
git clone https://github.com/bretthashimoto/lexical-profile-tool.git
cd lexical-profile-tool

# Create and activate a virtual environment (recommended, not required)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows (PowerShell/cmd)

# Install the package
pip install -e .
```

That installs the tool and spaCy (its only real dependency). No language
model download is required to get started — tokenization works out of the
box.

## 2. Smoke test (2 minutes)

Confirm everything's working using the sample files already in the repo:

```bash
python -m lexical_profiler \
    --reference-corpus examples/corpus \
    --target examples/targets/student_essay_1.txt
```

You should see a table breaking the essay down into frequency bands, an
"Off-list" row, and a sample of unfamiliar words. If you see that, setup
worked — move on to trying it for real.

For the full walkthrough of what everything means, see the
[README tutorial](README.md#tutorial-profiling-a-couple-of-nature-and-economy-essays).

## 3. Try it on your own text

This is the part that actually matters. Grab a few paragraphs of real
writing — an email, an essay, a blog post draft, anything you have as
plain text — save it as `my_text.txt`, and profile it:

```bash
python -m lexical_profiler \
    --reference-corpus examples/corpus \
    --target my_text.txt
```

The reference corpus in `examples/corpus/` is tiny (three short articles),
so almost everything will come back "off-list" — that's expected, not a
bug. If you want more realistic numbers, try building a reference from a
bigger source instead — a folder of your own documents, or an existing
frequency list (`--reference-wordlist`, see the README for format details).

Look at the output and ask yourself: **does this match my intuition about
the text?** Words you'd call "common" should mostly land in the earliest
bands; words you'd call "unusual" or technical should mostly land off-list.
If something looks backwards, that's worth flagging.

## 4. Things specifically worth trying

Work through as many of these as you have patience for — each one
exercises a different part of the tool:

- [ ] **Whole folder at once**: `--target-dir` on a folder of several
      `.txt` files instead of a single `--target`.
- [ ] **Export a report**: add `--out-csv report.csv --out-json report.json`
      and open the resulting files. Do they make sense on their own,
      without the CLI's console output next to them?
- [ ] **Ignore list**: try `--ignore-list examples/ignore_list.txt` (or
      write your own with a few proper nouns) and confirm those words move
      from "Off-list" to "Ignored" instead of disappearing.
- [ ] **Different band size**: `--band-size 500` vs the default `1000` —
      does the table stay readable?
- [ ] **A non-English text**, if you have one: `--language es` (or `de`,
      `fr`, etc.) with a Spanish/German/French `.txt` file. No model
      download needed for basic tokenization.
- [ ] **`--lemmatize`**: run the same target with and without this flag
      (needs a downloaded language model — `--download-model` fetches one
      automatically) and compare the off-list words. Does lemmatizing
      change which words get flagged as unknown, the way you'd expect?
- [ ] **Deliberately break it**: point `--target` at a `.pdf` or `.docx`
      file, or a file that doesn't exist, or an empty folder. Read the
      error message it gives you — is it clear what went wrong and what
      to do next, or would you have been stuck?
- [ ] **`--help`**: run `python -m lexical_profiler --help` cold, without
      reading the README first. Could you figure out how to do something
      useful from that alone?

## 5. If you're comfortable with Python

There's an equivalent Python API (`Reference`, `LexicalProfiler`, `report`)
that the CLI is a thin wrapper around — the
[README](README.md#tutorial-profiling-a-couple-of-nature-and-economy-essays)
walks through it step by step, and `example.py` is a runnable end-to-end
script (`python example.py`). If you try this route, I'm especially
interested in whether the API feels intuitive to call without hand-holding.

## What to tell me

Please don't just say "seems fine" — I'd rather hear specifics, even small
ones. Concretely:

1. **Where did you get stuck**, if anywhere — setup, a specific command, an
   error message, or interpreting the output?
2. **Was the summary table easy to read at a glance?** What would make it
   clearer (different wording, a different layout, more/less detail)?
3. **Did any error message confuse you**, or leave you unsure what to fix?
4. **Did anything behave differently than you expected** going in?
5. **Was there something you wanted to do that you couldn't figure out how
   to do** (or that isn't supported at all)?
6. **CLI vs. Python API** (if you tried both) — which did you prefer, and
   why?
7. On a scale of 1-5: how easy was it to get from "cloned the repo" to
   "got output I understood"?

Open a [GitHub Issue](https://github.com/bretthashimoto/lexical-profile-tool/issues)
on the repo if you can (that keeps feedback tracked in one place), or just
send it to Brett directly if that's easier. Screenshots or copy-pasted
terminal output are very welcome, especially for anything confusing.
