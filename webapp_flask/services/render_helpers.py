"""Pure-Python helpers that turn a ProfileResult (plus the LexicalProfiler
that produced it) into render-ready data for the results templates.

Ported from webapp/app.py's band_color/readable_text_color/word_table
logic and the inline highlight/legend-building code in its "Results"
section -- same colors, same math, no Streamlit/HTML-string building
(templates do the actual markup).
"""

from __future__ import annotations

from lexical_profiler.tokenizer import POS_DISPLAY_NAMES

# Sequential blue ramp for frequency bands (light = most frequent/easiest,
# dark = least frequent), plus fixed status colors -- identical to
# webapp/app.py's BAND_RAMP/OFF_LIST_COLOR/etc.
BAND_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
OFF_LIST_COLOR = "#d03b3b"
IGNORED_COLOR = "#898781"
PROPER_NOUN_COLOR = "#8a5cc7"
DIGIT_COLOR = "#c78a3a"


def band_color(band: int, num_bands: int) -> str:
    if num_bands <= 1:
        return BAND_RAMP[-1]
    idx = round((band - 1) / (num_bands - 1) * (len(BAND_RAMP) - 1))
    return BAND_RAMP[idx]


def readable_text_color(hex_color: str) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#0b0b0b" if luminance > 0.6 else "#ffffff"


def band_for_coverage(result, target_pct: float) -> int | None:
    """First band (in ascending order) whose cumulative token coverage
    reaches `target_pct`, or None if no band reaches it."""
    return next(
        (b for b in sorted(result.cumulative_token_pct)
         if result.cumulative_token_pct[b] >= target_pct),
        None,
    )


def max_coverage_pct(result) -> float:
    """Highest cumulative token coverage reached across all known bands
    (i.e. the coverage of the last band), for showing how close a text
    got to the 95%/98% thresholds when neither was reached."""
    if not result.cumulative_token_pct:
        return 0.0
    return result.cumulative_token_pct[max(result.cumulative_token_pct)]


def legend_entries(result, profiler) -> list[dict]:
    num_bands = result.num_bands
    bands = sorted(result.band_token_counts.keys())
    entries = []
    for b in bands:
        color = band_color(b, num_bands)
        entries.append({
            "label": result.band_label(b), "bg": color, "fg": readable_text_color(color),
        })
    entries.append({"label": "off-list", "bg": OFF_LIST_COLOR, "fg": "#ffffff"})
    entries.append({"label": "ignored", "bg": IGNORED_COLOR, "fg": "#ffffff"})
    if profiler.exclude_proper_nouns:
        entries.append({"label": "proper noun", "bg": PROPER_NOUN_COLOR, "fg": "#ffffff"})
    if profiler.exclude_digits:
        entries.append({"label": "digit", "bg": DIGIT_COLOR, "fg": "#ffffff"})
    return entries


_STATUS_STYLE = {
    "off_list": (OFF_LIST_COLOR, "#ffffff", "Off-list"),
    "ignored": (IGNORED_COLOR, "#ffffff", "Ignored"),
    "proper_noun": (PROPER_NOUN_COLOR, "#ffffff", "Proper noun"),
    "digit": (DIGIT_COLOR, "#ffffff", "Digit"),
}


def highlighted_tokens(profiler, text: str, result) -> list[dict]:
    num_bands = result.num_bands
    out = []
    for tok in profiler.highlight(text):
        entry = {"text": tok.text, "whitespace": tok.whitespace, "status": tok.status}
        if tok.status == "band":
            color = band_color(tok.band, num_bands)
            entry.update(
                bg=color, fg=readable_text_color(color),
                title=f"Band {result.band_label(tok.band)}",
            )
        elif tok.status in _STATUS_STYLE:
            bg, fg, title = _STATUS_STYLE[tok.status]
            entry.update(bg=bg, fg=fg, title=title)
        out.append(entry)
    return out


def word_table_rows(words: list[str], word_counts, pos_tagged: bool) -> list[dict]:
    """One row per off-list/ignored/proper-noun/digit word, with its
    count -- split into word + part-of-speech columns for a POS-tagged
    reference (each word is a "lemma_code" composite key in that case)."""
    if not pos_tagged:
        return [{"word": w, "count": word_counts[w]} for w in words]
    rows = []
    for w in words:
        lemma, _, code = w.rpartition("_")
        rows.append({
            "word": lemma if lemma else w,
            "pos": POS_DISPLAY_NAMES.get(code, code),
            "count": word_counts[w],
        })
    return rows
