"""
Streamlit web UI for lexical_profiler.

Lets a user build a reference from their own uploaded corpus (or word list),
profile target texts against it, and explore the results interactively:
a band-coverage chart, a LexTutor/AntWordProfiler-style color-coded text
view, and CSV/JSON export.

Run with:
    streamlit run webapp/app.py
"""

from __future__ import annotations

import base64
import sys
import tempfile
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Guarantees `import lexical_profiler` works regardless of whether the
# package was pip-installed into this environment -- the source directory
# sits right next to this file's parent, so this is robust to hosts (like
# Streamlit Community Cloud) that only install webapp/requirements.txt.
sys.path.insert(0, str(REPO_ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from components.file_dir_uploader import file_dir_uploader  # noqa: E402
from text_extract import extract_text  # noqa: E402

from lexical_profiler import (  # noqa: E402
    LexicalProfiler,
    Reference,
    download_model,
    lemmatizer_available,
)
from lexical_profiler import report as report_mod  # noqa: E402
from lexical_profiler.reference import BUILTIN_WORD_LISTS  # noqa: E402
from lexical_profiler.tokenizer import LANGUAGE_DISPLAY_NAMES, POS_DISPLAY_NAMES  # noqa: E402

st.set_page_config(page_title="LEAH — Lexical Analysis", page_icon="📖", layout="wide")

# Languages this deployment actually pre-installs a trained spaCy pipeline
# for (see the model wheels in webapp/requirements.txt) -- the dropdown is
# limited to these so lemmatization always works out of the box, rather
# than offering languages that would silently fall back to surface forms
# until a model is downloaded. Other languages still work fine via the
# library directly (Reference.from_corpus(..., language="ja"), etc.).
WEBAPP_LANGUAGES = ["en", "es", "fr", "de"]

# ---------------------------------------------------------------------------
# Color scheme: a validated sequential blue ramp for frequency bands
# (light = most frequent/easiest, dark = least frequent), plus fixed status
# colors for off-list (critical) and ignored (muted) words.
# ---------------------------------------------------------------------------
BAND_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
OFF_LIST_COLOR = "#d03b3b"
IGNORED_COLOR = "#898781"
PROPER_NOUN_COLOR = "#8a5cc7"
DIGIT_COLOR = "#c78a3a"

# The LEAH mark: an open book whose pages are bars decaying like a Zipf
# curve (tall/light on the left, falling into a long low/dark tail on the
# right) -- a nod to word-frequency distributions, and to the "L" in LEAH.
_LOGO_SVG_SRC = """
<svg width="60" height="60" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <clipPath id="leahClipLeft">
      <path d="M256,150 C210,140 150,155 118,190 L118,370 C118,398 165,415 256,406 Z"/>
    </clipPath>
    <clipPath id="leahClipRight">
      <path d="M256,150 C302,140 362,155 394,190 L394,370 C394,398 347,415 256,406 Z"/>
    </clipPath>
  </defs>
  <path d="M256,150 C210,140 150,155 118,190 L118,370 C118,398 165,415 256,406 Z"
        fill="#eef4fc" stroke="#184f95" stroke-width="3"/>
  <path d="M256,150 C302,140 362,155 394,190 L394,370 C394,398 347,415 256,406 Z"
        fill="#eef4fc" stroke="#184f95" stroke-width="3"/>
  <g clip-path="url(#leahClipLeft)">
    <rect x="132" y="195" width="20" height="190" fill="#86b6ef"/>
    <rect x="156" y="330" width="20" height="55" fill="#6da7ec"/>
    <rect x="180" y="359" width="20" height="26" fill="#5598e7"/>
    <rect x="204" y="369" width="20" height="16" fill="#3987e5"/>
    <rect x="228" y="375" width="20" height="10" fill="#2a78d6"/>
  </g>
  <g clip-path="url(#leahClipRight)">
    <rect x="264" y="377" width="20" height="8" fill="#256abf"/>
    <rect x="288" y="379" width="20" height="6" fill="#1c5cab"/>
    <rect x="312" y="380" width="20" height="5" fill="#184f95"/>
    <rect x="336" y="381" width="20" height="4" fill="#104281"/>
    <rect x="360" y="382" width="20" height="3" fill="#0d366b"/>
  </g>
  <path d="M256,150 C210,140 150,155 118,190 L118,370 C118,398 165,415 256,406 Z"
        fill="none" stroke="#184f95" stroke-width="3"/>
  <path d="M256,150 C302,140 362,155 394,190 L394,370 C394,398 347,415 256,406 Z"
        fill="none" stroke="#184f95" stroke-width="3"/>
  <line x1="256" y1="146" x2="256" y2="410" stroke="#0d366b" stroke-width="5"
        stroke-linecap="round"/>
</svg>
"""
# Collapsed to one line: a raw HTML block spanning multiple lines with a
# blank line in it gets split by Streamlit's markdown parser, so anything
# after the split renders as literal text instead of HTML.
LOGO_SVG = " ".join(line.strip() for line in _LOGO_SVG_SRC.strip().splitlines())


def band_color(band: int, num_bands: int) -> str:
    if num_bands <= 1:
        return BAND_RAMP[-1]
    idx = round((band - 1) / (num_bands - 1) * (len(BAND_RAMP) - 1))
    return BAND_RAMP[idx]


def readable_text_color(hex_color: str) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#0b0b0b" if luminance > 0.6 else "#ffffff"


def span_html(inner: str, bg: str, fg: str = "#fff", title: str = "",
              pad: str = "1px 6px", margin_right: str = "0") -> str:
    title_attr = f' title="{escape(title)}"' if title else ""
    style = (
        f"background:{bg};color:{fg};border-radius:3px;padding:{pad};"
        f"margin-right:{margin_right};"
    )
    return f'<span style="{style}"{title_attr}>{inner}</span>'


def word_table(words: list[str], word_counts, pos_tagged: bool) -> pd.DataFrame:
    """Build a {word, count} table for a list of off-list/ignored words --
    or, for a POS-tagged reference, split each "lemma_code" entry into
    separate "word" and "part of speech" columns instead of showing the
    raw composite string, since that's the one place such a word is
    displayed directly to the user."""
    if not pos_tagged:
        return pd.DataFrame({
            "word": words,
            "count": [word_counts[w] for w in words],
        })
    lemmas, parts_of_speech = [], []
    for w in words:
        lemma, _, code = w.rpartition("_")
        lemmas.append(lemma if lemma else w)
        parts_of_speech.append(POS_DISPLAY_NAMES.get(code, code))
    return pd.DataFrame({
        "word": lemmas,
        "part of speech": parts_of_speech,
        "count": [word_counts[w] for w in words],
    })


def export_to_bytes(export_fn, results, suffix) -> bytes:
    """Reuse the library's file-based export functions for downloads, by
    writing to a throwaway temp file and reading the bytes back."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        export_fn(results, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def build_with_progress(build_fn):
    """Run `build_fn(progress_callback)` -- a call to Reference.from_corpus
    or Reference.add_texts -- behind a progress bar with live status text
    ("Loading documents...", "Tokenizing...", "Lemmatizing...", etc.),
    since building a reference from a real corpus can take a while."""
    placeholder = st.empty()
    bar = placeholder.progress(0, text="Preparing...")

    def callback(current: int, total: int, message: str) -> None:
        bar.progress(current / total if total else 1.0, text=message)

    try:
        return build_fn(callback)
    finally:
        placeholder.empty()


def read_word_list(raw: str) -> list[str]:
    return [w.strip() for w in raw.splitlines() if w.strip() and not w.strip().startswith("#")]


def read_uploaded_texts_cached(files, cache_key: str) -> list[tuple[str, str]]:
    """Expand uploaded .txt/.docx/.pdf files into (name, text) pairs,
    skipping (with a warning) any file whose text couldn't be extracted.
    Shows a progress bar while extracting (.docx/.pdf parsing can take a
    noticeable moment for several files) and skips re-extracting when the
    same set of files (by Streamlit's own per-file id) was already decoded
    on a previous rerun -- otherwise every unrelated widget interaction/
    rerun would re-parse every file again."""
    files_sig = tuple(f.file_id for f in files) if files else ()
    sig_key, out_key = f"{cache_key}_sig", f"{cache_key}_out"
    if st.session_state.get(sig_key) != files_sig:
        placeholder = st.empty()
        bar = placeholder.progress(0, text="Preparing...")
        out = []
        for i, f in enumerate(files, start=1):
            try:
                out.append((f.name, extract_text(f.name, f.getvalue())))
            except ValueError as e:
                st.warning(str(e))
            bar.progress(i / len(files), text=f"Extracting {f.name} ({i} of {len(files)})...")
        placeholder.empty()
        st.session_state[out_key] = out
        st.session_state[sig_key] = files_sig
    return st.session_state.get(out_key, [])


def decode_picked_files(picked) -> dict[str, str]:
    """Turn file_dir_uploader's {"name", "content_b64"} entries into a
    {name: extracted_text} dict, skipping (with a warning) any file whose
    text couldn't be extracted."""
    out = {}
    for item in picked or []:
        try:
            data = base64.b64decode(item["content_b64"])
            out[item["name"]] = extract_text(item["name"], data)
        except ValueError as e:
            st.warning(str(e))
    return out


def cached_decode(picked, cache_key: str) -> dict[str, str]:
    """Like decode_picked_files, but skips re-extracting when the same set
    of files (by name) was already decoded on a previous rerun -- matters
    now that PDFs/DOCX are supported, since re-parsing them on every
    unrelated widget interaction/rerun would be wasteful."""
    names_sig = tuple(sorted(item["name"] for item in picked)) if picked else ()
    sig_key, texts_key = f"{cache_key}_sig", f"{cache_key}_texts"
    if st.session_state.get(sig_key) != names_sig:
        st.session_state[texts_key] = decode_picked_files(picked)
        st.session_state[sig_key] = names_sig
    return st.session_state[texts_key]


for key in ("reference", "profiler", "results", "target_texts"):
    st.session_state.setdefault(key, None)


def load_example_data(*, band_size=20, language="en", lemmatize=True,
                       fine_band_size=None, fine_grained_until=None,
                       coarse_band_size=None, coarse_grained_from=None):
    ref = Reference.from_corpus(
        str(REPO_ROOT / "examples" / "corpus"), band_size=band_size, language=language,
        lemmatize=lemmatize, fine_band_size=fine_band_size, fine_grained_until=fine_grained_until,
        coarse_band_size=coarse_band_size, coarse_grained_from=coarse_grained_from,
    )
    ignore_list_text = (REPO_ROOT / "examples" / "ignore_list.txt").read_text(encoding="utf-8")
    profiler = LexicalProfiler(ref, ignore_words=read_word_list(ignore_list_text))
    targets_dir = REPO_ROOT / "examples" / "targets"
    target_texts = {
        f.name: f.read_text(encoding="utf-8") for f in sorted(targets_dir.glob("*.txt"))
    }
    st.session_state.reference = ref
    st.session_state.profiler = profiler
    st.session_state.results = profiler.profile_texts(target_texts)
    st.session_state.target_texts = target_texts


# ---------------------------------------------------------------------------
# Header banner: fixed and full viewport width (breaks out of Streamlit's
# centered/padded block-container via the `key=` CSS hook below), so it
# stays pinned at the top of the screen edge-to-edge as the page scrolls.
# ---------------------------------------------------------------------------
# Shared between the real (fixed-position) banner and an invisible in-flow
# clone of it (see div.st-key-header_banner_spacer below) that reserves
# exactly the right amount of space for whatever the real banner's actual
# rendered height turns out to be.
_banner_inner_html = f"""
<div style="display:flex; align-items:center; gap:0.7rem;">
    {LOGO_SVG}
    <div style="display:flex; align-items:baseline; gap:0.7rem; flex-wrap:wrap;">
        <span style="
            font-family:'Space Grotesk', sans-serif;
            font-weight:700;
            font-style:italic;
            font-size:2.9rem;
            letter-spacing:0.02em;
            color:#ffffff;
        ">LEAH</span>
        <span style="color:#e8f0fc; font-size:1.5rem;">
            <b>LE</b>xical <b>A</b>nalysis — <b>H</b>ashimoto
        </span>
    </div>
</div>
<div style="color:#d3e2f7; font-size:1.0rem; margin-top:0.01rem; width:100%;">
    Profile the frequency of words in users' texts by determining the
    commonness/rarity of words, relative to a reference you build from your
    own corpus or common word lists in English, Spanish, French, and German.
</div>
"""
_banner_html = f"""
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&display=swap">
<style>
/* Streamlit's own header bar (Share/star/edit/GitHub/menu icons) is
   transparent and sits above everything (z-index far higher than any app
   content can reach) -- rather than reserving separate space below it, our
   banner now starts at the very top of the viewport and shows through
   behind those icons, so they visually sit inside the banner instead of on
   their own strip above it. */
header[data-testid="stHeader"] {{
    background: transparent;
}}
/* Those icons/links (Share, star, edit, GitHub, the "..." menu) are dark
   by default (meant for a light header) and near-invisible against our
   blue banner now that they overlay it. Not all of them are <button>s (the
   GitHub/star/fork indicators render as <a> links), so every descendant is
   covered here rather than just buttons; SVGs without an explicit fill use
   fill="currentColor" and pick up the color from this too, but a couple
   hard-code their own fill, hence the separate svg rule. */
header[data-testid="stHeader"] * {{
    color: #ffffff !important;
}}
header[data-testid="stHeader"] svg {{
    fill: #ffffff !important;
}}
/* Drop the "..." main menu button (Rerun/Settings/Record/Report a
   bug/About) but leave the Share button and Cloud-injected toolbar
   (star/fork/GitHub/edit) alone. */
[data-testid="stMainMenu"] {{
    display: none;
}}
/* Streamlit reserves ~96px of top padding on the main content area for its
   own (now-transparent) header bar. Our banner supplies its own spacer
   below instead, so left alone this padding just adds a blank gap between
   the banner and the tab menu. */
[data-testid="stMainBlockContainer"] {{
    padding-top: 0 !important;
}}
div.st-key-header_banner {{
    position: fixed !important;
    top: 0;
    left: 0;
    z-index: 1000;
    width: 100vw !important;
    box-sizing: border-box;
    background: linear-gradient(135deg, {BAND_RAMP[7]} 0%, {BAND_RAMP[4]} 100%);
    padding: 0.15rem 3vw 0.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
}}
/* `position: fixed` removes the banner from document flow, so a spacer is
   needed below it to keep whatever comes next from being covered. A fixed
   pixel-height spacer kept drifting out of sync, because the banner's
   real height is responsive -- the description text wraps to a different
   number of lines depending on viewport width -- leaving a gap at some
   widths and an overlap at others no matter what single number was
   chosen. This spacer instead renders an exact copy of the banner's own
   content (same padding/width/font-sizes, via the shared
   _banner_inner_html below), just invisible (`visibility: hidden`, which
   -- unlike `display: none` -- still takes up its normal layout space)
   and in normal flow instead of `position: fixed`, so it always reserves
   *exactly* the real banner's rendered height, at any viewport width,
   with no number to keep in sync. */
div.st-key-header_banner_spacer {{
    visibility: hidden;
    width: 100vw !important;
    max-width: 100vw !important;
    box-sizing: border-box;
    padding: 0.15rem 3vw 0.5rem;
}}
/* Streamlit's own default element gap (two of them stack here: one after
   the spacer container, one before this tab row) still shows up between
   the (invisible) spacer above and this tab row -- unlike the old height
   mismatch, that gap is a constant regardless of viewport width/content
   reflow, so it's safe to cancel out with a fixed negative margin here. */
div.st-key-top_menu [role="tablist"] {{
    background: var(--background-color, #ffffff);
    margin-top: -2rem;
}}
/* Streamlit's "running"/"file change" status widget (top-right, next to
   Deploy) shows a Material icon glyph via ligature text ("directions_run"
   while a script is executing) -- collapse that glyph and substitute a
   book emoji instead, on-theme with the rest of the app, with a Y-axis
   flip animation while running so it reads as a page turning rather than
   a static icon. */
[data-testid="stStatusWidget"] [data-testid="stIconMaterial"] {{
    font-size: 0 !important;
}}
[data-testid="stStatusWidget"] [data-testid="stIconMaterial"]::before {{
    content: "📖";
    font-size: 1rem;
    display: inline-block;
    animation: leahBookFlip 1.1s ease-in-out infinite;
}}
@keyframes leahBookFlip {{
    0%, 100% {{ transform: rotateY(0deg); }}
    50% {{ transform: rotateY(180deg); }}
}}
</style>
{_banner_inner_html}
"""
# Collapsed to one line for the same reason as LOGO_SVG above: a blank line
# inside a raw HTML block passed to st.markdown breaks it into two blocks,
# and everything after the break renders as literal text instead of HTML.
with st.container(key="header_banner"):
    st.markdown(
        " ".join(line.strip() for line in _banner_html.strip().splitlines()),
        unsafe_allow_html=True,
    )
# Invisible in-flow clone of the banner content, purely to reserve space
# below the real (position:fixed, out-of-flow) banner above -- see the
# div.st-key-header_banner_spacer CSS rule for why this replaced a fixed
# pixel-height spacer.
with st.container(key="header_banner_spacer"):
    st.markdown(
        " ".join(line.strip() for line in _banner_inner_html.strip().splitlines()),
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Top-level tabs
# ---------------------------------------------------------------------------
tab_build, tab_profile, tab_guide, tab_about, tab_cite, tab_about_me = st.tabs(
    [
        "Build a reference corpus or select a word list", "Profile a text",
        "Step-by-step guide", "About lexical frequency profiling", "How to cite",
        "About me",
    ],
    key="top_menu",
)

with tab_build:
    st.header("1. Build a reference or select a word list")
    source_kind = st.radio(
        "Reference source",
        ["Built-in word list", "Corpus of texts", "Word list", "Saved reference (.json)"],
        help="What to profile target texts against: a published word list bundled "
             "with this app, your own corpus of texts, your own word list file, or "
             "a reference you built earlier and downloaded as .json.",
    )

    band_size = int(st.number_input(
        "Band size", min_value=1, value=1000, step=100,
        help="How many words make up each frequency band, e.g. band 1 = the "
             "1000 most frequent words, band 2 = the next 1000, and so on. "
             "Smaller values give more (narrower) bands; doesn't affect which "
             "words end up off-list.",
    ))
    language_codes = sorted(WEBAPP_LANGUAGES, key=lambda code: LANGUAGE_DISPLAY_NAMES[code])
    language = st.selectbox(
        "Language", language_codes, index=language_codes.index("en"),
        format_func=lambda code: LANGUAGE_DISPLAY_NAMES[code],
        help="Used for tokenization and lemmatization (models for these languages are "
             "pre-installed).",
    )
    lemmatize = st.checkbox(
        "Lemmatize", value=True,
        help="Requires a spaCy pipeline installed for the language; silently falls back to "
             "surface forms otherwise. Uncheck to profile against surface word forms instead.",
    )
    language_name = LANGUAGE_DISPLAY_NAMES[language]
    if lemmatize and not lemmatizer_available(language):
        st.caption(
            f"⚠️ No lemmatizer model installed for {language_name} yet — words will use "
            f"their surface form instead of a lemma until one is installed."
        )
        if st.button(f"Download spaCy model for {language_name}"):
            with st.spinner(f"Downloading a spaCy model for {language_name}..."):
                installed = download_model(language)
            if installed:
                st.success(f"Installed a model for {language_name}.")
                st.rerun()
            else:
                st.error(
                    f"Couldn't download a model for {language_name}. Either this language "
                    f"doesn't have a trained spaCy pipeline (see spacy.io/models), "
                    f"or this host doesn't allow installing packages at runtime."
                )

    with st.expander("Fine-grained bands (optional)"):
        st.caption(
            "Split the most frequent words into narrower bands than the rest of "
            "the list, so a coverage curve doesn't lump e.g. the top 1000 words "
            "together when finer distinctions there matter most."
        )
        use_fine = st.checkbox(
            "Use narrower bands for the most frequent words",
            help="E.g. 100-word bands through rank 2000, then normal-size "
                 "(Band size) bands after that.",
        )
        fine_band_size = None
        fine_grained_until = None
        if use_fine:
            fine_band_size = int(st.number_input(
                "Fine band size", min_value=1, value=100,
                help="Width of each narrow band, e.g. 100 words per band.",
            ))
            fine_grained_until = int(st.number_input(
                "...through rank", min_value=1, value=2000,
                help="The rank up to which the narrower band size applies, e.g. "
                     "2000 for narrow bands through the 2000th most frequent "
                     "word. Normal-size (Band size) bands apply after that.",
            ))

    with st.expander("Coarse-grained bands (optional)"):
        st.caption(
            "Collapse the long tail of least-frequent words into a handful of wide bands "
            "instead of many normal-size ones."
        )
        use_coarse = st.checkbox(
            "Use wider bands for the least frequent words",
            help="E.g. one 10,000-word band covering everything past rank "
                 "50,000, instead of dozens of separate normal-size (Band "
                 "size) bands most profiling runs would never even reach.",
        )
        coarse_band_size = None
        coarse_grained_from = None
        if use_coarse:
            coarse_band_size = int(st.number_input(
                "Coarse band size", min_value=1, value=10000,
                help="Width of each wide band, e.g. 10,000 words per band.",
            ))
            coarse_grained_from = int(
                st.number_input(
                    "...from rank", min_value=1, value=50000,
                    help="The rank from which the wider band size applies, e.g. "
                         "50000 to start collapsing bands past the 50,000th "
                         "most frequent word. Normal-size (Band size) bands "
                         "apply before that (and after the fine-grained section "
                         "above, if any).",
                )
            )

    if source_kind == "Built-in word list":
        st.caption("Profile against a published reference list bundled with this app -- "
                   "no file to find or format.")
        builtin_names = sorted(BUILTIN_WORD_LISTS)
        builtin_choice = st.selectbox(
            "List", builtin_names,
            format_func=lambda name: BUILTIN_WORD_LISTS[name].get(
                "label", BUILTIN_WORD_LISTS[name]["description"],
            ),
            help="Which published reference word list to profile against. See the "
                 "\"How to cite\" tab for how to cite whichever one you use.",
        )
        if BUILTIN_WORD_LISTS[builtin_choice].get("pos_tagged"):
            st.caption(
                "This list matches by lemma *and* part of speech (e.g. \"record\" as a "
                "verb is scored separately from \"record\" as a noun), so lemmatization "
                "is always applied for it, regardless of the checkbox above."
            )
        # Auto-build (no button): rebuild whenever the selected list or any
        # of the build parameters change, tracked via a signature so we
        # don't redo the work on every unrelated widget interaction/rerun --
        # same pattern as "Corpus of texts"/"Word list" below. Without this,
        # toggling e.g. "Lemmatize" after already loading a list wouldn't
        # take effect until some other change happened to trigger a rebuild.
        # Also rebuilds if a *different* source tab last set the shared
        # st.session_state.reference -- otherwise switching to another tab
        # and back (with this tab's widgets unchanged, so this signature
        # matches its last-recorded value) would silently leave the
        # reference pointing at whatever the other tab built, even though
        # the UI still shows this tab's selection as active.
        builtin_sig = (
            builtin_choice, band_size, language, lemmatize, fine_band_size,
            fine_grained_until, coarse_band_size, coarse_grained_from,
        )
        if (st.session_state.get("_builtin_sig") != builtin_sig
                or st.session_state.get("_reference_source_kind") != source_kind):
            # Recorded before attempting the build (not just on success) so
            # a failed build doesn't retry -- and re-flash the same error --
            # on every unrelated rerun until something actually changes.
            st.session_state._builtin_sig = builtin_sig
            st.session_state._reference_source_kind = source_kind
            try:
                st.session_state.reference = Reference.from_builtin(
                    builtin_choice, band_size=band_size, language=language,
                    fine_band_size=fine_band_size, fine_grained_until=fine_grained_until,
                    coarse_band_size=coarse_band_size, coarse_grained_from=coarse_grained_from,
                    lemmatize=lemmatize,
                )
                st.session_state.results = None
            except ValueError as e:
                st.error(str(e))

    elif source_kind == "Corpus of texts":
        st.caption("Upload individual .txt/.docx/.pdf files, or use \"Choose folder...\" to "
                   "pick a whole directory from your computer. The reference builds "
                   "automatically.")
        corpus_picked = file_dir_uploader(label="Choose files...", key="corpus_picker")
        corpus_texts = cached_decode(corpus_picked, "_corpus_picked")
        if corpus_picked:
            st.caption(f"{len(corpus_picked)} file(s) ready.")
        pos_tag_corpus = st.checkbox(
            "Tag part of speech", value=False,
            help="Match target text by lemma *and* part of speech (e.g. \"record\" as a "
                 "verb scored separately from \"record\" as a noun), instead of by lemma "
                 "alone. Always uses lemmatization, regardless of the checkbox above.",
        )

        # Auto-build (no button): rebuild whenever the selected files or any
        # of the build parameters change, tracked via a signature so we
        # don't redo the work on every unrelated widget interaction/rerun.
        build_sig = (
            tuple(sorted(corpus_texts)),
            band_size, language, lemmatize, fine_band_size, fine_grained_until,
            coarse_band_size, coarse_grained_from, pos_tag_corpus,
        )
        if corpus_texts and (st.session_state.get("_corpus_build_sig") != build_sig
                              or st.session_state.get("_reference_source_kind") != source_kind):
            st.session_state._corpus_build_sig = build_sig
            st.session_state._reference_source_kind = source_kind
            try:
                st.session_state.reference = build_with_progress(
                    lambda cb: Reference.from_corpus(
                        corpus_texts, band_size=band_size, language=language,
                        lemmatize=lemmatize, fine_band_size=fine_band_size,
                        fine_grained_until=fine_grained_until, progress_callback=cb,
                        coarse_band_size=coarse_band_size, coarse_grained_from=coarse_grained_from,
                        pos_tagged=pos_tag_corpus,
                    )
                )
                st.session_state.results = None
            except ValueError as e:
                st.error(str(e))

        ref = st.session_state.reference
        if ref is not None and ref.counts:
            with st.expander("Add more texts to this reference"):
                st.caption(
                    "Adds documents to the corpus this reference was built from, and "
                    "recomputes ranks/bands from the combined word counts."
                )
                add_picked = file_dir_uploader(
                    label="Choose files to add...", key="corpus_add_picker",
                )
                add_texts_dict = cached_decode(add_picked, "_add_picked")
                if add_picked:
                    st.caption(f"{len(add_picked)} file(s) ready to add.")
                if st.button("Add to reference", disabled=not add_texts_dict):
                    try:
                        st.session_state.reference = build_with_progress(
                            lambda cb: ref.add_texts(add_texts_dict, progress_callback=cb)
                        )
                        st.session_state.results = None
                        st.success(f"Added {len(add_texts_dict)} file(s) to the reference.")
                    except ValueError as e:
                        st.error(str(e))

    elif source_kind == "Word list":
        wordlist_file = st.file_uploader(
            "Upload a word list .txt file", type=["txt"],
            help="One word per line, ranked most-frequent-first -- or "
                 "\"word,frequency\" pairs (frequency used only to break ties/"
                 "confirm ranking, not required to be exact counts).",
        )
        with st.expander("Advanced word list options"):
            freq_choice = st.selectbox(
                "Format", ["Auto-detect", "word,frequency pairs", "Plain word list"],
                help="\"Auto-detect\" checks whether each line has a "
                     "\"word,frequency\" pair or just a bare word, and works for "
                     "almost every file -- only override it if auto-detection "
                     "guesses wrong for your file.",
            )
            has_frequencies = {"Auto-detect": None, "word,frequency pairs": True,
                                "Plain word list": False}[freq_choice]
        wordlist_sig = (
            wordlist_file.file_id if wordlist_file else None,
            band_size, language, has_frequencies, fine_band_size, fine_grained_until,
            coarse_band_size, coarse_grained_from, lemmatize,
        )
        if wordlist_file and (st.session_state.get("_wordlist_sig") != wordlist_sig
                               or st.session_state.get("_reference_source_kind") != source_kind):
            st.session_state._wordlist_sig = wordlist_sig
            st.session_state._reference_source_kind = source_kind
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                tmp.write(wordlist_file.getvalue())
                tmp_path = tmp.name
            try:
                st.session_state.reference = Reference.from_word_list(
                    tmp_path, band_size=band_size, language=language,
                    has_frequencies=has_frequencies,
                    fine_band_size=fine_band_size, fine_grained_until=fine_grained_until,
                    coarse_band_size=coarse_band_size, coarse_grained_from=coarse_grained_from,
                    lemmatize=lemmatize,
                )
                st.session_state.results = None
            except ValueError as e:
                st.error(str(e))
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    else:  # Saved reference
        ref_file = st.file_uploader(
            "Upload a saved reference .json file", type=["json"],
            help="A reference previously exported with the \"Download this "
                 "reference (.json)\" button below, so you don't have to rebuild "
                 "it from a corpus/word list again.",
        )
        if ref_file and (st.session_state.get("_saved_ref_sig") != ref_file.file_id
                          or st.session_state.get("_reference_source_kind") != source_kind):
            st.session_state._saved_ref_sig = ref_file.file_id
            st.session_state._reference_source_kind = source_kind
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp.write(ref_file.getvalue())
                tmp_path = tmp.name
            try:
                st.session_state.reference = Reference.load(tmp_path)
                st.session_state.results = None
            except ValueError as e:
                st.error(str(e))
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    if st.session_state.reference:
        ref = st.session_state.reference
        st.success(ref.source_description)
        st.caption(f"{ref.num_bands} bands, {len(ref)} known words")
        st.download_button(
            "Download this reference (.json)",
            data=export_to_bytes(lambda results, path: ref.save(path), {}, ".json"),
            file_name="reference.json", mime="application/json",
        )

    st.divider()
    st.header("2. Ignore (optional)")
    st.caption("Words and categories to exclude from band/off-list scoring.")
    exclude_proper_nouns = st.checkbox(
        "Exclude proper nouns", value=True,
        help="Detected via the language's part-of-speech tagger, so this requires "
             "a trained pipeline to be installed for the selected language; with "
             "none installed, this has no effect (proper nouns are just profiled "
             "like any other word).",
    )
    exclude_digits = st.checkbox(
        "Exclude digits", value=True,
        help="Catches tokens containing a digit character (e.g. \"42\", \"3.14\") "
             "-- works regardless of language pipeline availability. Spelled-out "
             "number words (e.g. \"twelve\") are unaffected and stay profiled as "
             "ordinary vocabulary.",
    )
    ignore_file = st.file_uploader(
        "Upload ignore list (.txt)", type=["txt"], key="ignore_file",
        help="One word per line. Matched case-insensitively if the reference "
             "lowercases tokens (the default).",
    )
    ignore_text = st.text_area(
        "...or paste words, one per line", key="ignore_text",
        help="Merged with the uploaded file above, if both are given.",
    )

with tab_profile:
    if not st.session_state.reference:
        st.info(
            "Build a reference in the \"Build a reference corpus or select a word list\" tab "
            "(or use the Step-by-step guide tab's **Load bundled example data** button) to get "
            "started."
        )
    else:
        reference = st.session_state.reference

        st.header("3. Profile target text(s)")
        tab_upload, tab_paste = st.tabs(["Upload files", "Paste text"])
        target_texts: dict[str, str] = {}
        with tab_upload:
            target_files = st.file_uploader(
                "Upload .txt/.docx/.pdf files to profile", type=["txt", "docx", "pdf"],
                accept_multiple_files=True, key="targets",
                help="Select multiple files, or drag a whole folder onto this box.",
            )
            if target_files:
                target_texts.update(read_uploaded_texts_cached(target_files, "_targets"))
        with tab_paste:
            pasted_name = st.text_input(
                "Name for this text", value="pasted_text",
                help="Label used to identify this text in the results table and "
                     "exports -- change it if you paste more than one text so "
                     "they don't overwrite each other.",
            )
            pasted = st.text_area("Paste text to profile", height=200)
            if pasted.strip():
                target_texts[pasted_name] = pasted

        if st.button("Profile", type="primary", disabled=not target_texts):
            ignore_words = []
            if ignore_file:
                ignore_words += read_word_list(
                    ignore_file.getvalue().decode("utf-8", errors="ignore")
                )
            if ignore_text:
                ignore_words += read_word_list(ignore_text)

            profiler = LexicalProfiler(
                reference, ignore_words=ignore_words,
                exclude_proper_nouns=exclude_proper_nouns,
                exclude_digits=exclude_digits,
            )
            st.session_state.profiler = profiler
            st.session_state.results = build_with_progress(
                lambda cb: profiler.profile_texts(target_texts, progress_callback=cb)
            )
            st.session_state.target_texts = target_texts

        # ---------------------------------------------------------------------------
        # Results
        # ---------------------------------------------------------------------------
        if st.session_state.results:
            results = st.session_state.results
            all_target_texts = st.session_state.target_texts
            profiler = st.session_state.profiler

            st.header("4. Results")

            summary_rows = [
                {
                    "text": name,
                    "tokens": r.total_tokens,
                    "types": r.total_types,
                    "off-list %": f"{r.off_list_pct_tokens:.2f}%",
                    "ignored %": f"{r.ignored_pct_tokens:.2f}%",
                    "proper nouns %": f"{r.proper_noun_pct_tokens:.2f}%",
                    "digits %": f"{r.digit_pct_tokens:.2f}%",
                }
                for name, r in results.items()
            ]
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(
                summary_df, width="stretch", hide_index=True,
                column_config={
                    col: st.column_config.Column(alignment="left")
                    for col in summary_df.columns
                },
            )

            selected_name = st.selectbox("View text", list(results.keys()))
            result = results[selected_name]
            text = all_target_texts[selected_name]

            # Unequal widths: "Bands for 95%/98% coverage" are much
            # longer labels than "Tokens"/"Types"/"Off-list %", so equal
            # columns squish them -- give them proportionally more room.
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1.3, 2, 2])
            col1.metric("Tokens", result.total_tokens)
            col2.metric("Types", result.total_types)
            col3.metric("Off-list %", f"{result.off_list_pct_tokens:.1f}%")

            def band_for_coverage(target_pct: float):
                return next(
                    (
                        b for b in sorted(result.cumulative_token_pct)
                        if result.cumulative_token_pct[b] >= target_pct
                    ),
                    None,
                )

            band_95 = band_for_coverage(95)
            band_98 = band_for_coverage(98)
            col4.metric("Bands for 95% coverage", band_95 if band_95 else "N/A",
                        help=None if band_95 else "95% coverage is not reached by any band")
            col5.metric("Bands for 98% coverage", band_98 if band_98 else "N/A",
                        help=None if band_98 else "98% coverage is not reached by any band")

            st.subheader("Band coverage")
            bands = sorted(result.band_token_counts.keys())
            chart_df = pd.DataFrame({
                "band": [result.band_label(b) for b in bands],
                "band_num": bands,
                "pct_tokens": [result.band_token_pct[b] for b in bands],
                "cumulative_pct": [result.cumulative_token_pct[b] for b in bands],
            })
            chart_df["pct_tokens_label"] = chart_df["pct_tokens"].map(lambda v: f"{v:.2f}%")
            chart_df["cumulative_pct_label"] = chart_df["cumulative_pct"].map(lambda v: f"{v:.2f}%")
            bar = alt.Chart(chart_df).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
            ).encode(
                x=alt.X("band:N", sort=None, title="Frequency band", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("pct_tokens:Q", title="% of tokens", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("band_num:Q", scale=alt.Scale(range=BAND_RAMP), legend=None),
                tooltip=[
                    alt.Tooltip("band:N", title="Band"),
                    alt.Tooltip("pct_tokens_label:N", title="% tokens"),
                    alt.Tooltip("cumulative_pct_label:N", title="Cumulative %"),
                ],
            )
            line = alt.Chart(chart_df).mark_line(color="#eb6834", point=True).encode(
                x=alt.X("band:N", sort=None),
                y=alt.Y("cumulative_pct:Q", scale=alt.Scale(domain=[0, 100])),
            )
            threshold_df = pd.DataFrame({"y": [95, 98]})
            thresholds = alt.Chart(threshold_df).mark_rule(
                strokeDash=[4, 4], color="#898781",
            ).encode(
                y="y:Q",
            )
            st.altair_chart((bar + line + thresholds).properties(height=400), width="stretch")
            st.caption(
                "Bars: % of tokens in each band. Orange line: cumulative coverage. "
                "Dashed lines: 95%/98% reading-comprehension coverage thresholds "
                "(Laufer & Ravenhorst-Kalovski, 2010)."
            )

            with st.expander("Band coverage table"):
                table_df = chart_df[["band", "pct_tokens_label", "cumulative_pct_label"]].copy()
                table_df.columns = ["Band", "% tokens", "Cumulative %"]
                extra_rows = [{
                    "Band": "Off-list", "% tokens": f"{result.off_list_pct_tokens:.2f}%",
                    "Cumulative %": None,
                }]
                if result.ignored_tokens or result.ignored_words:
                    extra_rows.append({
                        "Band": "Ignored", "% tokens": f"{result.ignored_pct_tokens:.2f}%",
                        "Cumulative %": None,
                    })
                if result.proper_noun_tokens or result.proper_noun_words:
                    extra_rows.append({
                        "Band": "Proper nouns", "% tokens": f"{result.proper_noun_pct_tokens:.2f}%",
                        "Cumulative %": None,
                    })
                if result.digit_tokens or result.digit_words:
                    extra_rows.append({
                        "Band": "Digits", "% tokens": f"{result.digit_pct_tokens:.2f}%",
                        "Cumulative %": None,
                    })
                table_df = pd.concat([table_df, pd.DataFrame(extra_rows)], ignore_index=True)
                st.dataframe(table_df, width="stretch", hide_index=True)

            st.subheader("Highlighted text")
            num_bands = reference.num_bands
            legend_bands = bands
            legend_swatches = "".join(
                span_html(
                    result.band_label(b), band_color(b, num_bands),
                    readable_text_color(band_color(b, num_bands)), margin_right="6px",
                )
                for b in legend_bands
            )
            legend_swatches += span_html("off-list", OFF_LIST_COLOR, margin_right="6px")
            legend_swatches += span_html("ignored", IGNORED_COLOR, margin_right="6px")
            if profiler.exclude_proper_nouns:
                legend_swatches += span_html("proper noun", PROPER_NOUN_COLOR, margin_right="6px")
            if profiler.exclude_digits:
                legend_swatches += span_html("digit", DIGIT_COLOR)
            st.markdown(legend_swatches, unsafe_allow_html=True)

            spans = []
            for tok in profiler.highlight(text):
                surface = escape(tok.text)
                ws = escape(tok.whitespace)
                if tok.status == "band":
                    color = band_color(tok.band, num_bands)
                    fg = readable_text_color(color)
                    title = f"Band {result.band_label(tok.band)}"
                    spans.append(span_html(surface, color, fg, title, pad="0 1px") + ws)
                elif tok.status == "off_list":
                    spans.append(
                        span_html(surface, OFF_LIST_COLOR, title="Off-list", pad="0 1px") + ws
                    )
                elif tok.status == "ignored":
                    spans.append(
                        span_html(surface, IGNORED_COLOR, title="Ignored", pad="0 1px") + ws
                    )
                elif tok.status == "proper_noun":
                    spans.append(
                        span_html(surface, PROPER_NOUN_COLOR, title="Proper noun", pad="0 1px") + ws
                    )
                elif tok.status == "digit":
                    spans.append(span_html(surface, DIGIT_COLOR, title="Digit", pad="0 1px") + ws)
                else:
                    spans.append(f"{surface}{ws}")

            st.markdown(
                f'<div style="line-height:2.2; font-size:1.05rem; padding:1rem; '
                f'border:1px solid rgba(128,128,128,0.3); border-radius:8px; max-height:500px; '
                f'overflow-y:auto;">{"".join(spans)}</div>',
                unsafe_allow_html=True,
            )

            col_off, col_ign = st.columns(2)
            with col_off:
                off_label = (
                    f"Off-list words ({result.off_list_types} unique, "
                    f"{result.off_list_tokens} tokens)"
                )
                with st.expander(off_label):
                    off_df = word_table(
                        result.off_list_words, result.word_counts, reference.pos_tagged,
                    )
                    st.dataframe(off_df, width="stretch", hide_index=True)
            if result.ignored_words:
                with col_ign, st.expander(f"Ignored words ({result.ignored_types} unique)"):
                    ign_df = word_table(
                        result.ignored_words, result.word_counts, reference.pos_tagged,
                    )
                    st.dataframe(ign_df, width="stretch", hide_index=True)

            if result.proper_noun_words or result.digit_words:
                col_propn, col_num = st.columns(2)
                if result.proper_noun_words:
                    label = f"Proper nouns ({result.proper_noun_types} unique)"
                    with col_propn, st.expander(label):
                        propn_df = word_table(
                            result.proper_noun_words, result.word_counts, reference.pos_tagged,
                        )
                        st.dataframe(propn_df, width="stretch", hide_index=True)
                if result.digit_words:
                    with col_num, st.expander(f"Digits ({result.digit_types} unique)"):
                        num_df = word_table(
                            result.digit_words, result.word_counts, reference.pos_tagged,
                        )
                        st.dataframe(num_df, width="stretch", hide_index=True)

            st.subheader("Export")
            dl1, dl2, dl3, dl4 = st.columns(4)
            dl1.download_button(
                "Band coverage (CSV)", export_to_bytes(report_mod.export_csv, results, ".csv"),
                "band_coverage.csv", "text/csv",
            )
            dl2.download_button(
                "Full report (JSON)", export_to_bytes(report_mod.export_json, results, ".json"),
                "full_report.json", "application/json",
            )
            dl3.download_button(
                "Off-list words (CSV)",
                export_to_bytes(report_mod.export_off_list_csv, results, ".csv"),
                "off_list_words.csv", "text/csv",
            )
            dl4.download_button(
                "Ignored words (CSV)",
                export_to_bytes(report_mod.export_ignored_csv, results, ".csv"),
                "ignored_words.csv", "text/csv",
            )
            if any(r.proper_noun_words or r.digit_words for r in results.values()):
                dl5, dl6, _, _ = st.columns(4)
                if any(r.proper_noun_words for r in results.values()):
                    dl5.download_button(
                        "Proper nouns (CSV)",
                        export_to_bytes(report_mod.export_proper_nouns_csv, results, ".csv"),
                        "proper_nouns.csv", "text/csv",
                    )
                if any(r.digit_words for r in results.values()):
                    dl6.download_button(
                        "Digits (CSV)",
                        export_to_bytes(report_mod.export_digits_csv, results, ".csv"),
                        "digits.csv", "text/csv",
                    )

with tab_cite:
    st.header("How to cite")

    st.subheader("This tool")
    st.code(
        "Hashimoto, B. (2026). lexical_profiler (Version 0.2.0) [Computer software]. "
        "https://github.com/bretthashimoto/lexical-profile-tool",
        language=None,
    )

    st.subheader("Built-in word lists")
    st.caption(
        "Cite whichever list you actually used as your reference -- not this tool -- "
        "since the tool just reads a published list, it didn't create one."
    )
    for _name in sorted(BUILTIN_WORD_LISTS):
        _entry = BUILTIN_WORD_LISTS[_name]
        st.markdown(f"**{_entry['label']}**")
        st.code(_entry["citation"], language=None)

with tab_about:
    st.header("About lexical frequency profiling")
    st.markdown(
        """
Lexical frequency profiling measures how much of a text's vocabulary falls
into common vs. rare/unknown frequency bands, relative to a reference --
either a frequency-ranked word list (like the AVL or NGSL) or a corpus of
your own texts. It's widely used in vocabulary research, reading/materials
research, and ESL/EFL text leveling.

**Frequency bands.** Every word in the reference is ranked by frequency and
grouped into bands (band 1 = most frequent). Profiling a text tells you what
percentage of its tokens fall in each band -- a text dominated by band 1-2
words uses mostly very common vocabulary, while a text with a long tail in
higher bands uses more specialized or rare vocabulary.

**Off-list vs. ignored words.** A word that doesn't appear anywhere in the
reference is "off-list" -- it's outside the vocabulary the reference
describes (this is often what a profile is really trying to measure: how
much of a text a reader who knows the reference vocabulary would *not*
recognize). "Ignored" words are specific words you've deliberately
excluded (via the ignore list), so they don't get counted as off-list.

**Proper nouns and digits.** Names, places, and digit tokens (e.g. "42")
are usually not meaningful vocabulary knowledge -- most published Lexical
Frequency Profile tools exclude them by convention rather than counting
them off-list. This tool excludes both categories by default, reporting
each separately instead of folding it into off-list; uncheck either box
(the "Build a reference corpus or select a word list" tab, step 2) to
profile it like any other word instead, the same way an ignore list
works in reverse. Spelled-out number words (e.g. "twelve")
are unaffected either way and stay profiled as ordinary vocabulary.

**Coverage thresholds.** A common way to use band coverage: how many bands
does it take to reach 95% or 98% of a text's tokens? These particular
thresholds come from a line of studies linking lexical coverage to reading
comprehension. Laufer (1989) was the original work establishing a
threshold, finding that around 95% coverage was needed for a minimal level
of comprehension (55% on a comprehension test). Hu & Nation (2000) found
that 98% coverage was needed to reach 71% comprehension across two reading
tests. Schmitt, Jiang, & Grabe (2011) found that 98% coverage was the
optimal level for academic texts when 70% comprehension was expected. This
is where the 95% (minimal) and 98% (optimal) thresholds used in this tool
come from -- as framed by Laufer & Ravenhorst-Kalovski (2010), building on
Nation (2006), who estimated the vocabulary size needed to reach each
threshold for written and spoken English. For a broader look at this
research, see Laufer (2013).

**Lemmas, word forms, and part of speech.** Some references (like AVL,
NGSL, NAWL) match by lemma, so "runs", "running", and "ran" all count as
the word family "run". COCA-based profiling in this tool also tags part of
speech, so "record" as a verb is scored separately from "record" as a noun.

References:
- Laufer, B. (1989). What percentage of text-lexis is essential for
  comprehension? In C. Lauren & M. Nordman (Eds.), *Special language: From
  humans thinking to thinking machines* (pp. 316-323). Multilingual Matters.
- Hu, M., & Nation, P. (2000). Unknown vocabulary density and reading
  comprehension. *Reading in a Foreign Language, 13*(1), 403-430.
- Nation, P. (2006). How large a vocabulary is needed for reading and
  listening? *Canadian Modern Language Review, 63*(1), 59-82.
- Laufer, B., & Ravenhorst-Kalovski, G. C. (2010). Lexical threshold
  revisited: Lexical text coverage, learners' vocabulary size and reading
  comprehension. *Reading in a Foreign Language, 22*(1), 15-30.
- Schmitt, N., Jiang, X., & Grabe, W. (2011). The percentage of words known
  in a text and reading comprehension. *The Modern Language Journal, 95*(1),
  26-43.
- Laufer, B. (2013). Lexical thresholds for reading comprehension: What
  they are and how they can be used for teaching purposes. *TESOL
  Quarterly, 47*(4), 867-872.
"""
    )

with tab_guide:
    st.header("Step-by-step guide")

    st.subheader("Quick start")
    st.caption("Try the tool with the bundled sample corpus + essays before uploading your own.")
    load_example_clicked = st.button("Load bundled example data")
    if load_example_clicked:
        load_example_data(
            band_size=band_size, language=language, lemmatize=lemmatize,
            fine_band_size=fine_band_size, fine_grained_until=fine_grained_until,
            coarse_band_size=coarse_band_size, coarse_grained_from=coarse_grained_from,
        )
        st.rerun()
    st.markdown(
        """
1. **Build a reference** (the "Build a reference corpus or select a word list" tab, step 1).
   Pick one:
   - **Built-in word list** -- profile against a published list (AVL, NGSL,
     NAWL, or COCA) with no file to find or format.
   - **Corpus of texts** -- upload your own .txt/.docx/.pdf files (or pick a
     whole folder) and the reference builds automatically from their word
     frequencies.
   - **Word list** -- upload your own plain-text or word,frequency list.
   - **Saved reference (.json)** -- reload a reference you exported earlier.

   While you're at it, set the **band size**, **language**, and whether to
   **lemmatize** -- these affect how words are grouped and matched, so set
   them before building rather than after.

2. **Ignore (optional)** (the "Build a reference corpus or select a word list" tab,
   step 2). Proper nouns and digits are excluded by default -- uncheck either box to
   profile it like any other word instead. Upload or paste specific words (names,
   made-up words, etc.) you don't want counted as off-list.

3. **Profile target text(s)** (the "Profile a text" tab, step 3). Upload
   files or paste text directly, then click **Profile**.

4. **Read the results** (step 4):
   - The summary table and metrics show tokens, types, and off-list %
     (plus ignored/proper noun/digit %, if applicable) per text.
   - The **band coverage** chart shows what % of tokens fall in each band,
     plus cumulative coverage against the 95%/98% thresholds.
   - **Highlighted text** color-codes every word by band (or off-list/
     ignored/proper noun/digit), so you can see at a glance which words
     are driving the score.
   - **Export** lets you download the band coverage, full report, and
     off-list/ignored/proper-noun/digit word lists as CSV/JSON.

Not sure what any of this means? See the **About lexical frequency
profiling** tab for background on bands, coverage thresholds, and how
lemmas/POS matching work.
"""
    )

with tab_about_me:
    st.header("About me")
    st.markdown(
        """
**Brett Hashimoto**

Associate Professor of Linguistics at Brigham Young University. My research
uses corpus linguistics for applied linguistics purposes, including language
teaching and learning and legal linguistics.

- [BYU faculty page](https://hum.byu.edu/directory/brett-hashimoto)
- [Personal research website](https://sites.google.com/site/brettjameshashimoto/)
- [Google Scholar profile](https://scholar.google.com/citations?user=V4CzV4AAAAAJ&hl=en)

Contact: brett_hashimoto@byu.edu
"""
    )
