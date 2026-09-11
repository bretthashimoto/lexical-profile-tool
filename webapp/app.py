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

import io
import sys
import tempfile
import zipfile
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

from lexical_profiler import (  # noqa: E402
    LexicalProfiler,
    Reference,
    download_model,
    lemmatizer_available,
)
from lexical_profiler import report as report_mod  # noqa: E402

st.set_page_config(page_title="LEAH — Lexical Analysis", page_icon="📖", layout="wide")

# ---------------------------------------------------------------------------
# Color scheme: a validated sequential blue ramp for frequency bands
# (light = most frequent/easiest, dark = least frequent), plus fixed status
# colors for off-list (critical) and ignored (muted) words.
# ---------------------------------------------------------------------------
BAND_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
OFF_LIST_COLOR = "#d03b3b"
IGNORED_COLOR = "#898781"

# The LEAH mark: an open book whose pages are bars decaying like a Zipf
# curve (tall/light on the left, falling into a long low/dark tail on the
# right) -- a nod to word-frequency distributions, and to the "L" in LEAH.
_LOGO_SVG_SRC = """
<svg width="84" height="84" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
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
    <rect x="120" y="195" width="20" height="190" fill="#86b6ef"/>
    <rect x="147" y="330" width="20" height="55" fill="#6da7ec"/>
    <rect x="174" y="359" width="20" height="26" fill="#5598e7"/>
    <rect x="201" y="369" width="20" height="16" fill="#3987e5"/>
    <rect x="228" y="375" width="20" height="10" fill="#2a78d6"/>
  </g>
  <g clip-path="url(#leahClipRight)">
    <rect x="261" y="377" width="20" height="8" fill="#256abf"/>
    <rect x="288" y="379" width="20" height="6" fill="#1c5cab"/>
    <rect x="315" y="380" width="20" height="5" fill="#184f95"/>
    <rect x="342" y="381" width="20" height="4" fill="#104281"/>
    <rect x="369" y="382" width="20" height="3" fill="#0d366b"/>
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


def read_word_list(raw: str) -> list[str]:
    return [w.strip() for w in raw.splitlines() if w.strip() and not w.strip().startswith("#")]


def read_uploaded_texts(files) -> list[tuple[str, str]]:
    """Expand uploaded files into (name, text) pairs, decoding plain .txt
    files directly and unzipping any .zip archive into its .txt members --
    lets a user upload a whole directory of texts as one zipped file, since
    browsers don't offer a folder picker for a plain file input."""
    out = []
    for f in files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as zf:
                for member in zf.namelist():
                    name = Path(member).name
                    if member.endswith("/") or not name.lower().endswith(".txt"):
                        continue
                    if name.startswith(".") or member.startswith("__MACOSX/"):
                        continue
                    out.append((member, zf.read(member).decode("utf-8", errors="ignore")))
        else:
            out.append((f.name, f.getvalue().decode("utf-8", errors="ignore")))
    return out


for key in ("reference", "profiler", "results", "target_texts"):
    st.session_state.setdefault(key, None)


def load_example_data():
    ref = Reference.from_corpus(str(REPO_ROOT / "examples" / "corpus"), band_size=20)
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
# Sidebar: build/load a reference, quick start, ignore list
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Quick start")
    st.caption("Try the tool with the bundled sample corpus + essays before uploading your own.")
    if st.button("Load bundled example data"):
        load_example_data()
        st.rerun()

    st.divider()
    st.header("1. Build a reference")
    source_kind = st.radio(
        "Reference source", ["Corpus of texts", "Word list", "Saved reference (.json)"],
    )

    band_size = int(st.number_input("Band size", min_value=1, value=1000, step=100))
    language = st.text_input(
        "Language code", value="en", help="ISO 639-1 code (en, es, de, fr, ...)",
    )
    lemmatize = st.checkbox(
        "Lemmatize", value=False,
        help="Requires a spaCy pipeline installed for the language; silently falls back to "
             "surface forms otherwise.",
    )
    if lemmatize and language:
        if lemmatizer_available(language):
            st.caption(f"✅ Lemmatizer model available for '{language}'.")
        else:
            st.caption(
                f"⚠️ No lemmatizer model installed for '{language}' yet — words will use "
                f"their surface form instead of a lemma until one is installed."
            )
            if st.button(f"Download spaCy model for '{language}'"):
                with st.spinner(f"Downloading a spaCy model for '{language}'..."):
                    installed = download_model(language)
                if installed:
                    st.success(f"Installed a model for '{language}'.")
                    st.rerun()
                else:
                    st.error(
                        f"Couldn't download a model for '{language}'. Either this language "
                        f"code doesn't have a trained spaCy pipeline (see spacy.io/models), "
                        f"or this host doesn't allow installing packages at runtime."
                    )

    with st.expander("Fine-grained bands (optional)"):
        use_fine = st.checkbox("Use narrower bands for the most frequent words")
        fine_band_size = None
        fine_grained_until = None
        if use_fine:
            fine_band_size = int(st.number_input("Fine band size", min_value=1, value=100))
            fine_grained_until = int(st.number_input("...through rank", min_value=1, value=2000))

    if source_kind == "Corpus of texts":
        corpus_files = st.file_uploader(
            "Upload .txt corpus files", type=["txt", "zip"], accept_multiple_files=True,
            help="Select multiple files, drag a whole folder onto this box, or zip a "
                 "directory of .txt files and upload the .zip.",
        )
        if st.button("Build reference from corpus", disabled=not corpus_files):
            texts = [text for _, text in read_uploaded_texts(corpus_files)]
            try:
                st.session_state.reference = Reference.from_corpus(
                    texts, band_size=band_size, language=language, lemmatize=lemmatize,
                    fine_band_size=fine_band_size, fine_grained_until=fine_grained_until,
                )
                st.session_state.results = None
            except ValueError as e:
                st.error(str(e))

    elif source_kind == "Word list":
        wordlist_file = st.file_uploader("Upload a word list .txt file", type=["txt"])
        with st.expander("Advanced word list options"):
            freq_choice = st.selectbox(
                "Format", ["Auto-detect", "word,frequency pairs", "Plain word list"],
            )
            has_frequencies = {"Auto-detect": None, "word,frequency pairs": True,
                                "Plain word list": False}[freq_choice]
        if st.button("Build reference from word list", disabled=not wordlist_file):
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                tmp.write(wordlist_file.getvalue())
                tmp_path = tmp.name
            try:
                st.session_state.reference = Reference.from_word_list(
                    tmp_path, band_size=band_size, language=language,
                    has_frequencies=has_frequencies,
                    fine_band_size=fine_band_size, fine_grained_until=fine_grained_until,
                )
                st.session_state.results = None
            except ValueError as e:
                st.error(str(e))
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    else:  # Saved reference
        ref_file = st.file_uploader("Upload a saved reference .json file", type=["json"])
        if st.button("Load reference", disabled=not ref_file):
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
    st.header("2. Ignore list (optional)")
    st.caption("Proper nouns, names, or made-up words to exclude from band/off-list scoring.")
    ignore_file = st.file_uploader("Upload ignore list (.txt)", type=["txt"], key="ignore_file")
    ignore_text = st.text_area("...or paste words, one per line", key="ignore_text")
    min_length = int(st.number_input("Minimum token length", min_value=1, value=1))

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
_banner_html = f"""
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&display=swap">
<div style="
    background:linear-gradient(135deg, {BAND_RAMP[7]} 0%, {BAND_RAMP[4]} 100%);
    border-radius:12px;
    padding:1.6rem 2rem;
    margin-bottom:1.2rem;
">
    <div style="display:flex; align-items:center; gap:0.7rem;">
        {LOGO_SVG}
        <span style="
            font-family:'Space Grotesk', sans-serif;
            font-weight:700;
            font-style:italic;
            font-size:2.8rem;
            letter-spacing:0.02em;
            color:#ffffff;
        ">LEAH</span>
    </div>
    <div style="color:#e8f0fc; font-size:1.05rem; margin-top:0.25rem;">
        <b>LE</b>xical <b>A</b>nalysis — <b>H</b>ashimoto
    </div>
    <div style="color:#d3e2f7; font-size:0.9rem; margin-top:0.5rem; max-width:48rem;">
        Measure how much of a text's vocabulary falls into common vs. rare/unknown
        frequency bands, relative to a reference you build from your own corpus.
    </div>
</div>
"""
# Collapsed to one line for the same reason as LOGO_SVG above: a blank line
# inside a raw HTML block passed to st.markdown breaks it into two blocks,
# and everything after the break renders as literal text instead of HTML.
st.markdown(
    " ".join(line.strip() for line in _banner_html.strip().splitlines()),
    unsafe_allow_html=True,
)

if not st.session_state.reference:
    st.info(
        "Build a reference in the sidebar (or click **Load bundled example data**) to get started."
    )
    st.stop()

reference = st.session_state.reference

st.header("3. Profile target text(s)")
tab_upload, tab_paste = st.tabs(["Upload files", "Paste text"])
target_texts: dict[str, str] = {}
with tab_upload:
    target_files = st.file_uploader(
        "Upload .txt files to profile", type=["txt", "zip"], accept_multiple_files=True,
        key="targets",
        help="Select multiple files, drag a whole folder onto this box, or zip a "
             "directory of .txt files and upload the .zip.",
    )
    if target_files:
        for name, text in read_uploaded_texts(target_files):
            target_texts[name] = text
with tab_paste:
    pasted_name = st.text_input("Name for this text", value="pasted_text")
    pasted = st.text_area("Paste text to profile", height=200)
    if pasted.strip():
        target_texts[pasted_name] = pasted

if st.button("Profile", type="primary", disabled=not target_texts):
    ignore_words = []
    if ignore_file:
        ignore_words += read_word_list(ignore_file.getvalue().decode("utf-8", errors="ignore"))
    if ignore_text:
        ignore_words += read_word_list(ignore_text)

    profiler = LexicalProfiler(reference, min_length=min_length, ignore_words=ignore_words)
    st.session_state.profiler = profiler
    st.session_state.results = profiler.profile_texts(target_texts)
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
            "off-list %": round(r.off_list_pct_tokens, 2),
            "ignored %": round(r.ignored_pct_tokens, 2),
        }
        for name, r in results.items()
    ]
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    selected_name = st.selectbox("View text", list(results.keys()))
    result = results[selected_name]
    text = all_target_texts[selected_name]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tokens", result.total_tokens)
    col2.metric("Types", result.total_types)
    col3.metric("Off-list %", f"{result.off_list_pct_tokens:.1f}%")
    band_95 = next(
        (b for b in sorted(result.cumulative_token_pct) if result.cumulative_token_pct[b] >= 95),
        None,
    )
    col4.metric("Bands for 95% coverage", band_95 if band_95 else "not reached")

    st.subheader("Band coverage")
    bands = sorted(result.band_token_counts.keys())
    chart_df = pd.DataFrame({
        "band": [result.band_label(b) for b in bands],
        "band_num": bands,
        "pct_tokens": [result.band_token_pct[b] for b in bands],
        "cumulative_pct": [result.cumulative_token_pct[b] for b in bands],
    })
    bar = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("band:N", sort=None, title="Frequency band", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("pct_tokens:Q", title="% of tokens", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("band_num:Q", scale=alt.Scale(range=BAND_RAMP), legend=None),
        tooltip=[
            alt.Tooltip("band:N", title="Band"),
            alt.Tooltip("pct_tokens:Q", format=".2f", title="% tokens"),
            alt.Tooltip("cumulative_pct:Q", format=".2f", title="Cumulative %"),
        ],
    )
    line = alt.Chart(chart_df).mark_line(color="#eb6834", point=True).encode(
        x=alt.X("band:N", sort=None),
        y=alt.Y("cumulative_pct:Q", scale=alt.Scale(domain=[0, 100])),
    )
    threshold_df = pd.DataFrame({"y": [95, 98]})
    thresholds = alt.Chart(threshold_df).mark_rule(strokeDash=[4, 4], color="#898781").encode(
        y="y:Q",
    )
    st.altair_chart((bar + line + thresholds).properties(height=400), width="stretch")
    st.caption(
        "Bars: % of tokens in each band. Orange line: cumulative coverage. "
        "Dashed lines: 95%/98% reading-comprehension coverage thresholds "
        "(Laufer & Ravenhorst-Kalovski, 2010)."
    )

    with st.expander("Band coverage table"):
        table_df = chart_df[["band", "pct_tokens", "cumulative_pct"]].copy()
        table_df.columns = ["Band", "% tokens", "Cumulative %"]
        extra_rows = [{
            "Band": "Off-list", "% tokens": round(result.off_list_pct_tokens, 2),
            "Cumulative %": None,
        }]
        if result.ignored_tokens or result.ignored_words:
            extra_rows.append({
                "Band": "Ignored", "% tokens": round(result.ignored_pct_tokens, 2),
                "Cumulative %": None,
            })
        table_df = pd.concat([table_df, pd.DataFrame(extra_rows)], ignore_index=True)
        st.dataframe(table_df, width="stretch", hide_index=True)

    st.subheader("Highlighted text")
    num_bands = reference.num_bands
    legend_bands = sorted({1, max(1, num_bands // 2), num_bands})
    legend_swatches = "".join(
        span_html(
            result.band_label(b), band_color(b, num_bands),
            readable_text_color(band_color(b, num_bands)), margin_right="6px",
        )
        for b in legend_bands
    )
    legend_swatches += span_html("off-list", OFF_LIST_COLOR, margin_right="6px")
    legend_swatches += span_html("ignored", IGNORED_COLOR)
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
            spans.append(span_html(surface, OFF_LIST_COLOR, title="Off-list", pad="0 1px") + ws)
        elif tok.status == "ignored":
            spans.append(span_html(surface, IGNORED_COLOR, title="Ignored", pad="0 1px") + ws)
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
            f"Off-list words ({result.off_list_types} unique, {result.off_list_tokens} tokens)"
        )
        with st.expander(off_label):
            off_df = pd.DataFrame({
                "word": result.off_list_words,
                "count": [result.word_counts[w] for w in result.off_list_words],
            })
            st.dataframe(off_df, width="stretch", hide_index=True)
    if result.ignored_words:
        with col_ign:
            with st.expander(f"Ignored words ({result.ignored_types} unique)"):
                ign_df = pd.DataFrame({
                    "word": result.ignored_words,
                    "count": [result.word_counts[w] for w in result.ignored_words],
                })
                st.dataframe(ign_df, width="stretch", hide_index=True)

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
        "Off-list words (CSV)", export_to_bytes(report_mod.export_off_list_csv, results, ".csv"),
        "off_list_words.csv", "text/csv",
    )
    dl4.download_button(
        "Ignored words (CSV)", export_to_bytes(report_mod.export_ignored_csv, results, ".csv"),
        "ignored_words.csv", "text/csv",
    )
