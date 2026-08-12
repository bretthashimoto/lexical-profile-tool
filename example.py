"""
End-to-end example: run with `python example.py`.

Demonstrates both ways of building a Reference (from a corpus, and from
an existing word-frequency list), then profiles a target text against
each and prints a report.
"""

import os
import tempfile

from lexical_profiler import LexicalProfiler, Reference
from lexical_profiler import report as report_mod

SAMPLE_CORPUS = [
    "The quick brown fox jumps over the lazy dog. The dog barks at the fox.",
    "Economic policy influences market behavior significantly. Analysts "
    "predict that inflation will continue to affect consumer spending "
    "patterns this year.",
]

TARGET_TEXT = (
    "The quick fox jumped over an unprecedented obstacle while economists "
    "debated inflationary pressures near the forest. Zylophonic gizmos abound."
)


def demo_corpus_reference():
    print("=" * 60)
    print("DEMO 1: Reference built from a corpus")
    print("=" * 60)
    reference = Reference.from_corpus(SAMPLE_CORPUS, band_size=5)
    profiler = LexicalProfiler(reference)
    result = profiler.profile_text(TARGET_TEXT)
    print(result.summary())
    return result


def demo_word_list_reference():
    print()
    print("=" * 60)
    print("DEMO 2: Reference loaded from an existing word-frequency list")
    print("=" * 60)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(
            "the,150000\nbe,120000\nto,110000\nof,100000\nand,95000\n"
            "fox,50\ndog,45\nforest,40\nquick,30\neconomic,20\n"
        )
        wordlist_path = f.name

    try:
        reference = Reference.from_word_list(wordlist_path, band_size=3)
        profiler = LexicalProfiler(reference)
        result = profiler.profile_text(TARGET_TEXT)
        print(result.summary())
    finally:
        os.unlink(wordlist_path)
    return result


def demo_non_english():
    print()
    print("=" * 60)
    print("DEMO 3: Non-English text (Spanish), language='es'")
    print("=" * 60)
    from lexical_profiler import lemmatizer_available
    print(f"Spanish lemmatizer installed: {lemmatizer_available('es')} "
          f"(tokenization still works either way; install with "
          f"`python -m spacy download es_core_news_sm` to enable lemmatization)")
    corpus = [
        "El zorro rápido salta sobre el perro perezoso.",
        "El perro ladra al zorro. El zorro corre hacia el bosque.",
    ]
    reference = Reference.from_corpus(corpus, band_size=3, language="es", lemmatize=True)
    profiler = LexicalProfiler(reference)
    result = profiler.profile_text("El zorro rápido corre rápidamente por el bosque oscuro.")
    print(result.summary())
    return result


def demo_fine_grained_bands():
    print()
    print("=" * 60)
    print("DEMO 4: Fine-grained bands (100 through rank 2000, then 1000)")
    print("=" * 60)
    reference = Reference.from_corpus(
        SAMPLE_CORPUS, band_size=1000, fine_band_size=5, fine_grained_until=15,
    )
    profiler = LexicalProfiler(reference)
    result = profiler.profile_text(TARGET_TEXT)
    print(result.summary())
    return result


if __name__ == "__main__":
    r1 = demo_corpus_reference()
    r2 = demo_word_list_reference()
    r3 = demo_non_english()
    r4 = demo_fine_grained_bands()

    print()
    print("Exporting combined JSON/CSV report for both demos to the current directory ...")
    results = {"corpus_reference_demo": r1, "wordlist_reference_demo": r2,
               "spanish_demo": r3, "fine_grained_demo": r4}
    here = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(here, "lexical_profile_demo.json")
    csv_path = os.path.join(here, "lexical_profile_demo.csv")
    report_mod.export_json(results, json_path)
    report_mod.export_csv(results, csv_path)
    print(f"Done: {json_path}, {csv_path}")
