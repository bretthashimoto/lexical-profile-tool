from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, flash, redirect, render_template, session, url_for

from lexical_profiler import Reference
from lexical_profiler.reference import BUILTIN_WORD_LISTS

from ..services import profiling_service, session_store

pages_bp = Blueprint("pages", __name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pages_bp.route("/")
def index():
    return redirect(url_for("reference.build"))


@pages_bp.route("/reset", methods=["POST"])
def reset():
    """Wipe the current session's reference/targets/ignore-config and hand
    out a fresh session id, so refreshing the page (which keeps the same
    session cookie) can't bring old state back."""
    sessions_root = current_app.config["SESSION_DIR"]
    session_store.delete_session(sessions_root, session["session_id"])
    session.clear()
    flash("Started a new session.", "success")
    return redirect(url_for("reference.build"))


@pages_bp.route("/guide", methods=["GET"])
def guide():
    return render_template("pages/guide.html")


@pages_bp.route("/guide/load-example", methods=["POST"])
def load_example():
    """Build a reference from the bundled examples/ corpus, ignore list,
    and target essays, and profile them -- a one-click way to try the
    tool. Ported from webapp/app.py's load_example_data(), called there
    with the Build tab's live widget values -- band_size=1000 is that
    widget's default (its own default parameter value of 20 is never
    actually used by the real UI), so that's what's used here too."""
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]

    reference = Reference.from_corpus(
        str(REPO_ROOT / "examples" / "corpus"), band_size=1000, language="en", lemmatize=True,
    )
    ignore_words = profiling_service.read_word_list(
        (REPO_ROOT / "examples" / "ignore_list.txt").read_text(encoding="utf-8")
    )
    targets_dir = REPO_ROOT / "examples" / "targets"
    target_texts = {
        f.name: f.read_text(encoding="utf-8") for f in sorted(targets_dir.glob("*.txt"))
    }

    session_store.ensure_session_dir(sessions_root, session_id)
    reference.save(str(session_store.reference_path(sessions_root, session_id)))
    session_store.write_meta(sessions_root, session_id, {
        "reference_build": {
            "source_kind": "example", "band_size": 1000, "language": "en", "lemmatize": True,
        },
        # Streamlit's load_example_data() constructs LexicalProfiler directly
        # without passing exclude_proper_nouns/exclude_digits, so those take
        # the library's own defaults (False/False) here too -- not this
        # app's usual True/True (the ignore-config UI's own defaults).
        "ignore_config": {
            "exclude_proper_nouns": False, "exclude_digits": False, "ignore_words": ignore_words,
        },
        "target_texts": target_texts,
    })
    flash("Loaded the bundled example data.", "success")
    return redirect(url_for("profile.results"))


@pages_bp.route("/about")
def about():
    return render_template("pages/about.html")


@pages_bp.route("/cite")
def cite():
    return render_template("pages/cite.html", builtin_choices=BUILTIN_WORD_LISTS)


@pages_bp.route("/about-me")
def about_me():
    return render_template("pages/about_me.html")
