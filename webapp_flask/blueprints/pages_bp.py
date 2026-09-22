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
    """Build a reference from the COCA built-in word list and profile the
    bundled example target texts against it -- a one-click way to try the
    tool against a real published reference instead of hunting down a
    corpus or word list first. lemmatize is always True for a built-in
    list (see partials/_band_params.html's show_lemmatize=false), and
    exclude_proper_nouns/exclude_digits use this app's usual defaults
    (see session_store.DEFAULT_META)."""
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]

    reference = Reference.from_builtin("coca", band_size=1000, language="en", lemmatize=True)
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
        "reference_build": {"source_kind": "builtin", "builtin_name": "coca",
                             "band_size": 1000, "language": "en", "lemmatize": True},
        "ignore_config": {
            "exclude_proper_nouns": True, "exclude_digits": True, "ignore_words": ignore_words,
        },
        "target_texts": target_texts,
    })
    flash("Loaded the sample profile: two of Aesop's Fables against the COCA word list.", "success")
    return redirect(url_for("reference.build"))


@pages_bp.route("/about")
def about():
    return render_template("pages/about.html")


@pages_bp.route("/cite")
def cite():
    return render_template("pages/cite.html", builtin_choices=BUILTIN_WORD_LISTS)


@pages_bp.route("/about-me")
def about_me():
    return render_template("pages/about_me.html")
