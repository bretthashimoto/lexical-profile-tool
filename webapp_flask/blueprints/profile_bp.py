from __future__ import annotations

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from ..extensions import job_manager
from ..services import profiling_service, render_helpers, session_store, uploads

profile_bp = Blueprint("profile", __name__, url_prefix="/profile")


@profile_bp.route("", methods=["GET"])
def targets():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    if not session_store.has_reference(sessions_root, session_id):
        flash("Build a reference first.", "error")
        return redirect(url_for("reference.build"))
    meta = session_store.read_meta(sessions_root, session_id)
    return render_template("profile/targets.html", target_texts=meta["target_texts"])


@profile_bp.route("/upload", methods=["POST"])
def upload():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    meta = session_store.read_meta(sessions_root, session_id)

    new_texts, warnings = uploads.collect_uploaded_texts(request.files.getlist("target_files"))
    meta["target_texts"].update(new_texts)

    pasted_name = (request.form.get("pasted_name") or "pasted_text").strip()
    pasted_text = request.form.get("pasted_text") or ""
    if pasted_text.strip():
        meta["target_texts"][pasted_name] = pasted_text

    session_store.write_meta(sessions_root, session_id, meta)
    for w in warnings:
        flash(w, "error")
    return redirect(url_for("profile.targets"))


@profile_bp.route("/run", methods=["POST"])
def run():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    meta = session_store.read_meta(sessions_root, session_id)
    if not meta["target_texts"]:
        return jsonify({"error": "Upload or paste at least one text to profile."}), 400

    def target_fn(progress_callback):
        return profiling_service.get_results(
            sessions_root, session_id, progress_callback=progress_callback,
        )

    job_id = job_manager.start(session_id, target_fn, redirect_url=url_for("profile.results"))
    return jsonify({"job_id": job_id})


@profile_bp.route("/results", methods=["GET"])
def results():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    if not session_store.has_reference(sessions_root, session_id):
        flash("Build a reference first.", "error")
        return redirect(url_for("reference.build"))

    meta = session_store.read_meta(sessions_root, session_id)
    if not meta["target_texts"]:
        flash("Upload or paste at least one text to profile.", "error")
        return redirect(url_for("profile.targets"))

    try:
        profiler = profiling_service.get_profiler(sessions_root, session_id)
        results_by_name = profiler.profile_texts(meta["target_texts"])
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("profile.targets"))

    selected = request.args.get("text")
    if selected not in results_by_name:
        selected = next(iter(results_by_name))
    result = results_by_name[selected]
    text = meta["target_texts"][selected]
    pos_tagged = profiler.reference.pos_tagged

    return render_template(
        "profile/results.html",
        results=results_by_name,
        selected=selected,
        result=result,
        band_95=render_helpers.band_for_coverage(result, 95),
        band_98=render_helpers.band_for_coverage(result, 98),
        legend=render_helpers.legend_entries(result, profiler),
        highlighted=render_helpers.highlighted_tokens(profiler, text, result),
        off_list_words=render_helpers.word_table_rows(
            result.off_list_words, result.word_counts, pos_tagged,
        ),
        ignored_words=render_helpers.word_table_rows(
            result.ignored_words, result.word_counts, pos_tagged,
        ),
        proper_noun_words=render_helpers.word_table_rows(
            result.proper_noun_words, result.word_counts, pos_tagged,
        ),
        digit_words=render_helpers.word_table_rows(
            result.digit_words, result.word_counts, pos_tagged,
        ),
        pos_tagged=pos_tagged,
        has_proper_nouns=any(r.proper_noun_words for r in results_by_name.values()),
        has_digits=any(r.digit_words for r in results_by_name.values()),
    )
