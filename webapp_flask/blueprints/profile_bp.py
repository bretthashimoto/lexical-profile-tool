from __future__ import annotations

from flask import Blueprint, current_app, flash, jsonify, redirect, request, session, url_for

from ..extensions import job_manager
from ..services import profiling_service, session_store, uploads

profile_bp = Blueprint("profile", __name__, url_prefix="/profile")


@profile_bp.route("", methods=["GET"])
def targets():
    """Building a reference and profiling texts both live on one page now
    (see reference_bp.build) -- this just keeps old links/bookmarks
    working."""
    return redirect(url_for("reference.build"))


@profile_bp.route("/upload", methods=["POST"])
def upload():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    meta = session_store.read_meta(sessions_root, session_id)

    file_storages = request.files.getlist("target_files") + request.files.getlist("target_folder")
    new_texts, warnings = uploads.collect_uploaded_texts(file_storages)
    meta["target_texts"].update(new_texts)

    pasted_name = (request.form.get("pasted_name") or "pasted_text").strip()
    pasted_text = request.form.get("pasted_text") or ""
    if pasted_text.strip():
        meta["target_texts"][pasted_name] = pasted_text

    if not meta["target_texts"]:
        return jsonify({"error": "Upload or paste at least one text to profile."}), 400

    session_store.write_meta(sessions_root, session_id, meta)
    for w in warnings:
        flash(w, "warning")

    def target_fn(progress_callback):
        return profiling_service.get_results(
            sessions_root, session_id, progress_callback=progress_callback,
        )

    job_id = job_manager.start(
        session_id, target_fn, redirect_url=url_for("reference.build", _anchor="step-3"),
    )
    return jsonify({"job_id": job_id, "warnings": warnings})


@profile_bp.route("/results", methods=["GET"])
def results():
    """Results render inline on reference_bp.build now -- redirect there,
    preserving ?text= so a bookmarked/shared link still lands on the same
    text's results."""
    text = request.args.get("text")
    if text:
        return redirect(url_for("reference.build", text=text, _anchor="step-3"))
    return redirect(url_for("reference.build", _anchor="step-3"))
