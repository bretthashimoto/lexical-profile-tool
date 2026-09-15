from __future__ import annotations

from flask import Blueprint, jsonify, session

from ..extensions import job_manager

jobs_bp = Blueprint("jobs", __name__, url_prefix="/jobs")


@jobs_bp.route("/<job_id>/progress", methods=["GET"])
def progress(job_id):
    job = job_manager.get(job_id, session["session_id"])
    if job is None:
        return jsonify({"status": "not_found"}), 404

    if job.status == "running":
        return jsonify({
            "status": "running",
            "current": job.current,
            "total": job.total,
            "message": job.message,
        })
    if job.status == "done":
        return jsonify({"status": "done", "redirect_url": job.redirect_url})
    return jsonify({"status": "error", "error": job.error})
