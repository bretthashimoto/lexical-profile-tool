from __future__ import annotations

from flask import Blueprint, abort, current_app, jsonify, request, session

from ..services import chart_service, profiling_service

chart_bp = Blueprint("chart", __name__, url_prefix="/chart")


@chart_bp.route("/band-coverage.json", methods=["GET"])
def band_coverage():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    name = request.args.get("text")

    try:
        results = profiling_service.get_results(sessions_root, session_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if name not in results:
        abort(404)

    return jsonify(chart_service.build_band_chart(results[name]))
