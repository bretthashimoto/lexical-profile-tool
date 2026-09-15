from __future__ import annotations

from flask import Blueprint, Response, current_app, flash, redirect, session, url_for

from lexical_profiler import report as report_mod

from ..services import profiling_service, report_service

export_bp = Blueprint("export", __name__, url_prefix="/export")

# kind -> (export function, temp-file suffix, download filename, mimetype)
_EXPORTS = {
    "csv": (report_mod.export_csv, ".csv", "band_coverage.csv", "text/csv"),
    "json": (report_mod.export_json, ".json", "report.json", "application/json"),
    "off-list-csv": (report_mod.export_off_list_csv, ".csv", "off_list_words.csv", "text/csv"),
    "ignored-csv": (report_mod.export_ignored_csv, ".csv", "ignored_words.csv", "text/csv"),
    "proper-nouns-csv": (
        report_mod.export_proper_nouns_csv, ".csv", "proper_nouns.csv", "text/csv",
    ),
    "digits-csv": (report_mod.export_digits_csv, ".csv", "digits.csv", "text/csv"),
}


def _export(kind: str):
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    try:
        results = profiling_service.get_results(sessions_root, session_id)
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("profile.targets"))

    export_fn, suffix, filename, mimetype = _EXPORTS[kind]
    data = report_service.export_to_bytes(export_fn, results, suffix)
    return Response(data, mimetype=mimetype,
                     headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@export_bp.route("/csv")
def csv():
    return _export("csv")


@export_bp.route("/json")
def json_export():
    return _export("json")


@export_bp.route("/off-list-csv")
def off_list_csv():
    return _export("off-list-csv")


@export_bp.route("/ignored-csv")
def ignored_csv():
    return _export("ignored-csv")


@export_bp.route("/proper-nouns-csv")
def proper_nouns_csv():
    return _export("proper-nouns-csv")


@export_bp.route("/digits-csv")
def digits_csv():
    return _export("digits-csv")
