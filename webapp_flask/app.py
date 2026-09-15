"""Flask entrypoint for the LEAH web app.

Run with:
    flask --app webapp_flask.app run --debug --no-reload
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, session

REPO_ROOT = Path(__file__).resolve().parent.parent
# Guarantees `import lexical_profiler` works regardless of whether the
# package was pip-installed into this environment -- same rationale as
# webapp/app.py's identical sys.path insert.
sys.path.insert(0, str(REPO_ROOT))

from .blueprints.chart_bp import chart_bp  # noqa: E402
from .blueprints.export_bp import export_bp  # noqa: E402
from .blueprints.jobs_bp import jobs_bp  # noqa: E402
from .blueprints.pages_bp import pages_bp  # noqa: E402
from .blueprints.profile_bp import profile_bp  # noqa: E402
from .blueprints.reference_bp import reference_bp  # noqa: E402
from .config import Config  # noqa: E402
from .extensions import job_manager  # noqa: E402
from .services import session_store  # noqa: E402


def create_app(config_overrides: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if config_overrides:
        app.config.update(config_overrides)

    Path(app.config["SESSION_DIR"]).mkdir(parents=True, exist_ok=True)
    job_manager.configure(
        job_ttl=app.config["JOB_TTL"], run_inline=app.config.get("TESTING", False),
    )
    if not app.config.get("TESTING"):
        session_store.start_reaper(app.config["SESSION_DIR"], app.config["SESSION_TTL"])

    @app.before_request
    def ensure_session_id():
        if "session_id" not in session:
            session["session_id"] = uuid4().hex
        session.permanent = True
        session_store.touch(app.config["SESSION_DIR"], session["session_id"])

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(_e):
        return render_template("errors/500.html"), 500

    app.register_blueprint(pages_bp)
    app.register_blueprint(reference_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(chart_bp)
    app.register_blueprint(export_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
