import pytest

from webapp_flask.app import create_app


@pytest.fixture
def app(tmp_path):
    """A Flask app instance pointed at a throwaway per-test session
    directory. TESTING=True makes JobManager run jobs inline (see
    services/jobs.py) so tests don't need to poll a background thread."""
    return create_app({"SESSION_DIR": tmp_path / "sessions", "TESTING": True})


@pytest.fixture
def client(app):
    return app.test_client()
