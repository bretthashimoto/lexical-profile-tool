"""Thin smoke coverage for the Flask webapp: route reachability plus one
full flow (build a reference -> profile -> results -> export), using the
`app`/`client` fixtures from conftest.py (TESTING=True runs background
jobs inline, so no polling/sleeping is needed here).

Not exhaustive -- visual/branding parity has no automated coverage here,
matching how webapp/app.py (the Streamlit app) has never had automated
UI tests either. See the plan file for the fuller verification approach.
"""

import pytest

STATIC_PAGES = ["/reference/build", "/guide", "/about", "/cite", "/about-me"]


@pytest.mark.parametrize("path", STATIC_PAGES)
def test_static_pages_reachable(client, path):
    r = client.get(path)
    assert r.status_code == 200


@pytest.mark.parametrize("path", ["/", "/profile", "/profile/results"])
def test_pages_redirect_without_a_reference_yet(client, path):
    r = client.get(path)
    assert r.status_code == 302


def test_unknown_route_404s(client):
    r = client.get("/this-does-not-exist")
    assert r.status_code == 404


def _build_builtin_reference(client, **overrides):
    data = {"builtin_name": "ngsl", "band_size": "500", "language": "en"}
    data.update(overrides)
    r = client.post("/reference/build/builtin", data=data)
    assert r.status_code == 302
    return r


def test_full_flow_builtin_reference_to_results(client):
    _build_builtin_reference(client)

    r = client.get("/reference/build")
    assert b"known words" in r.data

    r = client.post("/profile/upload", data={
        "pasted_name": "sample", "pasted_text": "the cat sat on the mat",
    })
    assert r.status_code == 200
    assert "job_id" in r.get_json()

    r = client.get("/profile/results")
    assert r.status_code == 200
    body = r.data.decode()
    assert "sample" in body
    assert "Tokens" in body


def test_ignore_config_is_applied_at_profile_time(client):
    _build_builtin_reference(client)
    client.post("/reference/ignore-config", data={"ignore_text": "cat\nmat"})
    client.post("/profile/upload", data={
        "pasted_name": "sample", "pasted_text": "the cat sat on the mat",
    })

    r = client.get("/profile/results")
    body = r.data.decode()
    assert "Ignored words" in body


@pytest.fixture
def profiled_client(client):
    _build_builtin_reference(client)
    r = client.post("/profile/upload", data={
        "pasted_name": "sample",
        "pasted_text": "Dr. Smith bought 3 apples in 2024. The cat sat on the mat.",
    })
    assert r.status_code == 200
    return client


@pytest.mark.parametrize("path,content_type", [
    ("/export/csv", "text/csv"),
    ("/export/json", "application/json"),
    ("/export/off-list-csv", "text/csv"),
    ("/export/ignored-csv", "text/csv"),
    ("/export/proper-nouns-csv", "text/csv"),
    ("/export/digits-csv", "text/csv"),
])
def test_export_routes(profiled_client, path, content_type):
    r = profiled_client.get(path)
    assert r.status_code == 200
    assert content_type in r.headers["Content-Type"]
    assert "attachment" in r.headers["Content-Disposition"]
    assert len(r.data) > 0


def test_chart_json_endpoint(profiled_client):
    r = profiled_client.get("/chart/band-coverage.json?text=sample")
    assert r.status_code == 200
    spec = r.get_json()
    assert "layer" in spec


def test_chart_json_unknown_text_404s(profiled_client):
    r = profiled_client.get("/chart/band-coverage.json?text=does-not-exist")
    assert r.status_code == 404


def test_load_bundled_example_data(client):
    r = client.post("/guide/load-example")
    assert r.status_code == 302

    r = client.get("/profile/results")
    assert r.status_code == 200


def test_job_progress_endpoint_reports_done(client):
    r = client.post("/reference/build/builtin",
                     data={"builtin_name": "ngsl", "band_size": "500", "language": "en"})
    assert r.status_code == 302

    r = client.post("/profile/upload", data={"pasted_name": "s", "pasted_text": "the cat"})
    job_id = r.get_json()["job_id"]

    r = client.get(f"/jobs/{job_id}/progress")
    assert r.status_code == 200
    assert r.get_json()["status"] == "done"


def test_job_progress_is_session_scoped(client, app):
    r = client.post("/reference/build/builtin",
                     data={"builtin_name": "ngsl", "band_size": "500", "language": "en"})
    r = client.post("/profile/upload", data={"pasted_name": "s", "pasted_text": "the cat"})
    job_id = r.get_json()["job_id"]

    with app.test_client() as other_client:
        r = other_client.get(f"/jobs/{job_id}/progress")
        assert r.status_code == 404
