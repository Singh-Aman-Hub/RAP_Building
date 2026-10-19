"""pytest tests for the Flask API.   RUN: cd 09_api_development/09_testing_and_docs && pytest -v"""
import pytest

from flask_app import app


@pytest.fixture()                      # a fixture = reusable setup; pytest passes its result to tests that ask for it
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200      # `assert` = the test fails if the condition is False
    assert resp.get_json()["status"] == "ok"


def test_predict_positive(client):
    resp = client.post("/predict", json={"text": "I love this great film"})
    assert resp.status_code == 200
    assert resp.get_json()["label"] == "positive"


@pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": 5}])   # run the same test with 3 inputs
def test_predict_rejects_bad_input(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_predict_not_json(client):
    resp = client.post("/predict", data="hello", content_type="application/json")
    assert resp.status_code == 400


def test_max_length_comes_from_env(client, monkeypatch):
    monkeypatch.setenv("MAX_TEXT_LENGTH", "5")   # monkeypatch undoes the change after the test
    assert client.post("/predict", json={"text": "far too long"}).status_code == 422
