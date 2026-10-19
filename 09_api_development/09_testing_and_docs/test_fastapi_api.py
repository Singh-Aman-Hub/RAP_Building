"""pytest tests for the FastAPI API.   RUN: cd 09_api_development/09_testing_and_docs && pytest -v"""
import pytest
from fastapi.testclient import TestClient

from fastapi_app import app


@pytest.fixture()
def client():
    with TestClient(app) as c:          # `with` runs startup/lifespan (loads the model)
        yield c                         # yield = give the client to the test, then clean up afterwards


def test_health_ok(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_negative(client):
    body = client.post("/predict", json={"text": "terrible and boring"}).json()
    assert body["label"] == "negative"
    assert 0.0 <= body["confidence"] <= 1.0


@pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": 5}])
def test_predict_rejects_bad_input(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_openapi_lists_predict(client):
    assert "/predict" in client.get("/openapi.json").json()["paths"]


def test_docs_page_served(client):
    resp = client.get("/docs")
    assert resp.status_code == 200 and "swagger" in resp.text.lower()
