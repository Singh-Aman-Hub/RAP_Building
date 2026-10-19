"""
GOAL: Test the env-var behaviour without Docker.
RUN: cd 11_docker/03_ports_env_volumes && ../../.venv/bin/python check_locally.py
EXPECTED OUTPUT: model_name from the env var, visits counted in a temp dir, then OK.
"""
import os
import tempfile

# Env vars must be set BEFORE importing the app, because main.py reads them at import time.
os.environ["MODEL_NAME"] = "test-model"
os.environ["DATA_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
r = client.post("/predict", json={"text": "great stuff"}).json()
print(r)
print(client.get("/info").json())
assert r["model_name"] == "test-model" and client.get("/info").json()["visits"] == 1
print("OK")
