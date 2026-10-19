"""
GOAL: Prove the app works WITHOUT Docker (always test the app before blaming Docker).
RUN: cd 11_docker/01_first_image && ../../.venv/bin/python check_locally.py
EXPECTED OUTPUT: two lines of JSON, then "OK".
"""
from fastapi.testclient import TestClient   # TestClient calls the app in-process: no server needed
from app.main import app

client = TestClient(app)
print(client.get("/health").json())
print(client.post("/predict", json={"text": "I love it"}).json())
assert client.post("/predict", json={"txt": "x"}).status_code == 422   # validation works
print("OK")
