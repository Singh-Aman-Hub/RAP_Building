"""
Worked solutions (Flask side) for exercises: DELETE route (step 03) and GET /labels (step 06).
RUN: cd 09_api_development/solutions && python sol_flask_delete_and_labels.py
"""
import sys
from pathlib import Path

from flask import Flask, jsonify

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_model_api_flask_vs_fastapi"))
from model_loader import load_model  # noqa: E402

app = Flask(__name__)
ITEMS = {1: {"id": 1, "name": "pen"}}
MODEL, _ = load_model()


@app.delete("/items/<int:item_id>")
def delete_item(item_id):
    if ITEMS.pop(item_id, None) is None:          # pop(key, default) removes and returns, or default if missing
        return jsonify(error="not found"), 404
    return "", 204                                # 204 = success with NO body


@app.get("/labels")
def labels():
    return jsonify(labels=[str(c) for c in MODEL.classes_])


if __name__ == "__main__":
    c = app.test_client()
    print(c.delete("/items/1").status_code, c.delete("/items/1").status_code, c.get("/labels").get_json())
