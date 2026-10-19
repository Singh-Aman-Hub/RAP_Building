"""pytest tests: run `pytest -q` in this folder after train.py. Shows how ML code gets tested."""
import pytest
from predict import predict_texts


def test_positive_and_negative():
    out = predict_texts(["I love it, great and excellent", "terrible awful waste"])
    assert out[0]["label"] == "positive"
    assert out[1]["label"] == "negative"


def test_confidence_range():
    assert 0.0 <= predict_texts(["fine"])[0]["confidence"] <= 1.0


def test_bad_input():
    with pytest.raises(TypeError):
        predict_texts("not a list")
