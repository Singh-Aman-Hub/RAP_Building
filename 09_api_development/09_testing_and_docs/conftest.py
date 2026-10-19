"""pytest auto-loads conftest.py. We use it to make the step-06 apps importable from the tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_model_api_flask_vs_fastapi"))
