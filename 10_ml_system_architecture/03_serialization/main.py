"""
GOAL:   Learn how models are saved and loaded (joblib, pickle, safetensors), compare them,
        and understand WHY loading a pickle from an untrusted source is dangerous.

RUN:    cd 10_ml_system_architecture/03_serialization && python main.py

EXPECTED OUTPUT (shape): sizes of each format, round-trip equality checks, a harmless demonstration
        of code running during unpickling, and a gotcha with custom functions.

NEEDS:  scikit-learn, joblib, safetensors + numpy (safetensors comes with transformers)
"""

import os
import pickle                                   # built-in: serialize almost any Python object
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from safetensors.numpy import load_file, save_file   # safetensors: store ONLY named arrays (no code)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_pipeline_end_to_end"))
from data import get_test_data, get_training_data  # noqa: E402

tmp = Path(tempfile.mkdtemp())                  # throwaway folder for this demo
X, y = get_training_data()
X_test, _ = get_test_data()
model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))]).fit(X, y)
reference = model.predict_proba(X_test)         # remember what the ORIGINAL model says

# ---- 1. pickle and joblib: save the whole Python object (structure + learned numbers) ----
with open(tmp / "model.pkl", "wb") as f:        # "wb" = write binary
    pickle.dump(model, f)
joblib.dump(model, tmp / "model.joblib")        # joblib = pickle + efficient storage of big numpy arrays
for name in ("model.pkl", "model.joblib"):
    print(f"1. {name:<13} {os.path.getsize(tmp / name):>6} bytes")
with open(tmp / "model.pkl", "rb") as f:
    from_pickle = pickle.load(f)
from_joblib = joblib.load(tmp / "model.joblib")
print("   round-trip identical predictions:",
      np.allclose(reference, from_pickle.predict_proba(X_test)), np.allclose(reference, from_joblib.predict_proba(X_test)))

# ---- 2. safetensors: store only the numbers (weights), never code ----
weights = {"coef": model[-1].coef_, "intercept": model[-1].intercept_}   # model[-1] = last pipeline step (the classifier)
save_file(weights, str(tmp / "weights.safetensors"))
back = load_file(str(tmp / "weights.safetensors"))
print(f"2. safetensors    {os.path.getsize(tmp / 'weights.safetensors'):>6} bytes; arrays: {list(back)}; "
      f"identical: {np.array_equal(back['coef'], weights['coef'])}")
print("   (it stores arrays only: to use them you must rebuild the model code yourself -- that is the safety trade-off.)")
print("   Hugging Face LLM checkpoints (module 04 step 06, load pretrained model) use this format: model.safetensors.")

# ---- 3. SECURITY: unpickling RUNS CODE chosen by whoever made the file ----
class Innocent:
    def __reduce__(self):                       # tells pickle "to rebuild me, call this function with these args"
        return (print, ("!!! code ran during pickle.load -- an attacker could run ANYTHING here !!!",))

evil_bytes = pickle.dumps(Innocent())           # the 'model file' an attacker would send you
print("3. loading an untrusted pickle:")
pickle.loads(evil_bytes)                        # prints the message: proof that load == execute
print("   Rule: only load pickle/joblib files you created or fully trust. Prefer safetensors/ONNX for sharing.")

# ---- 4. GOTCHA: custom functions are pickled BY NAME ----
def shout(text):                                # defined in __main__ of THIS script
    return text.upper()

custom = Pipeline([("tfidf", TfidfVectorizer(preprocessor=shout)), ("clf", LogisticRegression(max_iter=1000))])
custom.fit(X, y)
joblib.dump(custom, tmp / "custom.joblib")
print("4. custom.joblib saved. Loading it from a DIFFERENT script fails unless that script can import `shout`:")
print("   try:  python -c \"import joblib; joblib.load('" + str(tmp / 'custom.joblib') + "')\"   -> AttributeError: Can't get attribute 'shout' on <module '__main__'>")
print("   Fix: keep custom code in an importable module (my_module.shout) and deploy that module with the model, or use only built-in sklearn parts.")
# TRY IT: train with n=2000 samples (get_training_data(n=2000)) and compare the pickle vs joblib sizes again.
# BREAK IT: run the `python -c` command printed above and read the AttributeError traceback.
