"""
config_loader.py -- load + validate config.json.  Imported by main.py here and by steps 03, 06, 07.
(Kept separate from main.py because every step's script is called main.py: importing "main" from
another folder would pick the wrong file - a classic name-collision bug, see 01_debugging module.)
"""

import json                       # standard library: read/write JSON text <-> Python dict/list
from pathlib import Path          # standard library: object-oriented file paths

CONFIG_PATH = Path(__file__).parent / "config.json"   # __file__ = this script; works from any cwd

# Plain-English meaning of each parameter (a dict of dicts). Keep this next to the loader so
# the documentation cannot drift away from the code.
MEANING = {
    "training": {
        "learning_rate": "step size of each weight update. Too big = unstable, too small = slow (step 06)",
        "batch_size": "how many examples are processed before ONE weight update (step 06)",
        "epochs": "how many full passes over the whole dataset (step 06)",
        "optimizer": "the algorithm that turns gradients into updates (adamw, sgd) (step 06)",
        "weight_decay": "small penalty on big weights; fights overfitting (step 06)",
    },
    "generation": {
        "do_sample": "False = greedy (always the top token); True = random sampling (step 02)",
        "temperature": "<1 sharper/safer, >1 flatter/more random (step 02)",
        "top_k": "only sample from the k most likely tokens (step 02)",
        "top_p": "only sample from the smallest set of tokens whose probabilities add to p (step 02)",
        "max_new_tokens": "hard cap on how many tokens to produce (step 04)",
        "repetition_penalty": ">1 makes already-used tokens less likely, reduces loops (step 02)",
    },
    "limits": {
        "context_window": "max tokens (prompt + output) the model can attend to (step 04)",
    },
}


def load_config(path=CONFIG_PATH):
    """Read the JSON file, check values are sensible, and return a plain dict.

    Raises ValueError with a message that says WHICH parameter is wrong and why.
    """
    with open(path, encoding="utf-8") as f:      # 'with' closes the file automatically, even on errors
        cfg = json.load(f)                       # JSON text -> Python dict

    g, t, lim = cfg["generation"], cfg["training"], cfg["limits"]   # shortcuts to the sub-dicts
    # Validation: catch silly values EARLY instead of getting confusing behaviour later.
    if g["temperature"] <= 0:
        raise ValueError("generation.temperature must be > 0 (use do_sample=false for greedy decoding)")
    if not 0 < g["top_p"] <= 1:
        raise ValueError(f"generation.top_p must be in (0, 1], got {g['top_p']}")
    if g["top_k"] < 0:
        raise ValueError("generation.top_k must be >= 0 (0 disables it)")
    if g["max_new_tokens"] < 1 or g["max_new_tokens"] > lim["context_window"]:
        raise ValueError("generation.max_new_tokens must be between 1 and limits.context_window")
    if t["learning_rate"] <= 0 or t["batch_size"] < 1 or t["epochs"] < 1:
        raise ValueError("training.learning_rate > 0, batch_size >= 1 and epochs >= 1 are required")
    return cfg


def generation_kwargs(cfg):
    """Return only the keys Hugging Face `model.generate(**kwargs)` understands."""
    return dict(cfg["generation"])               # dict(...) makes a copy so callers can't mutate the config
