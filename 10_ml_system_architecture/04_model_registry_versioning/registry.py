"""
registry.py -- a minimal MODEL REGISTRY on the local filesystem. Real tools (MLflow, W&B, SageMaker, HF Hub)
do the same job with more features.

Layout it creates:
    registry/
      v0001/model.joblib      <- the artifact (never modified after saving)
      v0001/metadata.json     <- metrics, params, library versions, creation time, data fingerprint
      v0002/...
      aliases.json            <- {"production": "v0001", "staging": "v0002"}  movable labels

Why aliases? Code says "load production"; ops moves the label. Rolling back = moving the label back,
no redeploy of code needed.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.alias_file = self.root / "aliases.json"

    # ---------- write side ----------
    def versions(self) -> list[str]:
        """All version names sorted oldest -> newest ('v0001' sorts correctly because of zero padding)."""
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and p.name.startswith("v"))

    def save(self, model, metrics: dict, params: dict, training_texts: list[str]) -> str:
        n = len(self.versions()) + 1
        version = f"v{n:04d}"                               # f-string format: pad number to 4 digits with zeros
        folder = self.root / version
        folder.mkdir()
        joblib.dump(model, folder / "model.joblib")
        fingerprint = hashlib.sha256("\n".join(training_texts).encode()).hexdigest()[:12]   # identifies the training data
        (folder / "metadata.json").write_text(json.dumps({
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": metrics,
            "params": params,
            "data_fingerprint": fingerprint,
            "sklearn_version": sklearn.__version__,
        }, indent=2))
        return version

    def _aliases(self) -> dict:
        return json.loads(self.alias_file.read_text()) if self.alias_file.exists() else {}

    def promote(self, version: str, alias: str = "production") -> None:
        if version not in self.versions():
            raise KeyError(f"unknown version {version!r}; have {self.versions()}")
        aliases = self._aliases()
        aliases[alias] = version
        self.alias_file.write_text(json.dumps(aliases, indent=2))

    # ---------- read side ----------
    def resolve(self, ref: str = "latest") -> str:
        """Turn 'latest' / 'production' / 'v0002' into a concrete version name."""
        if ref == "latest":
            return self.versions()[-1]
        aliases = self._aliases()
        if ref in aliases:
            return aliases[ref]
        if ref in self.versions():
            return ref                                      # pinned exact version
        raise KeyError(f"cannot resolve {ref!r}")

    def load(self, ref: str = "latest"):
        version = self.resolve(ref)
        folder = self.root / version
        return joblib.load(folder / "model.joblib"), json.loads((folder / "metadata.json").read_text())
