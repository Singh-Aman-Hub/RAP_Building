"""Solutions for step 05: 5.1 (plugin by name), 5.2 (break: duplicate), 5.3 (Registry + Chain from scratch, tiny)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.factory import RETRIEVERS, build_system
from ragkit.ingest import DirectoryLoader
from ragkit.patterns import Registry, UnknownPluginError
from ragkit.query import DenseRetriever

KB = Path(__file__).resolve().parents[2] / "14_rag_and_retrieval" / "data" / "kb"

# 5.1 add a plugin WITHOUT editing factory.py: a dense retriever restricted to one document through the store filter
@RETRIEVERS.register("refunds_doc_only")
def make_refunds_only(embedder, store, **_):
    from ragkit.ids import content_hash
    return DenseRetriever(embedder, store, where={"doc_id": content_hash("refunds_and_returns.md")})
system = build_system({"retriever": {"type": "refunds_doc_only"}}, DirectoryLoader(KB))
system.pipeline.run()
print("5.1 retrievers:", RETRIEVERS.names(), "| answer:", system.service.answer("How many days for a refund?").text[:50], "...")

# 5.2 duplicates and unknown names
try:
    RETRIEVERS.register("refunds_doc_only")(lambda **_: None)
except ValueError as e:
    print("5.2", e)
try:
    RETRIEVERS.create("nope")
except UnknownPluginError as e:
    print("    ", e)

# 5.3 a Registry in 10 lines (what you would write on a whiteboard)
class TinyRegistry:
    def __init__(self): self.f = {}
    def register(self, name):
        def deco(fn):
            assert name not in self.f, f"{name} already registered"
            self.f[name] = fn
            return fn
        return deco
    def create(self, name, **kw):
        if name not in self.f:
            raise KeyError(f"unknown {name!r}; available {sorted(self.f)}")
        return self.f[name](**kw)
r = TinyRegistry()
r.register("double")(lambda x: x * 2)
print("5.3 tiny registry:", r.create("double", x=21))
