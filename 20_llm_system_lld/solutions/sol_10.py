"""Solutions for step 10: 10.1 (calibrate the refusal threshold with the eval harness), 10.2 (break: bad input), 10.3 (agent loop)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import BaseModel

from ragkit.agent import AgentRuntime, ToolRegistry
from ragkit.evalkit import ContainsScorer, EvalCase, EvalRunner, RefusalScorer, RetrievalHitScorer
from ragkit.factory import build_system
from ragkit.gateway import Gateway, ScriptedProvider, ToolCall, ToolSpec
from ragkit.ingest import DirectoryLoader
from ragkit.memory import ConversationMemory

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "14_rag_and_retrieval" / "data" / "kb"
system = build_system({}, DirectoryLoader(KB))
system.pipeline.run()
by_key = {f"{Path(c.metadata['source']).stem}#{c.section}": c.chunk_id for c in system.store.all_chunks()}
raw = json.loads((ROOT / "14_rag_and_retrieval" / "data" / "eval_questions.json").read_text())
cases = [EvalCase(r["id"], r["question"], r.get("answer") if r["gold"] else None, tuple(by_key[g] for g in r["gold"]), bool(r["gold"])) for r in raw]

# 10.1 sweep the refusal threshold: false refusals vs false accepts
print("10.1 threshold  answerable-answered  unanswerable-refused  refusal_correct")
for th in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
    system.service.refusal.min_score = th
    rep = EvalRunner(system.service, [RefusalScorer()]).run(cases)
    rows = {r["id"]: r["refusal_correct"] for r in rep.rows}
    ans_ok = sum(rows[c.id] for c in cases if c.answerable) / sum(c.answerable for c in cases)
    unans_ok = sum(rows[c.id] for c in cases if not c.answerable) / sum(not c.answerable for c in cases)
    print(f"      {th:4.1f}        {ans_ok:5.2f}               {unans_ok:5.2f}                {rep.mean('refusal_correct'):.2f}")
system.service.refusal.min_score = 0.3

# 10.2 break: None content reaches the token counter. Validate at the boundary.
m = ConversationMemory(max_tokens=10)
def safe_add(sid, role, content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a non-empty string")
    m.add(sid, role, content)
try:
    safe_add("s", "user", None)
except ValueError as e:
    print("10.2 boundary validation:", e)

# 10.3 agent that keeps asking for the same tool with the same arguments
class A(BaseModel):
    q: str
reg = ToolRegistry(); reg.register(ToolSpec.from_model("search", "search", A), lambda q: "nothing")
same = ("", [ToolCall("1", "search", {"q": "x"})])
res = AgentRuntime(Gateway({"default": [ScriptedProvider("p", [same] * 9)]}), reg, max_steps=8).run("go")
print("10.3 stopped:", res.stopped, "after", res.steps, "steps (max_steps was 8)")
