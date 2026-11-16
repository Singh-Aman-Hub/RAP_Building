"""Solutions for step 09: 9.1 (percentile + histogram), 9.2 (break: no reset), 9.3 (propagate context to a thread)."""
import contextvars
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.observability import Tracer

# 9.1 nearest-rank percentile and a fixed-bucket histogram
def pct(xs, p):
    xs = sorted(xs)
    return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]
data = list(range(1, 101))
print("9.1 p50/p95/p99 of 1..100:", pct(data, 50), pct(data, 95), pct(data, 99))
latencies = [12, 15, 18, 22, 25, 40, 55, 90, 400, 1200]
buckets = [25, 50, 100, 500, 2000]
hist = {b: sum(x <= b for x in latencies) for b in buckets}       # cumulative, Prometheus-style "le" buckets
print("    cumulative histogram (le):", hist)

# 9.2 what 'no reset' does: simulate by never restoring the parent
var = contextvars.ContextVar("cur", default=None)
def span(name, reset):
    token = var.set(name)
    parent = None
    return token
t1 = var.set("request-1")                                          # request 1 starts and (bug) never resets
print("9.2 without reset, request 2 would see parent =", var.get(), "-> every later request nests under request 1")
var.reset(t1)
print("    after reset, parent =", var.get())

# 9.3 tracing across a worker thread: copy the context
tracer = Tracer()
with tracer.span("root") as root:
    plain = ThreadPoolExecutor(1).submit(Tracer.current).result()
    ctx = contextvars.copy_context()
    copied = ThreadPoolExecutor(1).submit(ctx.run, Tracer.current).result()
    def child_work():
        with tracer.span("worker") as s:
            return s.parent_id
    parent_in_worker = ThreadPoolExecutor(1).submit(contextvars.copy_context().run, child_work).result()
print("9.3 plain thread sees span:", plain is not None, "| copied context sees it:", copied is root,
      "| worker span parent is root:", parent_in_worker == root.span_id)
