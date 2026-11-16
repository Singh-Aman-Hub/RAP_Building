"""Solutions for step 08: 8.1 (third provider + budget), 8.2 (break: unknown alias), 8.3 (cost routing)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.gateway import BudgetExceeded, CostMeter, Gateway, LLMRequest, ScriptedProvider, Usage
from ragkit.resilience import TransientError

# 8.1 three providers: the first two are down
gw = Gateway({"default": [ScriptedProvider("a", [TransientError]), ScriptedProvider("b", [TransientError]), ScriptedProvider("c", ["third time lucky"])]})
r = gw.complete(LLMRequest.user("hi"))
print("8.1 answered by", r.provider, "| failures:", gw.last_chain.failures)

# 8.1b budget: the second call is refused BEFORE reaching any provider
gw2 = Gateway({"large": [ScriptedProvider("p", ["x " * 400, "never reached"])]}, CostMeter({"large": (3.0, 15.0)}, budget_usd=0.002))
gw2.complete(LLMRequest.user("hello there", model="large"))
try:
    gw2.complete(LLMRequest.user("again", model="large"))
except BudgetExceeded as e:
    print("    budget:", e)

# 8.2 unknown alias -> KeyError; a friendlier gateway would list valid aliases
try:
    gw.complete(LLMRequest.user("q", model="medium"))
except KeyError as e:
    print("8.2 KeyError:", e, "-> valid aliases:", sorted(gw.routes))

# 8.3 cost routing: send short questions to small, long ones to large
prices = {"small": (0.25, 1.25), "large": (3.0, 15.0)}
cm = CostMeter(prices)
traffic = [(300, 100)] * 800 + [(3000, 600)] * 200            # (input, output) tokens: 80% easy, 20% hard
all_large = sum(cm.cost("large", Usage(i, o)) for i, o in traffic)
routed = sum(cm.cost("small" if i < 1000 else "large", Usage(i, o)) for i, o in traffic)
print(f"8.3 1000 requests all on large: ${all_large:.2f}; routed by size: ${routed:.2f} ({100 * (1 - routed / all_large):.0f}% cheaper)")
