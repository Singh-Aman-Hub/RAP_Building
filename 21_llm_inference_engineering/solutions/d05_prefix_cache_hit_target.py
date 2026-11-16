"""Exercise D5: what prefix-cache hit rate cuts an API request's cost by 30%?
Prices (example): input $1.00, cached input $0.25, output $4.00 per 1M tokens. Request: 3000 input tokens, 300 output tokens."""
IN, CACHED, OUT = 1.00, 0.25, 4.00
tin, tout = 3000, 300


def cost(hit):
    c = int(tin * hit)
    return ((tin - c) * IN + c * CACHED + tout * OUT) / 1e6


base = cost(0.0)
# cost(h) = base - tin*h*(IN-CACHED)/1e6  =>  solve for 30% saving
h = 0.30 * base * 1e6 / (tin * (IN - CACHED))
print(f"D5: base ${base:.6f}/request; need hit rate {h:.1%} for -30% (cost {cost(h):.6f})")
assert abs(cost(h) / base - 0.70) < 0.001 and 0 < h < 1
