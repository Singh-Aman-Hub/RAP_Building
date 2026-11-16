"""Exercise D4: at what monthly volume does 2 always-on GPUs ($3/h each, example) beat an API at $0.60/1M input + $2.40/1M output
(80/20 mix)? How does the answer change if redundancy needs 50% more always-on capacity? """
GPU_H, GPUS, HOURS = 3.0, 2, 730
P_IN, P_OUT = 0.60, 2.40
blended = 0.8 * P_IN + 0.2 * P_OUT                     # 0.96 $/1M tokens
fixed = GPUS * GPU_H * HOURS                           # 4380 $/month, paid even when idle
be = fixed / blended                                   # millions of tokens per month
print(f"D4: blended API price ${blended:.2f}/1M, fixed ${fixed:,.0f}/month -> break-even {be:,.0f}M tokens/month")
assert abs(be - 4562.5) < 1e-9
# if half of the traffic needs a second, always-on redundancy copy, fixed cost rises 50% and break-even rises 50%
be_redundant = fixed * 1.5 / blended
print(f"    with 50% extra capacity for redundancy: {be_redundant:,.0f}M tokens/month")
assert abs(be_redundant / be - 1.5) < 1e-12
