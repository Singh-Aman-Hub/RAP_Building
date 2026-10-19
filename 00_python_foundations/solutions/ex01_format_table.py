"""Ex 01: aligned table with f-string width/format specs."""
models = [("distilgpt2", 81_912_576), ("gpt2", 124_439_808), ("gpt2-medium", 354_823_168)]
print(f"{'model':<14}{'parameters':>14}")           # < left-align, > right-align, number = width
for name, params in models:
    print(f"{name:<14}{params:>14,}")
