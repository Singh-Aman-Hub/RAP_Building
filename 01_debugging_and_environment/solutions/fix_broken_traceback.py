"""Solution for 01_reading_tracebacks/broken_traceback.py.
Root cause: 'carol' is not in the scores dict (bad INPUT), not a bug in report().
Fix choice: report missing names explicitly instead of crashing. Verified by the asserts at the bottom."""


def load_scores():
    return {"alice": 0.9, "bob": 0.7}


def report(names):
    scores = load_scores()
    lines = []
    for name in names:
        if name in scores:
            lines.append(f"{name}: {scores[name]}")
        else:
            lines.append(f"{name}: (no score)")    # visible, not silent
    return "\n".join(lines)


out = report(["alice", "bob", "carol"])
print(out)
assert out.splitlines()[-1] == "carol: (no score)"
print("verified")
