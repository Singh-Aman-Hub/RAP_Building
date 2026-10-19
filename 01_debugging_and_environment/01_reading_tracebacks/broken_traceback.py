"""
EXPECTED TO CRASH.  Do the drill: do NOT fix it until you can answer
  (a) what is the error type?  (b) which line RAISED it?  (c) which line CALLED the function?  (d) root cause?
RUN: python broken_traceback.py
"""


def load_scores():
    return {"alice": 0.9, "bob": 0.7}


def best_student(scores):
    best = max(scores, key=scores.get)
    return best, scores[best]


def report(names):
    scores = load_scores()
    lines = []
    for name in names:
        lines.append(f"{name}: {scores[name]}")   # one of these names is not in scores
    return "\n".join(lines)


print(best_student(load_scores()))
print(report(["alice", "bob", "carol"]))
