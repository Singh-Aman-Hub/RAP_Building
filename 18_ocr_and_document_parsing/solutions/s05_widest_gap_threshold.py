"""Exercise 5: reimplement widest_gap and show why min_gap_x=30 misses a 25 px gutter. RUN: python s05_widest_gap_threshold.py"""
import numpy as np


def widest_gap(nonempty, min_gap):
    idx = np.flatnonzero(nonempty)
    gaps = [(idx[i + 1] - idx[i] - 1, int(idx[i]) + 1, int(idx[i + 1])) for i in range(len(idx) - 1) if idx[i + 1] - idx[i] - 1 >= min_gap]
    if not gaps:
        return None
    _, a, b = max(gaps)
    return a, b


cols = np.zeros(300, bool)
cols[10:130] = True                     # left column text, words separated by 5 px gaps
cols[135:240] = True                    # (a word gap of 4 px at 130..134 inside the left column is NOT a cut)
cols[265:290] = True                    # right column starts after a 25 px gutter (240..264)
print("min_gap 30:", widest_gap(cols, 30), "(missed the gutter: columns merge)")
print("min_gap 20:", widest_gap(cols, 20), "(cuts the 25 px gutter at x=240..265)")
assert widest_gap(cols, 30) is None and widest_gap(cols, 20) == (240, 265)
print("ok")
