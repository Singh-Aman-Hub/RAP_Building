"""Exercise P4 solution: chained block hashes find the shared prefix; un-chained hashes give a WRONG hit."""
BLOCK = 16


def chained(tokens):
    h, out = 0, []
    for j in range(len(tokens) // BLOCK):                      # only full blocks
        h = hash((h, tuple(tokens[j * BLOCK:(j + 1) * BLOCK])))
        out.append(h)
    return out


def unchained(tokens):
    return [hash(tuple(tokens[j * BLOCK:(j + 1) * BLOCK])) for j in range(len(tokens) // BLOCK)]


def leading_hits(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


A = list(range(100))                                           # 6 full blocks + 4 tokens
B = A[:50] + [999] * 50                                        # shares exactly 50 tokens (3 full blocks + 2)
print("shared full blocks (chained):", leading_hits(chained(A), chained(B)), "-> tokens reusable:", leading_hits(chained(A), chained(B)) * BLOCK)
assert leading_hits(chained(A), chained(B)) == 3
# C has a DIFFERENT first block but the same second block as A: KV of block 2 differs (different context!)
C = [7] * 16 + A[16:32]
same_block2_unchained = unchained(A)[1] == unchained(C)[1]
same_block2_chained = chained(A)[1] == chained(C)[1]
print("block 2 key equal?  unchained:", same_block2_unchained, "(WRONG, KV differs)  chained:", same_block2_chained)
assert same_block2_unchained and not same_block2_chained
print("OK")
