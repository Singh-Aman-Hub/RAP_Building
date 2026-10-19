"""Solution for broken_types.py: fix all three bugs."""


def tokens_to_text(tokens):
    return " ".join(str(t) for t in tokens)      # bug 1: convert to str (or validate and reject)


def clean(text):
    return text.strip().lower()                   # bug 2: the missing `return`


def total_length(words):
    return sum(len(w) for w in words)


print(tokens_to_text([101, 7592, 2088]))
print(clean("  Hello World ").split()[0])
print(total_length(["a", "bb"]) + int("1"))       # bug 3: convert the string explicitly
