"""EXPECTED TO CRASH. Three typical type bugs; the script stops at the first. Fix one, re-run, repeat."""


def tokens_to_text(tokens):
    return " ".join(tokens)


def first_word(text):
    return text.split()[0]


def total_length(words):
    return sum(len(w) for w in words)


# Bug 1: a list of ints passed where list of str is expected
print(tokens_to_text(["hello", "world"]))
print(tokens_to_text([101, 7592, 2088]))

# Bug 2 (reached after fixing 1): a function that returns None implicitly
def clean(text):
    text.strip().lower()          # forgot `return`

print(first_word(clean("  Hello World ")))

# Bug 3: string where number expected (e.g. value read from input() or an env var)
print(total_length(["a", "bb"]) + "1")
