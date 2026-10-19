"""
GOAL:  See Python's basic value types, how to inspect them, and how to print them with f-strings.

RUN:   cd 00_python_foundations/01_variables_and_types && python main.py
       (or from repo root: python 00_python_foundations/01_variables_and_types/main.py)

EXPECTED OUTPUT (first lines):
    learning_rate = 0.001  -> <class 'float'>
    ...

NEEDS: nothing (standard library only), no internet.
"""

# A variable is just a name pointing at a value. No type declaration needed.
learning_rate = 0.001          # float  : number with a decimal point
epochs = 3                     # int    : whole number
model_name = "distilgpt2"      # str    : text
is_training = True             # bool   : True / False (capital first letter!)
nothing_yet = None             # NoneType : "no value" marker, often used for "not set"

# type() tells you what kind of value a name points to. Built-in function.
for name, value in [("learning_rate", learning_rate), ("epochs", epochs),
                    ("model_name", model_name), ("is_training", is_training),
                    ("nothing_yet", nothing_yet)]:
    # f-string: put f before the quotes and write {expressions} inside the text.
    # {value!r} shows the "repr" (strings get quotes) which is better for debugging.
    print(f"{name} = {value!r}  -> {type(value)}")

# ---- arithmetic ---------------------------------------------------------
print("\n--- arithmetic ---")
print("7 / 2  =", 7 / 2)      # true division always gives a float -> 3.5
print("7 // 2 =", 7 // 2)     # floor division -> 3
print("7 % 2  =", 7 % 2)      # remainder (modulo) -> 1
print("2 ** 10 =", 2 ** 10)   # power -> 1024

# ---- strings ------------------------------------------------------------
print("\n--- strings ---")
text = "  Hello, Tokenizer!  "
print(text.strip())            # strip() removes spaces at both ends (returns a NEW string)
print(text.lower())            # strings are immutable: the original is unchanged
print(text.strip().split(", "))  # split() -> list of pieces
print("-".join(["a", "b", "c"]))  # join() glues a list of strings together
print(model_name[0:4])         # slicing: characters 0,1,2,3 -> 'dist'
print(model_name[-1])          # negative index counts from the end -> '2'

# ---- converting between types (very common source of bugs) --------------
print("\n--- conversions ---")
print(int("42") + 1)           # str -> int
print(float("0.5") * 2)        # str -> float
print(str(123) + "!")          # int -> str
print(bool(0), bool(""), bool([]), bool("x"))  # "falsy" values: 0, "", [], None, {}

# f-string number formatting: {value:.2f} = 2 decimals, {value:,} = thousands commas
loss = 2.718281828
params = 124_439_808           # underscores in numbers are just for readability
print(f"loss={loss:.3f}  params={params:,}")

# TRY IT: change epochs to "3" (a string) and re-run section "arithmetic" ideas:
#         what happens with epochs + 1 ?
# BREAK IT: uncomment the next line to see a TypeError traceback (read the last line first!)
# print("epochs: " + epochs)
