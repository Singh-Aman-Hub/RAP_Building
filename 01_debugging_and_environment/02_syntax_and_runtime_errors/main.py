"""
GOAL:  Meet the most common built-in exception types and learn what each one usually means.

RUN:   cd 01_debugging_and_environment/02_syntax_and_runtime_errors && python main.py
       Crashing versions (EXPECTED to fail):  python broken_syntax.py   |   python broken_runtime.py

EXPECTED OUTPUT: a table  "ErrorType: message  -> usual cause".

NEEDS: nothing (standard library only), no internet.
"""

# Each case is (label, function that fails, usual cause). We run each and print the real message.
def index_error():    return [1, 2, 3][5]
def key_error():      return {"a": 1}["b"]
def type_error():     return "age: " + 30
def value_error():    return int("abc")
def attr_error():     return None.upper()          # classic: a function returned None
def name_error():     return undefined_variable    # noqa: F821  (typo or never defined)
def zero_error():     return 1 / 0
def file_error():     return open("no_such_file.txt")
def import_error():   import not_a_real_module     # noqa: F401
def recursion_error():
    def f(): return f()
    return f()


CASES = [
    ("IndexError", index_error, "index beyond list length (off-by-one?)"),
    ("KeyError", key_error, "dict key missing -> use .get() or check `in`"),
    ("TypeError", type_error, "wrong type for an operation / wrong number of args"),
    ("ValueError", value_error, "right type, impossible value"),
    ("AttributeError", attr_error, "object has no such attribute; often the object is None"),
    ("NameError", name_error, "typo, or used before assignment"),
    ("ZeroDivisionError", zero_error, "denominator is 0 (empty list average?)"),
    ("FileNotFoundError", file_error, "wrong path or wrong working directory"),
    ("ModuleNotFoundError", import_error, "package not installed in THIS interpreter"),
    ("RecursionError", recursion_error, "function calls itself without a base case"),
]

for label, func, cause in CASES:
    try:
        func()
    except Exception as err:               # Exception = base class of almost all runtime errors
        print(f"{type(err).__name__}: {err}")
        print(f"    -> usual cause: {cause}\n")

# ---- SyntaxError is different: it happens BEFORE the program runs (at compile time) ------
# compile() turns source text into code and raises SyntaxError if the text is not valid Python.
bad_source = "if True print('hi')"
try:
    compile(bad_source, "<example>", "exec")
except SyntaxError as err:
    print("SyntaxError:", err.msg, "| line", err.lineno)
    print("    -> a SyntaxError means NOTHING in the file ran. Look at the line shown AND the line before it.")

# TRY IT: add your own case (e.g. [].pop()) to CASES and predict the message first.
# BREAK IT: run broken_syntax.py and notice that nothing is printed before the error.
