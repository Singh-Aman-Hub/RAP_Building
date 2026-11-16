"""Exercise A2 solution: statically scan a pickle for dangerous imports WITHOUT executing it (what pickle scanners do)."""
import pickle, pickletools

ALLOW = {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"), ("torch", "FloatStorage")}

def scan(blob):
    """Return the (module, name) pairs the pickle would import. Handles GLOBAL and STACK_GLOBAL."""
    found, strings = [], []
    for op, arg, _ in pickletools.genops(blob):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            strings.append(arg)
        elif op.name == "GLOBAL":                                   # protocol <=3: "module name" in one argument
            found.append(tuple(arg.split(" ", 1)))
        elif op.name == "STACK_GLOBAL":                             # protocol >=4: the two last strings pushed
            found.append((strings[-2], strings[-1]))
    return found

class Evil:
    def __reduce__(self):
        return (print, ("pwned",))                                  # harmless here; could be os.system

class Good:
    pass

bad = pickle.dumps(Evil(), protocol=4)
ok = pickle.dumps({"w": [1, 2, 3]}, protocol=4)
for label, blob in (("evil", bad), ("plain dict", ok)):
    imports = scan(blob)
    risky = [i for i in imports if i not in ALLOW]
    print(f"{label:10s} imports={imports} risky={risky}")
assert scan(bad) == [("builtins", "print")] and scan(ok) == []
print("scanner flags the evil pickle and passes the data-only one without ever calling pickle.loads: OK")
print("(a scanner is a heuristic; the real fix is a format with no code path: safetensors / weights_only=True)")
