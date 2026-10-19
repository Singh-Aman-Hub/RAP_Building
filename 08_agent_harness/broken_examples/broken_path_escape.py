"""
DRILL 5 -- DOES NOT CRASH, BUT IS INSECURE.   Bug: file tool trusts the model-supplied path.
RUN: python broken_path_escape.py    Observe that a file OUTSIDE the sandbox folder is read successfully.

FIX: resolve the path and check it is inside the sandbox (step 07 safe_path).  Test absolute paths too.
"""
from pathlib import Path

SANDBOX = Path(__file__).resolve().parents[1] / "sandbox_files"

def read_file(path):
    return (SANDBOX / path).read_text()          # BUG: '..' segments (and absolute paths) escape the sandbox

attack_path = "../fake_llm.py"                    # a path the model could be tricked into asking for
print("first lines of a file OUTSIDE the sandbox:")
print("\n".join(read_file(attack_path).splitlines()[:3]))
