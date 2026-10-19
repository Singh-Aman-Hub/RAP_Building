"""Solution for broken_env.py: required variable with a clear message; int conversion."""
import os

api_key = os.environ.get("SERVICE_API_KEY")
if not api_key:
    print("SERVICE_API_KEY is not set -> using demo mode")   # or raise SystemExit(...) if it is required
else:
    print("key length:", len(api_key))

os.environ["PORT"] = "8000"
port = int(os.environ["PORT"])
print("next port:", port + 1)
assert port + 1 == 8001
