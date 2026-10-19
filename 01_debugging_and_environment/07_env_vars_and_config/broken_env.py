"""EXPECTED TO CRASH. Two config bugs. Fix with .get() + defaults and int() with a clear error (see main.py)."""
import os

api_key = os.environ["SERVICE_API_KEY"]      # KeyError if the variable is not set
print("key length:", len(api_key))

os.environ["PORT"] = "8000"                  # pretend the shell exported this
port = os.environ["PORT"]
print("next port:", port + 1)                # a string, not an int!
