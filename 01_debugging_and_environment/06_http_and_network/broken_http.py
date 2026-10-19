"""
EXPECTED TO CRASH. A client that forgot error handling.  Run it, read the traceback bottom-up, then
add specific except branches for Timeout and HTTPError and print a helpful message for each.
"""
from main import start_server           # reuse the local fake API from main.py
import requests

start_server(8766)
resp = requests.get("http://127.0.0.1:8766/slow", timeout=0.5)   # server needs 2 s
print(resp.json()["status"])
