"""Solution for broken_http.py: separate handling for Timeout, ConnectionError and HTTPError, one retry."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "06_http_and_network"))
from main import start_server      # works because main.py has an `if __name__ == "__main__"` guard
import requests

start_server(8767)
url = "http://127.0.0.1:8767/slow"

for attempt, timeout in enumerate([0.5, 5], start=1):     # second try is patient
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        print(f"attempt {attempt}: ok ->", resp.json()["status"])
        break
    except requests.Timeout:
        print(f"attempt {attempt}: server slower than {timeout}s")
    except requests.ConnectionError:
        print("server unreachable: check host/port"); break
    except requests.HTTPError as err:
        print("server answered with an error:", err); break
