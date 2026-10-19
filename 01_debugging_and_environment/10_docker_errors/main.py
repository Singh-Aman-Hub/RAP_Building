"""
GOAL:  A tiny "error lookup" that maps common Docker error messages to their usual cause and first command to run.
       (Docker itself is taught in module 11; this step is about the DEBUGGING MINDSET: message -> cause -> check.)

RUN:   cd 01_debugging_and_environment/10_docker_errors && python main.py
       or with your own text:  python main.py "bind: address already in use"

EXPECTED OUTPUT: for each sample message, the diagnosis and the command to confirm it.

NEEDS: nothing (standard library only). Docker is NOT required to run this script.
"""

import re
import sys

# (regex on the error text, usual cause, first check)
KNOWN_ERRORS = [
    (r"address already in use|port is already allocated",
     "Another process/container already uses the HOST port in -p HOST:CONTAINER.",
     "lsof -i :8000   |   docker ps   (then stop the other one or change the host port)"),
    (r"No such file or directory.*requirements|COPY failed|not found in build context",
     "COPY path is wrong or outside the build context / listed in .dockerignore.",
     "ls  from the folder you pass to `docker build`; check .dockerignore"),
    (r"ModuleNotFoundError|No module named",
     "Dependency not installed in the IMAGE (works on your laptop, not in the container).",
     "docker run --rm IMAGE pip list ; check requirements.txt + the RUN pip install line"),
    (r"Connection refused|connection reset|ERR_EMPTY_RESPONSE|Empty reply",
     "App listens on 127.0.0.1 inside the container, so the host cannot reach it.",
     "Start server with --host 0.0.0.0 ; confirm with: docker logs CONTAINER"),
    (r"exec format error",
     "Image built for a different CPU architecture (e.g. arm64 Mac -> amd64 server).",
     "docker build --platform linux/amd64 ... ; docker image inspect IMAGE | grep Architecture"),
    (r"permission denied",
     "File not executable or container user lacks rights on a mounted volume.",
     "ls -l ; check USER in Dockerfile and volume ownership"),
    (r"Exited \(1\)|exit code 1|OOMKilled|Killed",
     "Process crashed at startup (Exited 1) or ran out of memory (Killed/OOMKilled, e.g. big model).",
     "docker logs CONTAINER ; docker inspect CONTAINER --format '{{.State.OOMKilled}}'"),
    (r"pull access denied|manifest.*not found|repository does not exist",
     "Image name/tag typo, or the image is private.",
     "docker images ; check spelling and tag; docker login"),
    (r"Cannot connect to the Docker daemon",
     "Docker Desktop/daemon is not running.",
     "open -a Docker  (macOS) ; docker info"),
    (r"KeyError: '?[A-Z_]+'?|environment variable .* not set",
     "Env var exists on your laptop but was not passed to the container (-e / --env-file / compose environment:).",
     "docker exec CONTAINER env | grep NAME"),
]


def diagnose(message):
    for pattern, cause, check in KNOWN_ERRORS:
        if re.search(pattern, message, flags=re.IGNORECASE):
            return cause, check
    return "Unknown. Read the FIRST error in `docker logs`, not the last.", "docker logs --tail 50 CONTAINER"


SAMPLES = [
    "docker: Error response from daemon: Ports are not available: listen tcp 0.0.0.0:8000: bind: address already in use",
    "ModuleNotFoundError: No module named 'fastapi'",
    "curl: (52) Empty reply from server",
    "exec /usr/local/bin/python: exec format error",
    "container exited with Killed",
]

if __name__ == "__main__":
    messages = sys.argv[1:] or SAMPLES      # argv[0] is the script name; the rest are user arguments
    for m in messages:
        cause, check = diagnose(m)
        print(f"MESSAGE : {m}\nCAUSE   : {cause}\nCHECK   : {check}\n")

# TRY IT: add a new (regex, cause, check) tuple for "no space left on device".
# BREAK IT: write a regex with an unbalanced "(" and read the re.error traceback.
