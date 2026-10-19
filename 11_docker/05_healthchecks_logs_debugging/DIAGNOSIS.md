# Debug drills: diagnose first, THEN read the answer

Rules: no AI. Use only `docker ps -a`, `docker logs`, `docker exec`, `docker inspect`, and your eyes.
For each folder: build, run with the command shown, observe the symptom, write down (1) the symptom, (2) which
command revealed the cause, (3) root cause, (4) the fix, (5) how you verified it. Compare with the answers below.

```bash
cd broken_<name>
docker build -t broken:<name> .
docker run -d --name broken -p 8000:8000 broken:<name>
docker ps -a          # is it Up or Exited? what exit code?
docker logs broken    # what did the process print before dying (or while running)?
curl -v localhost:8000/health
docker rm -f broken   # always clean up before the next drill
```

## Drill 1 - `broken_wrong_port`
Symptom: container is Up, `curl localhost:8000` -> "Connection reset by peer" / empty reply.
<details><summary>Answer</summary>

`docker logs broken` says `Uvicorn running on http://0.0.0.0:8080`. The app listens on **8080** but you mapped
host 8000 -> container **8000**, where nothing listens. `docker port broken` shows the mapping, and
`docker inspect broken --format '{{.NetworkSettings.Ports}}'` too.
Fix: `-p 8000:8080` (or change `--port` to 8000). Verify: curl returns `{"status":"ok"}`.
Lesson: `-p HOST:CONTAINER` - the right-hand number must be the port the *app* listens on. `EXPOSE` is only documentation.
</details>

## Drill 2 - `broken_missing_dependency`
Symptom: `docker ps` shows nothing; `docker ps -a` shows `Exited (1)`.
<details><summary>Answer</summary>

`docker logs broken` ends with `ModuleNotFoundError: No module named 'uvicorn'` (read the traceback bottom-up:
last line = error, lines above = where). Dockerfile never ran `pip install`. Fix: add `COPY requirements.txt .` and
`RUN pip install -r requirements.txt` before copying code. Verify: container stays Up.
Lesson: "works on my machine" = your venv had it, the image did not. Images contain only what the Dockerfile puts in.
Debug trick: `docker run --rm -it --entrypoint sh broken:missing_dependency` then `pip list`.
</details>

## Drill 3 - `broken_bad_cmd_path`
Symptom: Exited (1) immediately.
<details><summary>Answer</summary>

Logs: `ERROR: Error loading ASGI app. Could not import module "src.main".` The folder in the image is `app/`
(`docker run --rm broken:bad_cmd_path ls /code` proves it). Fix: `app.main:app`. Format is `module.path:variable`.
Lesson: check the filesystem *inside* the image with `ls`, not the one on your laptop.
</details>

## Drill 4 - `broken_exits_immediately`
Symptom: `docker ps -a` -> `Exited (0)` seconds after start; logs only say "starting up...".
<details><summary>Answer</summary>

Exit code 0 = no crash, the process simply finished. A container lives as long as its main process (PID 1).
Fix: CMD must start a long-running server (`uvicorn ...`) in the foreground (never `&` or daemon mode).
Lesson: exit code 0 vs non-zero: 0 = finished normally, 1 = error, 137 = killed (often out of memory), 139 = segfault.
</details>

## Drill 5 - `broken_localhost_bind`
Symptom: container Up, logs look perfect (`Uvicorn running on http://127.0.0.1:8000`), curl fails.
<details><summary>Answer</summary>

127.0.0.1 inside a container is the container's *own* loopback; Docker's port forwarding arrives on the container's
network interface, not loopback. Fix: `--host 0.0.0.0`. Same applies to Flask (`app.run(host="0.0.0.0")`),
gunicorn (`-b 0.0.0.0:8000`) and Streamlit (`--server.address=0.0.0.0`).
Verify from inside: `docker exec broken python -c "import urllib.request as u;print(u.urlopen('http://127.0.0.1:8000/health').read())"` works
but from outside does not: that contrast is the proof.
</details>
