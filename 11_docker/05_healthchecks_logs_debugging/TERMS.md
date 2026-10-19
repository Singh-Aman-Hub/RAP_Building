# Terms — Logs, health checks and debugging broken containers

Folder: `11_docker/05_healthchecks_logs_debugging` · Files covered: `DIAGNOSIS.md`, `debug_cheatsheet.md`, `run_all_drills.sh`, `broken_wrong_port/`, `broken_missing_dependency/`, `broken_bad_cmd_path/`, `broken_exits_immediately/`, `broken_localhost_bind/` (each with `Dockerfile`, `app/main.py`, `requirements.txt`), `WALKTHROUGH.md`
Back to module: [11_docker README](../README.md)

**Terms in this folder:** [Container exit code (concept)](#container-exit-code-concept) · [docker ps -a (command)](#docker-ps--a-command) · [docker logs (command)](#docker-logs-command) · [docker exec (command)](#docker-exec-command) · [docker inspect (command)](#docker-inspect-command) · [Go template --format (command flag)](#go-template---format-command-flag) · [docker port (command)](#docker-port-command) · [HEALTHCHECK (Dockerfile instruction)](#healthcheck-dockerfile-instruction) · [urllib.request.urlopen (standard library)](#urllibrequesturlopen-standard-library) · [Health check vs process status (concept)](#health-check-vs-process-status-concept) · [Wrong port mapping (error pattern)](#wrong-port-mapping-error-pattern) · [--port 8080 (uvicorn flag)](#--port-8080-uvicorn-flag) · [ModuleNotFoundError (Python error)](#modulenotfounderror-python-error) · [Reading a traceback bottom-up (concept)](#reading-a-traceback-bottom-up-concept) · [python -m uvicorn (command)](#python--m-uvicorn-command) · [docker run --rm -it --entrypoint sh (command flags)](#docker-run---rm--it---entrypoint-sh-command-flags) · [Bad CMD module path (error pattern)](#bad-cmd-module-path-error-pattern) · [ASGI app import string (concept)](#asgi-app-import-string-concept) · [Container lifetime equals main process (concept)](#container-lifetime-equals-main-process-concept) · [python -c (command flag)](#python--c-command-flag) · [Binding to 127.0.0.1 vs 0.0.0.0 (concept)](#binding-to-127001-vs-0000-concept) · [uvicorn default host (concept)](#uvicorn-default-host-concept) · [EXPOSE vs publishing (concept)](#expose-vs-publishing-concept) · [curl -v / -m (command flags)](#curl--v---m-command-flags) · [docker build -q / docker run -d (command flags)](#docker-build--q--docker-run--d-command-flags) · [Shell for loop and parameter expansion (shell syntax)](#shell-for-loop-and-parameter-expansion-shell-syntax) · [docker stats (command)](#docker-stats-command) · [docker build --no-cache (command flag)](#docker-build---no-cache-command-flag) · [docker system df / prune (commands)](#docker-system-df--prune-commands) · [Systematic debugging order (concept)](#systematic-debugging-order-concept) · [<details> collapsible answers (Markdown/HTML)](#details-collapsible-answers-markdownhtml) · [FastAPI toy app (shared payload) (concept)](#fastapi-toy-app-shared-payload-concept) · [pip install --no-cache-dir -r requirements.txt (command)](#pip-install---no-cache-dir--r-requirementstxt-command)

---

## Container exit code `(concept)`
**1. Definition:** The integer status returned by a container's main process when it ends: 0 means it finished normally, 1 an application error, 125–127 a Docker or command problem, 137 a SIGKILL (often out of memory) and 139 a segmentation fault.
**2. In plain words:** It's the first clue when a container disappears. `Exited (0)` means nothing crashed but the program simply ended; `Exited (1)` means it hit an error, so go read the logs; `137` often means the system killed it for using too much memory.
**3. Example to quote:** > "`docker ps -a` showed `Exited (1)`, so I read the logs and found a `ModuleNotFoundError`; exit code 0 would have meant the process just finished."
**Seen in:** `DIAGNOSIS.md`, `WALKTHROUGH.md` syntax table

## docker ps -a `(command)`
**1. Definition:** Lists all containers including stopped ones, with their status and exit code.
**2. In plain words:** A container that crashed on start won't appear in plain `docker ps`. Adding `-a` shows it with how it ended, which is the first step of the diagnosis loop: is it alive at all?
**3. Example to quote:** > "My container wasn't in `docker ps`, but `docker ps -a` showed it as `Exited (1) 3 seconds ago`."
**Seen in:** `DIAGNOSIS.md`, `run_all_drills.sh`

## docker logs `(command)`
**1. Definition:** Prints the stdout and stderr captured from the container's main process; `-f` follows and `--tail N` limits output.
**2. In plain words:** Whatever your app prints, including Python tracebacks and uvicorn's "running on" line, lands here. Read tracebacks from the bottom: the last line is the error and the lines above say where.
**3. Example to quote:** > "`docker logs broken` ended with `ModuleNotFoundError: No module named 'uvicorn'`, which told me the image never installed its dependencies."
**Seen in:** `DIAGNOSIS.md`, `debug_cheatsheet.md`

## docker exec `(command)`
**1. Definition:** Runs a command inside an already running container, such as `docker exec -it NAME sh` for an interactive shell.
**2. In plain words:** It lets you look around inside the container: list files, print environment variables, test the app from the inside. Comparing what works from inside with what fails from outside often pinpoints a bind-address or port problem.
**3. Example to quote:** > "From inside the container `urlopen('http://127.0.0.1:8000/health')` worked, but from my laptop it failed; that contrast proved the 127.0.0.1 binding."
**Seen in:** `DIAGNOSIS.md`, `debug_cheatsheet.md`

## docker inspect `(command)`
**1. Definition:** Prints the full low-level configuration and state of a container or image as JSON, filterable with `--format` Go templates.
**2. In plain words:** It's the "everything about this container" command: ports, environment, mounts, exit code, health. The `--format '{{.State.ExitCode}}'` option picks out one field so you don't read pages of JSON.
**3. Example to quote:** > "I ran `docker inspect broken --format '{{.NetworkSettings.Ports}}'` and saw port 8000 was mapped, but the app listened on 8080."
**Seen in:** `DIAGNOSIS.md`, `WALKTHROUGH.md`

## Go template --format `(command flag)`
**1. Definition:** A `--format '{{ ... }}'` option that uses Go template syntax to select or print specific fields from a command's JSON output.
**2. In plain words:** Instead of scrolling through big JSON, you ask for just the piece you need, such as `{{.State.Health}}` or `{{json .State.Health}}` for a JSON rendering. Mind the quotes so the shell doesn't touch the braces.
**3. Example to quote:** > "`docker ps -a --format 'STATUS: {{.Status}}'` prints only the status column."
**Seen in:** `run_all_drills.sh`, `debug_cheatsheet.md`

## docker port `(command)`
**1. Definition:** Shows the published port mappings of a container.
**2. In plain words:** It answers "which host port goes to which container port?" in one line, which makes a wrong `-p` mapping obvious.
**3. Example to quote:** > "`docker port broken` showed `8000/tcp -> 0.0.0.0:8000`, but uvicorn was on 8080 inside, so the mapping pointed at nothing."
**Seen in:** `DIAGNOSIS.md`, `debug_cheatsheet.md`

## HEALTHCHECK `(Dockerfile instruction)`
**1. Definition:** Defines a command Docker runs periodically inside the container; exit 0 marks it healthy and exit 1 unhealthy, with `--interval`, `--timeout`, `--start-period` and `--retries` options.
**2. In plain words:** It checks the app, not just the process. A container can be "Up" while the web server inside is broken, and the health status reveals that. Plain Docker only reports unhealthy; it won't restart the container, but Compose's `depends_on` conditions and orchestrators act on it.
**3. Example to quote:** > "I added a HEALTHCHECK that fetches `/health`, and `docker ps` changed from Up to `(unhealthy)` when I made the endpoint return 500."
```dockerfile
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1
```
**Seen in:** `debug_cheatsheet.md`

## urllib.request.urlopen `(standard library)`
**1. Definition:** Opens a URL and returns a response object; it raises an exception for HTTP errors and connection failures.
**2. In plain words:** It's the built-in way to make a quick web request, used in the health check because `curl` isn't installed in `python:slim`. If the request fails, the exception makes the Python process exit non-zero, which Docker reads as unhealthy.
**3. Example to quote:** > "I use Python's own `urlopen` for the probe so I don't have to install curl in the image."
**Seen in:** `debug_cheatsheet.md`, `DIAGNOSIS.md`

## Health check vs process status `(concept)`
**1. Definition:** Process status says whether the main process is running; health status says whether the application is functioning, as judged by a probe.
**2. In plain words:** "Up" only means PID 1 hasn't exited. The server may be deadlocked or listening on the wrong port. Health checks measure what users actually care about.
**3. Example to quote:** > "My wrong-port container was Up but became unhealthy after the retries; health measures the app, not the process."
**Seen in:** `WALKTHROUGH.md`, experiment 1

## Wrong port mapping `(error pattern)`
**1. Definition:** A mismatch between the host-to-container mapping in `-p HOST:CONTAINER` and the port the application actually listens on inside the container.
**2. In plain words:** The container is running and healthy-looking, but curl gets "connection reset" or an empty reply because the right-hand number in `-p` points to a port where nothing listens. `EXPOSE` is only documentation and can't fix it.
**3. Example to quote:** > "The app listened on 8080 but I mapped 8000:8000; the fix was `-p 8000:8080`."
**Seen in:** `broken_wrong_port/Dockerfile`, `DIAGNOSIS.md` drill 1

## --port 8080 `(uvicorn flag)`
**1. Definition:** Sets the TCP port on which uvicorn listens inside the container.
**2. In plain words:** In `broken_wrong_port` this is the deliberate bug: the app uses 8080 while the learner maps 8000. The log line `Uvicorn running on http://0.0.0.0:8080` reveals the real port.
**3. Example to quote:** > "I compared the port in the uvicorn log line with the right-hand side of my `-p` flag and they didn't match."
**Seen in:** `broken_wrong_port/Dockerfile`

## ModuleNotFoundError `(Python error)`
**1. Definition:** The exception raised when an `import` statement names a module that isn't installed or can't be found on `sys.path`.
**2. In plain words:** In `broken_missing_dependency` the Dockerfile never runs `pip install`, so even `uvicorn` is missing. "Works on my machine" means your virtualenv had it but the image did not. An image contains only what its Dockerfile puts in.
**3. Example to quote:** > "The traceback ended with `No module named 'uvicorn'`, so I added `RUN pip install -r requirements.txt` before copying the code."
**Seen in:** `broken_missing_dependency/Dockerfile`, `DIAGNOSIS.md` drill 2

## Reading a traceback bottom-up `(concept)`
**1. Definition:** The practice of reading a Python traceback from its last line, which states the exception type and message, upward through the call chain.
**2. In plain words:** The last line tells you what went wrong; the lines above tell you where. Starting from the bottom gets you to the cause fastest.
**3. Example to quote:** > "I read the logs bottom-up: the last line named the error and the lines above pointed to the failing import."
**Seen in:** `DIAGNOSIS.md` drill 2

## python -m uvicorn `(command)`
**1. Definition:** Runs uvicorn as a module via the Python interpreter, equivalent to the `uvicorn` command but using the current interpreter's environment.
**2. In plain words:** It guarantees uvicorn is run by the same Python that has your packages. In `broken_missing_dependency` it still fails because uvicorn was never installed in the image.
**3. Example to quote:** > "`python -m uvicorn` uses that interpreter's installed packages; there were none, hence the error."
**Seen in:** `broken_missing_dependency/Dockerfile`

## docker run --rm -it --entrypoint sh `(command flags)`
**1. Definition:** Starts a throwaway interactive container that runs `sh` instead of the image's configured entrypoint; `--rm` deletes it on exit and `-it` attaches a terminal.
**2. In plain words:** When the normal command crashes the container immediately, you can't `exec` into it. Overriding the entrypoint gives you a shell in the same image so you can `ls` the filesystem or run `pip list`.
**3. Example to quote:** > "I ran `docker run --rm -it --entrypoint sh broken:missing_dependency` and `pip list` showed no uvicorn."
**Seen in:** `DIAGNOSIS.md`, `debug_cheatsheet.md`

## Bad CMD module path `(error pattern)`
**1. Definition:** A `CMD` whose import string, such as `src.main:app`, doesn't match the real module layout inside the image, so uvicorn can't import the application.
**2. In plain words:** The image has a folder named `app/`, but the command asks for `src.main`. Uvicorn exits with "Error loading ASGI app. Could not import module". The cure is to look at the filesystem inside the image, not the one on your laptop.
**3. Example to quote:** > "`docker run --rm broken:bad_cmd_path ls /code` showed `app`, so I changed `src.main:app` to `app.main:app`."
**Seen in:** `broken_bad_cmd_path/Dockerfile`, `DIAGNOSIS.md` drill 3

## ASGI app import string `(concept)`
**1. Definition:** The `module.path:variable` format that tells uvicorn which application object to import and serve.
**2. In plain words:** Left of the colon is the dotted module path, right of it is the variable name. If either is wrong, the server can't start and prints an import error.
**3. Example to quote:** > "`app.main:app` means module `app/main.py`, variable `app`."
**Seen in:** all five `Dockerfile`s

## Container lifetime equals main process `(concept)`
**1. Definition:** A container runs only as long as its main process (PID 1); when that process exits, the container stops.
**2. In plain words:** In `broken_exits_immediately` the command just prints a line and finishes, so the container ends with exit code 0 within a second. A server must run in the foreground, never backgrounded with `&` or a daemon mode.
**3. Example to quote:** > "Exit code 0 told me nothing crashed; my CMD simply printed 'starting up...' and ended, so I replaced it with uvicorn."
**Seen in:** `broken_exits_immediately/Dockerfile`, `DIAGNOSIS.md` drill 4

## python -c `(command flag)`
**1. Definition:** Runs the string that follows as a short Python program and then exits.
**2. In plain words:** It's handy for one-liners such as health probes and quick checks. As a container `CMD` it ends as soon as the code finishes, which is the bug in the "exits immediately" example.
**3. Example to quote:** > "`python -c "print('starting up...')"` ends instantly, so it can't keep a container alive."
**Seen in:** `broken_exits_immediately/Dockerfile`

## Binding to 127.0.0.1 vs 0.0.0.0 `(concept)`
**1. Definition:** 127.0.0.1 is the loopback interface reachable only from the same network namespace; 0.0.0.0 means listen on all interfaces.
**2. In plain words:** Inside a container, a server bound to loopback only accepts connections from itself. Docker's port forwarding arrives on the container's network interface, so it never reaches the server. The logs look perfect, but curl fails.
**3. Example to quote:** > "Logs said `Uvicorn running on http://127.0.0.1:8000` and curl failed; adding `--host 0.0.0.0` fixed it."
**Seen in:** `broken_localhost_bind/Dockerfile`, `DIAGNOSIS.md` drill 5

## uvicorn default host `(concept)`
**1. Definition:** If `--host` is omitted uvicorn binds to `127.0.0.1`.
**2. In plain words:** That default is sensible on a laptop but wrong in a container. The same trap exists elsewhere: Flask `app.run(host="0.0.0.0")`, gunicorn `-b 0.0.0.0:8000` and Streamlit `--server.address=0.0.0.0`.
**3. Example to quote:** > "Every framework I containerise gets an explicit 0.0.0.0 bind: uvicorn `--host`, gunicorn `-b`, Streamlit `--server.address`."
**Seen in:** `broken_localhost_bind/Dockerfile`

## EXPOSE vs publishing `(concept)`
**1. Definition:** `EXPOSE` documents a port in image metadata but doesn't publish it; publishing happens with `-p` or Compose `ports:`.
**2. In plain words:** All five broken Dockerfiles contain `EXPOSE 8000`, yet some are unreachable. This is why EXPOSE alone never fixes a connection problem.
**3. Example to quote:** > "I had `EXPOSE 8000` in the Dockerfile but still got no response, since I never published the port with `-p`."
**Seen in:** all five `Dockerfile`s, `DIAGNOSIS.md` drill 1

## curl -v / -m `(command flags)`
**1. Definition:** `-v` prints verbose connection details and headers; `-m SECONDS` sets a maximum time for the whole request.
**2. In plain words:** `-v` shows whether the connection was reset or refused, which helps tell a wrong port from a wrong bind. `-m 3` stops curl from hanging forever when the container is broken.
**3. Example to quote:** > "`curl -v localhost:8000/health` showed 'Connection reset by peer', so I knew something accepted the connection but nothing answered."
**Seen in:** `DIAGNOSIS.md`, `run_all_drills.sh`

## docker build -q / docker run -d `(command flags)`
**1. Definition:** `-q` makes `docker build` quiet, printing only the image ID; `-d` runs a container detached in the background.
**2. In plain words:** The drill script uses them to build and start each broken image without flooding the terminal, then inspects the result with `ps`, `logs` and `curl`.
**3. Example to quote:** > "The script builds quietly with `-q`, runs detached with `-d`, sleeps three seconds and then prints status, logs and curl output."
**Seen in:** `run_all_drills.sh`

## Shell for loop and parameter expansion `(shell syntax)`
**1. Definition:** `for d in broken_*/; do ... done` iterates over matching directories; `${d%/}` strips the trailing slash from a variable.
**2. In plain words:** It lets one script run the same steps for every broken example. `set -u` makes the script fail if it uses an unset variable, catching typos.
**3. Example to quote:** > "`name=${d%/}` turns `broken_wrong_port/` into `broken_wrong_port` for the image tag."
**Seen in:** `run_all_drills.sh`

## docker stats `(command)`
**1. Definition:** Shows live CPU, memory and network usage of running containers.
**2. In plain words:** It reveals whether a container is close to its memory limit, which would explain an exit code 137.
**3. Example to quote:** > "I ran `docker stats` and saw the container sitting at its memory cap right before it was killed."
**Seen in:** `debug_cheatsheet.md`

## docker build --no-cache `(command flag)`
**1. Definition:** Forces Docker to ignore the layer cache and execute every instruction.
**2. In plain words:** Use it to rule out a stale cached layer when a build behaves strangely. It makes the build slower.
**3. Example to quote:** > "I rebuilt with `--no-cache` to make sure an old layer wasn't hiding my fix."
**Seen in:** `debug_cheatsheet.md`

## docker system df / prune `(commands)`
**1. Definition:** `docker system df` reports disk used by images, containers and volumes; `docker system prune` deletes stopped containers, unused networks and dangling images.
**2. In plain words:** Images and containers accumulate quickly. Prune frees space, but it deletes things, so read what it will remove first.
**3. Example to quote:** > "My disk filled up from old builds; `docker system df` showed where, and prune cleaned up the stopped containers."
**Seen in:** `debug_cheatsheet.md`

## Systematic debugging order `(concept)`
**1. Definition:** A fixed checklist: is it running (exit code), read the logs, is the app listening where expected, is the port mapped as expected, is the config right, then reproduce minimally changing one thing at a time.
**2. In plain words:** It's the same discipline as debugging Python code: observe before changing anything, and change one variable at a time so you know what fixed it.
**3. Example to quote:** > "I wrote each drill up as: symptom, command that revealed it, root cause, fix, verification."
**Seen in:** `debug_cheatsheet.md`, `WALKTHROUGH.md`

## `<details>` collapsible answers `(Markdown/HTML)`
**1. Definition:** An HTML element, usable inside Markdown, that hides content until the reader expands it.
**2. In plain words:** It hides each drill's answer so you attempt the diagnosis first and only then peek.
**3. Example to quote:** > "I wrote my own diagnosis for drill 3 before opening its `<details>` answer."
**Seen in:** `DIAGNOSIS.md`

## FastAPI toy app (shared payload) `(concept)`
**1. Definition:** The same small FastAPI app copied into every broken example, so only the Dockerfile differs between drills.
**2. In plain words:** Because the app is identical and correct, any failure must come from the Dockerfile or the run command, which keeps each drill focused.
**3. Example to quote:** > "Each broken folder has the same working `app/main.py`; the bug is always in the Dockerfile."
**Seen in:** `broken_*/app/main.py`

## pip install --no-cache-dir -r requirements.txt `(command)`
**1. Definition:** Installs pinned packages while not keeping pip's download cache.
**2. In plain words:** It appears in four of the five Dockerfiles; its absence in `broken_missing_dependency` is the bug.
**3. Example to quote:** > "The missing `RUN pip install` line is the whole bug of the missing-dependency drill."
**Seen in:** `broken_*/Dockerfile`
