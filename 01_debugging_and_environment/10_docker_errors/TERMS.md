# Terms — Debugging Docker Problems

Folder: `01_debugging_and_environment/10_docker_errors` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [01 Debugging and environment README](../README.md)

**Terms in this folder:** [Regular expression (re.search)](#regular-expression-research) · [re.IGNORECASE](#reignorecase) · [Regex alternation (a|b)](#regex-alternation-ab) · [sys.argv[1:] and the or default](#sysargv1-and-the-or-default) · [List of tuples as a lookup table](#list-of-tuples-as-a-lookup-table) · [Message, cause, check method](#message-cause-check-method) · [Port already in use](#port-already-in-use) · [docker ps -a](#docker-ps--a) · [docker logs](#docker-logs) · [docker exec](#docker-exec) · [docker inspect](#docker-inspect) · [docker build --progress=plain](#docker-build---progressplain) · [docker run --rm -p](#docker-run---rm--p) · [COPY failed / build context](#copy-failed--build-context) · [Dependencies missing in the image](#dependencies-missing-in-the-image) · [Binding to 0.0.0.0 vs 127.0.0.1](#binding-to-0000-vs-127001) · [Exec format error (architecture)](#exec-format-error-architecture) · [Permission denied](#permission-denied) · [Exited (1) and OOMKilled](#exited-1-and-oomkilled) · [Image pull errors](#image-pull-errors) · [Docker daemon not running](#docker-daemon-not-running) · [Environment variable not passed to the container](#environment-variable-not-passed-to-the-container) · [lsof -i](#lsof--i)

---

## Regular expression (re.search)  `(standard library)`
**1. Definition:** A regular expression (regex) is a pattern language for matching text; `re.search(pattern, text)` returns a match if the pattern occurs anywhere in `text`, otherwise `None`.
**2. In plain words:** The lookup table stores a pattern for each known Docker error message. `diagnose` tests each pattern against the message you give it and returns the cause of the first one that matches. Regexes are powerful but a malformed one (such as an unbalanced bracket) raises `re.error`.
**3. Example to quote:**
> "`re.search('address already in use', message)` matched my port error, so the script told me another process uses the host port."
```python
import re
print(bool(re.search(r"No module named", "No module named 'fastapi'")))   # True
```
**Seen in:** `main.py`, `diagnose`

## re.IGNORECASE  `(standard library)`
**1. Definition:** The flag `re.IGNORECASE` makes a regex match regardless of upper or lower case.
**2. In plain words:** Error messages vary in capitalisation between tools. With the flag, "Permission Denied" and "permission denied" both match.
**3. Example to quote:**
> "I pass `flags=re.IGNORECASE` so differences in capital letters don't stop a match."
```python
re.search("denied", "Permission DENIED", flags=re.IGNORECASE)
```
**Seen in:** `main.py`, `diagnose`

## Regex alternation (a|b)  `(syntax)`
**1. Definition:** In a regex, `a|b` matches either `a` or `b`.
**2. In plain words:** One entry can cover several wordings of the same problem: `address already in use|port is already allocated` catches both messages for a busy port.
**3. Example to quote:**
> "One pattern with `|` covers both 'address already in use' and 'port is already allocated'."
```python
re.search(r"cat|dog", "hotdog")
```
**Seen in:** `main.py`, `KNOWN_ERRORS`

## sys.argv[1:] and the or default  `(syntax)`
**1. Definition:** `sys.argv[1:]` is the list of command-line arguments after the script name, and `x or y` evaluates to `x` if it is truthy, otherwise `y`.
**2. In plain words:** `messages = sys.argv[1:] or SAMPLES` means "use the arguments you gave me; if there are none (empty list is falsy), use the built-in samples".
**3. Example to quote:**
> "`python main.py` with no arguments runs the samples, but `python main.py 'bind: address already in use'` diagnoses my own text."
```python
messages = sys.argv[1:] or ["default"]
```
**Seen in:** `main.py`, end of file

## List of tuples as a lookup table  `(concept)`
**1. Definition:** A list of `(pattern, cause, check)` tuples acts as a small rule table that code loops over, returning the first matching row.
**2. In plain words:** To support a new error you add one line of data, not new logic. The "TRY IT" task adds a tuple for "no space left on device".
**3. Example to quote:**
> "To add 'no space left on device' I only append one more tuple to `KNOWN_ERRORS`."
```python
KNOWN_ERRORS.append((r"no space left", "Disk full", "docker system df"))
```
**Seen in:** `main.py`, `KNOWN_ERRORS`

## Message, cause, check method  `(concept)`
**1. Definition:** A debugging pattern of reading the error message, naming its usual cause, and running the command that confirms it.
**2. In plain words:** This is the same mindset as module 01 steps 01 to 09: read, hypothesise, test. If the message is unknown, read the first error in `docker logs`, not the last.
**3. Example to quote:**
> "For 'address already in use' the usual cause is a busy host port, and my first check is `lsof -i :8000` or `docker ps`."
```text
message -> cause -> check
```
**Seen in:** `main.py`, docstring and `diagnose`

## Port already in use  `(concept)`
**1. Definition:** The error `bind: address already in use` or `port is already allocated` means the host port in `-p HOST:CONTAINER` is already used by another process or container.
**2. In plain words:** Two programs can't listen on the same host port. Stop the other one or map a different host port, e.g. `-p 8001:8000`.
**3. Example to quote:**
> "I got 'address already in use' because an old container was still running on host port 8000, so I stopped it."
```bash
docker run -p 8001:8000 image
```
**Seen in:** `main.py`, first rule and sample

## docker ps -a  `(command)`
**1. Definition:** `docker ps -a` lists all containers, running or stopped, with their status and exit codes.
**2. In plain words:** It answers "is it running, and if not, how did it end?". `Exited (1)` means it crashed at startup.
**3. Example to quote:**
> "My container disappeared, so I ran `docker ps -a` and saw `Exited (1)` 5 seconds after start."
```bash
docker ps -a
```
**Seen in:** `WALKTHROUGH.md`, six commands

## docker logs  `(command)`
**1. Definition:** `docker logs NAME` prints the output (stdout and stderr) of a container; `--tail 50` shows only the last 50 lines.
**2. In plain words:** For Python apps it is just an ordinary traceback, so everything from reading tracebacks applies. Make your app log to stdout/stderr so Docker can see it.
**3. Example to quote:**
> "`docker logs --tail 50 api` showed a `ModuleNotFoundError`, so I added the missing package to requirements."
```bash
docker logs --tail 50 mycontainer
```
**Seen in:** `WALKTHROUGH.md`; `main.py`, fallback advice

## docker exec  `(command)`
**1. Definition:** `docker exec -it NAME sh` opens an interactive shell inside a running container.
**2. In plain words:** You can look around with `ls`, `env`, `pwd` and `pip list` to compare the container's files, variables and packages with your laptop's.
**3. Example to quote:**
> "I ran `docker exec -it api sh` and `env | grep API` to see whether my variable reached the container."
```bash
docker exec -it api sh
```
**Seen in:** `WALKTHROUGH.md`

## docker inspect  `(command)`
**1. Definition:** `docker inspect NAME` prints detailed JSON about a container or image, including ports, environment variables, mounts and state; `--format` extracts one field.
**2. In plain words:** It reveals facts like `{{.State.OOMKilled}}`, which tells you if the container was killed for using too much memory.
**3. Example to quote:**
> "`docker inspect api --format '{{.State.OOMKilled}}'` returned true, so the model was too big for the memory limit."
```bash
docker inspect api --format '{{.State.OOMKilled}}'
```
**Seen in:** `main.py`, `Exited (1)` rule

## docker build --progress=plain  `(command)`
**1. Definition:** `docker build --progress=plain .` builds an image printing the full, unabridged output of every step.
**2. In plain words:** When a build fails you can see exactly which Dockerfile instruction broke and its complete error text.
**3. Example to quote:**
> "I rebuilt with `--progress=plain` to see that the `pip install` step was the one failing."
```bash
docker build --progress=plain .
```
**Seen in:** `WALKTHROUGH.md`

## docker run --rm -p  `(command)`
**1. Definition:** `docker run --rm -p 8000:8000 IMAGE` starts a container, mapping host port 8000 to container port 8000, and removes the container when it exits.
**2. In plain words:** A quick way to reproduce a problem. `--rm` keeps stopped containers from piling up. The mapping is `HOST:CONTAINER`.
**3. Example to quote:**
> "I reproduce issues quickly with `docker run --rm -p 8000:8000 image` and nothing is left behind afterwards."
```bash
docker run --rm -p 8000:8000 myimage
```
**Seen in:** `WALKTHROUGH.md`

## COPY failed / build context  `(concept)`
**1. Definition:** `COPY failed` or `not found in build context` means a file named in a Dockerfile `COPY` isn't inside the folder passed to `docker build` or is excluded by `.dockerignore`.
**2. In plain words:** Docker can only copy files from the build context. Check with `ls` from the folder you build in, and read `.dockerignore`.
**3. Example to quote:**
> "`COPY requirements.txt .` failed because I ran the build from a parent folder, outside the context."
```bash
ls
cat .dockerignore
```
**Seen in:** `main.py`, second rule

## Dependencies missing in the image  `(concept)`
**1. Definition:** `ModuleNotFoundError` inside a container means a required package wasn't installed in the image, even though it works on your laptop.
**2. In plain words:** The image only contains what the Dockerfile installs. Check `docker run --rm IMAGE pip list`, `requirements.txt` and the `RUN pip install` line.
**3. Example to quote:**
> "`No module named 'fastapi'` in the container meant my `requirements.txt` was missing fastapi."
```bash
docker run --rm myimage pip list
```
**Seen in:** `main.py`, third rule and samples

## Binding to 0.0.0.0 vs 127.0.0.1  `(concept)`
**1. Definition:** A server bound to `127.0.0.1` accepts connections only from inside its own network namespace, while `0.0.0.0` accepts connections on all interfaces.
**2. In plain words:** Inside a container, `127.0.0.1` is the container itself, so the host can't reach it, giving "Connection refused" or "Empty reply". Start the server with `--host 0.0.0.0`.
**3. Example to quote:**
> "`curl: (52) Empty reply from server` was because uvicorn listened on 127.0.0.1; adding `--host 0.0.0.0` fixed it."
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
**Seen in:** `main.py`, fourth rule and sample

## Exec format error (architecture)  `(concept)`
**1. Definition:** `exec format error` means the binary in the image was built for a different CPU architecture than the machine running it, e.g. arm64 (Apple Silicon) versus amd64.
**2. In plain words:** An image built on an M1 Mac may not run on an Intel server. Build with `--platform linux/amd64` and check with `docker image inspect IMAGE`.
**3. Example to quote:**
> "My M1-built image failed on the server with 'exec format error', so I rebuilt with `--platform linux/amd64`."
```bash
docker build --platform linux/amd64 -t myimage .
```
**Seen in:** `main.py`, fifth rule and sample

## Permission denied  `(concept)`
**1. Definition:** `permission denied` means the file isn't executable or the container's user lacks rights on a file or mounted volume.
**2. In plain words:** Check `ls -l`, the `USER` instruction in the Dockerfile and ownership of mounted volumes.
**3. Example to quote:**
> "A non-root user couldn't write to my mounted folder, so I checked `ls -l` and the volume's owner."
```bash
ls -l
```
**Seen in:** `main.py`, sixth rule

## Exited (1) and OOMKilled  `(concept)`
**1. Definition:** `Exited (1)` means the container's main process crashed at startup; `Killed` or `OOMKilled` means it was stopped for running out of memory.
**2. In plain words:** Read `docker logs` for the crash cause. For OOM, a large model may not fit in the memory available; `docker inspect` can confirm.
**3. Example to quote:**
> "The container showed `Killed`; `docker inspect` said OOMKilled true, so the model was too big for the memory limit."
```bash
docker logs mycontainer
```
**Seen in:** `main.py`, seventh rule; sample 5

## Image pull errors  `(concept)`
**1. Definition:** `pull access denied`, `manifest not found` or `repository does not exist` indicate a typo in the image name or tag, or a private image that needs login.
**2. In plain words:** Check the spelling and tag with `docker images`, and use `docker login` for private registries.
**3. Example to quote:**
> "I mistyped the tag, so Docker said 'manifest not found'; correcting `python:3.12-slim` fixed it."
```bash
docker login
```
**Seen in:** `main.py`, eighth rule

## Docker daemon not running  `(concept)`
**1. Definition:** `Cannot connect to the Docker daemon` means the Docker background service (Docker Desktop on macOS) isn't running.
**2. In plain words:** Start Docker Desktop (`open -a Docker`) and confirm with `docker info`.
**3. Example to quote:**
> "Docker commands failed with 'Cannot connect to the Docker daemon', so I started Docker Desktop and checked `docker info`."
```bash
docker info
```
**Seen in:** `main.py`, ninth rule

## Environment variable not passed to the container  `(concept)`
**1. Definition:** A variable set on your laptop isn't automatically available in a container; it must be passed with `-e`, `--env-file` or Compose `environment:`.
**2. In plain words:** A `KeyError: 'API_KEY'` in the container usually means this. Check with `docker exec CONTAINER env | grep NAME`.
**3. Example to quote:**
> "The app worked locally but crashed with `KeyError: 'API_KEY'` in Docker, since I hadn't passed `-e API_KEY`."
```bash
docker run -e API_KEY=... myimage
```
**Seen in:** `main.py`, tenth rule

## lsof -i  `(command)`
**1. Definition:** `lsof -i :PORT` lists the processes that have the given network port open.
**2. In plain words:** When a port is "already in use", this shows which process holds it so you can stop it or pick another port.
**3. Example to quote:**
> "`lsof -i :8000` showed an old uvicorn using port 8000."
```bash
lsof -i :8000
```
**Seen in:** `main.py`, first rule
