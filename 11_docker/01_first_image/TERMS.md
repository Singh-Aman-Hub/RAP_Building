# Terms — First image and container

Folder: `11_docker/01_first_image` · Files covered: `Dockerfile`, `app/main.py`, `check_locally.py`, `commands.sh`, `requirements.txt`, `.dockerignore`, `WALKTHROUGH.md`
Back to module: [11_docker README](../README.md)

**Terms in this folder:** [Docker image (concept)](#docker-image-concept) · [Docker container (concept)](#docker-container-concept) · [Containers vs virtual machines (concept)](#containers-vs-virtual-machines-concept) · [Dockerfile (concept)](#dockerfile-concept) · [FROM (Dockerfile instruction)](#from-dockerfile-instruction) · [python:3.12-slim (base image)](#python312-slim-base-image) · [WORKDIR (Dockerfile instruction)](#workdir-dockerfile-instruction) · [COPY (Dockerfile instruction)](#copy-dockerfile-instruction) · [RUN (Dockerfile instruction)](#run-dockerfile-instruction) · [pip install --no-cache-dir (command flag)](#pip-install---no-cache-dir-command-flag) · [requirements.txt (file)](#requirementstxt-file) · [EXPOSE (Dockerfile instruction)](#expose-dockerfile-instruction) · [CMD (Dockerfile instruction)](#cmd-dockerfile-instruction) · [--host 0.0.0.0 (uvicorn flag)](#--host-0000-uvicorn-flag) · [app.main:app (uvicorn import string)](#appmainapp-uvicorn-import-string) · [docker build (command)](#docker-build-command) · [Build context (concept)](#build-context-concept) · [.dockerignore (file)](#dockerignore-file) · [docker run (command)](#docker-run-command) · [Port mapping (-p HOST:CONTAINER) (docker run flag)](#port-mapping--p-hostcontainer-docker-run-flag) · [docker ps (command)](#docker-ps-command) · [docker logs (command)](#docker-logs-command) · [docker stop and docker rm (commands)](#docker-stop-and-docker-rm-commands) · [FastAPI (third-party: fastapi)](#fastapi-third-party-fastapi) · [BaseModel (third-party: pydantic)](#basemodel-third-party-pydantic) · [Decorator @app.get / @app.post (syntax)](#decorator-appget--apppost-syntax) · [Generator expression with sum() (syntax)](#generator-expression-with-sum-syntax) · [str.lower() and str.split() (Python built-in)](#strlower-and-strsplit-python-built-in) · [TestClient (third-party: starlette via fastapi)](#testclient-third-party-starlette-via-fastapi) · [HTTP 422 Unprocessable Entity (concept)](#http-422-unprocessable-entity-concept) · [curl (command)](#curl-command)

---

## Docker image `(concept)`
**1. Definition:** A read-only, layered template containing a filesystem snapshot plus metadata (default command, environment, exposed ports) from which containers are created.
**2. In plain words:** Think of an image as a frozen, ready-to-run copy of your app together with everything it needs: Python, the libraries and your code. You build it once from a Dockerfile, and it never changes afterwards. If you edit your code, the running container does not notice, because the image is a snapshot; you must build a new image.
**3. Example to quote:** > "I built the image `toy-api:v1` once. Editing `POSITIVE` in `main.py` did nothing to a container started from it, because an image is a snapshot; I had to rebuild to get the new behaviour."
**Seen in:** `WALKTHROUGH.md`, experiment 4

## Docker container `(concept)`
**1. Definition:** A running (or stopped) instance of an image: an isolated process with its own filesystem view, network namespace and process tree, sharing the host kernel.
**2. In plain words:** If the image is the recipe card, the container is the dish being cooked. You can start many containers from one image, and each has its own writable layer. When you delete a container, anything it wrote inside itself disappears unless you stored it in a volume.
**3. Example to quote:** > "I ran two containers from the same image on host ports 8000 and 9000. One image, many independent containers."
**Seen in:** `commands.sh`, `docker run`

## Containers vs virtual machines `(concept)`
**1. Definition:** Containers virtualise at the operating-system level and share the host kernel, while virtual machines virtualise hardware and each run a full guest operating system.
**2. In plain words:** A VM is like renting a whole separate house with its own plumbing; a container is like a locked room in a shared house. Containers start in seconds and use far less memory, but they are less strongly isolated than VMs because they share the kernel. That is why a Linux container on a Mac actually runs inside a small hidden Linux VM.
**3. Example to quote:** > "I chose a container over a VM because my API only needs Python and a few libraries; I don't need a whole guest OS, and it starts in about a second."
**Seen in:** `WALKTHROUGH.md`, "Say it aloud"

## Dockerfile `(concept)`
**1. Definition:** A text file of ordered instructions (`FROM`, `COPY`, `RUN`, `CMD`, ...) that `docker build` executes to produce an image.
**2. In plain words:** It is the written recipe for your image. Each line does one step, and Docker runs them top to bottom. Because the recipe lives in a file, anyone can rebuild exactly the same environment, which is the whole point of using Docker.
**3. Example to quote:** > "My Dockerfile is seven lines: base image, working directory, copy requirements, pip install, copy code, expose, and the start command."
**Seen in:** `Dockerfile`

## FROM `(Dockerfile instruction)`
**1. Definition:** Sets the base image on which all following instructions build; every Dockerfile begins with it.
**2. In plain words:** You rarely start from nothing. `FROM python:3.12-slim` says "start from a Debian system that already has Python 3.12". The part after the colon is the tag, i.e. the version of that base image. Pinning a tag keeps your build repeatable.
**3. Example to quote:** > "I used `FROM python:3.12-slim` so I get Python without hundreds of megabytes of compilers I don't need."
**Seen in:** `Dockerfile`, first line

## python:3.12-slim `(base image)`
**1. Definition:** An official Docker Hub image of Debian with Python 3.12 installed and without most extra build tools, around 120 MB.
**2. In plain words:** It is the middle choice between the huge full `python` image and the tiny `alpine` one. It is small enough to pull quickly and uses the standard glibc library, so most Python packages install from ready-made wheels.
**3. Example to quote:** > "For ML code I use `-slim`, not alpine, because alpine's musl libc often has no prebuilt wheels for numpy or torch."
**Seen in:** `Dockerfile`

## WORKDIR `(Dockerfile instruction)`
**1. Definition:** Creates a directory if needed and sets it as the current directory for subsequent `COPY`, `RUN` and `CMD` instructions and for the running container.
**2. In plain words:** It is `mkdir` plus `cd` in one line. After `WORKDIR /code`, a `COPY requirements.txt .` puts the file in `/code`, and the app starts from there. It saves you from writing long absolute paths.
**3. Example to quote:** > "`WORKDIR /code` means my `COPY app ./app` lands in `/code/app`, and uvicorn finds `app.main` because it starts in `/code`."
**Seen in:** `Dockerfile`

## COPY `(Dockerfile instruction)`
**1. Definition:** Copies files or directories from the build context on the host into the image's filesystem, creating a new layer.
**2. In plain words:** The container cannot see your laptop's folders, so you must copy your files into the image explicitly. The source is relative to the build context (the folder you pass to `docker build`), and the destination is inside the image.
**3. Example to quote:** > "I copy `requirements.txt` first and the `app` folder later, so the dependency layer stays cached when only my code changes."
```dockerfile
COPY requirements.txt .
COPY app ./app
```
**Seen in:** `Dockerfile`

## RUN `(Dockerfile instruction)`
**1. Definition:** Executes a shell command at build time and stores the resulting filesystem changes as a new image layer.
**2. In plain words:** `RUN` is for preparing the image, such as installing packages. It runs while building, not when the container starts. Don't confuse it with `CMD`, which runs when you start a container.
**3. Example to quote:** > "`RUN pip install` happens once during the build, so the packages are baked into the image and every container starts instantly."
**Seen in:** `Dockerfile`, pip install line

## pip install --no-cache-dir `(command flag)`
**1. Definition:** `pip install -r requirements.txt` installs the listed packages; `--no-cache-dir` tells pip not to store downloaded archives in its cache directory.
**2. In plain words:** Pip normally keeps a copy of everything it downloads in case you reinstall. Inside an image that cache is dead weight, because you never reinstall in the same layer. Turning it off makes the image smaller.
**3. Example to quote:** > "I add `--no-cache-dir` so pip's download cache doesn't get baked into the image and inflate its size."
**Seen in:** `Dockerfile`

## requirements.txt `(file)`
**1. Definition:** A plain-text list of Python packages, optionally pinned to versions, that `pip install -r` reads.
**2. In plain words:** It is the shopping list for your Python environment. Pinning versions such as `fastapi==0.146.0` means a build next month installs exactly the same thing, so the app doesn't break because a library changed.
**3. Example to quote:** > "I pinned `fastapi==0.146.0` and `uvicorn[standard]==0.54.0` so the image is reproducible."
**Seen in:** `requirements.txt`

## EXPOSE `(Dockerfile instruction)`
**1. Definition:** Documents which port the containerised application listens on; it does not publish or open that port.
**2. In plain words:** `EXPOSE 8000` is a note for humans and tools saying "this app talks on 8000". It does not make the port reachable from your laptop. You still need `docker run -p` to publish it. This misunderstanding is a very common interview trap.
**3. Example to quote:** > "I wrote `EXPOSE 8000`, but curl still failed until I added `-p 8000:8000`, because EXPOSE is only documentation."
**Seen in:** `Dockerfile`

## CMD `(Dockerfile instruction)`
**1. Definition:** Defines the default command executed when a container starts; the JSON-list (exec) form runs the program directly without a shell.
**2. In plain words:** After the image is built, `CMD` tells Docker what to launch. In exec form (`["uvicorn", ...]`) the server is the main process, so `docker stop` signals reach it and it can shut down cleanly. Anything you pass after the image name in `docker run` replaces it.
**3. Example to quote:** > "My `CMD` starts uvicorn in exec form, so uvicorn is PID 1 and shuts down gracefully when I run `docker stop`."
```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
**Seen in:** `Dockerfile`, last line

## --host 0.0.0.0 `(uvicorn flag)`
**1. Definition:** Makes uvicorn bind to all network interfaces rather than only the loopback interface `127.0.0.1`.
**2. In plain words:** Inside a container, `127.0.0.1` means "this container only". If the server listens only there, the port mapping from your laptop can never reach it. Binding to `0.0.0.0` means "accept connections from anywhere", which is what you want inside a container.
**3. Example to quote:** > "My container started fine but curl got connection reset; the fix was `--host 0.0.0.0` because the default 127.0.0.1 is unreachable from outside the container."
**Seen in:** `Dockerfile`, `CMD`

## app.main:app `(uvicorn import string)`
**1. Definition:** A module path `package.module:variable` telling uvicorn which ASGI application object to serve.
**2. In plain words:** The part before the colon is the Python module (`app/main.py`), and the part after is the variable inside it (`app = FastAPI(...)`). If either is wrong, uvicorn fails at start with an import error.
**3. Example to quote:** > "`app.main:app` means: in the `app` package, module `main`, find the object called `app`."
**Seen in:** `Dockerfile`, `main.py` docstring

## docker build `(command)`
**1. Definition:** Builds an image from a Dockerfile and a build context, with `-t name:tag` to name the result and `-f` to choose a non-default Dockerfile.
**2. In plain words:** This is the command that turns your recipe into an image. The final `.` is the build context, the folder Docker packages and sends to the engine. If a file is outside that folder, `COPY` cannot see it.
**3. Example to quote:** > "I ran `docker build -t toy-api:v1 .` and got an image named `toy-api` with the tag `v1`."
```bash
docker build -t toy-api:v1 .
```
**Seen in:** `commands.sh`

## Build context `(concept)`
**1. Definition:** The set of files at the path given to `docker build` (minus anything in `.dockerignore`) that is sent to the Docker engine and is the only source `COPY` can read.
**2. In plain words:** Docker can't reach into random places on your computer; it only sees the folder you hand it. That is why the trailing `.` in the build command matters, and why a "file not found" during `COPY` usually means the file isn't in that folder.
**3. Example to quote:** > "The trailing dot in `docker build -t x .` is the build context; my `COPY requirements.txt .` works because that file sits in the same folder."
**Seen in:** `WALKTHROUGH.md`, execution trace step 1

## .dockerignore `(file)`
**1. Definition:** A file listing patterns of paths excluded from the build context.
**2. In plain words:** It works like `.gitignore` for Docker. Excluding `.venv`, `.git`, `__pycache__` and `.env` makes the build context smaller and faster, and keeps secrets such as `.env` out of the image.
**3. Example to quote:** > "I list `.venv` and `.env` in `.dockerignore` so my 500 MB virtualenv and my secrets never get copied into the image."
**Seen in:** `.dockerignore`

## docker run `(command)`
**1. Definition:** Creates and starts a container from an image; common flags are `-d` (detached), `-p HOST:CONTAINER` (publish a port), `--name`, `-e` and `-v`.
**2. In plain words:** This turns an image into a running process. `-d` frees your terminal, `--name` gives it a memorable handle, and `-p` connects a port on your laptop to a port inside the container.
**3. Example to quote:** > "`docker run -d --name toy -p 8000:8000 toy-api:v1` starts my API in the background and makes it reachable at localhost:8000."
```bash
docker run -d --name toy -p 8000:8000 toy-api:v1
```
**Seen in:** `commands.sh`

## Port mapping (-p HOST:CONTAINER) `(docker run flag)`
**1. Definition:** Publishes a container port on a host port so that traffic to the host port is forwarded to the container.
**2. In plain words:** The container has its own private network. `-p 9000:8000` means "when someone visits port 9000 on my laptop, send them to port 8000 inside the container". The left number is the host side, the right is the container side; swapping them is a classic mistake.
**3. Example to quote:** > "With `-p 9000:8000` my API answers at localhost:9000 while uvicorn still listens on 8000 inside."
**Seen in:** `commands.sh`, `WALKTHROUGH.md` experiment 2

## docker ps `(command)`
**1. Definition:** Lists running containers; `-a` also lists stopped ones.
**2. In plain words:** It is your "what is running right now" check. If a container isn't in the list, add `-a` to see if it started and then exited, which usually means an error you can read with `docker logs`.
**3. Example to quote:** > "My container wasn't in `docker ps`, but `docker ps -a` showed it as Exited, so I ran `docker logs` to see why."
**Seen in:** `commands.sh`

## docker logs `(command)`
**1. Definition:** Prints the stdout and stderr captured from a container's main process; `-f` follows new output and `--tail N` limits lines.
**2. In plain words:** Anything your app prints ends up here, including Python tracebacks. It is the first thing to check when a container misbehaves.
**3. Example to quote:** > "I ran `docker logs toy` and saw uvicorn's startup line plus every request, which told me the app was healthy."
**Seen in:** `commands.sh`

## docker stop and docker rm `(commands)`
**1. Definition:** `docker stop` sends a termination signal to a running container; `docker rm` deletes a stopped container; the image is unaffected.
**2. In plain words:** Stopping pauses the process, removing deletes the container and its writable layer. The image stays so you can run it again. `docker rmi` is the one that deletes images.
**3. Example to quote:** > "After testing I ran `docker stop toy && docker rm toy`, then `docker rmi toy-api:v1` to clean up the image too."
**Seen in:** `commands.sh`

## FastAPI `(third-party: fastapi)`
**1. Definition:** A Python web framework for building HTTP APIs with automatic validation from type hints and auto-generated documentation.
**2. In plain words:** You write ordinary functions and FastAPI turns them into web endpoints. It checks incoming data against your type hints and returns a clear error if it's wrong. Docker doesn't care which framework you use; it just runs the process.
**3. Example to quote:** > "I used FastAPI because it gives me `/health` and `/predict` in a few lines, with validation for free."
```python
app = FastAPI(title="Toy sentiment API")
```
**Seen in:** `app/main.py`

## BaseModel `(third-party: pydantic)`
**1. Definition:** The Pydantic base class used to declare a typed data schema whose instances are validated when created.
**2. In plain words:** You describe what the request body must look like (`text: str`) and Pydantic rejects anything else. FastAPI uses it to turn JSON into a Python object, or to answer with a 422 error when the shape is wrong.
**3. Example to quote:** > "Sending `{"txt": "hi"}` instead of `{"text": ...}` returns 422, because `PredictRequest` requires a `text` field."
```python
class PredictRequest(BaseModel):
    text: str
```
**Seen in:** `app/main.py`

## Decorator `@app.get` / `@app.post` `(syntax)`
**1. Definition:** A function applied with `@` syntax above another function to wrap or register it; here it registers the function as the handler for an HTTP method and path.
**2. In plain words:** The line `@app.get("/health")` is a label saying "when someone sends GET /health, run the function below". The function itself stays normal Python.
**3. Example to quote:** > "`@app.post("/predict")` registers my `predict` function for POST requests to `/predict`."
**Seen in:** `app/main.py`

## Generator expression with sum() `(syntax)`
**1. Definition:** An inline expression like `sum(w in POSITIVE for w in words)` that lazily produces values; `True` counts as 1 and `False` as 0 when summed.
**2. In plain words:** It counts how many words are in the positive set without building a list. Python treats booleans as numbers, so adding up `True` values gives a count.
**3. Example to quote:** > "`sum(w in POSITIVE for w in words)` counted one positive word in 'i love it', so the score was 1."
```python
sum([True, False, True])  # 2
```
**Seen in:** `app/main.py`, `predict_label`

## str.lower() and str.split() `(Python built-in)`
**1. Definition:** `lower()` returns a lowercase copy of a string; `split()` with no argument splits on whitespace and returns a list of words.
**2. In plain words:** Lowercasing makes "LOVE" match "love", and splitting turns a sentence into individual words you can look up in a set.
**3. Example to quote:** > "`"I LOVE it".lower().split()` gives `['i', 'love', 'it']`."
**Seen in:** `app/main.py`

## TestClient `(third-party: starlette via fastapi)`
**1. Definition:** A test helper that calls a FastAPI/Starlette app in-process, returning response objects without starting a real server.
**2. In plain words:** You can test your endpoints like normal function calls. It is how `check_locally.py` proves the app works before Docker enters the picture, so you know any later failure is a Docker problem, not an app bug.
**3. Example to quote:** > "Before blaming Docker I ran `check_locally.py`; TestClient showed `/health` returns ok and a bad body returns 422."
```python
client = TestClient(app)
client.get("/health").json()
```
**Seen in:** `check_locally.py`

## HTTP 422 Unprocessable Entity `(concept)`
**1. Definition:** The status code FastAPI returns when a request is well-formed HTTP but its body fails validation against the declared schema.
**2. In plain words:** It means "I understood your request, but the data is wrong". The JSON error body names the missing or invalid field, so you can fix your request.
**3. Example to quote:** > "Posting `{"txt": "x"}` returned 422 with a message that `text` was required."
**Seen in:** `check_locally.py`, `main.py` BREAK IT

## curl `(command)`
**1. Definition:** A command-line tool for making HTTP requests; `-X POST`, `-H` and `-d` set the method, headers and body.
**2. In plain words:** It lets you talk to your API from the terminal without a browser. It's the quickest way to check that a published port really works.
**3. Example to quote:** > "I tested the container with `curl localhost:8000/health` and got `{"status":"ok"}`."
**Seen in:** `commands.sh`
