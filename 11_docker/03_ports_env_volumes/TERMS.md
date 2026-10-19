# Terms — Ports, environment variables, volumes and bind mounts

Folder: `11_docker/03_ports_env_volumes` · Files covered: `Dockerfile`, `app/main.py`, `check_locally.py`, `commands.sh`, `sample.env`, `requirements.txt`, `WALKTHROUGH.md`
Back to module: [11_docker README](../README.md)

**Terms in this folder:** [Environment variable (concept)](#environment-variable-concept) · [os.getenv (Python built-in module function)](#osgetenv-python-built-in-module-function) · [ENV (Dockerfile instruction)](#env-dockerfile-instruction) · [docker run -e (command flag)](#docker-run--e-command-flag) · [--env-file (docker run flag)](#--env-file-docker-run-flag) · [sample.env (file)](#sampleenv-file) · [Volume (named volume) (concept)](#volume-named-volume-concept) · [docker run -v (command flag)](#docker-run--v-command-flag) · [Bind mount (concept)](#bind-mount-concept) · [VOLUME (Dockerfile instruction)](#volume-dockerfile-instruction) · [docker volume ls / rm (commands)](#docker-volume-ls--rm-commands) · [Port mapping (docker run -p HOST:CONTAINER)](#port-mapping-docker-run--p-hostcontainer) · [docker rm -f (command flag)](#docker-rm--f-command-flag) · [docker run --name (command flag)](#docker-run---name-command-flag) · [Ephemeral container filesystem (concept)](#ephemeral-container-filesystem-concept) · [pathlib.Path (standard library)](#pathlibpath-standard-library) · [Path.mkdir(parents=True, exist_ok=True) (standard library)](#pathmkdirparentstrue-exist_oktrue-standard-library) · [logging.basicConfig (standard library)](#loggingbasicconfig-standard-library) · [logging.getLogger and log.info (standard library)](#logginggetlogger-and-loginfo-standard-library) · [Persisting a counter in a file (concept)](#persisting-a-counter-in-a-file-concept) · [Conditional expression (syntax)](#conditional-expression-syntax) · [Environment set before import (concept)](#environment-set-before-import-concept) · [tempfile.mkdtemp (standard library)](#tempfilemkdtemp-standard-library) · [$(pwd) (shell syntax)](#pwd-shell-syntax) · [Shell && and ; chaining (shell syntax)](#shell--and--chaining-shell-syntax) · [PermissionError at startup (error)](#permissionerror-at-startup-error) · [FastAPI GET /info endpoint (concept)](#fastapi-get-info-endpoint-concept) · [requirements.txt (file)](#requirementstxt-file)

---

## Environment variable `(concept)`
**1. Definition:** A named string value held in a process's environment, inherited from its parent and readable by the program at run time.
**2. In plain words:** It is a setting you pass in from outside the code, like `MODEL_NAME=cli-model`. The same image can behave differently in development and production without editing or rebuilding anything. Because they are plain text visible to anyone who can inspect the container, they are not a safe place for real secrets.
**3. Example to quote:** > "I changed `MODEL_NAME` with `-e` and `/info` reported the new model name, with no code change and no rebuild."
**Seen in:** `app/main.py`, `commands.sh`

## os.getenv `(Python built-in module function)`
**1. Definition:** `os.getenv(name, default)` returns the value of environment variable `name` as a string, or `default` (or `None`) if it is not set.
**2. In plain words:** It is how Python reads settings from outside. The default means the app still runs on your laptop where nothing is set. Remember the result is always a string, so convert numbers yourself.
**3. Example to quote:** > "`os.getenv("MODEL_NAME", "keyword-sentiment-v1")` gives me the container's setting, or a sensible default when I run locally."
```python
MODEL_NAME = os.getenv("MODEL_NAME", "keyword-sentiment-v1")
```
**Seen in:** `app/main.py`

## ENV `(Dockerfile instruction)`
**1. Definition:** Sets one or more environment variables with default values inside the image, available at build time and in containers.
**2. In plain words:** It bakes defaults into the image. At `docker run` time, `-e` or `--env-file` can override them, so `ENV` is the fallback and the command line is the override.
**3. Example to quote:** > "I set `ENV MODEL_NAME=keyword-sentiment-v1` as a default in the image and override it per container with `-e`."
```dockerfile
ENV MODEL_NAME=keyword-sentiment-v1 DATA_DIR=/data LOG_LEVEL=INFO
```
**Seen in:** `Dockerfile`

## docker run -e `(command flag)`
**1. Definition:** Sets a single environment variable for the container being started, overriding any `ENV` default in the image.
**2. In plain words:** Use it for one-off settings, like `-e MODEL_NAME=cli-model`. When the same variable also appears in an `--env-file`, the explicit `-e` wins.
**3. Example to quote:** > "`docker run -e MODEL_NAME=cli-model ...` and then `curl /info` returned `cli-model`."
**Seen in:** `commands.sh`

## --env-file `(docker run flag)`
**1. Definition:** Loads many environment variables for a container from a file with one `KEY=value` per line.
**2. In plain words:** When you have more than a couple of settings, put them in a file such as `sample.env`. Lines have no quotes and no `export`. Keep real secrets files out of git and out of the image.
**3. Example to quote:** > "I passed `--env-file sample.env` and the logs switched to DEBUG level without touching the image."
**Seen in:** `commands.sh`, `sample.env`

## sample.env `(file)`
**1. Definition:** A dotenv-style text file of `KEY=value` lines (`MODEL_NAME=model-from-env-file`, `LOG_LEVEL=DEBUG`) meant to be passed with `--env-file`.
**2. In plain words:** It is the settings sheet for a container. Naming it "sample" signals that it holds example values you can commit, while a real `.env` with secrets should be ignored by git and `.dockerignore`.
**3. Example to quote:** > "The sample file sets the model name and log level; I copy it to `.env` and edit the values locally."
**Seen in:** `sample.env`

## Volume (named volume) `(concept)`
**1. Definition:** Docker-managed persistent storage mounted into a container at a path, living independently of any container's lifecycle.
**2. In plain words:** A container's own writable layer is deleted with the container. A named volume survives, so a brand-new container mounted to the same volume sees the old data. It is the right choice for databases and application state.
**3. Example to quote:** > "With `-v api-data:/data` the visit counter was still 1 after I removed the container and started a new one."
**Seen in:** `commands.sh`, `WALKTHROUGH.md`

## docker run -v `(command flag)`
**1. Definition:** Mounts storage into a container as `-v source:container_path`; a source without a leading `/` or `./` is a named volume, an absolute host path is a bind mount.
**2. In plain words:** The same flag does two jobs depending on the left-hand side. `-v api-data:/data` makes or reuses a Docker-managed volume; `-v "$(pwd)/host-data:/data"` shares a folder from your laptop. Mixing them up is a common reason for "my files didn't appear".
**3. Example to quote:** > "I expected files on my disk, but `-v host-data:/data` without a path created a named volume; I needed `$(pwd)/host-data`."
**Seen in:** `commands.sh`, `WALKTHROUGH.md` debugging

## Bind mount `(concept)`
**1. Definition:** A mount that maps a specific host directory or file into the container's filesystem, so both see the same files.
**2. In plain words:** It's a shortcut to a real folder on your laptop. It is perfect in development, for example mounting `app/` so edits appear instantly, but it ties the setup to your machine's paths and so isn't portable for production.
**3. Example to quote:** > "I bind-mounted `./host-data` and `cat ./host-data/visits.txt` showed the container's file on my own disk."
**Seen in:** `commands.sh`

## VOLUME `(Dockerfile instruction)`
**1. Definition:** Declares a path inside the image as a mount point for externally managed storage; if none is supplied Docker creates an anonymous volume.
**2. In plain words:** It documents "state lives here". If you forget `-v`, Docker silently creates an anonymous volume that is easy to lose track of, so you normally name the volume explicitly at run time.
**3. Example to quote:** > "I marked `/data` with `VOLUME` so anyone reading the Dockerfile knows that's where the app keeps its state."
**Seen in:** `Dockerfile`

## docker volume ls / rm `(commands)`
**1. Definition:** `docker volume ls` lists volumes; `docker volume rm NAME` deletes one.
**2. In plain words:** Volumes outlive containers, so they need explicit cleanup. Listing them also helps you check whether you made a named volume or a bind mount.
**3. Example to quote:** > "After the experiment I ran `docker volume rm api-data` to delete the persisted counter."
**Seen in:** `commands.sh`, `WALKTHROUGH.md`

## Port mapping `(docker run -p HOST:CONTAINER)`
**1. Definition:** Publishes a container port on a host port so traffic to the host port is forwarded to the container.
**2. In plain words:** Containers have a private network, so you map a port from your laptop to one inside. Two containers can listen on the same container port as long as their host ports differ, as with 8000 and 9000 here.
**3. Example to quote:** > "I ran the same image twice, `-p 8000:8000` and `-p 9000:8000`, and both answered on their own host ports."
**Seen in:** `commands.sh`

## docker rm -f `(command flag)`
**1. Definition:** Forcibly stops (if running) and removes one or more containers in a single step.
**2. In plain words:** It is the quick cleanup between experiments, so the container name and the host port are free for the next run.
**3. Example to quote:** > "`docker rm -f a b` removed both demo containers so port 8000 was free again."
**Seen in:** `commands.sh`

## docker run --name `(command flag)`
**1. Definition:** Assigns a chosen name to a container instead of a random one.
**2. In plain words:** Names let you write `docker logs d` or `docker rm -f e` instead of copying long IDs. A name must be unique, so starting a second container with the same name fails until you remove the first.
**3. Example to quote:** > "I named the containers a, b, c so the logs and cleanup commands were short."
**Seen in:** `commands.sh`

## Ephemeral container filesystem `(concept)`
**1. Definition:** The writable layer of a container exists only as long as the container does; removing the container deletes its contents.
**2. In plain words:** Anything your app writes inside the container, like the visit counter, vanishes when the container is removed unless it's stored in a volume or bind mount. This is why state must be externalised.
**3. Example to quote:** > "Without `-v`, I removed the container and the visit counter reset to zero."
**Seen in:** `WALKTHROUGH.md`, experiment 1

## pathlib.Path `(standard library)`
**1. Definition:** An object-oriented representation of filesystem paths; `/` joins path parts and methods like `mkdir`, `exists`, `read_text` and `write_text` operate on files.
**2. In plain words:** It's a nicer alternative to gluing strings together. `Path("/data") / "visits.txt"` builds the path in a portable way and the object knows how to read and write that file.
**3. Example to quote:** > "`DATA_DIR / "visits.txt"` gives me the file path, and `.write_text()` stores the counter."
```python
VISITS_FILE = Path("/data") / "visits.txt"
```
**Seen in:** `app/main.py`

## Path.mkdir(parents=True, exist_ok=True) `(standard library)`
**1. Definition:** Creates a directory; `parents=True` also creates missing parent directories and `exist_ok=True` suppresses the error if it already exists.
**2. In plain words:** It makes the data folder safely at startup whether or not it is already there, which matters when a volume is mounted over it.
**3. Example to quote:** > "I call `DATA_DIR.mkdir(parents=True, exist_ok=True)` so the app starts the same on a fresh machine and on a mounted volume."
**Seen in:** `app/main.py`

## logging.basicConfig `(standard library)`
**1. Definition:** Configures the root logger's level, output format and handler; by default it writes to stderr.
**2. In plain words:** It sets how log lines look and which severity is shown. Logging to the console (not a file) is exactly what Docker wants, because `docker logs` shows whatever the process prints.
**3. Example to quote:** > "With `LOG_LEVEL=DEBUG` from the env file, `docker logs d` showed extra detail; the level comes from an environment variable."
```python
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
```
**Seen in:** `app/main.py`

## logging.getLogger and log.info `(standard library)`
**1. Definition:** `getLogger(name)` returns a named logger; `info(msg, *args)` emits an INFO-level record, formatting `%s`/`%d` placeholders lazily.
**2. In plain words:** It's the structured replacement for `print`. Each request logs which model produced which label, and you can raise or lower the verbosity without editing the code.
**3. Example to quote:** > "`log.info("predict #%d model=%s label=%s", n, MODEL_NAME, label)` shows up in `docker logs`."
**Seen in:** `app/main.py`

## Persisting a counter in a file `(concept)`
**1. Definition:** Reading an integer from a file, adding one, and writing it back so the value survives process restarts, provided the file's directory is on persistent storage.
**2. In plain words:** It's a deliberately simple way to demonstrate state. It only survives container removal if the directory is a volume or bind mount, which makes the lesson visible: restart with and without `-v` and compare.
**3. Example to quote:** > "`count_visit()` reads `visits.txt`, adds one and writes it back; with a volume the count continued across containers."
```python
n = int(VISITS_FILE.read_text()) if VISITS_FILE.exists() else 0
```
**Seen in:** `app/main.py`, `count_visit`

## Conditional expression `(syntax)`
**1. Definition:** `a if condition else b` evaluates to `a` when the condition is true and `b` otherwise.
**2. In plain words:** A one-line if/else that produces a value. It's used here to read the file only if it exists, and to pick the label "positive", "negative" or "neutral".
**3. Example to quote:** > "`label = "positive" if score > 0 else "negative" if score < 0 else "neutral"` picks the label in one line."
**Seen in:** `app/main.py`

## Environment set before import `(concept)`
**1. Definition:** Module-level code such as `os.getenv(...)` runs at import time, so environment variables must be set before the module is imported.
**2. In plain words:** `main.py` reads settings the moment it is loaded. In `check_locally.py` the variables are set first and the app is imported afterwards; reversing the order would make the app use its defaults.
**3. Example to quote:** > "I set `MODEL_NAME` in `os.environ` before `from app.main import app`, otherwise the app would already have read the defaults."
**Seen in:** `check_locally.py`

## tempfile.mkdtemp `(standard library)`
**1. Definition:** Creates a unique temporary directory and returns its path.
**2. In plain words:** It gives the test a clean, throwaway folder for `DATA_DIR`, so test runs don't pollute each other or your real data.
**3. Example to quote:** > "The local check uses `tempfile.mkdtemp()` as `DATA_DIR`, so the visit count starts at zero every run."
**Seen in:** `check_locally.py`

## $(pwd) `(shell syntax)`
**1. Definition:** Command substitution that is replaced by the output of `pwd`, the current directory's absolute path.
**2. In plain words:** Bind mounts need an absolute host path. `"$(pwd)/host-data"` builds it from wherever you are, and the quotes protect paths containing spaces.
**3. Example to quote:** > "I wrote `-v "$(pwd)/host-data:/data"` so Docker received the full path to the folder on my disk."
**Seen in:** `commands.sh`

## Shell && and ; chaining `(shell syntax)`
**1. Definition:** `a && b` runs `b` only if `a` succeeds; `a; b` runs both regardless.
**2. In plain words:** Used to combine cleanup steps and several curl calls on one line.
**3. Example to quote:** > "`curl localhost:8000/health; curl localhost:9000/health` checks both ports in one line."
**Seen in:** `commands.sh`

## PermissionError at startup `(error)`
**1. Definition:** A Python exception raised when the process lacks rights to read or write a path, for example creating a directory under `/` as a non-root user.
**2. In plain words:** It's the failure the BREAK IT hint predicts: pointing `DATA_DIR` at a location the container user can't write to. The traceback in `docker logs` names the exact path.
**3. Example to quote:** > "The container exited at startup with PermissionError on `/nonexistent/deep`; `docker logs` showed the path and the failing `mkdir` line."
**Seen in:** `app/main.py`, BREAK IT comment

## FastAPI GET /info endpoint `(concept)`
**1. Definition:** An endpoint returning the running configuration (model name, data directory and visit count) as JSON.
**2. In plain words:** It makes the effect of environment variables and volumes visible with a single curl, so you can prove that a setting took effect.
**3. Example to quote:** > "`curl localhost:8000/info` returned `cli-model` after I passed `-e MODEL_NAME=cli-model`."
**Seen in:** `app/main.py`

## requirements.txt `(file)`
**1. Definition:** A pinned list of Python packages for `pip install -r`.
**2. In plain words:** The same two pinned packages, fastapi and uvicorn, as in step 01; installed in their own cached layer before the code.
**3. Example to quote:** > "The pinned versions guarantee the image I build today matches the one I build next month."
**Seen in:** `requirements.txt`
