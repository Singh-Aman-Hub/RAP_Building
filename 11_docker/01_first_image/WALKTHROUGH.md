# Step 01 - Your first image and container

## The story so far
Chapter 22 ([module 10](../../10_ml_system_architecture/)) ended with a blueprint for a serving system and one honest problem: it works on my machine only. This step is the first answer. We take the toy FastAPI service from [module 09](../../09_api_development/) and make it run anywhere.

## The pain point
A teammate clones your repo. Their Python is a different version, `fastapi` is not installed, port 8000 is busy, and your `.venv` is not in git. The same code behaves differently on every laptop and server. Writing a longer README does not fix it.

## The spark
Do not ship instructions, ship the machine. A **Dockerfile** is a recipe that starts from a known base (Python 3.12 slim), installs pinned requirements and copies your code. Building it gives an **image**; running the image gives a **container**, the same everywhere.

## Goal
Package a small FastAPI app into an image, run it as a container, and reach it from your laptop.

## Run it
```bash
cd 11_docker/01_first_image
../../.venv/bin/python check_locally.py     # 1. prove the app works WITHOUT Docker (real output below)
cat commands.sh                              # 2. then run its lines one by one (Docker Desktop must be running)
```
Real output of `check_locally.py`:
```
{'status': 'ok'}
{'label': 'positive', 'score': 1}
OK
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `FROM image:tag` | Start from a base image | `FROM python:3.12-slim` |
| `WORKDIR /code` | `cd` + `mkdir -p` for later steps | `WORKDIR /code` |
| `COPY src dst` | Host (build context) -> image | `COPY requirements.txt .` |
| `RUN cmd` | Run at **build** time, creates a layer | `RUN pip install -r requirements.txt` |
| `EXPOSE 8000` | Documentation only | does not publish anything |
| `CMD [...]` | Default **run-time** command, exec form | `CMD ["uvicorn", "app.main:app"]` |
| `-t name:tag` | Name the built image | `docker build -t toy-api:v1 .` |
| `-p HOST:CONTAINER` | Publish a port | `-p 8000:8000` |
| `-d` | Detached (background) | `docker run -d ...` |
| the trailing `.` in `docker build` | The **build context** (folder sent to Docker) | `docker build -t x .` |
| Python: `@app.get("/health")` | Decorator: registers the function as a route | see main.py |
| Python: `sum(w in POSITIVE for w in words)` | Generator expression; `True` counts as 1 | `sum([True, False, True])` = 2 |

## Libraries & commands
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `docker build` | Docker CLI | Builds an image from a Dockerfile | `-t`, `-f` (file), context path | image ID |
| `docker run` | Docker CLI | Creates + starts a container | `-d -p -e -v --name --rm` | container ID |
| `docker ps [-a]` | Docker CLI | Lists (all) containers | `-a` includes exited | table |
| `docker logs` | Docker CLI | Shows container stdout/stderr | `-f` follow, `--tail N` | text |
| `FastAPI` | 3rd-party | Web framework (routes, validation, docs) | `title=` | app object |
| `BaseModel` (pydantic) | 3rd-party | Validates request body shape | type-hinted fields | model instance |
| `TestClient` | 3rd-party (starlette) | Calls the app in-process for tests | app | `Response` |

## Execution trace (input -> processing -> output)
1. **Input**: `docker build -t toy-api:v1 .` - the CLI zips the folder (minus `.dockerignore`) = build context.
2. **Processing**: Docker executes each Dockerfile line; `FROM` pulls python:3.12-slim, `RUN pip install` makes a layer with fastapi+uvicorn, `COPY app` adds your code.
3. **Output**: an image `toy-api:v1` (typically ~150-200 MB; Python slim ~120 MB + packages).
4. `docker run -d -p 8000:8000 toy-api:v1`: Docker makes a container (isolated process), runs `uvicorn ... --host 0.0.0.0 --port 8000`, and forwards host port 8000 to container port 8000.
5. `curl localhost:8000/health` -> `{"status":"ok"}`. (Typical output - the daemon was not running when this was written, so Docker commands were not captured; the local check above was.)

## Experiments (one change at a time)
1. Remove `--host 0.0.0.0` -> rebuild -> curl fails (see `05.../broken_localhost_bind`).
2. Run with `-p 9000:8000` -> reachable at 9000, not 8000.
3. Run two containers from the same image with different host ports: one image, many containers.
4. Edit `POSITIVE` in main.py, `docker run` again WITHOUT rebuilding -> unchanged behaviour. Why? (Image is a snapshot.)

## Debugging practice
`docker run -d -p 8000:8000 toy-api:v1` when port 8000 is already used -> `Bind for 0.0.0.0:8000 failed: port is already allocated`. Read: *host* port taken. Find with `docker ps` or `lsof -i :8000`; choose another host port.

## Rebuild checklist
- [ ] Write the 7-line Dockerfile from memory (FROM, WORKDIR, COPY req, RUN pip, COPY app, EXPOSE, CMD).
- [ ] Write build + run commands from memory.

## The new problem (cliffhanger)
The image works, but every time you change one line of code, `docker build` seems to reinstall every package. A 1-second edit costs a 30-second build. Why, and how do we make rebuilds fast? Next: [step 02, layers and cache](../02_layers_and_cache/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** An image is a read-only template built from a Dockerfile layer by layer, and a container is a running instance of it with its own filesystem and network. I used a slim Python base, installed pinned requirements, copied my code and started uvicorn on 0.0.0.0 so the published port is reachable. The trade-off is that every code change needs a rebuild, which is why layer ordering matters.
