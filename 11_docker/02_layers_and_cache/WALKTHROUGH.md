# Step 02 - Layers, the build cache and base-image choice

## The story so far
[Step 01](../01_first_image/WALKTHROUGH.md) gave us a working image, but a one-line code edit made the rebuild feel as slow as the first build.

## The pain point
`Dockerfile.bad_order` runs `COPY . .` before `pip install`. Any edit to any file changes that layer, and Docker then rebuilds every layer after it, including the slow dependency install. You wait, every time.

## The spark
Each instruction is a cached layer, and a layer is reused only if everything before it is unchanged. So order instructions from least to most frequently changing: copy `requirements.txt`, install, and only then copy the source. Pick `slim`, not `alpine`, as the base so ML wheels install.

## Goal
Understand that each Dockerfile instruction is a cached layer, order them to make rebuilds fast, and choose a base image.

## Run it
```bash
cd 11_docker/02_layers_and_cache
cat Dockerfile.bad_order Dockerfile.good_order
./commands.sh      # run line by line; Docker Desktop must be running
```

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `docker build -f FILE` | Use a Dockerfile with a non-default name | `-f Dockerfile.good_order` |
| `CACHED` in build output | Layer reused because instruction + inputs are unchanged | `=> CACHED [3/5] RUN pip install ...` |
| `COPY . .` | Copy whole build context (first `.`) into WORKDIR (second `.`) | cache-busting when anything changes |
| Comment placement | `#` only starts a comment at the *beginning* of a line | `COPY a b # x` would pass `#`, `x` as files! |

## Concepts
| Name | What it does |
|---|---|
| Layer | Filesystem diff produced by one instruction; stacked read-only |
| Cache rule | Docker reuses a layer if the instruction text AND (for COPY) file contents are unchanged AND all previous layers were reused. First change invalidates everything below it |
| `docker history` | Lists layers and sizes |
| `python:3.12` vs `-slim` vs `-alpine` | ~1 GB (tools, gcc) / ~120 MB (Python only) / ~50 MB (musl libc, wheels often missing) |

## Execution trace
- First build of both: every step runs (pip install ~10-30 s).
- Edit one comment in `app/main.py`, rebuild:
  - `bad_order`: `COPY . .` changed -> pip install runs again.
  - `good_order`: `COPY requirements.txt` unchanged -> pip layer `CACHED`; only `COPY app` re-runs (<1 s).
- Typical numbers (not captured, daemon off): bad ~20 s, good ~1 s on rebuild.

## Experiments
1. Add a package to requirements.txt -> good_order re-runs pip (correct: dependencies changed).
2. Move `COPY app` above `RUN pip install` in good_order -> you recreated the bad order.
3. Build `Dockerfile.alpine`, add `numpy` to requirements: observe the slow compile or failure. Why do ML projects avoid alpine?

## Debugging practice
`COPY requirements.txt .` fails with `"/requirements.txt": not found` -> the file is not in the *build context* (wrong folder, or listed in `.dockerignore`). Check the context path (last argument of `docker build`).

## Rebuild checklist
- [ ] Explain why dependencies are copied and installed before application code.
- [ ] Write a `.dockerignore` that excludes `.venv`, `.git`, `__pycache__`, `.env`.

## The new problem (cliffhanger)
The image builds fast now, but it has everything baked in: port, model name, and any data it writes disappears when the container is removed. How do we configure a container without rebuilding, and keep data? Next: [step 03, ports, env and volumes](../03_ports_env_volumes/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** Docker caches each layer, and once an instruction's inputs change that layer and every later one rebuilds. So I copy requirements first, install, then copy source, because code changes constantly and dependencies rarely. The trade-off for base images is that slim is compatible and small, while alpine is smaller but its musl libc breaks many scientific wheels.
