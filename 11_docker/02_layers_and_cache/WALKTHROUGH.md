# Step 02 - Layers, the build cache and base-image choice

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

## Say it aloud
"Docker caches each layer; once an instruction's inputs change, that layer and all later ones rebuild. So I copy requirements first, install, then copy source, because code changes constantly and dependencies rarely. Trade-off: slim images are small and compatible; alpine is smaller but its musl libc breaks many scientific wheels."
