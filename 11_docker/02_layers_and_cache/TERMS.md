# Terms — Layers, build cache and base images

Folder: `11_docker/02_layers_and_cache` · Files covered: `Dockerfile.bad_order`, `Dockerfile.good_order`, `Dockerfile.alpine`, `commands.sh`, `app/main.py`, `requirements.txt`, `.dockerignore`, `WALKTHROUGH.md`
Back to module: [11_docker README](../README.md)

**Terms in this folder:** [Image layer (concept)](#image-layer-concept) · [Build cache (concept)](#build-cache-concept) · [CACHED (docker build output)](#cached-docker-build-output) · [Layer ordering (least-changing first) (concept)](#layer-ordering-least-changing-first-concept) · [COPY . . (Dockerfile instruction)](#copy---dockerfile-instruction) · [docker build -f (command flag)](#docker-build--f-command-flag) · [Image tag (name:tag) (concept)](#image-tag-nametag-concept) · [docker history (command)](#docker-history-command) · [docker images (command)](#docker-images-command) · [docker rmi (command)](#docker-rmi-command) · [Slim base image (concept)](#slim-base-image-concept) · [Alpine base image (concept)](#alpine-base-image-concept) · [musl libc vs glibc (concept)](#musl-libc-vs-glibc-concept) · [Python wheel (concept)](#python-wheel-concept) · [Dockerfile comment placement (syntax)](#dockerfile-comment-placement-syntax) · [.dockerignore (file)](#dockerignore-file) · [CMD (exec form) (Dockerfile instruction)](#cmd-exec-form-dockerfile-instruction) · [FROM, WORKDIR, RUN (recap) (Dockerfile instructions)](#from-workdir-run-recap-dockerfile-instructions) · [pip install --no-cache-dir (command flag)](#pip-install---no-cache-dir-command-flag) · [FastAPI toy app (concept)](#fastapi-toy-app-concept) · [Pydantic BaseModel (third-party: pydantic)](#pydantic-basemodel-third-party-pydantic) · [uvicorn (third-party: uvicorn)](#uvicorn-third-party-uvicorn) · [Dockerfile variants for comparison (concept)](#dockerfile-variants-for-comparison-concept)

---

## Image layer `(concept)`
**1. Definition:** A read-only filesystem difference produced by one Dockerfile instruction; an image is a stack of such layers.
**2. In plain words:** Every `COPY` or `RUN` line adds one more layer on top of the previous ones, like sheets of transparent film stacked into one picture. Because layers are separate, Docker can reuse unchanged ones between builds and share them between images, which saves time and disk.
**3. Example to quote:** > "My image has a layer for the Python base, one for the pip install and one for my code; only the last one changes when I edit a file."
**Seen in:** `WALKTHROUGH.md`, Concepts table

## Build cache `(concept)`
**1. Definition:** Docker's mechanism of reusing a previously built layer when the instruction and its inputs are unchanged and all earlier layers were reused.
**2. In plain words:** If Docker has already done an identical step, it skips it and prints `CACHED`. The first step that changed forces every step after it to run again. That is why the order of lines in a Dockerfile decides how fast your rebuilds are.
**3. Example to quote:** > "After I edited a comment in `main.py`, the good-order build said `CACHED` for pip install and finished in about a second."
**Seen in:** `commands.sh`, `WALKTHROUGH.md`

## CACHED `(docker build output)`
**1. Definition:** The marker in BuildKit's build output showing that a step's result was reused from the cache instead of being executed.
**2. In plain words:** It is the green light that your layer ordering works. If you see `CACHED` on the slow pip step after a code-only change, you did it right; if it re-runs, something earlier in the file changed or the order is wrong.
**3. Example to quote:** > "I checked the output for `CACHED [3/5] RUN pip install`; seeing it told me my dependencies weren't reinstalled."
**Seen in:** `WALKTHROUGH.md`, Syntax table

## Layer ordering (least-changing first) `(concept)`
**1. Definition:** The practice of placing instructions whose inputs change rarely (base image, dependencies) before those that change often (application code) so the cache stays valid longer.
**2. In plain words:** Dependencies change a few times a month; your code changes every few minutes. If code is copied first, every edit throws away the slow pip layer. Copy `requirements.txt`, install, and only then copy the code.
**3. Example to quote:** > "In `Dockerfile.good_order` I copy requirements first and code last, so a code edit rebuilds in under a second instead of re-running pip for twenty."
```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
```
**Seen in:** `Dockerfile.good_order`

## COPY . . `(Dockerfile instruction)`
**1. Definition:** Copies the whole build context (first dot) into the current working directory of the image (second dot).
**2. In plain words:** It is convenient but it makes one big layer that changes whenever any file changes. Putting it before `pip install`, as `Dockerfile.bad_order` does, busts the cache on every edit.
**3. Example to quote:** > "`COPY . .` before `pip install` meant that touching one comment reinstalled all my packages."
**Seen in:** `Dockerfile.bad_order`

## docker build -f `(command flag)`
**1. Definition:** Selects a Dockerfile other than the default `Dockerfile` by file name.
**2. In plain words:** Normally Docker looks for a file named `Dockerfile`. When you keep several variants side by side, like `Dockerfile.bad_order`, `-f` tells it which one to use.
**3. Example to quote:** > "I compared the two orderings with `docker build -f Dockerfile.bad_order -t cache-demo:bad .` and the same for good."
```bash
docker build -f Dockerfile.good_order -t cache-demo:good .
```
**Seen in:** `commands.sh`

## Image tag (name:tag) `(concept)`
**1. Definition:** A label after the colon in an image reference, such as `cache-demo:good`, identifying a version or variant of an image.
**2. In plain words:** The name says which project, the tag says which version. If you omit a tag Docker assumes `latest`, which can silently change, so use explicit tags for repeatability.
**3. Example to quote:** > "I built three images named `cache-demo` with tags `bad`, `good` and `alpine` so I could compare them in one `docker images` listing."
**Seen in:** `commands.sh`

## docker history `(command)`
**1. Definition:** Lists the layers of an image with the instruction that created each and its size.
**2. In plain words:** It shows you where the megabytes went. You can spot that the pip install layer is the biggest and check that you did not accidentally copy something huge.
**3. Example to quote:** > "`docker history cache-demo:good` showed me one row per instruction; the pip layer was by far the largest."
**Seen in:** `commands.sh`

## docker images `(command)`
**1. Definition:** Lists local images with repository, tag, ID and size.
**2. In plain words:** It is how you compare image sizes, for example slim against alpine, and see what is taking disk space.
**3. Example to quote:** > "I ran `docker images cache-demo` to compare the three variants' sizes side by side."
**Seen in:** `commands.sh`

## docker rmi `(command)`
**1. Definition:** Removes one or more images from the local machine.
**2. In plain words:** Images pile up with every build. `rmi` is the cleanup step; it fails if a container still uses the image, so remove containers first.
**3. Example to quote:** > "At the end of the experiment I removed all three demo images with `docker rmi cache-demo:bad cache-demo:good cache-demo:alpine`."
**Seen in:** `commands.sh`

## Slim base image `(concept)`
**1. Definition:** A reduced Debian-based variant of an official image, for example `python:3.12-slim` (about 120 MB), that excludes most build tools and documentation.
**2. In plain words:** It is the sweet spot for most Python apps: much smaller than the full image (about 1 GB) yet using the same standard libraries, so pip finds ready-made wheels. If a package must be compiled you may have to install build tools yourself.
**3. Example to quote:** > "I picked `python:3.12-slim` because it is a tenth the size of the full image but still installs numpy from a wheel."
**Seen in:** `Dockerfile.good_order`, `WALKTHROUGH.md`

## Alpine base image `(concept)`
**1. Definition:** A minimal Linux distribution (about 5 MB base) that uses the musl C library instead of glibc; `python:3.12-alpine` is the Python variant.
**2. In plain words:** Alpine is tiny, but many Python packages ship wheels built for glibc only. On Alpine pip often compiles them from source, which is slow and sometimes fails. For ML projects that is usually not worth the saved megabytes.
**3. Example to quote:** > "I tried adding numpy to the alpine Dockerfile and the build crawled while compiling; that's why ML projects stick with slim."
**Seen in:** `Dockerfile.alpine`

## musl libc vs glibc `(concept)`
**1. Definition:** Two implementations of the C standard library; Alpine uses musl, most other Linux distributions including Debian use glibc, and compiled Python wheels are tagged for one or the other.
**2. In plain words:** Compiled packages are built against a specific C library. A wheel built for glibc won't run on musl, so pip falls back to compiling. That explains why Alpine builds are slower for scientific packages.
**3. Example to quote:** > "There's no prebuilt musl wheel for many numeric packages, so on Alpine pip had to build them from source."
**Seen in:** `Dockerfile.alpine` comment

## Python wheel `(concept)`
**1. Definition:** A pre-built binary distribution of a Python package (`.whl`) that pip installs without compiling.
**2. In plain words:** Think of a wheel as a ready-baked cake and a source package as a recipe. Wheels install in seconds; source builds need compilers and take minutes, and can fail if headers or libraries are missing.
**3. Example to quote:** > "On `python:3.12-slim` pip downloads a wheel for numpy; on Alpine there is none, so it compiles."
**Seen in:** `Dockerfile.alpine` comment

## Dockerfile comment placement `(syntax)`
**1. Definition:** In a Dockerfile a `#` starts a comment only at the beginning of a line; a `#` after an instruction is passed to that instruction as part of its arguments.
**2. In plain words:** Unlike Python, you can't put an inline comment after a command. `COPY a b # note` would try to copy files named `#` and `note`. Put comments on their own line above the instruction.
**3. Example to quote:** > "Both of my Dockerfiles keep comments on separate lines; an inline `#` after `COPY` would have been treated as a file name."
**Seen in:** `Dockerfile.bad_order`, `WALKTHROUGH.md` Syntax table

## .dockerignore `(file)`
**1. Definition:** A file listing paths excluded from the build context sent to the Docker engine.
**2. In plain words:** Without it, `COPY . .` would send your virtualenv, git history and `.env` secrets into the image. Excluding them shrinks the context, speeds the build and avoids baking in secrets. It also affects caching, because irrelevant file changes no longer invalidate the `COPY` layer.
**3. Example to quote:** > "I ignore `.venv`, `.git` and `.env` so editing them can't bust my cache and my secrets never reach the image."
**Seen in:** `.dockerignore`

## CMD (exec form) `(Dockerfile instruction)`
**1. Definition:** Sets the container's default start command; the JSON-array form runs the executable directly rather than through `/bin/sh -c`.
**2. In plain words:** Each of the three Dockerfiles ends with the same uvicorn command. Exec form means uvicorn is the main process and receives stop signals, so shutdown is clean.
**3. Example to quote:** > "I always write `CMD` as a JSON list so uvicorn gets the SIGTERM from `docker stop` directly."
**Seen in:** all three Dockerfiles

## FROM, WORKDIR, RUN (recap) `(Dockerfile instructions)`
**1. Definition:** `FROM` picks the base image, `WORKDIR` sets the working directory, and `RUN` executes a build-time shell command that becomes a layer.
**2. In plain words:** These three appear in every Dockerfile here. Each is a layer-creating step, and the cache logic of this step applies to all of them: change one, and everything after it rebuilds.
**3. Example to quote:** > "In the good-order file, `FROM` and `WORKDIR` rarely change, so they and the pip `RUN` all stay cached."
**Seen in:** all three Dockerfiles

## pip install --no-cache-dir `(command flag)`
**1. Definition:** Installs packages from `requirements.txt` while telling pip not to keep its download cache.
**2. In plain words:** The cache is useless inside an image and just makes the layer bigger. Note that this is different from Docker's build cache: one is pip's cache, the other is layer reuse.
**3. Example to quote:** > "`--no-cache-dir` keeps pip's own cache out of my image; Docker's layer cache is a separate thing that I want to keep."
**Seen in:** all three Dockerfiles

## FastAPI toy app `(concept)`
**1. Definition:** A minimal FastAPI application with `GET /health` and `POST /predict`, used only as a payload to build into an image.
**2. In plain words:** The app is deliberately trivial: a keyword counter. Docker doesn't care what the app does, so a tiny app lets you focus on layers and caching rather than waiting for heavy libraries.
**3. Example to quote:** > "The app itself is unimportant; what I'm measuring is how long the image rebuilds after I change one line of it."
**Seen in:** `app/main.py`

## Pydantic BaseModel `(third-party: pydantic)`
**1. Definition:** A base class for declaring typed request schemas that are validated automatically.
**2. In plain words:** `PredictRequest(BaseModel)` with `text: str` means requests missing `text`, or sending the wrong type, are rejected with a 422 before your code runs.
**3. Example to quote:** > "`PredictRequest` documents and enforces that the body must be `{"text": "<string>"}`."
**Seen in:** `app/main.py`

## uvicorn `(third-party: uvicorn)`
**1. Definition:** An ASGI web server that runs Python async web applications such as FastAPI.
**2. In plain words:** The framework (FastAPI) defines the routes; the server (uvicorn) listens on a port and passes requests to it. In the Dockerfile `CMD`, uvicorn is the process that keeps the container alive.
**3. Example to quote:** > "FastAPI is the app, uvicorn is the server; my container's only job is to keep uvicorn running."
**Seen in:** all `CMD` lines, `requirements.txt`

## Dockerfile variants for comparison `(concept)`
**1. Definition:** Multiple Dockerfiles in one folder, selected with `-f`, used to measure the effect of one design choice at a time.
**2. In plain words:** Changing one thing, such as the instruction order or the base image, and comparing build times or sizes is the same one-change-at-a-time habit used for all experiments in this pack.
**3. Example to quote:** > "I changed only the order of two lines between the two files and measured twenty seconds against one."
**Seen in:** `commands.sh`
