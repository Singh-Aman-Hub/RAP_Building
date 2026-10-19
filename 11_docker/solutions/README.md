# Solutions - Module 11

## Step 01
```bash
docker run -d --name toy -p 9100:8000 toy-api:v1 && curl localhost:9100/health
```
Break-it: without `--host 0.0.0.0` uvicorn binds 127.0.0.1; logs say `Uvicorn running on http://127.0.0.1:8000`; port forwarding cannot reach loopback inside the container.

## Step 02
Adding `httpx` changes `requirements.txt` -> layers from `COPY requirements.txt` onward rebuild (including pip install) in `Dockerfile.good_order`; `FROM`/`WORKDIR` stay cached.
Break-it: `COPY app ./app # comment` -> error like `failed to compute cache key: "#" not found` (the `#` is treated as a source file).
`.dockerignore`: `.venv`, `.git`, `__pycache__`, `*.pyc`, `.env`.

## Step 03
```bash
docker run -d --name e -p 8000:8000 --env-file sample.env -v api-data:/data env-api:v1
```
`./host-data:/data` = bind mount (host folder); `host-data:/data` = named volume called host-data.

## Step 04
Fix: `REDIS_HOST: redis`. Log error is `redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379` - inside the api container, localhost is itself.

## Step 05
See DIAGNOSIS.md answers and [drill_notes_template.md](drill_notes_template.md).
Wrong-port health check: `HEALTHCHECK CMD python -c "import urllib.request as u;u.urlopen('http://localhost:9999/health')"` -> status goes `starting` then `unhealthy`.

## Step 06
Make `/data` writable: `RUN mkdir /data && chown appuser:appuser /data` before `USER appuser`.
Hardening: slim base, multi-stage, non-root, pinned versions, `.dockerignore`, no secrets in image, health check.
