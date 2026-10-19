# Step 03 - Ports, environment variables, volumes and bind mounts

## Goal
Configure a container at run time (no rebuild) and persist data beyond the container's life.

## Run it
```bash
cd 11_docker/03_ports_env_volumes
../../.venv/bin/python check_locally.py   # no Docker needed
./commands.sh                              # line by line, Docker Desktop running
```
Real output of `check_locally.py` (temp dir path varies):
```
{'label': 'positive', 'score': 1, 'model_name': 'test-model'}
{'model_name': 'test-model', 'data_dir': '/var/folders/.../tmpXXXX', 'visits': 1}
OK
```

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `-p 9000:8000` | host 9000 -> container 8000 | `curl localhost:9000` |
| `-e KEY=value` | set one env var | `-e MODEL_NAME=x` |
| `--env-file f` | many env vars from file | `--env-file sample.env` |
| `-v name:/path` | named volume (Docker-managed) | `-v api-data:/data` |
| `-v /abs/host:/path` | bind mount (host folder) | `-v "$(pwd)/host-data:/data"` |
| `ENV A=1 B=2` | default env in the image | overridable at run time |
| `VOLUME /data` | marks a mount point | |
| `$(pwd)` | shell substitution: current directory | |
| Python `os.getenv("X", "default")` | read env var or default | |
| Python `Path("a") / "b"` | join paths | `Path("/data") / "visits.txt"` |

## Concepts: volume vs bind mount
| | Named volume | Bind mount |
|---|---|---|
| Location | Managed by Docker | Any host path you choose |
| Best for | Databases, app state, production | Dev: live code edits, sharing files with host |
| Portable | Yes (same on any host) | No (depends on host paths) |
| Container's own writable layer | Deleted with the container | n/a |

## Execution trace
1. `docker run -e MODEL_NAME=cli-model ...`: Docker injects the variable into the process environment; `os.getenv` in main.py reads it at import time; `/info` returns `cli-model`.
2. POST `/predict` -> `count_visit()` writes `/data/visits.txt`.
3. With `-v api-data:/data` the file lives in the volume, so a brand-new container sees `visits: 1`. Without `-v`, it would reset to 0.
(Typical output; daemon was off when written.)

## Experiments
1. Run without `-v`, predict, `docker rm -f`, run again: visits resets.
2. Pass both `-e MODEL_NAME=a` and `--env-file sample.env`: which wins? (Explicit `-e` overrides the file.)
3. Bind-mount `app/` over `/code/app` and edit main.py: restart picks up the change without rebuild (dev workflow).

## Debugging practice
`docker run -v host-data:/data` vs `-v ./host-data:/data`: a path *without* a leading `/` or `./` is treated as a **named volume**, not a folder. If you expect files on your disk and see none, check which one you made (`docker volume ls`, `docker inspect <c> --format '{{json .Mounts}}'`).

## Rebuild checklist
- [ ] Run a container with a different host port, an env var and a named volume from memory.
- [ ] Say when you would pick a bind mount over a volume.

## Say it aloud
"I keep configuration in environment variables so one image works in dev and prod, injected with -e or an env file. Containers are ephemeral, so state goes in a named volume, or a bind mount while developing. Trade-off: env vars are visible to anyone who can inspect the container, so real secrets need a secrets mechanism."
