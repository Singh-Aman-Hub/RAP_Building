# Exercises - Module 11 Docker

Rules: do one change at a time, predict the result before running, and use `docker logs`/`ps -a`/`exec` rather than AI when something fails. Solutions in [solutions/](solutions/).

## Step 01 - first image
- **Modify:** run the container on host port 9100 and call `/predict`.
- **Break it & read:** remove `--host 0.0.0.0` from CMD; rebuild; explain what `docker logs` shows and why curl fails.
- **Rebuild from memory:** write the Dockerfile and build/run commands with this file closed.
- **Explain aloud:** image vs container in 30 seconds.

## Step 02 - layers
- **Modify:** add `httpx` to requirements; predict which layers rebuild; verify.
- **Break it:** put a `# comment` after a `COPY` line and read the build error.
- **Rebuild:** write a `.dockerignore`.
- **Explain aloud:** why slim, not alpine, for ML.

## Step 03 - ports/env/volumes
- **Modify:** run with `--env-file sample.env` and a named volume; show visits survive container removal.
- **Break it:** use `-v ./host-data:/data` vs `-v host-data:/data` and explain the difference.
- **Rebuild:** run command with port, env var, volume (no looking).
- **Explain aloud:** volume vs bind mount.

## Step 04 - Compose
- **Modify:** add a `LOG_LEVEL` env var to api via compose and `.env`.
- **Break it:** set `REDIS_HOST: localhost`; read `docker compose logs api`; fix.
- **Rebuild:** a 2-service compose file (api + redis) from memory.
- **Explain aloud:** how does ui reach api and why not `localhost`?

## Step 05 - debugging
- **Modify:** do all five drills; write the 5-point diagnosis note for each (solutions/drill_notes_template.md).
- **Break it:** add a HEALTHCHECK that tests the wrong port; observe `unhealthy`.
- **Explain aloud:** exit code 0 vs 1 vs 137.

## Step 06 - multi-stage/security
- **Modify:** add `numpy`; compare image sizes `single` vs `multi`.
- **Break it:** make `/data` unwritable for non-root; read the PermissionError.
- **Rebuild:** a 2-stage Dockerfile with non-root user.
- **Explain aloud:** five hardening practices.

## Capstone for this module
Containerise your module-09 API with Compose (api + ui from [../12_streamlit_gradio/](../12_streamlit_gradio/)); document the commands in its README.
