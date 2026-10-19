#!/usr/bin/env bash
# Step 06: multi-stage build + security checks. (Needs Docker Desktop running.)
docker build -f Dockerfile.single -t sec-demo:single .
docker build -t sec-demo:multi .
docker images sec-demo                                  # compare SIZE: single (full python image) vs multi (slim)
docker run -d --name sec -p 8000:8000 sec-demo:multi
docker exec sec whoami                                  # -> appuser (not root)
sleep 12; docker ps --format '{{.Names}} {{.Status}}'   # -> (healthy)
curl localhost:8000/health
# Override CMD arguments (ENTRYPOINT stays uvicorn):  
docker run --rm sec-demo:multi --version
docker rm -f sec; docker rmi sec-demo:single sec-demo:multi
