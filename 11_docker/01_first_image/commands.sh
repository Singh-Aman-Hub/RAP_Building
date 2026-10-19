#!/usr/bin/env bash
# Step 01 commands. Run them ONE AT A TIME from this folder and read each output. (Needs Docker Desktop running.)
docker build -t toy-api:v1 .                      # build image from ./Dockerfile; -t = name:tag
docker images toy-api                             # list it; note the SIZE
docker run -d --name toy -p 8000:8000 toy-api:v1  # -d detached, --name, -p HOST:CONTAINER
docker ps                                         # running containers
curl localhost:8000/health
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text":"I love it"}'
docker logs toy                                   # stdout/stderr of the container
docker stop toy && docker rm toy                  # stop, then delete the container (image stays)
docker rmi toy-api:v1                             # delete the image (cleanup)
