#!/usr/bin/env bash
# Step 03: ports, env vars, volumes. One line at a time. (Needs Docker Desktop running.)
docker build -t env-api:v1 .

# --- PORTS: -p HOST:CONTAINER ---------------------------------------------------------------
docker run -d --name a -p 8000:8000 env-api:v1        # reachable at localhost:8000
docker run -d --name b -p 9000:8000 env-api:v1        # same app, host port 9000 -> container 8000
curl localhost:8000/health; curl localhost:9000/health
docker rm -f a b                                      # -f = stop + remove

# --- ENV VARS: -e one var, --env-file many --------------------------------------------------
docker run -d --name c -p 8000:8000 -e MODEL_NAME=cli-model env-api:v1
curl localhost:8000/info                              # model_name = cli-model
docker rm -f c
docker run -d --name d -p 8000:8000 --env-file sample.env env-api:v1
curl localhost:8000/info; docker logs d
docker rm -f d

# --- VOLUMES: data that outlives the container -----------------------------------------------
# Named volume: Docker manages where it lives. Best for databases / app state.
docker run -d --name e -p 8000:8000 -v api-data:/data env-api:v1
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text":"great"}'
docker rm -f e                                        # container gone...
docker run -d --name e2 -p 8000:8000 -v api-data:/data env-api:v1
curl localhost:8000/info                              # ...visits is still 1: the volume survived
docker rm -f e2; docker volume rm api-data

# Bind mount: a HOST folder appears inside the container. Best for development / sharing files.
mkdir -p ./host-data
docker run -d --name f -p 8000:8000 -v "$(pwd)/host-data:/data" env-api:v1
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text":"great"}'
cat ./host-data/visits.txt                            # the file is on YOUR disk
docker rm -f f; rm -rf ./host-data; docker rmi env-api:v1
