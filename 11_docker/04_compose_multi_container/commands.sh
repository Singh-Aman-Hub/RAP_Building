#!/usr/bin/env bash
# Step 04: multi-container app with Compose. (Needs Docker Desktop running.)
docker compose config                  # validate + print the final YAML with ${VARS} filled in (no daemon needed)
docker compose up --build -d           # build images, create network+volume, start all services in dependency order
docker compose ps                      # STATUS shows (healthy) once health checks pass
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text":"great"}'
curl localhost:8000/stats              # counter_backend = redis
open http://localhost:8501             # the Streamlit UI (macOS); or visit it in a browser
docker compose logs -f api             # follow API logs (Ctrl+C to stop following)
docker compose exec api sh -c "python -c \"import socket;print(socket.gethostbyname('redis'))\""  # DNS proof
docker compose down                    # stop + remove containers & network (volume kept)
docker compose down -v                 # ...and ALSO delete the named volume (counter resets)
