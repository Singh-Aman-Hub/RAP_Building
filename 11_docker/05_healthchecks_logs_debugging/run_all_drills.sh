#!/usr/bin/env bash
# Builds every broken_* image, runs it, prints what a beginner would see, and cleans up. (Needs Docker Desktop.)
set -u
for d in broken_*/; do
  name=${d%/}
  echo "=================== $name ==================="
  docker build -q -t "drill:$name" "$d" >/dev/null
  docker run -d --name drill -p 8000:8000 "drill:$name" >/dev/null
  sleep 3
  docker ps -a --filter name=drill --format 'STATUS: {{.Status}}'
  echo "--- logs ---"; docker logs drill 2>&1 | tail -5
  echo "--- curl ---"; curl -s -m 3 localhost:8000/health || echo "curl FAILED"
  echo
  docker rm -f drill >/dev/null; docker rmi "drill:$name" >/dev/null
done
