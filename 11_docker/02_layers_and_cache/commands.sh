#!/usr/bin/env bash
# Step 02: watch the layer cache work. Run one line at a time. (Needs Docker Desktop running.)
docker build -f Dockerfile.bad_order  -t cache-demo:bad  .   # -f = which Dockerfile to use
docker build -f Dockerfile.good_order -t cache-demo:good .
echo "# edit a comment in app/main.py, then rebuild both and compare the output:" 
docker build -f Dockerfile.bad_order  -t cache-demo:bad  .   # pip install runs AGAIN (slow)
docker build -f Dockerfile.good_order -t cache-demo:good .   # steps 1-4 say "CACHED" (fast)
docker history cache-demo:good                               # one row per layer, with sizes
docker build -f Dockerfile.alpine -t cache-demo:alpine . && docker images cache-demo   # compare sizes
docker rmi cache-demo:bad cache-demo:good cache-demo:alpine  # cleanup
