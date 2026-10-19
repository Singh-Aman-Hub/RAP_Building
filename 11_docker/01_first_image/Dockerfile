# ---- Step 01: the smallest Dockerfile that works ----
# FROM: the base image we build on. python:3.12-slim = Debian + Python 3.12, without extra build tools.
FROM python:3.12-slim

# WORKDIR: create /code and make it the current directory for all later instructions (and at runtime).
WORKDIR /code

# COPY <from-host> <to-image>: copy files from the build context (this folder) into the image.
COPY requirements.txt .

# RUN: execute a shell command at BUILD time. The result is saved as a new image layer.
# --no-cache-dir: do not keep pip's download cache in the image (smaller image).
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (after the install -- see step 02 for why the order matters).
COPY app ./app

# EXPOSE: documentation only! It does NOT publish the port. Publishing happens with `docker run -p`.
EXPOSE 8000

# CMD: the default command when a container starts. Exec form (JSON list) = no shell, signals work.
# --host 0.0.0.0 is ESSENTIAL: 127.0.0.1 inside a container is unreachable from outside it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
