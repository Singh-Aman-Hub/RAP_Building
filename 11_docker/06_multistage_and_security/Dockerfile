# ======================= Stage 1: builder =======================
# Has compilers / build tools if wheels need compiling. Everything here is THROWN AWAY except what we COPY out.
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
# --prefix=/install puts installed packages in one folder that we can copy wholesale into the final image.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ======================= Stage 2: runtime =======================
FROM python:3.12-slim AS runtime

# Security: create an unprivileged user. If the app is hacked, the attacker is not root in the container.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /code
# Copy ONLY the installed packages from the builder stage (no pip cache, no build tools).
COPY --from=builder /install /usr/local
# --chown so the unprivileged user owns its code.
COPY --chown=appuser:appuser app ./app

# Python hygiene: no .pyc files; unbuffered stdout so `docker logs` shows lines immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

# Docker will mark the container "unhealthy" if /health stops answering.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1

# ENTRYPOINT = the fixed program; CMD = its default arguments (easy to override at `docker run`).
ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]
