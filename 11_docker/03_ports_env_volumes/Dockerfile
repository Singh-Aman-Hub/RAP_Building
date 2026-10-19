FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app

# ENV sets a DEFAULT inside the image. `docker run -e` or --env-file overrides it at run time.
ENV MODEL_NAME=keyword-sentiment-v1 \
    DATA_DIR=/data \
    LOG_LEVEL=INFO

# VOLUME documents that /data holds state worth persisting (Docker creates an anonymous volume if you forget).
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
