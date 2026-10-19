FROM python:3.12-slim
WORKDIR /app
# dependencies first: this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MODEL_PATH=/app/artifacts/model.joblib
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
# 0.0.0.0 so the port mapping works from outside the container
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
