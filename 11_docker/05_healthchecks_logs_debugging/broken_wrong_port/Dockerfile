FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
# SYMPTOM TARGET: the app listens on 8080 inside the container, but we will map 8000:8000.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
