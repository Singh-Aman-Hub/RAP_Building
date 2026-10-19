FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
# BUG: uvicorn's default host is 127.0.0.1 = "inside this container only". -p 8000:8000 cannot reach it.
CMD ["uvicorn", "app.main:app", "--port", "8000"]
