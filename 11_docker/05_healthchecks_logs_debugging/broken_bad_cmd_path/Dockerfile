FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
# BUG: the module is app.main, not src.main.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
