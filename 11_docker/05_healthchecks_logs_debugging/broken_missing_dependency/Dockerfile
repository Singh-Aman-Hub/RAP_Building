FROM python:3.12-slim
WORKDIR /code
# BUG: requirements.txt is never installed, so `import fastapi` fails at start-up.
COPY app ./app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
