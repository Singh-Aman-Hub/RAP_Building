FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
# BUG: this runs a script that finishes instantly, so PID 1 exits and the container stops (exit code 0).
# A container lives exactly as long as its main process.
CMD ["python", "-c", "print('starting up...')"]
