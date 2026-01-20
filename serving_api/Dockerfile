FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY serving_api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

CMD ["python", "-m", "serving_api.app"]
