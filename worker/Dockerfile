FROM grzgrzgrz/gaussian-splatting:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .
RUN uv pip install --system .

COPY . .

# ONNX models are mounted at runtime via docker-compose volume: ./models:/app/models
# Run scripts/download_models.py on the host to populate ./models/ before starting.

CMD ["bash", "-c", \
     "Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset & \
      export DISPLAY=:99 && \
      python3 -m celery -A entrypoints.consumer.tasks worker --loglevel=info --concurrency=1"]
