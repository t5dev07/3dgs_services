FROM grzgrzgrz/gaussian-splatting:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .
RUN uv pip install --system .

COPY . .

# tmp_process_video.py is reference only — not needed at runtime
# Celery worker, single concurrency to prevent GPU contention
CMD ["python3", "-m", "celery", "-A", "entrypoints.consumer.tasks", "worker", \
     "--loglevel=info", "--concurrency=1"]
