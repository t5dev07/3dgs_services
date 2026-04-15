from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from settings import settings
from logging_config import setup_logging
from entrypoints.http.routers import jobs, uploads

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = aioredis.from_url(
        settings.redis_job_store_url,
        decode_responses=False,
    )
    yield
    await app.state.redis.aclose()


app = FastAPI(title="3DGS API", version="1.0.0", lifespan=lifespan)

# Serve output artifacts directly (ply, preview video)
# Path: /files/outputs/{job_id}/model.ply  etc.
app.mount(
    "/files",
    StaticFiles(directory=settings.storage.data_dir),
    name="files",
)

app.include_router(uploads.router)
app.include_router(jobs.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
