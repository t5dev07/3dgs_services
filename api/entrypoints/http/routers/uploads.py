import logging

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from domain.jobs.schemas import CreateJobResponse, UploadUrlRequest
from domain.jobs.service import JobService
from entrypoints.http.deps import get_job_service

log = logging.getLogger(__name__)
router = APIRouter(tags=["uploads"])

_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def _check_extension(filename: str) -> None:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext!r}. Allowed: {sorted(_ALLOWED_EXTENSIONS)}")


_VALID_PRESETS = {"fast", "balanced", "high", "ultra"}

@router.post("/upload", response_model=CreateJobResponse, status_code=202)
async def upload_video(
    file: UploadFile = File(...),
    quality: str = Form("balanced"),
    svc: JobService = Depends(get_job_service),
):
    _check_extension(file.filename or "unknown")
    if quality not in _VALID_PRESETS:
        raise HTTPException(400, f"Invalid quality preset: {quality!r}. Valid: {sorted(_VALID_PRESETS)}")
    job = await svc.create_from_upload(file.filename, file.file, quality_preset=quality)
    return CreateJobResponse(job_id=job.id)


@router.post("/upload-url", response_model=CreateJobResponse, status_code=202)
async def upload_from_url(
    body: UploadUrlRequest,
    svc: JobService = Depends(get_job_service),
):
    url = body.url
    _check_extension(url.split("?")[0])  # ignore query params for ext check

    filename = url.split("/")[-1].split("?")[0] or "video.mp4"

    try:
        async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise HTTPException(400, f"Could not fetch URL: HTTP {resp.status_code}")

                import io
                buf = io.BytesIO()
                async for chunk in resp.aiter_bytes(65536):
                    buf.write(chunk)
                buf.seek(0)

        job = await svc.create_from_upload(filename, buf)
        return CreateJobResponse(job_id=job.id)

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Failed to download video from URL: %s", url)
        raise HTTPException(500, f"Download failed: {exc}") from exc
