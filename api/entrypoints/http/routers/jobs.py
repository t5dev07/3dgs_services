import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from domain.jobs.schemas import JobResponse
from domain.jobs.service import JobService
from entrypoints.http.deps import get_job_service

log = logging.getLogger(__name__)
router = APIRouter(tags=["jobs"])


def _to_response(job, svc: JobService) -> JobResponse:
    ply_url = svc.get_download_url(job.ply_key) if job.ply_key else None
    preview_url = svc.get_download_url(job.preview_key) if job.preview_key else None
    return JobResponse(
        id=job.id,
        status=job.status.value,
        stage=job.stage.value,
        progress=job.progress,
        message=job.message,
        ply_url=ply_url,
        preview_url=preview_url,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs(svc: JobService = Depends(get_job_service)):
    jobs = await svc.list_all()
    return [_to_response(j, svc) for j in jobs]


@router.get("/job/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, svc: JobService = Depends(get_job_service)):
    job = await svc.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id!r} not found")
    return _to_response(job, svc)


@router.get("/job/{job_id}/download/ply")
async def download_ply(job_id: str, svc: JobService = Depends(get_job_service)):
    job = await svc.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.ply_key:
        raise HTTPException(404, "PLY not ready")
    path = svc.get_download_path(job.ply_key)
    if not path.exists():
        raise HTTPException(404, "PLY file missing from storage")
    return FileResponse(path, filename="model.ply", media_type="application/octet-stream")


@router.get("/job/{job_id}/download/preview")
async def download_preview(job_id: str, svc: JobService = Depends(get_job_service)):
    job = await svc.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.preview_key:
        raise HTTPException(404, "Preview not ready")
    path = svc.get_download_path(job.preview_key)
    if not path.exists():
        raise HTTPException(404, "Preview file missing from storage")
    return FileResponse(path, filename="preview.mp4", media_type="video/mp4")
