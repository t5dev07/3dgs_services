from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class JobResponse(BaseModel):
    id: str
    status: str
    stage: str
    progress: int
    message: str
    ply_url: Optional[str] = None
    preview_url: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class CreateJobResponse(BaseModel):
    job_id: str


class UploadUrlRequest(BaseModel):
    url: str
