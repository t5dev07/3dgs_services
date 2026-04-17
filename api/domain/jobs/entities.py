from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class JobStage(str, Enum):
    QUEUED = "queued"
    EXTRACT = "extract"
    MASK = "mask"
    COLMAP = "colmap"
    DEPTH = "depth"
    TRAIN = "train"
    POSTPROCESS = "postprocess"
    VIDEO = "video"
    DONE = "done"


@dataclass
class Job:
    id: str
    status: JobStatus
    stage: JobStage
    input_key: str                    # storage key for uploaded video
    quality_preset: str = "balanced"  # fast | balanced | high | ultra
    progress: int = 0
    message: str = ""
    ply_key: Optional[str] = None     # storage key for model.ply
    splat_key: Optional[str] = None   # storage key for model.splat (web format)
    preview_key: Optional[str] = None # storage key for preview.mp4
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "stage": self.stage.value,
            "input_key": self.input_key,
            "quality_preset": self.quality_preset,
            "progress": self.progress,
            "message": self.message,
            "ply_key": self.ply_key or "",
            "splat_key": self.splat_key or "",
            "preview_key": self.preview_key or "",
            "error": self.error or "",
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        return cls(
            id=data["id"],
            status=JobStatus(data["status"]),
            stage=JobStage(data["stage"]),
            input_key=data.get("input_key", ""),
            quality_preset=data.get("quality_preset", "balanced"),
            progress=int(data.get("progress", 0)),
            message=data.get("message", ""),
            ply_key=data.get("ply_key") or None,
            splat_key=data.get("splat_key") or None,
            preview_key=data.get("preview_key") or None,
            error=data.get("error") or None,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )
