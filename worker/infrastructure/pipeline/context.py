from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# stage, progress (0-100), message
ProgressCallback = Callable[[str, int, str], None]


@dataclass
class PipelineContext:
    job_id: str
    video_path: Path
    out_dir: Path
    quality_preset: str
    iterations: int | None  # None = use preset default

    # Derived paths — set once out_dir is known
    frames_dir: Path = field(init=False)
    masks_dir: Path = field(init=False)
    depths_dir: Path = field(init=False)
    colmap_dir: Path = field(init=False)
    ply_path: Path = field(init=False)
    splat_path: Path = field(init=False)
    preview_path: Path = field(init=False)
    log_file: Path = field(init=False)

    def __post_init__(self) -> None:
        self.frames_dir = self.out_dir / "frames"
        self.masks_dir = self.out_dir / "masks"
        self.depths_dir = self.out_dir / "depths"
        self.colmap_dir = self.out_dir / "colmap"
        self.ply_path = self.out_dir / "model.ply"
        self.splat_path = self.out_dir / "model.splat"
        self.preview_path = self.out_dir / "preview.mp4"
        self.log_file = self.out_dir / "log.txt"

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)
        self.colmap_dir.mkdir(exist_ok=True)
