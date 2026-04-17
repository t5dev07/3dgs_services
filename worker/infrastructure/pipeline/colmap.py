"""COLMAP SfM runner — adapted from tmp_process_video.py."""
import logging
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Optional

from .context import PipelineContext, ProgressCallback

log = logging.getLogger(__name__)


def _should_use_gpu() -> bool:
    return os.environ.get("COLMAP_USE_GPU", "1").strip().lower() not in ("0", "false", "no")


def _db_stats(db_path: Path) -> dict:
    stats = {"images": 0, "keypoints": 0, "descriptors": 0}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM images")
        row = cur.fetchone()
        stats["images"] = int(row[0]) if row else 0
        for tbl in ("keypoints", "descriptors"):
            cur.execute(f"SELECT COALESCE(SUM(rows), 0) FROM {tbl}")
            row = cur.fetchone()
            stats[tbl] = int(row[0]) if row and row[0] else 0
        conn.close()
    except Exception as exc:
        log.warning("Could not read COLMAP DB stats: %s", exc)
    return stats


def _reset_workspace(db_path: Path, sparse_dir: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)


def _run_colmap(
    frames_dir: Path,
    colmap_dir: Path,
    use_gpu: bool,
    matcher: str,
    sequential_overlap: int,
    loop_closure: bool,
    max_num_features: Optional[int],
    max_image_size: Optional[int],
    guided_matching: bool,
    log_file: Path,
    masks_dir: Optional[Path] = None,
) -> bool:
    db_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    gpu_flag = "1" if use_gpu else "0"

    def _exec(cmd: list) -> subprocess.CompletedProcess:
        with open(log_file, "a") as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    # Feature extraction
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", gpu_flag,
    ]
    if max_num_features:
        cmd += ["--SiftExtraction.max_num_features", str(max_num_features)]
    if max_image_size:
        cmd += ["--SiftExtraction.max_image_size", str(max_image_size)]
    if masks_dir and masks_dir.exists() and list(masks_dir.glob("*.png")):
        cmd += ["--ImageReader.mask_path", str(masks_dir)]
        log.info("COLMAP masking enabled: %s", masks_dir)

    result = _exec(cmd)
    if result.returncode != 0:
        if use_gpu:
            log.warning("GPU feature extraction failed, retrying with CPU")
            _reset_workspace(db_path, sparse_dir)
            return _run_colmap(
                frames_dir, colmap_dir, False, matcher, sequential_overlap,
                loop_closure, max_num_features, max_image_size, guided_matching, log_file,
                masks_dir=masks_dir,
            )
        return False

    stats = _db_stats(db_path)
    log.info("COLMAP DB: images=%d kp=%d desc=%d", stats["images"], stats["keypoints"], stats["descriptors"])
    if stats["images"] < 2 or stats["keypoints"] == 0:
        log.error("Insufficient features extracted")
        return False

    # Feature matching
    matcher_key = matcher.strip().lower()
    if matcher_key not in ("exhaustive", "sequential", "sequential_loop"):
        matcher_key = "sequential_loop"

    if matcher_key in ("sequential", "sequential_loop"):
        base_cmd = [
            "colmap", "sequential_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", gpu_flag,
            "--SequentialMatching.overlap", str(sequential_overlap),
        ]
        if guided_matching:
            base_cmd += ["--SiftMatching.guided_matching", "1"]
        if matcher_key == "sequential_loop":
            vocab_tree = Path("/app/models/vocab_tree_flickr100K_words256K.bin")
            if vocab_tree.exists():
                base_cmd += [
                    "--SequentialMatching.loop_detection", "1",
                    "--SequentialMatching.vocab_tree_path", str(vocab_tree),
                ]
                log.info("Sequential matching with vocab tree loop closure enabled")
            else:
                log.warning("Vocab tree not found at %s — loop closure disabled", vocab_tree)
        result = _exec(base_cmd)
    else:
        cmd = ["colmap", "exhaustive_matcher", "--database_path", str(db_path), "--SiftMatching.use_gpu", gpu_flag]
        if guided_matching:
            cmd += ["--SiftMatching.guided_matching", "1"]
        result = _exec(cmd)

    if result.returncode != 0:
        if use_gpu:
            log.warning("GPU matching failed, retrying with CPU")
            _reset_workspace(db_path, sparse_dir)
            return _run_colmap(
                frames_dir, colmap_dir, False, matcher, sequential_overlap,
                loop_closure, max_num_features, max_image_size, guided_matching, log_file,
                masks_dir=masks_dir,
            )
        return False

    # Sparse reconstruction
    result = _exec([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.init_min_tri_angle", "4",       # default 16° — relax for walkthrough/slow camera
        "--Mapper.init_min_num_inliers", "50",    # default 100
        "--Mapper.abs_pose_min_num_inliers", "15",
        "--Mapper.abs_pose_min_inlier_ratio", "0.1",
    ])
    if result.returncode != 0:
        return False

    recon_dirs = list(sparse_dir.iterdir())
    if not recon_dirs:
        log.error("COLMAP mapper produced no reconstruction")
        return False

    log.info("COLMAP complete: %d reconstruction(s)", len(recon_dirs))
    return True


def run(ctx: PipelineContext, params: dict, on_progress: ProgressCallback) -> None:
    on_progress("colmap", 20, "Running COLMAP reconstruction...")

    ok = _run_colmap(
        frames_dir=ctx.frames_dir,
        colmap_dir=ctx.colmap_dir,
        use_gpu=_should_use_gpu(),
        matcher=str(params.get("colmap_matcher", "sequential")),
        sequential_overlap=int(params.get("sequential_overlap", 12)),
        loop_closure=bool(params.get("loop_closure", False)),
        max_num_features=int(params.get("max_num_features", 0)) or None,
        max_image_size=int(params.get("max_image_size", 0)) or None,
        guided_matching=bool(params.get("guided_matching", True)),
        log_file=ctx.log_file,
        masks_dir=ctx.masks_dir,
    )
    if not ok:
        raise RuntimeError("COLMAP reconstruction failed — check log for details")

    on_progress("colmap", 40, "COLMAP complete")
    log.info("[%s] COLMAP done", ctx.job_id)
