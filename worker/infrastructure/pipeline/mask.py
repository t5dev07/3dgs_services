"""Person masking via YOLOv8n — generates per-frame PNG masks for COLMAP."""
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .context import PipelineContext, ProgressCallback

log = logging.getLogger(__name__)

MODEL_PATH = Path("/app/models/yolov8n.onnx")
PERSON_CLASS = 0
CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640        # YOLOv8 standard input resolution
BBOX_PAD = 30           # pixel padding around detected bounding boxes


# ---------------------------------------------------------------------------
# ONNX runtime loader
# ---------------------------------------------------------------------------

def _load_session():
    """Load YOLOv8n ONNX session. Returns None if model not found."""
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        log.warning("onnxruntime not installed, skipping person masking")
        return None

    if not MODEL_PATH.exists():
        log.warning("YOLOv8n model not found at %s, skipping masking", MODEL_PATH)
        return None

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
        active = session.get_providers()[0]
        log.info("YOLOv8n loaded — provider: %s", active)
        return session
    except Exception as exc:
        log.warning("Failed to load YOLOv8n: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _preprocess(img: np.ndarray) -> np.ndarray:
    """Resize + normalise image for YOLOv8 input. Returns (1,3,640,640) float32."""
    resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    rgb = resized[:, :, ::-1]          # BGR → RGB
    norm = rgb.astype(np.float32) / 255.0
    chw = np.transpose(norm, (2, 0, 1))
    return np.expand_dims(chw, 0)      # (1,3,640,640)


def _nms(boxes: list, iou_thresh: float) -> list:
    """Simple NMS. Each entry: (x1,y1,x2,y2,score)."""
    if not boxes:
        return []
    boxes_arr = sorted(boxes, key=lambda b: -b[4])
    kept = []
    while boxes_arr:
        best = boxes_arr.pop(0)
        kept.append(best)
        bx1, by1, bx2, by2 = best[:4]
        remaining = []
        for b in boxes_arr:
            ix1 = max(bx1, b[0])
            iy1 = max(by1, b[1])
            ix2 = min(bx2, b[2])
            iy2 = min(by2, b[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_best = (bx2 - bx1) * (by2 - by1)
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = area_best + area_b - inter
            if union <= 0 or inter / union < iou_thresh:
                remaining.append(b)
        boxes_arr = remaining
    return kept


def _detect_persons(session, img: np.ndarray) -> list[tuple]:
    """Run YOLOv8n inference and return person bounding boxes in image coords.

    Returns list of (x1, y1, x2, y2, score).
    """
    h, w = img.shape[:2]
    blob = _preprocess(img)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})[0]  # (1, 84, 8400)

    # Reshape: (8400, 84) where 84 = 4 bbox + 80 class scores
    preds = output[0].T

    boxes = []
    for pred in preds:
        scores = pred[4:]
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])
        if class_id != PERSON_CLASS or score < CONF_THRESHOLD:
            continue
        cx, cy, bw, bh = pred[:4]
        x1 = int((cx - bw / 2) / INPUT_SIZE * w)
        y1 = int((cy - bh / 2) / INPUT_SIZE * h)
        x2 = int((cx + bw / 2) / INPUT_SIZE * w)
        y2 = int((cy + bh / 2) / INPUT_SIZE * h)
        boxes.append((x1, y1, x2, y2, score))

    return _nms(boxes, IOU_THRESHOLD)


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------

def run(ctx: PipelineContext, on_progress: ProgressCallback) -> None:
    """Generate per-frame PNG masks for detected persons.

    COLMAP convention: mask filename = {image_name}.png
    COLMAP mask polarity: white (255) = feature extraction OK, black (0) = ignored.

    Strategy: white background (all room pixels usable), black rectangles over
    detected person bounding boxes → COLMAP ignores person regions entirely.

    Non-fatal: if model is missing or inference fails, logs a warning and
    continues without masking. COLMAP will run on unmasked frames.
    """
    session = _load_session()
    if session is None:
        return

    on_progress("mask", 16, "Detecting persons for COLMAP masking...")

    frames = sorted(ctx.frames_dir.glob("frame_*.jpg"))
    if not frames:
        log.warning("[%s] No frames found for masking", ctx.job_id)
        return

    ctx.masks_dir.mkdir(parents=True, exist_ok=True)

    total_masked = 0
    errors = 0

    for frame_path in frames:
        try:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            boxes = _detect_persons(session, img)
            if not boxes:
                continue

            h, w = img.shape[:2]
            # White (255) = COLMAP extracts features (room); black (0) = COLMAP ignores (person)
            mask = np.full((h, w), 255, dtype=np.uint8)
            for x1, y1, x2, y2, _ in boxes:
                x1p = max(0, x1 - BBOX_PAD)
                y1p = max(0, y1 - BBOX_PAD)
                x2p = min(w, x2 + BBOX_PAD)
                y2p = min(h, y2 + BBOX_PAD)
                cv2.rectangle(mask, (x1p, y1p), (x2p, y2p), 0, -1)

            mask_path = ctx.masks_dir / f"{frame_path.name}.png"
            cv2.imwrite(str(mask_path), mask)
            total_masked += 1

        except Exception as exc:
            errors += 1
            log.debug("Mask error on %s: %s", frame_path.name, exc)

    on_progress("mask", 18, f"Masked {total_masked}/{len(frames)} frames")
    log.info(
        "[%s] Person masking: %d/%d frames masked, %d errors",
        ctx.job_id, total_masked, len(frames), errors,
    )
