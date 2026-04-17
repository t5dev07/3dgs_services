"""Post-training PLY cleanup + .splat format export."""
import logging
from pathlib import Path

import numpy as np

from .context import PipelineContext, ProgressCallback

log = logging.getLogger(__name__)

OPACITY_THRESHOLD = 0.005   # sigmoid scale — removes near-invisible Gaussians
SH_C0 = 0.28209479177387814  # degree-0 spherical harmonics normalization constant

_PLY_DTYPE_MAP = {
    "float": np.float32, "float32": np.float32,
    "double": np.float64, "float64": np.float64,
    "int": np.int32, "int32": np.int32,
    "uint": np.uint32, "uint32": np.uint32,
    "uchar": np.uint8, "uint8": np.uint8,
    "char": np.int8, "int8": np.int8,
}


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

def _parse_ply_header(ply_path: Path) -> tuple[int, list[tuple[str, type]], int]:
    """Return (num_vertices, [(name, numpy_dtype), ...], header_byte_length)."""
    num_vertices = 0
    properties: list[tuple[str, type]] = []
    in_vertex = False

    with open(ply_path, "rb") as fh:
        header_bytes = 0
        while True:
            line = fh.readline()
            header_bytes += len(line)
            tok = line.decode("ascii", errors="replace").strip().split()

            if not tok:
                continue
            if tok[0] == "end_header":
                break
            elif tok[0] == "element":
                in_vertex = (tok[1] == "vertex")
                if in_vertex:
                    num_vertices = int(tok[2])
            elif tok[0] == "property" and in_vertex and tok[1] != "list":
                dtype = _PLY_DTYPE_MAP.get(tok[1], np.float32)
                properties.append((tok[2], dtype))

    return num_vertices, properties, header_bytes


def _read_ply(ply_path: Path) -> tuple[np.ndarray, list[str], int]:
    """Load PLY into structured numpy array. Returns (data, prop_names, header_bytes)."""
    n, props, hdr = _parse_ply_header(ply_path)
    dt = np.dtype([(name, t) for name, t in props])
    with open(ply_path, "rb") as fh:
        fh.seek(hdr)
        raw = fh.read(n * dt.itemsize)
    data = np.frombuffer(raw, dtype=dt).copy()  # copy → writeable
    return data, [p[0] for p in props], hdr


def _write_ply(ply_path: Path, original_path: Path, data: np.ndarray) -> None:
    """Write filtered Gaussians back to PLY with corrected vertex count."""
    with open(original_path, "rb") as src:
        header_lines: list[bytes] = []
        while True:
            line = src.readline()
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("element vertex"):
                header_lines.append(f"element vertex {len(data)}\n".encode())
            elif text == "end_header":
                header_lines.append(b"end_header\n")
                break
            else:
                header_lines.append(line)

    with open(ply_path, "wb") as dst:
        for h in header_lines:
            dst.write(h)
        dst.write(data.tobytes())


# ---------------------------------------------------------------------------
# .splat conversion  (antimatter15 format — 32 bytes/Gaussian)
# ---------------------------------------------------------------------------

def _to_splat(data: np.ndarray, prop_names: list[str]) -> bytes:
    """
    Convert filtered Gaussian array to .splat binary.

    Layout per Gaussian (32 bytes):
      pos   x,y,z       3 × float32  = 12 bytes
      scale sx,sy,sz    3 × float32  = 12 bytes  (linear: exp of log-scale)
      rgba  r,g,b,a     4 × uint8    =  4 bytes
      rot   qx,qy,qz,qw 4 × uint8   =  4 bytes  (q*127.5 + 127.5)

    Sorted by (opacity × volume) descending for correct alpha compositing.
    """
    n = len(data)

    def _f32(name: str) -> np.ndarray:
        return data[name].astype(np.float32)

    pos = np.stack([_f32("x"), _f32("y"), _f32("z")], axis=1)          # (N,3)
    scales = np.exp(np.stack([
        _f32("scale_0"), _f32("scale_1"), _f32("scale_2"),
    ], axis=1))                                                            # (N,3) linear

    opacity_raw = _f32("opacity")
    alpha = 1.0 / (1.0 + np.exp(-opacity_raw))                           # sigmoid

    r = np.clip(0.5 + SH_C0 * _f32("f_dc_0"), 0.0, 1.0)
    g = np.clip(0.5 + SH_C0 * _f32("f_dc_1"), 0.0, 1.0)
    b = np.clip(0.5 + SH_C0 * _f32("f_dc_2"), 0.0, 1.0)
    rgba = np.stack([
        (r * 255).astype(np.uint8),
        (g * 255).astype(np.uint8),
        (b * 255).astype(np.uint8),
        (alpha * 255).astype(np.uint8),
    ], axis=1)                                                             # (N,4) uint8

    q = np.stack([_f32("rot_0"), _f32("rot_1"), _f32("rot_2"), _f32("rot_3")], axis=1)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    rot_u8 = np.clip(q_norm * 127.5 + 127.5, 0, 255).astype(np.uint8)   # (N,4)

    # Sort: largest/most-visible first
    sort_key = alpha * np.exp(
        _f32("scale_0") + _f32("scale_1") + _f32("scale_2")
    )
    order = np.argsort(-sort_key)

    # Vectorised byte assembly: (N, 32)
    pos_b   = np.ascontiguousarray(pos[order]).view(np.uint8).reshape(n, 12)
    scale_b = np.ascontiguousarray(scales[order]).view(np.uint8).reshape(n, 12)
    rgba_b  = rgba[order]
    rot_b   = rot_u8[order]

    record = np.concatenate([pos_b, scale_b, rgba_b, rot_b], axis=1)    # (N, 32)
    return record.tobytes()


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------

def run(ctx: PipelineContext, on_progress: ProgressCallback) -> None:
    if not ctx.ply_path.exists():
        log.warning("[%s] model.ply not found, skipping postprocess", ctx.job_id)
        return

    on_progress("postprocess", 78, "Filtering model and exporting .splat...")

    try:
        data, prop_names, _ = _read_ply(ctx.ply_path)
    except Exception as exc:
        log.warning("[%s] PLY parse failed: %s — skipping postprocess", ctx.job_id, exc)
        return

    if "opacity" not in prop_names:
        log.warning("[%s] No opacity field in PLY, skipping postprocess", ctx.job_id)
        return

    original_count = len(data)

    # ------------------------------------------------------------------
    # Opacity filter: keep Gaussians with sigmoid(opacity_raw) > threshold
    # ------------------------------------------------------------------
    opacity_raw = data["opacity"].astype(np.float32)
    alpha = 1.0 / (1.0 + np.exp(-opacity_raw))
    data = data[alpha > OPACITY_THRESHOLD]
    kept = len(data)

    log.info(
        "[%s] Opacity filter %.4f: %d → %d Gaussians (removed %d)",
        ctx.job_id, OPACITY_THRESHOLD, original_count, kept, original_count - kept,
    )

    if kept == 0:
        log.warning("[%s] All Gaussians removed by opacity filter — keeping original PLY", ctx.job_id)
        return

    # ------------------------------------------------------------------
    # Write filtered PLY (in-place)
    # ------------------------------------------------------------------
    try:
        _write_ply(ctx.ply_path, ctx.ply_path, data)
    except Exception as exc:
        log.warning("[%s] PLY write failed: %s", ctx.job_id, exc)

    # ------------------------------------------------------------------
    # Export .splat
    # ------------------------------------------------------------------
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"}
    if not required.issubset(set(prop_names)):
        missing = required - set(prop_names)
        log.warning("[%s] Missing PLY properties for .splat: %s", ctx.job_id, missing)
        return

    try:
        splat_bytes = _to_splat(data, prop_names)
        ctx.splat_path.write_bytes(splat_bytes)
        size_mb = ctx.splat_path.stat().st_size / (1024 ** 2)
        on_progress("postprocess", 82, f"Model ready — {size_mb:.1f} MB .splat")
        log.info("[%s] .splat written: %s (%.1f MB)", ctx.job_id, ctx.splat_path, size_mb)
    except Exception as exc:
        log.warning("[%s] .splat export failed: %s", ctx.job_id, exc)
