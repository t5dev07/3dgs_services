FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.3

# Pinned to 1.1.3 because dn-splatter requires nerfstudio==1.1.3 exactly.
# Base image ships: gsplat, splatfacto, COLMAP, GLOMAP, ffmpeg, torch CUDA 11.8.
# Missing: git + CA certs (pip install from github) + build-essential (PyMCubes 0.1.2 has no wheel).

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        ca-certificates curl build-essential python3-dev \
        libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
        libswscale-dev libswresample-dev libavfilter-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# dn-splatter source is vendored into the build context (worker/vendor_dn_splatter/)
# because both git-clone and curl-from-github fail from inside the Docker network
# on this host. Vendor-copy sidesteps the outbound TLS/connectivity issue entirely.
COPY vendor_dn_splatter /tmp/dn-splatter

# ViewCrafter source vendored (Phase 2). Opt-in: skipped at runtime unless
# settings.pipeline.viewcrafter.enabled=true. Weights (~6GB) must be mounted
# at /app/models/viewcrafter/ (see scripts/download_models.py).
COPY vendor_viewcrafter /app/vendor/viewcrafter

# Vendored wheels: pytorch3d pre-built for torch 2.1 + CUDA 11.8 + Python 3.10.
# Downloaded on host because Docker daemon's network can't reach FAIR/GitHub.
COPY vendor_wheels /tmp/vendor_wheels

COPY pyproject.toml .
RUN uv pip install --system . && \
    uv pip install --system /tmp/dn-splatter && \
    rm -rf /tmp/dn-splatter

# ViewCrafter Python deps. Torch/xformers come from base image (2.1.2 + CUDA 11.8);
# we skip the torch==1.13.1 pin in ViewCrafter's requirements.txt.
# Use av>=12 (has wheels) instead of av==10.0.0 which requires Cython<3 build.
# pytorch3d v0.7.5 installed from vendored wheel.
RUN uv pip install --system \
        "av>=12.0" "decord>=0.6" einops "imageio>=2.27" imageio-ffmpeg \
        kornia "moviepy>=1,<2" open-clip-torch \
        "pytorch-lightning>=1.9,<2.3" PyYAML roma timm trimesh omegaconf && \
    uv pip install --system --no-deps /tmp/vendor_wheels/pytorch3d-*.whl && \
    rm -rf /tmp/vendor_wheels

COPY . .

# ONNX + ViewCrafter models are mounted at runtime via docker-compose volume: ./models:/app/models
# Run scripts/download_models.py on the host to populate ./models/ before starting.

CMD ["python3", "-m", "celery", "-A", "entrypoints.consumer.tasks", \
     "worker", "--loglevel=info", "--concurrency=1"]
