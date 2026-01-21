# CycleGAN Port + Benchmarking — Progress Log

Date: 2026-01-21 (UTC)

## Summary
We set up a reproducible Docker workflow for both the modern (PyTorch) port and a legacy TensorFlow environment, then validated that:
- The modern container can import/run the PyTorch training CLI.
- A legacy TF container can load and execute the GraphDef benchmark on the checked-in `optimized-graph.pb`.
- The checked-in `optimized-graph.pb` is *not* compatible with the MSc-era 256×256 baseline shape; it is hard-coded for 600×600.

## What We Did

### 1) Modern container (PyTorch / `uv`)
- Added a `Dockerfile` that:
  - Installs `uv`
  - Installs repo deps via `uv sync --frozen`
  - Installs CPU-only PyTorch wheels (`torch`, `torchvision`) from the PyTorch CPU index
- Added `docker-compose.yml` service `cyclegan`.
- Fixed a venv masking issue:
  - The bind mount `.:/app` was masking the image’s `/app/.venv` with the host repo’s `.venv`.
  - Added an anonymous volume mount at `/app/.venv` so the container’s venv remains intact.

**Smoke tests (inside container)**
- Verified runtime:
  - Python `3.11.14`
  - `torch 2.10.0+cpu`, `torchvision 0.25.0+cpu`
- Verified training entrypoint starts:
  - `cyclegan-train --help`
- Verified forward pass runs:
  - `GeneratorResNet9()` produces output shape `(1, 3, 256, 256)`

### 2) Legacy container (TensorFlow 1.x)
Goal: run inference benchmarks comparable to MSc-era baselines.

- Attempted TF1.7/Py2 image; tag was unavailable on Docker Hub.
- Implemented a “legacy” container using `tensorflow/tensorflow:1.15.0-py3` (Python 3.6.8, TF 1.15.0).
- Added `Dockerfile.legacy` + `docker-compose.yml` service `cyclegan_legacy`.

### 3) Benchmarking `optimized-graph.pb`
The repo contains a checked-in GraphDef: `optimized-graph.pb`.

#### a) Python benchmark run (legacy container)
- Ran: `pb_benchmark_legacy.py` against `datasets/horse2zebra/testA`.
- Important: this graph requires input shape `(?, 600, 600, 3)`, so `--crop_size 600` is required.

Result (sample run):
- `Images: 5`
- `All: 9.6122s (0.5202 fps)`

#### b) Graph inspection
We inspected `optimized-graph.pb` in the legacy container:
- Placeholder: `inputA` dtype `float32`, shape `(-1, 600, 600, 3)`
- Found tensors:
  - `inputA:0 -> (?, 600, 600, 3)`
  - `a2b_generator/output_image:0 -> (?, 600, 600, 3)`
- Not found in this GraphDef:
  - `inputB:0`
  - `b2a_generator/output_image:0`

This explains why MSc-era commands expecting `1,256,256,3` won’t match this particular checked-in `.pb`.

## Files Added / Modified (Git)

### Modified (tracked)
Per `git diff --stat`:
- `README.md`
- `download_dataset.sh`
- `image_utils.py`
- `pb_a2b.py`

### New / Untracked (present in working tree)
Per `git status --porcelain` (not exhaustive details, but key new items):
- `Dockerfile`
- `docker-compose.yml`
- `Dockerfile.legacy`
- `pb_benchmark.py`
- `pb_benchmark_legacy.py`
- `pyproject.toml`, `uv.lock`
- `cyclegan_torch/` (PyTorch port package + scripts)
- `configs/`, `docs/`, `scripts/`
- Various local/derived artifacts: `outputs_torch_*`, `cyclegan_torch.egg-info/`, `.venv_tf/`

## Next Decisions / Open Questions
1) **Baseline alignment**: MSc-era baselines appear to assume `256×256` GraphDef input shape. We need to locate or regenerate the correct 256×256 optimized graph for apples-to-apples comparisons.
2) **Which benchmark should be “authoritative”?**
   - MSc docs reference TensorFlow’s `benchmark_model` (Bazel-built) and python scripts (`pb_a2b.py`).
   - We currently have `pb_benchmark_legacy.py` for timing loops on `.jpg` folders.
3) **GPU vs CPU**: Current Docker runs are CPU-only (TF logs show no CUDA driver). If GPU baselines exist, we’ll need an NVIDIA-enabled runtime and matching container.

## Handy Commands
- Modern container shell: `docker compose exec cyclegan bash`
- Legacy container shell: `docker compose exec cyclegan_legacy bash`
- Legacy benchmark example:
  - `docker compose exec -T cyclegan_legacy python pb_benchmark_legacy.py --graph optimized-graph.pb --dataset datasets/horse2zebra/testA --crop_size 600 --limit 100`
- Inspect pb placeholder shapes:
  - `docker compose exec -T cyclegan_legacy python -c "import tensorflow as tf; ..."`
