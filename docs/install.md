# Install (uv)

This repo is designed to be installed with `uv`, and to let users choose the correct PyTorch wheels
for their hardware (CPU vs CUDA vs Apple Silicon / MPS).

PyTorch uses the same dependency names (`torch`, `torchvision`) across platforms; the difference is
*which wheel index/source* you install from.

## 0) Create the venv

```bash
uv venv
```

## 1) Install project deps (without PyTorch)

```bash
uv pip install -e .
```

## 2) Install PyTorch for your hardware

Pick **one**:

### macOS (Apple Silicon / MPS)

```bash
uv add --optional mps torch torchvision
uv sync --extra mps
```

### Linux CPU-only

```bash
uv add --optional cpu torch torchvision --default-index https://download.pytorch.org/whl/cpu
uv sync --extra cpu
```

### Linux + NVIDIA CUDA

Choose the right CUDA wheel index for your system (example: cu121):

```bash
uv add --optional cuda torch torchvision --default-index https://download.pytorch.org/whl/cu121
uv sync --extra cuda
```

## Notes

- `uv add` updates `pyproject.toml` and (when you run `uv sync`) locks and installs.
- You generally should **not** install multiple extras at once (e.g., `cpu` and `cuda`).
- The training script selects device at runtime: CUDA if available, else MPS, else CPU.
