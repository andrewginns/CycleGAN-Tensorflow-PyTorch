# CycleGAN-Tensorflow-PyTorch → Modern Rewrite Proposal

## Goals
- Rebuild this repo in a modern, readable training stack.
- First-class Apple Silicon acceleration (Metal / MPS).
- Keep behavior close to the current implementation where it matters (architecture + losses), while modernizing the data pipeline, training loop, checkpointing, and export.

## Requirements (from thread)
- **Standard PyTorch** training loop (no Lightning).
- **Match TF `tf.contrib.slim` padding semantics** for now (fidelity > "best practice").
- **Skip identity loss** (only adversarial + cycle).
- Keep **TensorBoard logging** (PyTorch `torch.utils.tensorboard` is the industry-standard default).

## Dataset Download: Current Status
The repo’s `download_dataset.sh` no longer works as-is:
- The original Berkeley URL now returns **404** for at least `horse2zebra`.
- The script depends on `unzip`, which isn’t always installed in minimal environments.

Fix implemented in this workspace:
- New `download_dataset.py` tries a known mirror and extracts using Python’s `zipfile`.
- `download_dataset.sh` is now a thin wrapper calling the Python script.

---

## Framework Choice

### Recommendation: **PyTorch (2.x) + native MPS support**
PyTorch is the best fit for “modern understandability” + “accelerates on many GPUs” with Apple Silicon as a primary target.

Why PyTorch:
- **Apple Silicon**: `torch.device("mps")` works out of the box for many ops; `torch.compile` and mixed precision keep improving.
- **CUDA + multi-GPU**: still the best-supported path for NVIDIA, with DDP, FSDP, AMP, etc.
- **Ecosystem**: torchvision transforms/datasets, pretrained backbones, experiment tooling, export paths.

### Why not tinygrad (for this rewrite)
Tinygrad is interesting (and could be a future experiment), but for a CycleGAN production-quality rewrite it’s currently a worse match:
- Less mature coverage for the full training ecosystem (data loading/augmentations, distributed, logging, exporters).
- More edge cases on Apple Metal kernels and operator support.

If you want “tinygrad as a second backend” later, we can design the code so the **model definition is backend-agnostic-ish** (e.g., minimal PyTorch-isms), but the initial port should be PyTorch-first.

---

## What This Repo Actually Implements (from code)

This repository is a **CycleGAN** variant implemented in TF1 + `tf.contrib.slim`.

### Architecture Summary
Source: `models.py`

#### Generator: ResNet (9 residual blocks)
- Input: image tensor `[N, H, W, 3]` in **[-1, 1]**.
- Layers:
  1. `7x7` conv, `dim=64`, stride 1, **batch norm**, ReLU
  2. `3x3` conv, `dim*2`, stride 2, BN, ReLU
  3. `3x3` conv, `dim*4`, stride 2, BN, ReLU
  4. **9 residual blocks** at `dim*4`:
     - `3x3` conv BN ReLU
     - `3x3` conv BN
     - skip connection
  5. `3x3` deconv (transpose conv), `dim*2`, stride 2, BN, ReLU
  6. `3x3` deconv, `dim`, stride 2, BN, ReLU
  7. `7x7` conv to 3 channels, stride 1
  8. `tanh` output

Notes:
- No explicit reflection padding is visible; `tf.contrib.slim.conv2d` defaults to `padding='SAME'`. Per the thread, the PyTorch port should **match Slim padding semantics for now**.
- This is a fidelity choice (sub-optimal vs modern CycleGAN practice, which often uses reflection padding for generators).

#### Discriminator: PatchGAN-style
Source: `models.py`
- Layers:
  1. `4x4` conv, `dim=64`, stride 2, leaky ReLU (no BN)
  2. `4x4` conv, `dim*2`, stride 2, BN, leaky ReLU
  3. `4x4` conv, `dim*4`, stride 2, BN, leaky ReLU
  4. `4x4` conv, `dim*8`, stride 1, BN, leaky ReLU
  5. `4x4` conv to 1 channel, stride 1 (patch logits)

This is the standard CycleGAN discriminator family.

### Training Graph + Losses
Source: `train.py`

Models:
- `G_A2B`, `G_B2A`
- `D_A`, `D_B`

Forward passes:
- `a2b = G_A2B(a_real)`
- `b2a = G_B2A(b_real)`
- `a2b2a = G_B2A(a2b)` (cycle)
- `b2a2b = G_A2B(b2a)` (cycle)

Losses:
- **Adversarial loss uses LSGAN** (MSE to 1/0):
  - `G_A2B`: `MSE(D_B(a2b), 1)`
  - `G_B2A`: `MSE(D_A(b2a), 1)`
  - `D_A`: `MSE(D_A(a_real), 1) + MSE(D_A(b2a_sample), 0)`
  - `D_B`: `MSE(D_B(b_real), 1) + MSE(D_B(a2b_sample), 0)`
- **Cycle-consistency loss**: L1
  - `L1(a_real, a2b2a)` and `L1(b_real, b2a2b)`
- Total generator objective:
  - `g_loss = g_loss_a2b + g_loss_b2a + 10 * cyc_loss_a + 10 * cyc_loss_b`

Notably missing vs many CycleGAN implementations:
- **No identity loss** (often used with weight ~5 or 10). If you want closer-to-paper behavior, we can add it behind a flag.

Optimization:
- Adam, `lr=2e-4`, `beta1=0.5`.
- Training schedule appears constant LR (no decay logic in code).

### Fake-image Pool (Image Buffer)
Source: `utils.py` (`ItemPool`)
- Maintains up to 50 previous generated samples.
- Each iteration, 50% chance to swap a random old sample into the batch.

This is standard CycleGAN stabilization; we should keep it.

### Data Pipeline (Current)
Source: `data.py`

TF `tf.data.Dataset` pipeline per-domain:
- Read JPEG
- Random horizontal flip
- Resize to `load_size` (default 286)
- **Per-image min/max normalization**:
  - `(img - min(img)) / (max(img) - min(img))`
- Random crop to `crop_size` (default 256)
- Scale to `[-1, 1]` via `img * 2 - 1`

Important note:
- That min/max normalization is unusual for images and can change contrast per image. Most CycleGAN pipelines normalize by 255 and then to `[-1, 1]`. In the rewrite we should decide whether to:
  - **Match current behavior** for fidelity, or
  - **Switch to standard normalization** for stability (recommended), possibly behind a flag.

Batch size:
- default is `1`.

---

## Proposed Rewrite: High-Level Design

### Repo Structure (suggested)
- `cyclegan/`
  - `models/`
    - `generator_resnet.py`
    - `discriminator_patchgan.py`
  - `data/`
    - `datasets.py` (paired/unpaired folder datasets)
    - `transforms.py`
  - `losses.py`
  - `train_step.py`
  - `replay_buffer.py`
  - `checkpointing.py`
  - `utils/`
- `scripts/`
  - `train.py`
  - `infer.py`
  - `export.py`
- `configs/` (YAML or TOML)
- `outputs/` (checkpoints, samples, logs)

### Modern Python Tooling (recommended)
- Packaging: `pyproject.toml`
- Dependency/runtime management: **uv** (`uv venv`, `uv pip install`, `uv run`)
- Lockfile: `uv.lock` committed
- Lint/format: `ruff` (and optionally `black` if you want)
- Type checks (optional): `pyright`

### Training Loop Stack
Two good options:
1. **Plain PyTorch** (recommended): simplest to read, easiest to control.
2. **Lightning**: faster iteration, but adds abstraction. If your priority is “understandability”, I’d keep it plain.

Logging:
- TensorBoard (`torch.utils.tensorboard`) or Weights & Biases (optional).

### Device + Acceleration Plan
- Auto device selection:
  - Prefer `cuda` if available
  - Else `mps` if available
  - Else `cpu`
- Mixed precision:
  - CUDA: `torch.cuda.amp.autocast`
  - MPS: AMP support is more limited; we can gate AMP by device and PyTorch version.

---

## Smoke Test Notes (this workspace)
- Dataset download works via `python3 download_dataset.py horse2zebra` (mirror fallback).
- Training smoke test ran with `uv run cyclegan-train --epochs 1 --max_steps 5` and produced:
  - TensorBoard event file under `outputs_torch_smoketest/<dataset>/tb/`
  - Checkpoints under `outputs_torch_smoketest/<dataset>/checkpoints/`
  - Sample grids under `outputs_torch_smoketest/<dataset>/samples/`

### Issues / Deviations / Sub-optimal Choices
- **PyTorch wheel selection**: `uv` pulled a CUDA-enabled PyTorch wheel (`+cu128`) even though CUDA isn’t available in this environment. This is functional on CPU but wastes a lot of download size/time.
  - Follow-up improvement: document / pin CPU-only wheels for Linux (`--index-url https://download.pytorch.org/whl/cpu`) or provide optional dependency groups.
- **Padding fidelity**: matching TF `SAME` padding is intentionally non-modern for CycleGAN generators; reflection padding is common best practice. Keeping TF semantics is per-thread request.

## Core Components We Must Port (Reference Map)

### 1) Model definitions (must match)
- `models.py:generator` → `GeneratorResNet9`
- `models.py:discriminator` → `PatchDiscriminator`

Key fidelity details to verify during port:
- Padding behavior (`SAME` vs reflection padding).
- BatchNorm behavior (TF slim BN params: `decay=0.9`, `epsilon=1e-5`, `scale=True`).

### 2) Losses + objective weighting
- LSGAN adversarial losses (MSE)
- Cycle L1 with weight 10
- Identity loss: **intentionally not implemented** (per thread)

### 3) Replay buffer
- Port `utils.py:ItemPool` behavior exactly.

### 4) Data pipeline
- Replace TF `tf.data` with a PyTorch `Dataset` + `DataLoader`:
  - JPEG decode via PIL / torchvision
  - Deterministic & fast transforms via torchvision v2
  - `num_workers`, `pin_memory` (CUDA), `persistent_workers`

Unpaired sampling:
- CycleGAN trains on independent A and B batches. We can implement two datasets or one dataset returning `(imgA, imgB)` with independent indexing.

Normalization decision:
- Provide two modes:
  - `--norm=minmax` to match current repo
  - `--norm=standard` (default): `img = img/255; img = img*2-1`

### 5) Checkpointing & export
Current repo supports:
- TF checkpoints + graph freezing to `.pb` (`freeze.py`, `optimized-graph.pb`, `pb_a2b.py`)

In PyTorch rewrite:
- Training checkpoints: `state_dict` for `G_A2B`, `G_B2A`, `D_A`, `D_B`, optimizers, and step.
- Export options:
  - **TorchScript** (good for portability)
  - **ONNX** (optional; depends on op support)

Inference scripts should mirror:
- Input tensor name `inputA` equivalent
- Output `a2b` image in `[-1,1]` (then save as uint8)

---

## Concrete Rewrite Plan (phased)

### Phase 1 — Minimal faithful PyTorch port
Deliverables:
- PyTorch implementations of generator/discriminator.
- Training step that reproduces:
  - LSGAN losses
  - cycle losses
  - replay buffer for discriminator inputs
- Single-device training supporting `cuda|mps|cpu`.
- Basic sampling image grid output.

Success criteria:
- Loss curves behave sensibly.
- Inference generates reasonable translations on a small dataset.

### Phase 2 — Modern data pipeline + ergonomics
Deliverables:
- Clean dataset layout handling (`trainA/trainB/testA/testB`).
- Fast `DataLoader` settings + reproducibility.
- Config system (YAML) and CLI overrides.

### Phase 3 — Export + tooling
Deliverables:
- `infer.py` to run folder inference.
- `export.py` to TorchScript (and possibly ONNX).
- Optional: resume-from-checkpoint, EMA (not standard for CycleGAN but sometimes helpful).

---

## Risks / Open Questions
- **Padding differences** can materially change results. We should decide early whether to match TF Slim `SAME` or move to reflection padding (common in CycleGAN).
- **Normalization mismatch**: the current min/max normalization may be accidental. I recommend defaulting to standard image normalization but allowing strict-compat mode.
- **BatchNorm vs InstanceNorm**: many CycleGANs use InstanceNorm in generators/discriminators. This repo uses BatchNorm. Port should keep BatchNorm initially; we can add a switch later.

---

## “Core Reference” Checklist (from this repo)
- `models.py` (authoritative): generator + discriminator definitions.
- `train.py` (authoritative): exact losses, weights, update order, replay buffer usage.
- `utils.py` (authoritative): replay buffer (`ItemPool`) and basic helpers.
- `data.py` (authoritative): current preprocessing/augmentation (note min/max normalization).
- `freeze.py` / `pb_a2b.py` (secondary): shows intended inference IO naming and export expectations.

---

## Next Step
If you confirm PyTorch as the target, I can:
- Draft the new repo skeleton and implement Phase 1 (models + training loop + dataloader + MPS device selection).
- Add a small “smoke test” training run that executes 10–50 iterations to verify everything works on CPU/MPS.
