# AGENTS.md - CycleGAN-Tensorflow-PyTorch

This file provides quick guidance for coding agents working in this repo.

## Project
- Primary language: Python
- Entry points: `train.py`, `single_a2b.py`, `pb_a2b.py`, `ckpt_a2b.py`

## Conventions
- Keep changes minimal and focused.
- Prefer existing utility helpers in `utils.py` / `image_utils.py`.
- Avoid adding heavyweight dependencies unless necessary.

## Running
- Training: `python train.py` (see `README.md` for flags and data layout)
- Inference: `python single_a2b.py` or `python pb_a2b.py`

## Notes
- Large assets: `optimized-graph.pb` is checked in; avoid rewriting it.
