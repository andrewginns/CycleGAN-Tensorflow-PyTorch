# For use in https://github.com/andrewginns/MSc-Project

### Trained models available in releases https://github.com/andrewginns/CycleGAN-Tensorflow-PyTorch/releases

***Changes from LynnHo/CycleGAN-Tensorflow-PyTorch***

-Added creation of an GrafDef proto called 'graph.pb' when training a new model

-Input and output nodes named

-PyTorch code removed --> Name preserved for traceability

# CycleGAN
Tensorflow implementation of CycleGAN.

Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)
Author: [Jun-Yan Zhu ](https://people.eecs.berkeley.edu/~junyanz/) *et al.*

## Exemplar results on testset
- gif: horse -> zebra
![](./pics/horse2zebra.gif)
- row 1: horse -> zebra -> reconstructed horse, row 2: zebra -> horse -> reconstructed zebra
![](./pics/example_horse2zebra_1.jpg)
- row 1: apple -> orange -> reconstructed apple, row 2: orange -> apple -> reconstructed orange
![](./pics/example_apple2orange_1.jpg)

# Prerequisites

## Legacy (TF1)
- tensorflow r1.7
- python 2.7

## Modern (PyTorch)
- Python >= 3.10
- Recommended: `uv` for env/deps

Setup:
- See `docs/install.md`

# Usage
```
cd CycleGAN-Tensorflow-PyTorch-master
```

## Download Datasets
- Download the horse2zebra dataset:
```bash
sh ./download_dataset.sh horse2zebra
```
- Download the apple2orange dataset:
```bash
sh ./download_dataset.sh apple2orange
```
- See download_dataset.sh for more datasets

## Train Example

### Legacy (TF1)
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=horse2zebra
```

### Modern (PyTorch)
```bash
# Install first (see docs/install.md)
uv run cyclegan-train --config configs/horse2zebra.yaml

# Resume example
uv run cyclegan-train --config configs/horse2zebra.yaml --resume outputs_torch/horse2zebra/checkpoints/ckpt_step_0001000.pt

# Inference example (A->B)
uv run cyclegan-infer --ckpt outputs_torch/horse2zebra/checkpoints/ckpt_step_0001000.pt --direction a2b --input datasets/horse2zebra/testA --output outputs_torch/horse2zebra/infer_a2b

# Benchmark example (A->B)
python torch_benchmark.py --ckpt outputs_torch/horse2zebra/checkpoints/ckpt_step_0001000.pt --direction a2b --dataset datasets/horse2zebra/testA
```

## Test Example
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset=horse2zebra
```
