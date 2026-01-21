from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data import UnpairedDataset
from ..losses import LSGANLoss, l1_cycle_loss
from ..models import GeneratorResNet9, PatchDiscriminator
from ..replay_buffer import ReplayBuffer
from ..utils import save_image_grid


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def _apply_yaml_defaults(parser: argparse.ArgumentParser, config_path: str | None) -> None:
    if not config_path:
        return
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    parser.set_defaults(**cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="YAML config file")

    # Core hyperparams / paths
    parser.add_argument("--dataset", default="horse2zebra")
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--load_size", type=int, default=286)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=2)

    # Logging/output
    parser.add_argument("--logdir", default="outputs_torch")
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1000)

    # Behavior toggles
    parser.add_argument(
        "--legacy_minmax_norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Match TF per-image min/max normalization",
    )

    # Resume/smoke testing
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps (0 = no limit)")

    # Parse config first, then CLI overrides.
    config_ns, remaining = parser.parse_known_args(argv)
    _apply_yaml_defaults(parser, config_ns.config)
    args = parser.parse_args(argv)

    dataset_dir = Path(args.data_root) / args.dataset
    train_a = dataset_dir / "trainA"
    train_b = dataset_dir / "trainB"
    test_a = dataset_dir / "testA"
    test_b = dataset_dir / "testB"

    device = _device()

    out_root = Path(args.logdir) / args.dataset
    ckpt_dir = out_root / "checkpoints"
    sample_dir = out_root / "samples"
    tb_dir = out_root / "tb"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = UnpairedDataset(
        train_a,
        train_b,
        load_size=args.load_size,
        crop_size=args.crop_size,
        train=True,
        legacy_minmax_norm=args.legacy_minmax_norm,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    test_ds = UnpairedDataset(
        test_a,
        test_b,
        load_size=args.load_size,
        crop_size=args.crop_size,
        train=False,
        legacy_minmax_norm=args.legacy_minmax_norm,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    g_a2b = GeneratorResNet9().to(device)
    g_b2a = GeneratorResNet9().to(device)
    d_a = PatchDiscriminator().to(device)
    d_b = PatchDiscriminator().to(device)

    g_opt = Adam(list(g_a2b.parameters()) + list(g_b2a.parameters()), lr=args.lr, betas=(0.5, 0.999))
    d_a_opt = Adam(d_a.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_b_opt = Adam(d_b.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lsgan = LSGANLoss()

    a_buffer = ReplayBuffer()
    b_buffer = ReplayBuffer()

    writer = SummaryWriter(tb_dir)

    start_epoch = 0
    step = 0
    start_time = time.time()

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        g_a2b.load_state_dict(ckpt["g_a2b"])
        g_b2a.load_state_dict(ckpt["g_b2a"])
        d_a.load_state_dict(ckpt["d_a"])
        d_b.load_state_dict(ckpt["d_b"])
        g_opt.load_state_dict(ckpt["g_opt"])
        d_a_opt.load_state_dict(ckpt["d_a_opt"])
        d_b_opt.load_state_dict(ckpt["d_b_opt"])

        step = int(ckpt.get("step", 0)) + 1
        start_epoch = int(ckpt.get("epoch", 0))

    g_a2b.train()
    g_b2a.train()
    d_a.train()
    d_b.train()

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for a_real, b_real in pbar:
            if args.max_steps and step >= args.max_steps:
                break
            a_real = a_real.to(device)
            b_real = b_real.to(device)

            # -------------------------
            # Generators
            # -------------------------
            _set_requires_grad(d_a, False)
            _set_requires_grad(d_b, False)
            g_opt.zero_grad(set_to_none=True)

            a2b = g_a2b(a_real)
            b2a = g_b2a(b_real)
            a2b2a = g_b2a(a2b)
            b2a2b = g_a2b(b2a)

            g_loss_a2b = lsgan.g_loss(d_b(a2b))
            g_loss_b2a = lsgan.g_loss(d_a(b2a))
            cyc_a = l1_cycle_loss(a_real, a2b2a)
            cyc_b = l1_cycle_loss(b_real, b2a2b)

            g_loss = g_loss_a2b + g_loss_b2a + 10.0 * cyc_a + 10.0 * cyc_b
            g_loss.backward()
            g_opt.step()

            # -------------------------
            # Discriminators (with replay)
            # -------------------------
            _set_requires_grad(d_a, True)
            _set_requires_grad(d_b, True)

            with torch.no_grad():
                a2b_detached = a2b.detach()
                b2a_detached = b2a.detach()
                a2b_replay = a_buffer.push_and_pop(a2b_detached)
                b2a_replay = b_buffer.push_and_pop(b2a_detached)

            d_a_opt.zero_grad(set_to_none=True)
            d_loss_a = lsgan.d_loss(d_a(a_real), d_a(b2a_replay))
            d_loss_a.backward()
            d_a_opt.step()

            d_b_opt.zero_grad(set_to_none=True)
            d_loss_b = lsgan.d_loss(d_b(b_real), d_b(a2b_replay))
            d_loss_b.backward()
            d_b_opt.step()

            # -------------------------
            # Logging
            # -------------------------
            writer.add_scalar("loss/g_total", float(g_loss.item()), step)
            writer.add_scalar("loss/g_a2b", float(g_loss_a2b.item()), step)
            writer.add_scalar("loss/g_b2a", float(g_loss_b2a.item()), step)
            writer.add_scalar("loss/cyc_a", float(cyc_a.item()), step)
            writer.add_scalar("loss/cyc_b", float(cyc_b.item()), step)
            writer.add_scalar("loss/d_a", float(d_loss_a.item()), step)
            writer.add_scalar("loss/d_b", float(d_loss_b.item()), step)
            writer.add_scalar("meta/epoch", float(epoch), step)

            elapsed = time.time() - start_time
            pbar.set_postfix({"step": step, "g": f"{g_loss.item():.3f}", "t": f"{elapsed/60:.1f}m"})

            # Samples
            if (step + 1) % args.sample_every == 0:
                g_a2b.eval()
                g_b2a.eval()
                with torch.no_grad():
                    a_test, b_test = next(iter(test_loader))
                    a_test = a_test.to(device)
                    b_test = b_test.to(device)
                    a2b_t = g_a2b(a_test)
                    a2b2a_t = g_b2a(a2b_t)
                    b2a_t = g_b2a(b_test)
                    b2a2b_t = g_a2b(b2a_t)

                    # Layout matches TF sample concat: 2 rows x 3 cols
                    sample = torch.cat(
                        [a_test, a2b_t, a2b2a_t, b_test, b2a_t, b2a2b_t], dim=0
                    )
                    save_image_grid(sample, sample_dir / f"step_{step:07d}.jpg", rows=2, cols=3)

                g_a2b.train()
                g_b2a.train()

            # Checkpoints
            if (step + 1) % args.save_every == 0:
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "g_a2b": g_a2b.state_dict(),
                        "g_b2a": g_b2a.state_dict(),
                        "d_a": d_a.state_dict(),
                        "d_b": d_b.state_dict(),
                        "g_opt": g_opt.state_dict(),
                        "d_a_opt": d_a_opt.state_dict(),
                        "d_b_opt": d_b_opt.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_dir / f"ckpt_step_{step:07d}.pt",
                )

            step += 1

        if args.max_steps and step >= args.max_steps:
            break

    writer.flush()
    writer.close()
    return 0
