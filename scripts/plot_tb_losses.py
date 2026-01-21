#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="TensorBoard logdir (contains events.out.tfevents.*)")
    parser.add_argument("--out", default="tb_plots", help="Output directory")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    wanted = [t for t in tags if t.startswith("loss/")]
    if not wanted:
        raise SystemExit(f"No loss/* scalar tags found. Found tags: {tags}")

    # Lazy import so script can still inspect tags without matplotlib.
    import matplotlib.pyplot as plt

    def plot_group(group_name: str, group_tags: list[str]) -> Path:
        plt.figure(figsize=(9, 5))
        for tag in group_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            vals = [e.value for e in events]
            plt.plot(steps, vals, label=tag)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title(group_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_path = out_dir / f"{group_name.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return out_path

    # Two plots: generators vs discriminators (+cycle) so it’s readable.
    gen_tags = [t for t in wanted if t in {"loss/g_total", "loss/g_a2b", "loss/g_b2a", "loss/cyc_a", "loss/cyc_b"}]
    disc_tags = [t for t in wanted if t in {"loss/d_a", "loss/d_b"}]

    created = []
    if gen_tags:
        created.append(plot_group("loss_generators", gen_tags))
    if disc_tags:
        created.append(plot_group("loss_discriminators", disc_tags))

    # Also dump a small CSV for convenience.
    import csv

    csv_path = out_dir / "loss_scalars.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "step", "value"])
        for tag in sorted(wanted):
            for e in ea.Scalars(tag):
                w.writerow([tag, e.step, e.value])

    print("Created:")
    for p in created + [csv_path]:
        print(p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
