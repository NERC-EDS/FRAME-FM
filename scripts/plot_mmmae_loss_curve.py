#!/usr/bin/env python3
"""Plot training/validation loss curves from an MLflow run directory.

Examples:
    python scripts/plot_mmmae_loss_curve.py --run-id 323794e77cc04101bcf496dc30990b00
    python scripts/plot_mmmae_loss_curve.py --run-dir mlruns/629506779529726635/<run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _find_run_dir_by_id(repo_root: Path, run_id: str) -> Path:
    matches = list(repo_root.glob(f"mlruns/*/{run_id}"))
    if not matches:
        raise FileNotFoundError(f"Could not find run_id '{run_id}' under mlruns/*/")
    return matches[0]


def _find_latest_run_dir(repo_root: Path) -> Path:
    metric_paths = sorted(
        repo_root.glob("mlruns/*/*/metrics/train/loss_step"),
        key=lambda p: p.stat().st_mtime,
    )
    if not metric_paths:
        raise FileNotFoundError("No runs with metrics/train/loss_step found under mlruns")
    return metric_paths[-1].parents[2]


def _read_metric_file(path: Path):
    if not path.exists():
        return np.array([]), np.array([])

    steps = []
    values = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            # Format: <timestamp_ms> <value> <step>
            values.append(float(parts[1]))
            steps.append(int(parts[2]))

    if not steps:
        return np.array([]), np.array([])

    return np.asarray(steps), np.asarray(values)


def _summary(name: str, steps: np.ndarray, vals: np.ndarray) -> str:
    if len(vals) == 0:
        return f"{name}: no points"
    first = vals[0]
    last = vals[-1]
    min_v = float(vals.min())
    min_step = int(steps[int(vals.argmin())])
    drop_pct = 100.0 * (first - last) / first if first != 0 else float("nan")
    return (
        f"{name}: points={len(vals)}, first={first:.4f}, last={last:.4f}, "
        f"min={min_v:.4f} at step={min_step}, drop={drop_pct:.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MMMAE train/val loss curves from MLflow metrics")
    parser.add_argument("--run-id", type=str, default=None, help="MLflow run ID")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to MLflow run directory")
    parser.add_argument("--output", type=str, default=None, help="Output image path (default: plots/loss_<runid>.png)")
    parser.add_argument("--train-metric", type=str, default="metrics/train/loss_step")
    parser.add_argument("--val-metric", type=str, default="metrics/val/loss")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    repo_root = Path.cwd()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.run_id:
        run_dir = _find_run_dir_by_id(repo_root, args.run_id)
    else:
        run_dir = _find_latest_run_dir(repo_root)

    run_id = run_dir.name
    output = Path(args.output) if args.output else Path(f"plots/loss_{run_id}.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    train_steps, train_vals = _read_metric_file(run_dir / args.train_metric)
    val_steps, val_vals = _read_metric_file(run_dir / args.val_metric)

    if len(train_vals) == 0 and len(val_vals) == 0:
        raise RuntimeError(f"No metric points found in {run_dir}")

    plt.figure(figsize=(12, 6))

    if len(train_vals) > 0:
        plt.plot(train_steps, train_vals, label="train/loss_step", linewidth=1.8)
    if len(val_vals) > 0:
        plt.plot(val_steps, val_vals, label="val/loss", linewidth=1.2, alpha=0.8)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"MMMAE Loss Curve\nrun_id={run_id}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=args.dpi)

    print(f"Run dir: {run_dir}")
    print(f"Saved: {output}")
    print(_summary("train", train_steps, train_vals))
    print(_summary("val", val_steps, val_vals))


if __name__ == "__main__":
    main()
