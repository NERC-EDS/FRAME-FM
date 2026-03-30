#!/usr/bin/env python3
"""Plot train/validation loss curves from FRAME-FM CSV logger metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a loss curve plot from a FRAME-FM metrics.csv file.")
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to metrics.csv (from csv_lightweight logger).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("loss_curve.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="FRAME-FM Training: Loss Curves",
        help="Plot title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI.",
    )
    return parser.parse_args()


def build_epoch_dataframe(metrics: pd.DataFrame) -> pd.DataFrame:
    required = ["epoch", "train/loss_epoch", "val/loss"]
    missing = [col for col in required if col not in metrics.columns]
    if missing:
        raise ValueError(f"Missing required columns in metrics CSV: {missing}")

    epochs_data: list[dict[str, float | int]] = []
    for epoch in sorted(metrics["epoch"].dropna().unique()):
        epoch_df = metrics[metrics["epoch"] == epoch]

        train_loss = epoch_df[epoch_df["train/loss_epoch"].notna()]["train/loss_epoch"].values
        val_loss = epoch_df[epoch_df["val/loss"].notna()]["val/loss"].values

        if len(train_loss) > 0 and len(val_loss) > 0:
            epochs_data.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss[0]),
                    "val_loss": float(val_loss[0]),
                }
            )

    if not epochs_data:
        raise ValueError("No epochs with both train/loss_epoch and val/loss were found.")

    return pd.DataFrame(epochs_data)


def plot_losses(df_epochs: pd.DataFrame, output: Path, title: str, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        df_epochs["epoch"],
        df_epochs["train_loss"],
        "b-o",
        label="Train Loss",
        linewidth=2,
        markersize=4,
    )
    plt.plot(
        df_epochs["epoch"],
        df_epochs["val_loss"],
        "r-s",
        label="Val Loss",
        linewidth=2,
        markersize=4,
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches="tight")


def print_summary(df_epochs: pd.DataFrame, output: Path) -> None:
    print(f"Plot saved: {output}")
    print("\nTraining Summary:")
    print(f"  Initial train loss: {df_epochs['train_loss'].iloc[0]:.6f}")
    print(f"  Final train loss:   {df_epochs['train_loss'].iloc[-1]:.6f}")
    print(f"  Initial val loss:   {df_epochs['val_loss'].iloc[0]:.6f}")
    print(f"  Final val loss:     {df_epochs['val_loss'].iloc[-1]:.6f}")
    print(f"  Epochs completed:   {len(df_epochs)}")


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics)
    df_epochs = build_epoch_dataframe(metrics)
    plot_losses(df_epochs, args.output, args.title, args.dpi)
    print_summary(df_epochs, args.output)


if __name__ == "__main__":
    main()
