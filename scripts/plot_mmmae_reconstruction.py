#!/usr/bin/env python3
"""Generate MMMAE reconstruction visualizations from a checkpoint.

Example:
    python scripts/plot_mmmae_reconstruction.py \
      --checkpoint outputs/2026-03-16/17-48-46/.../checkpoints/epoch=536-step=31645.ckpt \
      --output plots/era5_mmmae_reconstruction_trained.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _find_latest_checkpoint(root: Path) -> Path:
    checkpoints = sorted(root.glob("outputs/**/checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found under outputs/**/checkpoints/*.ckpt")
    return checkpoints[-1]


def _load_model_from_checkpoint(model_cfg_path: Path, checkpoint_path: Path):
    model_cfg = OmegaConf.load(str(model_cfg_path))
    model = instantiate(model_cfg)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _prepare_batch(data_cfg_path: Path, batch_size: int, sample_index: int):
    data_cfg = OmegaConf.load(str(data_cfg_path))
    data_cfg.model_ready_inputs = True
    data_cfg.batch_size = batch_size
    data_cfg.debug = False

    dm = instantiate(data_cfg)
    dm.setup("fit")

    loader = dm.train_dataloader()
    batch = next(iter(loader))
    inputs, coords = batch[0]

    if sample_index >= inputs.shape[0]:
        raise IndexError(f"sample_index={sample_index} out of range for batch size {inputs.shape[0]}")

    return inputs, coords


def _build_composite_tokens(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, mask: torch.Tensor):
    # mask convention: 0 = visible/kept token, 1 = masked token
    composite = pred_tokens.clone()
    visible = mask == 0
    composite[visible] = target_tokens[visible]
    return composite


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MMMAE reconstruction from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .ckpt (default: latest under outputs)")
    parser.add_argument("--data-config", type=str, default="configs/data/era5_spatial_pixels_demo.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model/era5_demo_mmmae.yaml")
    parser.add_argument("--output", type=str, default="plots/era5_mmmae_reconstruction.png")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--mask-ratio", type=float, default=None, help="Override model default mask ratio")
    parser.add_argument(
        "--reconstruction-mode",
        choices=["pure", "composite"],
        default="pure",
        help="pure: decoder predictions everywhere; composite: ground truth on visible tokens + predictions on masked tokens",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _find_latest_checkpoint(repo_root)
    data_cfg_path = Path(args.data_config)
    model_cfg_path = Path(args.model_config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using checkpoint: {checkpoint_path}")
    model = _load_model_from_checkpoint(model_cfg_path, checkpoint_path)
    inputs, coords = _prepare_batch(data_cfg_path, args.batch_size, args.sample_index)

    mask_ratio = model.default_mask_ratio if args.mask_ratio is None else args.mask_ratio

    with torch.no_grad():
        _, predictions, mask = model([(inputs, coords)], mask_ratio=mask_ratio)
        pred_tokens = predictions[0]  # [B, L, patch_dim]
        embedder = model.input_embedders[0]

        if args.reconstruction_mode == "composite":
            target_tokens = embedder.tokenify((inputs, coords))
            render_tokens = _build_composite_tokens(pred_tokens, target_tokens, mask)
        else:
            render_tokens = pred_tokens

        recon = embedder.untokenify(render_tokens)

    # Pick sample/channel
    s = args.sample_index
    x = inputs[s, 0].cpu().numpy()       # [T,H,W]
    y = recon[s, 0].cpu().numpy()         # [T,H,W]
    m = mask[s].cpu().numpy().reshape(embedder.grid_shape)  # [T_patch,H_patch,W_patch]

    n_time = x.shape[0]
    n_cols = min(4, n_time)

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, max(n_cols, 1), figure=fig, hspace=0.40, wspace=0.35)

    # Shared color range for fair comparison
    vmin = min(float(np.nanmin(x)), float(np.nanmin(y)))
    vmax = max(float(np.nanmax(x)), float(np.nanmax(y)))

    for col in range(n_cols):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(x[col], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"Input (t={col})", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        ax = fig.add_subplot(gs[1, col])
        mask_t = m[col]
        patch_h = x.shape[1] // mask_t.shape[0]
        patch_w = x.shape[2] // mask_t.shape[1]
        mask_display = np.repeat(np.repeat(mask_t, patch_h, axis=0), patch_w, axis=1)
        ax.imshow(mask_display, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Mask (t={col})", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        pct_masked = 100.0 * mask_t.mean()
        pct_visible = 100.0 - pct_masked
        ax.text(0.5, -0.15, f"{pct_visible:.1f}% visible / {pct_masked:.1f}% masked", transform=ax.transAxes, ha="center", fontsize=9)

        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(y[col], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        recon_label = "Reconstruction (Pure)" if args.reconstruction_mode == "pure" else "Reconstruction (Composite)"
        ax.set_title(f"{recon_label} (t={col})", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    fig.text(0.02, 0.83, "Input", ha="right", va="center", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.50, "Mask", ha="right", va="center", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.17, "Reconstruction", ha="right", va="center", fontsize=12, fontweight="bold")

    title_mode = "Decoder-only" if args.reconstruction_mode == "pure" else "MAE Composite"
    plt.suptitle(
        f"ERA5 MMMAE: Input -> Mask -> Reconstruction ({title_mode})\ncheckpoint={checkpoint_path.name}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(output_path, dpi=180, bbox_inches="tight")

    mse = float(np.mean((x - y) ** 2))
    mae = float(np.mean(np.abs(x - y)))

    print(f"Saved: {output_path}")
    print(f"Mode: {args.reconstruction_mode}")
    print(f"Mask ratio (requested): {mask_ratio:.2f}")
    print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Reconstruction range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
