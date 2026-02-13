# src/FRAME_FM/models/demo_autoencoder.py
"""
EuroSAT Autoencoder (torchvision EuroSAT friendly)

This module defines a simple convolutional autoencoder intended for use with
torchvision.datasets.EuroSAT and a dataloader that yields batches as:

    batch = (x, y)

where:
  - x is a float Tensor of shape [B, C, H, W]
  - y is the class label Tensor [B] (not used for reconstruction loss)

Important:
- Your transforms should convert PIL -> Tensor (e.g., ToTensor()).
- For this architecture, it's simplest to resize images to 64x64 so that
  4x MaxPool(2) leads to a 4x4 spatial map.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from FRAME_FM.utils.LightningModuleWrapper import BaseModule


class EuroSATAutoencoder(BaseModule):
    """
    Convolutional autoencoder:

        x -> encoder -> z -> decoder -> x_recon

    Uses MSE reconstruction loss.

    > Thappitla, R.S., Villuri, V.G.K. & Kumar, S. An autoencoder driven deep learning geospatial approach to flood vulnerability analysis
      in the upper and middle basin of river Damodar. Sci Rep 15, 33741 (2025). https://doi.org/10.1038/s41598-025-96781-2

    Hydra config (example):
        _target_: FRAME_FM.model_code.demo_model.EuroSATAutoencoder
        in_channels: 3
        base_ch: 32
        k_size: 5
        latent_dim: 256
        lr: 1e-3
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 32,
        k_size: int = 5,
        latent_dim: int = 256,
        lr: float = 1e-3,
        **kwargs,
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB EuroSAT via torchvision).
            base_ch: Base channel width for conv blocks.
            k_size: Kernel size for convolutions.
            latent_dim: Dimensionality of the latent vector z.
            lr: Learning rate for Adam.
            **kwargs: Ignored extra Hydra keys (keeps config flexible).
        """
        super().__init__()
        self.save_hyperparameters()

        # ---- Encoder ----
        # Four downsampling steps: H,W -> H/16,W/16 (64 -> 4 if resized to 64)
        chs = [in_channels, base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        pad = k_size // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(chs[0], chs[1], kernel_size=k_size, stride=1, padding=pad),
            nn.BatchNorm2d(chs[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(chs[1], chs[2], kernel_size=k_size, stride=1, padding=pad),
            nn.BatchNorm2d(chs[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(chs[2], chs[3], kernel_size=k_size, stride=1, padding=pad),
            nn.BatchNorm2d(chs[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(chs[3], chs[4], kernel_size=k_size, stride=1, padding=pad),
            nn.BatchNorm2d(chs[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # ---- Latent projection ----
        # Assumes encoder output spatial size is 4x4 (e.g., input 64x64).
        self.to_latent = nn.Sequential(
            nn.Flatten(),  # [B, chs[4]*4*4]
            nn.Linear(chs[4] * 4 * 4, latent_dim),  # [B, latent_dim]
        )

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, chs[4] * 4 * 4),  # [B, chs[4]*4*4]
            nn.Unflatten(1, (chs[4], 4, 4)),  # [B, chs[4], 4, 4]
        )

        # ---- Decoder ----
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 4->8
            nn.Conv2d(chs[4], chs[3], kernel_size=k_size, padding=pad),
            nn.BatchNorm2d(chs[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 8->16
            nn.Conv2d(chs[3], chs[2], kernel_size=k_size, padding=pad),
            nn.BatchNorm2d(chs[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 16->32
            nn.Conv2d(chs[2], chs[1], kernel_size=k_size, padding=pad),
            nn.BatchNorm2d(chs[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 32->64
            nn.Conv2d(chs[1], chs[0], kernel_size=k_size, padding=pad),
            # No final activation: depends on how you normalize inputs.
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor [B, C, H, W]

        Returns:
            x_recon: Tensor [B, C, H, W]
            z: Tensor [B, latent_dim]
        """
        h = self.encoder(x)
        z = self.to_latent(h)
        x_recon = self.decoder(self.from_latent(z))
        return x_recon, z

    def _shared_step(self, batch, stage: str):
        """
        Shared step for train/val/test.

        Expects torchvision-style batches: (x, y)
        y is unused for the loss, but kept for compatibility/logging.
        """
        x = batch["image"]  # torchgeo retuens {"image": x, "label": y},
        y = batch["label"]
        x_recon, _ = self(x)

        loss = self.loss_fn(x_recon, x)

        # Simple reconstruction metric: fraction of pixels close to target
        with torch.no_grad():
            recon_acc = ((x_recon - x).abs() < 0.1).float().mean()

        logs: Dict[str, torch.Tensor] = {
            f"{stage}/loss": loss,
            f"{stage}/recon_acc": recon_acc,
        }
        return loss, logs

    # BaseModule wrapper is expecting these *_step_body methods
    def training_step_body(self, batch, batch_idx: int = 0):
        return self._shared_step(batch, stage="train")

    def validation_step_body(self, batch, batch_idx: int = 0):
        return self._shared_step(batch, stage="val")

    def test_step_body(self, batch, batch_idx: int = 0):
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.lr))
