# SPDX-FileCopyrightText: 2026 2026 FRAME-FM Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytorch_lightning as pl
import torch
from hydra import main as hydra_main
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra_main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra-driven ERA5 dataloader + MMMAE encoder smoke test.

    Expected usage example:
        python src/FRAME_FM/training/era5_mmmae_demo.py \
            data=era5_spatial_pixels_demo model=era5_demo_mmmae
    """

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    datamodule = instantiate(cfg.data)
    print("Setting up DataModule...")
    datamodule.setup()

    print("\nDataset sizes:")
    print("Train:", len(datamodule.train_dataset))
    print("Val  :", len(datamodule.val_dataset))
    print("Test :", 0 if datamodule.test_dataset is None else len(datamodule.test_dataset))

    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))

    train_values, train_times, train_pos = train_batch
    val_values, val_times, val_pos = val_batch

    print("\nTrain batch contents:")
    print("Train batch length:", len(train_batch), "Expected: 3 (values, times, positions)")
    print("values:", train_values.shape, "Expected: (B, C, T, H, W)")
    print("times :", train_times.shape, "Expected: (B, T)")
    print("pos   :", train_pos.shape, "Expected: (B, 3, T, H, W)")

    print("\nSingle sample shapes:")
    print("values sample:", train_values[0].shape)
    print("times sample :", train_times[0].shape)
    print("pos sample   :", train_pos[0].shape)

    print("\nVal batch shapes:")
    print("values:", val_values.shape)
    print("times :", val_times.shape)
    print("pos   :", val_pos.shape)

    print("\nTesting forward pass through MMMAE encoder...")

    _, c_dim, t_dim, h_dim, w_dim = val_values.shape
    print(f"Input batch shape: {val_values.shape} (B, C, T, H, W)")
    print("val_pos.shape:", val_pos.shape)

    t_min = float(val_pos[:, 0].min().item())
    t_max = float(val_pos[:, 0].max().item())
    lat_min = float(val_pos[:, 1].min().item())
    lat_max = float(val_pos[:, 1].max().item())
    lon_min = float(val_pos[:, 2].min().item())
    lon_max = float(val_pos[:, 2].max().item())

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.input_shapes = [[int(t_dim), int(h_dim), int(w_dim)]]
    model_cfg.n_channels = [int(c_dim)]
    model_cfg.position_space = [[t_min, t_max], [lat_min, lat_max], [lon_min, lon_max]]

    model = instantiate(model_cfg)
    model.eval()

    embed = model.input_embedders[0]
    print("embed.input_shape:", embed.input_shape)
    print("embed.patch_shape:", embed.patch_shape)
    print("len(position_space):", len(embed.position_space))
    print("pos_conv_kernel shape:", embed.pos_conv_kernel.shape)

    conv_fn = torch.nn.functional.conv3d
    pconv = conv_fn(
        val_pos,
        embed.pos_conv_kernel,
        stride=embed.patch_shape,
        groups=len(embed.position_space),
    )
    print("raw position conv shape:", pconv.shape)

    with torch.no_grad():
        x_tokens = embed.proj(val_values).flatten(start_dim=2).transpose(1, 2)
        position_tokens = embed.pos_embed(val_pos)
        print("value token shape:", x_tokens.shape)
        print("position token shape:", position_tokens.shape)

    with torch.no_grad():
        latent, decoder_metadata_embed, mask, ids_restore = model.forward_encoder(
            inputs=[(val_values, val_pos)],
            mask_ratio=cfg.get("demo_mask_ratio", 0.5),
        )

    print("\nEncoder forward pass successful.")
    print("Latent shape     :", latent.shape)
    print("pos_embed shape  :", decoder_metadata_embed.shape)
    print("Mask shape       :", mask.shape)
    print("ids_restore shape:", ids_restore.shape)


if __name__ == "__main__":
    main()
