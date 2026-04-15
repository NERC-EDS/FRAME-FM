# SPDX-FileCopyrightText: 2026 FRAME-FM Contributors
#
# SPDX-License-Identifier: Apache-2.0

# src/FRAME_FM/training/train.py
from __future__ import annotations

from hydra import main as hydra_main
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch.multiprocessing as mp


@hydra_main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Ensure reproducibility
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    # Instantiate Data + Model from config
    data = instantiate(cfg.data, _convert_="partial")
    model = instantiate(cfg.model, _convert_="partial")

    # Configure MLflow logger (if provided)
    logger = None
    if "logging" in cfg:
        logger = instantiate(cfg.logging)

    # Instantiate PL Trainer
    trainer = instantiate(cfg.trainer, logger=logger)

    # Train
    if isinstance(data, pl.LightningDataModule):
        trainer.fit(model, datamodule=data)
    else:
        trainer.fit(model, train_dataloaders=data['training'], val_dataloaders=data['validation'])

    # Optional: test after training
    if hasattr(cfg.trainer, "run_test") and cfg.trainer.run_test:
        if isinstance(data, pl.LightningDataModule):
            trainer.test(model, datamodule=data)
        else:
            trainer.test(model, dataloaders=data['test'])


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
