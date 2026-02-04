# src/FRAME_FM/training/train.py
from __future__ import annotations

import pytorch_lightning as pl
from hydra import main as hydra_main
from omegaconf import DictConfig
from hydra.utils import instantiate

@hydra_main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Ensure reproducibility
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # Instantiate DataModule + Model from config
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)

    # Configure MLflow logger (if provided)
    logger = None
    if "logging" in cfg:
        logger = instantiate(cfg.logging)

    # Instantiate PL Trainer
    trainer = instantiate(cfg.trainer, logger=logger)

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Optional: test after training
    if hasattr(cfg.trainer, "run_test") and cfg.trainer.run_test:
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
