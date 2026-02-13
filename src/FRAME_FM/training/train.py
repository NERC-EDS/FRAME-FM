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

    # Output some debug info about the model and data (optional)
    print(f"Model instantiated: {model.__class__.__name__}")
    print(f"DataModule instantiated: {datamodule.__class__.__name__}")
    print(f"Trainer instantiated: {trainer.__class__.__name__}")
    if logger is not None:
        print(f"Logger instantiated: {logger.__class__.__name__}")

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Optional: test after training
    if cfg.get("run_test", False):
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
