import matplotlib
matplotlib.use("Agg") #Ensure a non-interactive Matplotlib backend
import matplotlib.pyplot as plt
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import random_split
import mlflow
import mlflow.pytorch


# add src to python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))



#load configurations through Hydra - tryin the example that is pointing to the
#right config files
config_dir = os.path.join(os.getcwd(), 'configs')
@hydra.main(config_path=config_dir, config_name="config_convAE", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    
    # Initialize MLflow setup --params, metrics, and model artifacts get captured automatically during training
    mlflow.set_tracking_uri(cfg.logging.tracking_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)
    # set input example for better model signature in MLflow UI
    
    mlflow.pytorch.autolog(log_model_signatures=True, log_models=True) # log schema for reproducibility

    # #Load Datamodule and Model - could be loaded explicitly with individual
    # parameters passed
    # datamodule = XarrayStaticDataModule(
    #     data_root=cfg.data.data_root,
    #     batch_size=cfg.data.batch_size,
    #     num_workers=cfg.data.num_workers,
    #     pin_memory=cfg.data.pin_memory,
    #     persistent_workers=cfg.data.persistent_workers,
    #     train_split=cfg.data.train_split,
    #     val_split=cfg.data.val_split,
    #     test_split=cfg.data.test_split,
    #     split_strategy=cfg.data.split_strategy,
    #     train_transforms=instantiate(cfg.data.train_transforms),
    #     val_transforms=instantiate(cfg.data.val_transforms),
    #     test_transforms=instantiate(cfg.data.test_transforms),
    #     tile_size=cfg.data.tile_size,
    # )
    # model = ConvAutoencoder(cfg.model.in_channels, cfg.model.base_channels,
    # cfg.model.kernel_size, cfg.model.latent_dim)
    
        # #load and initialize MLflow logger from config
    # with mlflow.start_run():
    #     trainer = pl.Trainer(
    #         max_epochs=cfg.trainer.max_epochs,
    #         accelerator=cfg.trainer.accelerator,
    #         log_every_n_steps=cfg.trainer.log_every_n_steps,
    #         # limit_val_batches=2, 
    #         # num_sanity_val_steps=0,
    #         enable_progress_bar=True,
    #     )
    #     trainer.fit(model, datamodule=datamodule)
    
    
    # Trying to instantiate everything from config as shown in main
    
    # Instantiate DataModule + Model from config
    datamodule = instantiate(cfg.data, _recursive_=True) # _recursive_=True ensures that nested configs are also instantiated
    model = instantiate(cfg.model, _recursive_=True)
    
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