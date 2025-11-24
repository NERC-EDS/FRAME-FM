import os
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import tensorflow_datasets as tfds
from tensorflow_datasets.core import dataset_builders
import mlflow

class CroissantDataset(IterableDataset):
    """PyTorch IterableDataset wrapper for TFDS CroissantBuilder.

    Args:
        builder: The initialized TFDS CroissantBuilder.
        split: The split to load (e.g., 'default').
        batch_size: Batch size for the dataset.
    """

    def __init__(self, builder, split: str, batch_size: int):
        self.builder = builder
        self.split = split
        self.batch_size = batch_size
        # Ensure the dataset is prepared
        self.builder.download_and_prepare()

    def __iter__(self):
        """Yields batches of data from the TFDS dataset.

        Returns:
            Iterator yielding dictionaries of numpy arrays.
        """
        # Get the tf.data.Dataset
        ds = self.builder.as_dataset(split=self.split, shuffle_files=True)
        # Batch and convert to numpy
        ds = ds.batch(self.batch_size)
        ds = tfds.as_numpy(ds)
        
        for batch in ds:
            # Convert numpy arrays to torch tensors
            # We expect specific columns based on the regression task: PRES, PSAL -> TEMP
            # The batch is a dict of numpy arrays
            
            # Handle potential missing values or types if necessary, 
            # but for this example we assume clean data or handle simple casting.
            
            # Extract features and target
            # Note: The field names must match the JSON-LD file field names (sanitized by TFDS usually)
            # Based on the JSON-LD:
            # 'dataRecordSet/PRES' -> Pressure
            # 'dataRecordSet/PSAL' -> Salinity
            # 'dataRecordSet/TEMP' -> Temperature
            
            # TFDS CroissantBuilder usually flattens or keeps structure. 
            # Let's assume it keeps the structure or we access by key.
            # We'll need to verify the keys at runtime or check the builder info.
            # For now, we'll try to access them directly.
            
            try:
                # TFDS might sanitize keys. Let's assume standard sanitization.
                pres = torch.tensor(batch['dataRecordSet/PRES'], dtype=torch.float32).unsqueeze(1)
                psal = torch.tensor(batch['dataRecordSet/PSAL'], dtype=torch.float32).unsqueeze(1)
                temp = torch.tensor(batch['dataRecordSet/TEMP'], dtype=torch.float32).unsqueeze(1)
                
                # Simple imputation for NaNs if any (replace with 0 for this example)
                pres = torch.nan_to_num(pres)
                psal = torch.nan_to_num(psal)
                temp = torch.nan_to_num(temp)

                x = torch.cat([pres, psal], dim=1)
                y = temp
                
                yield {"x": x, "y": y}
            except KeyError as e:
                 # Fallback or debug print if keys are different
                 print(f"Available keys in batch: {batch.keys()}")
                 raise e


class CroissantTFDSDataModule(pl.LightningDataModule):
    """LightningDataModule for loading Croissant datasets via TFDS.

    Args:
        jsonld_path: Path to the Croissant JSON-LD file.
        record_set: Name of the record set to use.
        batch_size: Batch size for the dataloaders.
    """

    def __init__(self, jsonld_path: str, record_set: str, batch_size: int = 32):
        super().__init__()
        self.jsonld_path = jsonld_path
        self.record_set = record_set
        self.batch_size = batch_size
        self.builder = None

    def setup(self, stage=None):
        """Initializes the TFDS CroissantBuilder."""
        self.builder = dataset_builders.CroissantBuilder(
            jsonld=self.jsonld_path,
            record_set_ids=[self.record_set],
            file_format='array_record', # or 'tfrecord'
        )
        self.builder.download_and_prepare()

    def train_dataloader(self):
        """Returns the training dataloader.

        Returns:
            DataLoader: PyTorch DataLoader containing the training data.
        """
        # TFDS CroissantBuilder usually creates a 'default' split if not specified otherwise
        return DataLoader(
            CroissantDataset(self.builder, split='default', batch_size=self.batch_size),
            batch_size=None, # Batching is done in the dataset
            num_workers=0 # TFDS handles parallelism
        )


class SimpleRegressionModel(pl.LightningModule):
    """A simple linear regression model using PyTorch Lightning.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output targets.
        learning_rate: Learning rate for the optimizer.
    """

    def __init__(self, input_dim: int, output_dim: int, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.

        Returns:
            Loss tensor.
        """
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configures the optimizers.

        Returns:
            Optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


@hydra.main(config_path="conf", config_name="config_croissant", version_base="1.3")
def main(cfg: DictConfig):
    """Main entry point for training.

    Args:
        cfg: Hydra configuration object.
    """
    pl.seed_everything(cfg.trainer.seed)

    # MLflow setup
    tracking_uri = cfg.mlflow.tracking_uri
    experiment_name = cfg.mlflow.experiment_name
    run_name = cfg.mlflow.run_name

    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
    )
    
    print(f"MLflow tracking enabled. View results at: {tracking_uri}")

    # DataModule
    datamodule = CroissantTFDSDataModule(
        jsonld_path=cfg.data.jsonld,
        record_set=cfg.data.record_set,
        batch_size=cfg.data.batch_size
    )

    # Model
    model = SimpleRegressionModel(
        input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim,
        learning_rate=cfg.model.learning_rate
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=mlf_logger,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
