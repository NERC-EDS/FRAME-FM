import os
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchvision import models

from torchgeo.datasets import EuroSAT

import mlflow
import mlflow.pytorch


# -----------------------------
# 1. LightningModule with ResNet
# -----------------------------
class EuroSATResNet(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Pretrained ResNet18 backbone
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # FIX 1: Modify the first convolutional layer for 13-channel EuroSAT data
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=13,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=(original_conv1.bias is not None)
        )

        # FIX 2: Modify the final fully connected layer for the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# -----------------------------
# 2. DataModule for EuroSAT
# -----------------------------
class EuroSATDataModule(pl.LightningDataModule):
    def __init__(self, root: str = "data", batch_size: int = 32):
        super().__init__()
        self.root = root
        self.batch_size = batch_size

    def prepare_data(self):
        EuroSAT(self.root, download=True)

    def setup(self, stage=None):
        dataset = EuroSAT(self.root, transforms=None)
        n_train = int(len(dataset) * 0.8)
        n_val = len(dataset) - n_train
        self.train_dataset, self.val_dataset = random_split(dataset, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.trainer.seed)

    tracking_uri = cfg.mlflow.tracking_uri
    experiment_name = cfg.mlflow.experiment_name
   
    # MLflow setup
    mlflow.set_tracking_uri(tracking_uri)  # Local tracking
    mlflow.set_experiment(experiment_name)

    datamodule = EuroSATDataModule(batch_size=cfg.data.batch_size)
    model = EuroSATResNet(num_classes=cfg.model.num_classes, lr=cfg.model.learning_rate)

    # MLflow autologging
    mlflow.pytorch.autolog(log_models=True)

    with mlflow.start_run():
        trainer = pl.Trainer(
            max_epochs=cfg.trainer.max_epochs,
            accelerator=cfg.trainer.accelerator,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
        )
        trainer.fit(model, datamodule=datamodule)

# -----------------------------
# 3. Main training with MLflow
# -----------------------------
if __name__ == "__main__":
   main()
