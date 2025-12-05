
# %% Import necessary libraries
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

torch.set_float32_matmul_precision("medium")



# %% 
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
# 1.a Autoencoder Module (AK)
# -----------------------------
class BaselineConvAE(pl.LightningModule):
    def __init__(self, in_channels: int = 13, base_ch: int = 32, k_size: int = 5, latent_dim: int = 256, num_classes = 3, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        # encoder part - 4 conv layers with ReLU and maxpooling
        chs = [in_channels, base_ch, base_ch*2, base_ch*4, base_ch*8]
       
        # assume input is the tensor size (B, in_channels, W, H)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=chs[0], out_channels=chs[1], kernel_size=k_size, stride=1, padding=1), # expected output size (B, base_ch, W - k_size + 1, H - k_size + 1)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=chs[1], out_channels=chs[2], kernel_size=k_size, stride=1, padding=1), # expected output size (B, base_ch*2, W - 2*k_size + 2, H - 2*k_size + 2)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=chs[2], out_channels=chs[3], kernel_size=k_size, stride=1, padding=1), # expected output size (B, base_ch*4,  W - 3*k_size + 3, H - 3*k_size + 3)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=chs[3], out_channels=chs[4], kernel_size=k_size, stride=1, padding=1), # expected output size (B, base_ch*8,  W - 4*k_size + 4, H - 4*k_size + 4)
            nn.ReLU(inplace=True),
        )
       
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # expected output size (B, chs[4], 1, 1)
            nn.Flatten(), # expected output size (B, chs[4] * 1 * 1)
            nn.Linear(in_features= chs[4], out_features=latent_dim), # expected output size (B, latent_dim)
        )
       
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, chs[4]),
            nn.Unflatten(1, (chs[4], 1, 1)), # need to make sure the dimensions match
        )

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

       
        self.classifier = nn.Linear(latent_dim, num_classes)
    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.to_latent(encoded)
        classified = self.classifier(latent)
        return classified
    
    
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

# %%
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

    # tracking_uri = cfg.mlflow.tracking_uri
    # experiment_name = cfg.mlflow.experiment_name

    # tracking_uri = "file:/gws/ssde/j25b/eds_ai/frame-fm/users/jercar/mlruns"
    # experiment_name = cfg.mlflow.experiment_name
   
    # # MLflow setup
    # mlflow.set_tracking_uri(tracking_uri)  # Local tracking
    # mlflow.set_experiment(experiment_name)

    datamodule = EuroSATDataModule(batch_size=cfg.data.batch_size)
    # model = EuroSATResNet(num_classes=cfg.model.num_classes, lr=cfg.model.learning_rate)

    model = BaselineConvAE(in_channels=13, base_ch=32, k_size=5, latent_dim=256, num_classes=cfg.model.num_classes, lr=cfg.model.learning_rate)

    # # MLflow autologging
    # mlflow.pytorch.autolog(log_models=True)

    with mlflow.start_run():
        trainer = pl.Trainer(
            max_epochs=cfg.trainer.max_epochs,
            accelerator=cfg.trainer.accelerator,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
        )
        trainer.fit(model, datamodule=datamodule)

# %%
# -----------------------------
# 3. Main training with MLflow
# -----------------------------
if __name__ == "__main__":
   main()

# %%
