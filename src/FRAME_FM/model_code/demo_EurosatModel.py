# src/FRAME_FM/model_code/demo_EurosatModel.py
"""
Example Model code for processing EuroSAT dataset

Two models are available.
1. a Pretrained ResNet18 with weights from PyTorch Model Zoo
2. An autoencoder from a [Nature Paper on Flood Analysis](https://www.nature.com/articles/s41598-025-96781-2)

"""

from typing import Any
import torch
import torch.nn as nn
from torchmetrics import Accuracy

from src.FRAME_FM.utils.LightningModuleWrapper import BaseModule


class BaselineConvAE(BaseModule):
    """
    This model consists of a convolutional encoder that progressively increases feature channels through stacked Conv-ReLU blocks,
      followed by a latent projection layer that compresses the final feature map into a fixed-size vector via adaptive average pooling
      and a linear layer. A corresponding inverse mapping (`self.from_latent`) expands latent vectors back into channel-first feature tensors,
      supporting downstream decoding or reconstruction.

    > Thappitla, R.S., Villuri, V.G.K. & Kumar, S. An autoencoder driven deep learning geospatial approach to flood vulnerability analysis
      in the upper and middle basin of river Damodar. Sci Rep 15, 33741 (2025). https://doi.org/10.1038/s41598-025-96781-2

    """

    def __init__(
        self,
        in_channels: int = 13,
        base_ch: int = 32,
        k_size: int = 5,
        latent_dim: int = 256,
        num_classes=3,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        # encoder part - 4 conv layers with ReLU and maxpooling
        chs = [in_channels, base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        # assume input is the tensor size (B, in_channels, W, H)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=chs[0],
                out_channels=chs[1],
                kernel_size=k_size,
                stride=1,
                padding=1,
            ),  # expected output size (B, base_ch, W - k_size + 1, H - k_size + 1)
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=chs[1],
                out_channels=chs[2],
                kernel_size=k_size,
                stride=1,
                padding=1,
            ),  # expected output size (B, base_ch*2, W - 2*k_size + 2, H - 2*k_size + 2)
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=chs[2],
                out_channels=chs[3],
                kernel_size=k_size,
                stride=1,
                padding=1,
            ),  # expected output size (B, base_ch*4,  W - 3*k_size + 3, H - 3*k_size + 3)
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=chs[3],
                out_channels=chs[4],
                kernel_size=k_size,
                stride=1,
                padding=1,
            ),  # expected output size (B, base_ch*8,  W - 4*k_size + 4, H - 4*k_size + 4)
            nn.ReLU(inplace=True),
        )

        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # expected output size (B, chs[4], 1, 1)
            nn.Flatten(),  # expected output size (B, chs[4] * 1 * 1)
            nn.Linear(
                in_features=chs[4], out_features=latent_dim
            ),  # expected output size (B, latent_dim)
        )

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, chs[4]),
            nn.Unflatten(1, (chs[4], 1, 1)),  # need to make sure the dimensions match
        )

        self.classifier = nn.Linear(latent_dim, num_classes)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward Loop for model data flow

        Method returns 'x' processed through:
         Encoder -> Latent Dims -> Classifier
        """
        return self.classifier(self.to_latent(self.encoder(x)))

    def _sharedStep(self, batch, stage):
        """
        Shared step method for train/val/test stages

        param: batch - Batch Data
        param: stage - `str` of 'train'/'val'/'test'
        """

        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)

        if stage == "train":
            acc = self.train_acc(logits, y)
        elif stage == "val":
            acc = self.val_acc(logits, y)
        else:  # "test"
            acc = self.test_acc(logits, y)

        logs = {"ce loss": loss, "acc": acc}

        return loss, logs

    def training_step_body(self, batch):
        return self._sharedStep(batch, stage="train")

    def validation_step_body(self, batch):
        return self._sharedStep(batch, stage="val")

    def test_step_body(self, batch):
        return self._sharedStep(batch, stage="test")
