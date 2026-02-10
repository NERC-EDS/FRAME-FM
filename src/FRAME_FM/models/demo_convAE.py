"""
This demo shows the application of convolutional autoencoder to a stack of
geospatial tiles. Two classes are defined - ConvAutoencoder and
ConvAutoencoderWithLocation. Bof have the same forward method
The first one takes only the tile data as input, while the second one also takes
the coordinates of the tile centroids for potential use in aligning data from
different sources. """

import matplotlib
matplotlib.use("Agg") #Ensure a non-interactive Matplotlib backend
import matplotlib.pyplot as plt
import os
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import random_split
import mlflow
import mlflow.pytorch



from FRAME_FM.utils.LightningModuleWrapper import BaseModule


class ConvAutoencoder(BaseModule):

    "Class for defining the AE, train and validation steps"

    def __init__(self, in_channels: int=3, base_channels: int=32, kernel_size: int=3, latent_dim: int=256, lr=0e-3, weight_decay=1e-5):
        super().__init__()  # stores config in self.hparams per as per wrapper convention -- no need to directly save here

        self.in_channels = in_channels
        self.base_ch     = base_channels
        self.k_size      = kernel_size
        self.latent_dim  = latent_dim

        #These are to store per epoch vectors from latent space, input tiles and reconstructed tiles for plotting and visualisation at the end of each epoch
        self.latent_buffer = []
        self.input_tile_buffer = []
        self.reconstructed_tile_buffer = []
        self.max_latents_per_epoch = 2000  

        #Number of channels
        chs = [self.in_channels, self.base_ch, self.base_ch * 2, self.base_ch * 4, self.base_ch * 8]

        #Encoder
        # input =  [batch_size, chs[0], W, H] #Batch, InChannel, Width, Height; Expected input size (B, InChannel, 64, 64)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=chs[0], out_channels=chs[1], kernel_size=self.k_size, stride=1, padding=1), # Output - (B, 32, W, H)
            nn.BatchNorm2d(chs[1]), # Normalise each output to smoothen the loss plot wrt parameters - loss will converge better #output shape unchanged as above
            nn.ReLU(inplace=True), # Activation, shape unchanged
            nn.MaxPool2d(kernel_size=2, stride=2), #Reduces feature maps by taking the max value in each region, output shape is (B, 32, 32, 32)
            
            #Layer 2 (B, 32, 32, 32) --> (B, 64, 16,16)
            nn.Conv2d(in_channels=chs[1], out_channels=chs[2], kernel_size=self.k_size, stride=1, padding=1),
            nn.BatchNorm2d(chs[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Layer 3 (B, 64, 16,16) --> (B, 128, 8,8)
            nn.Conv2d(in_channels=chs[2], out_channels=chs[3], kernel_size=self.k_size, stride=1, padding=1),
            nn.BatchNorm2d(chs[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Layer 4 (B, 128, 8,8) --> (B, 256, 4,4)
            nn.Conv2d(in_channels=chs[3], out_channels=chs[4], kernel_size=self.k_size, stride=1, padding=1), 
            nn.BatchNorm2d(chs[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        #Input to Latent space is of size (B, 256, 4, 4)
        #Flatten and compress to latent space
        self.to_latent = nn.Sequential(
            nn.Flatten(), # expected output size (B, 256 * 4 * 4)
            nn.Linear(in_features=chs[4] * 2 * 2, out_features=self.latent_dim), # expected output size (B, 256)
        )

        #From Latent space back to feature maps -- input to decoder
        self.from_latent = nn.Sequential(
            nn.Linear(self.latent_dim, chs[4] * 2 * 2),               # (B, 256*4)
            nn.Unflatten(1, (chs[4], 2, 2)),                     # (B, 256, 4, 4)
        )
        #Decoder
        self.decoder = nn.Sequential(
            
            #Layer 1 -- (B, 256, 2, 2) -> (B, 128, 4, 4) All *2 now since the input is (B, 256, 4, 4) not (B, 256, 2, 2)
            nn.Upsample(scale_factor=2, mode='nearest'), # (B, 256, 4, 4)
            nn.Conv2d(chs[4], chs[3], kernel_size=self.k_size, padding=1), # (B, 128, 4, 4)
            nn.BatchNorm2d(chs[3]),
            nn.ReLU(inplace=True), 

            #Layer 2 -- (B, 128, 4, 4) -> (B, 64, 8, 8)
            nn.Upsample(scale_factor=2, mode='nearest'), # (B, 256, 4, 4)
            nn.Conv2d(chs[3], chs[2], kernel_size=self.k_size, padding=1), # (B, 128, 4, 4)
            nn.BatchNorm2d(chs[2]),
            nn.ReLU(inplace=True), 

            #Layer 3 -- (B, 64, 8, 8) -> (B, 32, 16, 16)
            nn.Upsample(scale_factor=2, mode='nearest'), # (B, 256, 4, 4)
            nn.Conv2d(chs[2], chs[1], kernel_size=self.k_size, padding=1), # (B, 128, 4, 4)
            nn.BatchNorm2d(chs[1]),
            nn.ReLU(inplace=True), 

            #Layer 4 -- (B, 32, 16, 16) -> (B, inchannels, 32, 32)
            nn.Upsample(scale_factor=2, mode='nearest'), # (B, 256, 4, 4)
            nn.Conv2d(chs[1], self.in_channels, kernel_size=self.k_size, padding=1), # (B, 128, 4, 4) #No other layers as final layer; output not passed
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # print("Input shape:", x.shape)
        encoded = self.encoder(x)
        # print("After Encoding: ", x.shape)
        z = self.to_latent(encoded)
        # print("Latent Vis Output: ", z.shape)
        reconstructed = self.decoder(self.from_latent(z))
        # print("Reconstructed Output: ", reconstructed.shape)
        return reconstructed, z

    #What happens in each trainning step
    def training_step_body(self, batch, batch_idx):
        x = batch # note that there is no label - entire batch is the input
        # y = batch.get("label", None) #In case label is not present
        # print(x.shape)
        reconstructed, z = self(x) #self(x) is equivalent to self.forward(x), but we should call it as self(x) (that’s the PyTorch usual way).
        loss = self.loss_fn(reconstructed, x)

        with torch.no_grad():
            acc = ((reconstructed - x).abs() < 0.1).float().mean()
            tp = (((reconstructed > 0.5) & (x > 0.5)).float().sum())
            fp = (((reconstructed > 0.5) & (x <= 0.5)).float().sum())
            fn = (((reconstructed <= 0.5) & (x > 0.5)).float().sum())
            precision = tp / (tp + fp + 1e-8)
            recall   = tp / (tp + fn + 1e-8)

        logs = {
            "train_loss": loss,
            "train_acc": acc,
            "train_precision": precision,
            "train_recall": recall,
        }
        return loss, logs

    def on_validation_epoch_start(self):
        self.latent_buffer.clear()


    def validation_step_body(self, batch, batch_idx):        
        x = batch # note that there is no label - entire batch is the input

        reconstructed, z = self(x) #self(x) is equivalent to self.forward(x)

        loss = self.loss_fn(reconstructed, x)

        # Collect a capped sample of latents for plotting
        if len(self.latent_buffer) < self.max_latents_per_epoch:
            with torch.no_grad():
                # Keep only as many as we can fit in the cap
                remaining = self.max_latents_per_epoch - len(self.latent_buffer)
                take = min(remaining, z.size(0))
                z_take = z[:remaining].detach().cpu()
                # self.latent_buffer.append(z_take)
                # print(f"[val_step] collected={take}, total_so_far={sum(t.size(0) for t in self.latent_buffer)}")
                
        # Collect a single batch of the input and output per epoch for visualisation
        if len(self.input_tile_buffer) == 0 and len(self.reconstructed_tile_buffer) == 0:
            with torch.no_grad():
                self.input_tile_buffer.append(x.detach().cpu())
                self.reconstructed_tile_buffer.append(reconstructed.detach().cpu())

        with torch.no_grad():
            acc = ((reconstructed - x).abs() < 0.1).float().mean()
            tp = (((reconstructed > 0.5) & (x > 0.5)).float().sum())
            fp = (((reconstructed > 0.5) & (x <= 0.5)).float().sum())
            fn = (((reconstructed <= 0.5) & (x > 0.5)).float().sum())
            precision = tp / (tp + fp + 1e-8)
            recall   = tp / (tp + fn + 1e-8)

        logs = {
            "val_loss": loss,
            "val_acc": acc,
            "val_precision": precision,
            "val_recall": recall,
        }
        return loss, logs
    

    def on_validation_epoch_end(self):
        # plor input and reconstrtucted tiles every now and then
        if self.input_tile_buffer and self.reconstructed_tile_buffer:
            if self.current_epoch % 50 == 0:
                # Plot the first batch of input and reconstructed tiles side by side for visual comparison
                input_tiles = self.input_tile_buffer[0]  # (B, C, W, H)
                recon_tiles = self.reconstructed_tile_buffer[0]  # (B, C, W, H)

                # For simplicity, we'll just plot the first tile in the batch
                nChannels = input_tiles.shape[1]
                # each channel is a separate image, so we can plot them in a grid
                fig, axes = plt.subplots(2, nChannels, figsize=(3 * nChannels, 6))
                for i in range(nChannels):
                    axes[0, i].imshow(input_tiles[0, i].cpu(), cmap='viridis')
                    axes[0, i].set_title(f"Input Channel {i}")
                    axes[0, i].axis('off')

                    axes[1, i].imshow(recon_tiles[0, i].cpu(), cmap='viridis')
                    axes[1, i].set_title(f"Reconstructed Channel {i}")
                    axes[1, i].axis('off')
                    
                # Save locally
                out_dir = os.path.join(os.getcwd(), "tile_viz")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"tiles_epoch_{self.current_epoch:03d}.png")
                fig.tight_layout()
                fig.savefig(out_path, dpi=150)
                plt.close(fig)

                # # Log to MLflow under the active run
                # try:
                #     mlflow.log_artifact(out_path, artifact_path="tile_viz")
                # except Exception as e:
                #     print(f"[val_epoch_end] MLflow artifact log failed: {e}")

            self.latent_buffer.clear()
            self.input_tile_buffer.clear()
            self.reconstructed_tile_buffer.clear()

    def configure_optimizers(self):
        lr = getattr(self.hparams, "lr", 0.001)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/val_loss"},
        }



# for this class the only change is tha the batch contains data values and coordinates
# of grid cell centroids - has the same forward pas as ConvAutoencoder
class ConvAutoencoderWithLocation(ConvAutoencoder):
    def __init__(self, in_channels: int=3, base_channels: int=32, kernel_size: int=3, latent_dim: int=256, lr=0e-3, weight_decay=1e-5):
        super().__init__(in_channels, base_channels, kernel_size, latent_dim)
        # Additional layers or modifications to incorporate location information
        # can be added here
        
       
    #What happens in each trainning step  
    def training_step_body(self, batch, batch_idx):
        x, coords = batch # note that there is no label - entire batch is the input
        # y = batch.get("label", None) #In case label is not present
        # print(x.shape)
        reconstructed, z = self(x) #self(x) is equivalent to self.forward(x), but we should call it as self(x) (that’s the PyTorch usual way).
        loss = self.loss_fn(reconstructed, x)

        with torch.no_grad():
            acc = ((reconstructed - x).abs() < 0.1).float().mean()
            tp = (((reconstructed > 0.5) & (x > 0.5)).float().sum())
            fp = (((reconstructed > 0.5) & (x <= 0.5)).float().sum())
            fn = (((reconstructed <= 0.5) & (x > 0.5)).float().sum())
            precision = tp / (tp + fp + 1e-8)
            recall   = tp / (tp + fn + 1e-8)

        logs = {
            "train_loss": loss,
            "train_acc": acc,
            "train_precision": precision,
            "train_recall": recall,
        }
        return loss, logs
    
    def validation_step_body(self, batch, batch_idx):
        
        x, coords = batch # input and coords - there must be a better way to call batch elements, perhaps calling them by pre-specified keys
        reconstructed, z = self(x) #self(x) is equivalent to self.forward(x)

        loss = self.loss_fn(reconstructed, x)

        # Collect a capped sample of latents for plotting
        if len(self.latent_buffer) < self.max_latents_per_epoch:
            with torch.no_grad():
                # Keep only as many as we can fit in the cap
                remaining = self.max_latents_per_epoch - len(self.latent_buffer)
                take = min(remaining, z.size(0))
                z_take = z[:remaining].detach().cpu()
                # self.latent_buffer.append(z_take)
                # print(f"[val_step] collected={take}, total_so_far={sum(t.size(0) for t in self.latent_buffer)}")
                
        # Collect a single batch of the input and output per epoch for visualisation
        if len(self.input_tile_buffer) == 0 and len(self.reconstructed_tile_buffer) == 0:
            with torch.no_grad():
                self.input_tile_buffer.append(x.detach().cpu())
                self.reconstructed_tile_buffer.append(reconstructed.detach().cpu())

        with torch.no_grad():
            acc = ((reconstructed - x).abs() < 0.1).float().mean()
            tp = (((reconstructed > 0.5) & (x > 0.5)).float().sum())
            fp = (((reconstructed > 0.5) & (x <= 0.5)).float().sum())
            fn = (((reconstructed <= 0.5) & (x > 0.5)).float().sum())
            precision = tp / (tp + fp + 1e-8)
            recall   = tp / (tp + fn + 1e-8)

        logs = {
            "val_loss": loss,
            "val_acc": acc,
            "val_precision": precision,
            "val_recall": recall,
        }
        return loss, logs
