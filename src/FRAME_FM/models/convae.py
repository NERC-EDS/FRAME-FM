"""
This demo shows the application of convolutional autoencoder to a stack of
geospatial tiles. A ConvAutoencoder class is defined to readh tiles form the
input batch and pass them through the convolutional encoder-decodernetwork. 
"""

from click.core import batch
import matplotlib
matplotlib.use("Agg") #Ensure a non-interactive Matplotlib backend
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl



from FRAME_FM.utils.LightningModuleWrapper import BaseModule


class ConvAutoencoder(BaseModule):

    "Class for defining the AE, train and validation steps"

    def __init__(self, 
                 in_channels: int=3, 
                 base_channels: int=32, 
                 kernel_size: int=3, 
                 latent_dim: int=256, 
                 input_dim: int= 32,
                 plotting: bool = False,
                 lr=0e-3, 
                 weight_decay=1e-5):
        super().__init__()  # stores config in self.hparams per as per wrapper convention -- no need to directly save here

        self.in_channels = in_channels
        self.base_ch     = base_channels
        self.k_size      = kernel_size
        self.latent_dim  = latent_dim
        self.input_dim   = input_dim
        self.encoder_output_dim = self.input_dim // 2 ** 4  # After 4 max-pool layers, the spatial dimensions are reduced by a factor of 16 (2^4)
        self.plotting = plotting

        #These are to store input tiles and reconstructed tiles
        self.input_tile_buffer = []
        self.reconstructed_tile_buffer = []

        #Number of channels
        chs = [self.in_channels, self.base_ch, self.base_ch * 2, self.base_ch * 4, self.base_ch * 8]

        #Encoder
        # input =  [batch_size, chs[0], W, H] #Batch, InChannel, Width, Height; Expected input size (B, InChannel, 64, 64)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=chs[0], out_channels=chs[1], kernel_size=self.k_size, stride=1, padding=1), # Output - (batch_size, chs[1], W', H')
            nn.BatchNorm2d(chs[1]), # Normalise each output to smoothen the loss plot wrt parameters - loss will converge better #output shape unchanged as above
            nn.ReLU(inplace=True), # Activation, shape unchanged
            nn.MaxPool2d(kernel_size=2, stride=2), #Reduces feature maps by taking the max value in each region, output shape is (batch_size, chs[1], W', H')
            
            #Layer 2 (batch_size, chs[1], W', H') --> (batch_size, chs[2], W'', H'')
            nn.Conv2d(in_channels=chs[1], out_channels=chs[2], kernel_size=self.k_size, stride=1, padding=1),
            nn.BatchNorm2d(chs[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Layer 3 (batch_size, chs[2], W'', H'') --> (batch_size, chs[3], W''', H''')
            nn.Conv2d(in_channels=chs[2], out_channels=chs[3], kernel_size=self.k_size, stride=1, padding=1),
            nn.BatchNorm2d(chs[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Layer 4 (batch_size, chs[3], W''', H''') --> (batch_size, chs[4], W'''', H'''')
            nn.Conv2d(in_channels=chs[3], out_channels=chs[4], kernel_size=self.k_size, stride=1, padding=1), 
            nn.BatchNorm2d(chs[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        #Input to Latent space is of size (batch_size, chs[3], 4, 4)
        #Flatten and compress to latent space
        self.to_latent = nn.Sequential(
            nn.Flatten(), # expected output size (batch_size, chs[4] * W''''* H'''')
            nn.Linear(in_features=chs[4] * self.encoder_output_dim * self.encoder_output_dim, out_features=self.latent_dim), # expected output size (batch_size, latent_dim)
        )

        #From Latent space back to feature maps -- input to decoder
        self.from_latent = nn.Sequential(
            nn.Linear(self.latent_dim, chs[4] * self.encoder_output_dim * self.encoder_output_dim),               # (batch_size, chs[4] * W'''', H'''')
            nn.Unflatten(1, (chs[4], self.encoder_output_dim, self.encoder_output_dim)),                     # (batch_size, chs[4], W'''', H'''')
        )
        #Decoder
        self.decoder = nn.Sequential(
            
            #Layer 1 -- (batch_size, chs[4], W'''', H'''') -> (batch_size, chs[3], W''', H''')
            nn.Upsample(scale_factor=2, mode='nearest'), # (batch_size, chs[4], W''', H''')
            nn.Conv2d(chs[4], chs[3], kernel_size=self.k_size, padding=1), # (batch_size, chs[3], W''', H''')
            nn.BatchNorm2d(chs[3]),
            nn.ReLU(inplace=True), 

            #Layer 2 -- (batch_size, chs[3], W''', H''') -> (batch_size, chs[2], W'', H'')
            nn.Upsample(scale_factor=2, mode='nearest'), # (batch_size, chs[4], W'', H'')
            nn.Conv2d(chs[3], chs[2], kernel_size=self.k_size, padding=1), # (batch_size, chs[2], W'', H'')
            nn.BatchNorm2d(chs[2]),
            nn.ReLU(inplace=True), 

            #Layer 3 -- (batch_size, chs[2], W'', H'') -> (batch_size, chs[1], W', H')
            nn.Upsample(scale_factor=2, mode='nearest'), # (batch_size, chs[4], W', H')
            nn.Conv2d(chs[2], chs[1], kernel_size=self.k_size, padding=1), # (batch_size, chs[1], W', H')
            nn.BatchNorm2d(chs[1]),
            nn.ReLU(inplace=True), 

            #Layer 4 -- (batch_size, chs[1], W', H') -> (batch_size, self.in_channels, W, H)
            nn.Upsample(scale_factor=2, mode='nearest'), # (batch_size, chs[4], W, H)
            nn.Conv2d(chs[1], self.in_channels, kernel_size=self.k_size, padding=1), # (batch_size, self.in_channels, W, H)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # print("Input shape:", x.shape)
        encoded = self.encoder(x)
        # print("After Encoding: ", x.shape)
        latent = self.to_latent(encoded)
        # print("Latent Vis Output: ", z.shape)
        reconstructed = self.decoder(self.from_latent(latent))
        # print("Reconstructed Output: ", reconstructed.shape)
        return reconstructed, latent

    #What happens in each trainning step
    def training_step_body(self, batch, batch_idx):
        # batch is a batch of tiles of shape (B, C, W, H)
        reconstructed, _ = self(batch) #self(x) is equivalent to self.forward(x)
        loss = self.loss_fn(reconstructed, batch)

        with torch.no_grad():
            acc = ((reconstructed - batch).abs() < 0.1).float().mean()
            tp = (((reconstructed > 0.5) & (batch > 0.5)).float().sum())
            fp = (((reconstructed > 0.5) & (batch <= 0.5)).float().sum())
            fn = (((reconstructed <= 0.5) & (batch > 0.5)).float().sum())
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
        self.input_tile_buffer.clear()
        self.reconstructed_tile_buffer.clear()


    def validation_step_body(self, batch, batch_idx):   
        reconstructed, _ = self(batch) #self(x) is equivalent to self.forward(x)
        loss = self.loss_fn(reconstructed, batch)

        # collect a single batch of the input and output per epoch for visualisation
        if len(self.input_tile_buffer) == 0 and len(self.reconstructed_tile_buffer) == 0:
            with torch.no_grad():
                self.input_tile_buffer.append(batch.detach().cpu())
                self.reconstructed_tile_buffer.append(reconstructed.detach().cpu())

        with torch.no_grad():
            acc = ((reconstructed - batch).abs() < 0.1).float().mean()
            tp = (((reconstructed > 0.5) & (batch > 0.5)).float().sum())
            fp = (((reconstructed > 0.5) & (batch <= 0.5)).float().sum())
            fn = (((reconstructed <= 0.5) & (batch > 0.5)).float().sum())
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
            if self.current_epoch % 10 == 0 and self.plotting:  
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