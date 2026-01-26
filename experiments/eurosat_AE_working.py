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
from torchgeo.datasets import EuroSAT
import mlflow
import mlflow.pytorch


#For the data module
from src.FRAME_FM.utils.LightningModuleWrapper import BaseModule
from src.FRAME_FM.utils.LightningDataModuleWrapper import BaseDataModule


class EuroSATAutoencoder(BaseModule):

    "Class for defining the AE, train and validation steps"

    def __init__(self, config):
        super().__init__(config)  # stores config in self.hparams per as per wrapper convention -- no need to directly save here

        in_channels = config.in_channels
        base_ch     = config.base_ch
        k_size      = config.k_size
        latent_dim  = config.latent_dim

        #These are to store per epoch vectors from latent space
        self.latent_buffer = []
        self.label_buffer = []
        self.max_latents_per_epoch = 2000  

        #Number of channels
        chs = [in_channels, base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        #Encoder
        # input =  [batch_size, chs[0], W, H] #Batch, InChannel, Width, Height; Expected input size (B, 13, 64, 64)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=chs[0], out_channels=chs[1], kernel_size=k_size, stride=1, padding=1), # Output - (B, 32, W, H)
            nn.BatchNorm2d(chs[1]), # Normalise each output to smoothen the loss plot wrt parameters - loss will converge better #output shape unchanged as above
            nn.ReLU(inplace=True), # Activation, shape unchanged
            nn.MaxPool2d(kernel_size=2, stride=2), #Reduces feature maps by taking the max value in each region, output shape is (B, 32, 32, 32)
            
            #Layer 2 (B, 32, 32, 32) --> (B, 64, 16,16)
            nn.Conv2d(in_channels=chs[1], out_channels=chs[2], kernel_size=k_size, stride=1, padding=1),
            nn.BatchNorm2d(chs[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Layer 3 (B, 64, 16,16) --> (B, 128, 8,8)
            nn.Conv2d(in_channels=chs[2], out_channels=chs[3], kernel_size=k_size, stride=1, padding=1),
            nn.BatchNorm2d(chs[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Layer 4 (B, 128, 8,8) --> (B, 256, 4,4)
            nn.Conv2d(in_channels=chs[3], out_channels=chs[4], kernel_size=k_size, stride=1, padding=1), 
            nn.BatchNorm2d(chs[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        #Input to Latent space is of size (B, 256, 4, 4)
        #Flatten and compress to latent space
        self.to_latent = nn.Sequential(
            nn.Flatten(), # expected output size (B, 256 * 4 * 4)
            nn.Linear(in_features=chs[4] * 4 * 4, out_features=latent_dim), # expected output size (B, 256)
        )

        #From Latent space back to feature maps -- input to decoder
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, chs[4] * 4 * 4),               # (B, 256*4)
            nn.Unflatten(1, (chs[4], 4, 4)),                     # (B, 256, 2, 2)
        )
        #Decoder
        self.decoder = nn.Sequential(
            
            #Layer 1 -- (B, 256, 2, 2) -> (B, 128, 4, 4) All *2 now since the input is (B, 256, 4, 4) not (B, 256, 2, 2)
            nn.Upsample(scale_factor=2, mode='nearest'), # (B, 256, 4, 4)
            nn.Conv2d(chs[4], chs[3], kernel_size=k_size, padding=1), # (B, 128, 4, 4)
            nn.BatchNorm2d(chs[3]),
            nn.ReLU(inplace=True), 

            #Layer 2 -- (B, 128, 4, 4) -> (B, 64, 8, 8)
            nn.Upsample(scale_factor=2, mode='nearest'), # (B, 256, 4, 4)
            nn.Conv2d(chs[3], chs[2], kernel_size=k_size, padding=1), # (B, 128, 4, 4)
            nn.BatchNorm2d(chs[2]),
            nn.ReLU(inplace=True), 

            #Layer 3 -- (B, 64, 8, 8) -> (B, 32, 16, 16)
            nn.Upsample(scale_factor=2, mode='nearest'), # (B, 256, 4, 4)
            nn.Conv2d(chs[2], chs[1], kernel_size=k_size, padding=1), # (B, 128, 4, 4)
            nn.BatchNorm2d(chs[1]),
            nn.ReLU(inplace=True), 

            #Layer 4 -- (B, 32, 16, 16) -> (B, inchannels, 32, 32)
            nn.Upsample(scale_factor=2, mode='nearest'), # (B, 256, 4, 4)
            nn.Conv2d(chs[1], in_channels, kernel_size=k_size, padding=1), # (B, 128, 4, 4) #No other layers as final layer; output not passed
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # print("Input shape:", x.shape)
        encoded = self.encoder(x)
        # print("After Encoding: ", x.shape)
        z = self.to_latent(encoded)
        # print("Latent Vis Output: ", z.shape)
        x_recons = self.decoder(self.from_latent(z))
        # print("Reconstructed Output: ", x_recons.shape)
        return x_recons, z

    #What happens in each trainning step
    def training_step_body(self, batch, batch_idx):
        x = batch["image"]
        y = batch.get("label", None) #Incase label is not present
        # print(x.shape)
        x_recons, z = self(x) #self(x) is equivalent to self.forward(x), but we should call it as self(x) (that’s the PyTorch usual way).
        loss = self.loss_fn(x_recons, x)

        with torch.no_grad():
            acc = ((x_recons - x).abs() < 0.1).float().mean()
            tp = (((x_recons > 0.5) & (x > 0.5)).float().sum())
            fp = (((x_recons > 0.5) & (x <= 0.5)).float().sum())
            fn = (((x_recons <= 0.5) & (x > 0.5)).float().sum())
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

        #Need to clear for more than one epochs (and for the sanity check)
        # print(f"[val_epoch_start] Epoch={self.current_epoch}  - clearing latent buffers")
        self.latent_buffer.clear()
        self.label_buffer.clear()


    def validation_step_body(self, batch, batch_idx):

        # Inspect batch structure
        # keys = list(batch.keys())
        # print(f"[val_step] batch_idx={batch_idx}, keys={keys}")
        
        x = batch["image"]
        y = batch.get("label", None) #Incase label is not present
        if y is None:
            y = torch.zeros((x.size(0),), dtype=torch.long, device=x.device)

        # print(f"[val_step] x(Input).shape={tuple(x.shape)}, y(label).shape={tuple(y.shape)}, y.min={int(y.min())}, y.max={int(y.max())}")
        x_recons, z = self(x) #self(x) is equivalent to self.forward(x)

        # print(f"[val_step] z.shape={tuple(z.shape)}, collecting latents")

        loss = self.loss_fn(x_recons, x)
        # print(f"[val_step] loss={float(loss.detach().cpu())}")

        # Collect a capped sample of latents for plotting
        if len(self.latent_buffer) < self.max_latents_per_epoch:
            with torch.no_grad():
                # Keep only as many as we can fit in the cap
                remaining = self.max_latents_per_epoch - len(self.latent_buffer)
                take = min(remaining, z.size(0))
                z_take = z[:remaining].detach().cpu()
                y_take = y[:remaining].detach().cpu()
                self.latent_buffer.append(z_take)
                self.label_buffer.append(y_take)
                print(f"[val_step] collected={take}, total_so_far={sum(t.size(0) for t in self.latent_buffer)}")

        with torch.no_grad():
            acc = ((x_recons - x).abs() < 0.1).float().mean()
            tp = (((x_recons > 0.5) & (x > 0.5)).float().sum())
            fp = (((x_recons > 0.5) & (x <= 0.5)).float().sum())
            fn = (((x_recons <= 0.5) & (x > 0.5)).float().sum())
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

        total = sum(t.size(0) for t in self.latent_buffer)
        # print(f"[val_epoch_end] Epoch={self.current_epoch}, total_collected={total}")

        if not self.latent_buffer:
            # print("[val_epoch_end] No latents collected - skipping plot.")
            return

        # If we collected latents, make a plot
        if self.latent_buffer:
            Z = torch.cat(self.latent_buffer, dim=0)  # (N, 256)
            y = torch.cat(self.label_buffer, dim=0)   # (N)
            # print(f"[val_epoch_end] Z.shape={tuple(Z.shape)}, y.shape={tuple(y.shape)}")

            #Subtract the average latent vector from every point
            Z_centered = Z - Z.mean(dim=0, keepdim=True)

            # 2D PCA with torch (no extra deps), V has the eigenvectors of shape = (256,q) V[:, 0] → direction of maximum variation, V[:, 1] → second‑most variation (orthogonal to the first)
            U, S, V = torch.pca_lowrank(Z_centered, q=2)
            # print(f"[val_epoch_end] PCA S (singular values)={S.tolist()}")

            #Matrix multiplication to do PCA; above just found the best eigenvectors
            Z_2d = Z_centered @ V[:, :2]
            # print(f"[val_epoch_end] Z_2d.shape={tuple(Z_2d.shape)} - plotting")

            # Plot
            fig, ax = plt.subplots(figsize=(6, 6))
            sc = ax.scatter(
                Z_2d[:, 0].numpy(), Z_2d[:, 1].numpy(),
                c=y.numpy(), cmap='tab10', s=6, alpha=0.8
            )
            ax.set_title("Latent space (PCA to 2D)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Class label")

            # Save locally
            out_dir = os.getcwd()
            out_path = os.path.join(out_dir, f"latent_pca_epoch_{self.current_epoch:03d}.png")
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            # Log to MLflow under the active run
            try:
                mlflow.log_artifact(out_path, artifact_path="latent_viz")
            except Exception as e:
                print(f"MLflow artifact log failed: {e}")

            # Clear buffers for the next epoch
            self.latent_buffer.clear()
            self.label_buffer.clear()

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, mode="min", factor=0.5, patience=5
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
    #     }

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



class EuroSATDataModule(BaseDataModule):
    """
    Wrap EuroSAT in BaseDataModule: implement _create_datasets and assign
    self.train_dataset, self.val_dataset. Dataloaders are provided by BaseDataModule.
    """

    def __init__(self, data_root: str = "data", batch_size: int = 32, num_workers: int = 4, **base_kwargs):
        super().__init__(data_root=data_root, batch_size=batch_size, num_workers=num_workers, **base_kwargs)

    def _create_datasets(self, stage=None):
        dataset = EuroSAT(self.data_root, transforms=None, download=True)
        n_train = int(len(dataset) * 0.8)
        n_val = len(dataset) - n_train
        self.train_dataset, self.val_dataset = random_split(dataset, [n_train, n_val])

#load configurations through Hydra
@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.trainer.seed)

    # Initialize MLflow setup --params, metrics, and model artifacts get captured automatically during training
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.pytorch.autolog(log_models=True)

    #Load Datamodule from EuroSATDataModule -- this initiates create dataset which downloads the EUROSAT data in cfg.data.data_root
    datamodule = EuroSATDataModule(
        data_root=cfg.data.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
    )
    model = EuroSATAutoencoder(cfg.model)

    #This is where the trainer starts
    with mlflow.start_run():
        trainer = pl.Trainer(
            max_epochs=cfg.trainer.max_epochs,
            accelerator=cfg.trainer.accelerator,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            # limit_val_batches=2, 
            # num_sanity_val_steps=0,
            enable_progress_bar=True,
        )
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()