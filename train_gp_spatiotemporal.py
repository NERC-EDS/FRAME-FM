"""
Training script for Gaussian Process spatiotemporal modeling on oceanographic glider data.

This example demonstrates:
- Loading and preprocessing oceanographic time series data
- Training a Gaussian Process with ARD kernel using GPyTorch
- PyTorch Lightning integration for training loop
- Hydra for configuration management
- MLflow for experiment tracking

The model predicts temperature from latitude, longitude, time, and depth.
"""

import os
from typing import Optional, Tuple

import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import gpytorch

import mlflow


class GliderDataModule(pl.LightningDataModule):
    """DataModule for oceanographic glider data.
    
    Handles loading, preprocessing, and temporal splitting of glider sensor data.
    Features include latitude, longitude, time (days since start), and depth.
    Target is temperature.
    
    Args:
        csv_path: Path to the glider CSV file.
        batch_size: Batch size for dataloaders.
        val_split: Fraction of data to use for validation (temporal split).
        num_workers: Number of dataloader workers.
    """
    
    def __init__(
        self,
        csv_path: str,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 0
    ):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        
        # Will be set during setup()
        self.scaler_X = None
        self.min_timestamp = None
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Load and preprocess the glider data.
        
        Preprocessing steps:
        1. Load CSV
        2. Select relevant columns
        3. Convert time to days since first observation
        4. Remove NaN values
        5. Sort by time (critical for temporal split)
        6. Split into train/val (temporal: first 80%, last 20%)
        7. Normalize features (fit on train only)
        
        Args:
            stage: Training stage ('fit', 'validate', 'test', or None).
        """
        print(f"Loading data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        
        # Select relevant columns
        columns_needed = ['latitude', 'longitude', 'time', 'GLIDER_DEPTH', 'TEMP']
        df = df[columns_needed].copy()
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Convert time to days since first observation
        self.min_timestamp = df['time'].min()
        df['time_days'] = (df['time'] - self.min_timestamp).dt.total_seconds() / (24 * 3600)
        
        # Remove rows with NaN values
        df_clean = df.dropna(subset=['latitude', 'longitude', 'time_days', 'GLIDER_DEPTH', 'TEMP'])
        print(f"Removed {len(df) - len(df_clean)} rows with NaN values")
        print(f"Remaining rows: {len(df_clean)}")
        
        # Sort by time (critical for temporal split)
        df_clean = df_clean.sort_values('time')
        
        # Temporal split: first 80% train, last 20% validation
        split_idx = int(len(df_clean) * (1 - self.val_split))
        train_df = df_clean.iloc[:split_idx]
        val_df = df_clean.iloc[split_idx:]
        
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
        
        # Extract features and targets
        feature_cols = ['latitude', 'longitude', 'time_days', 'GLIDER_DEPTH']
        X_train = train_df[feature_cols].values
        y_train = train_df['TEMP'].values.reshape(-1, 1)
        X_val = val_df[feature_cols].values
        y_val = val_df['TEMP'].values.reshape(-1, 1)
        
        # Normalize features: fit on train only, then transform both
        # This prevents data leakage
        self.scaler_X = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        
        print(f"Feature means: {self.scaler_X.mean_}")
        print(f"Feature stds: {self.scaler_X.scale_}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        # Create datasets
        self.train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    def train_dataloader(self):
        """Return training dataloader.
        
        Returns:
            DataLoader for training data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        """Return validation dataloader.
        
        Returns:
            DataLoader for validation data.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffle for validation
            num_workers=self.num_workers
        )


class GPSpatioTemporalModel(gpytorch.models.ApproximateGP, pl.LightningModule):
    """Gaussian Process model for spatiotemporal prediction.
    
    Uses variational inference with inducing points for scalability.
    Implements RBF kernel with ARD (Automatic Relevance Determination) to learn
    separate lengthscales for each input dimension (lat, lon, time, depth).
    
    Args:
        inducing_points: Tensor of inducing point locations (num_inducing x input_dim).
        learning_rate: Learning rate for Adam optimizer.
    """
    
    def __init__(
        self,
        inducing_points: torch.Tensor,
        learning_rate: float = 0.01
    ):
        # Initialize variational distribution
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        
        # Initialize variational strategy
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        
        # Initialize parent classes
        # Initialize parent classes
        super().__init__(variational_strategy)
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Mean and covariance modules
        self.mean_module = gpytorch.means.ConstantMean()
        
        # RBF kernel with ARD for all 4 dimensions (lat, lon, time, depth)
        # ARD learns separate lengthscales for each dimension
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=4)
        )
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
    def forward(self, x):
        """Forward pass through the GP.
        
        Args:
            x: Input tensor (batch_size x input_dim).
            
        Returns:
            MultivariateNormal distribution over function values.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def training_step(self, batch, batch_idx):
        """Training step.
        
        Computes the variational ELBO (Evidence Lower Bound) loss.
        
        Args:
            batch: Tuple of (x, y) tensors.
            batch_idx: Batch index.
            
        Returns:
            Loss tensor.
        """
        x, y = batch
        y = y.squeeze()  # Remove extra dimension
        
        output = self(x)
        
        # Variational ELBO loss
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood,
            self,
            num_data=len(self.trainer.train_dataloader.dataset)
        )
        loss = -mll(output, y)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step.
        
        Computes ELBO loss and interpretable metrics (MAE, RMSE) in temperature units.
        
        Args:
            batch: Tuple of (x, y) tensors.
            batch_idx: Batch index.
            
        Returns:
            Loss tensor.
        """
        x, y = batch
        y = y.squeeze()
        
        output = self(x)
        
        # Compute ELBO for validation
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood,
            self,
            num_data=y.size(0)
        )
        loss = -mll(output, y)
        
        # Also compute interpretable metrics
        with torch.no_grad():
            pred = self.likelihood(output).mean
            mae = torch.abs(pred - y).mean()
            rmse = torch.sqrt(((pred - y) ** 2).mean())
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)
        self.log('val_rmse', rmse, prog_bar=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        """Prediction step for inference.
        
        Returns predictions with uncertainty estimates.
        
        Args:
            batch: Tuple of (x, y) tensors.
            batch_idx: Batch index.
            
        Returns:
            Dictionary with 'mean' and 'variance' predictions.
        """
        x, _ = batch
        self.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self(x))
            mean = pred_dist.mean
            variance = pred_dist.variance
        
        return {'mean': mean, 'variance': variance}
    
    def configure_optimizers(self):
        """Configure Adam optimizer.
        
        Returns:
            Adam optimizer for model parameters.
        """
        print(self.hparams)
        return torch.optim.Adam(self.parameters(), lr=0.01)
    
    def on_train_end(self):
        """Called at the end of training.
        
        Logs learned kernel hyperparameters to MLflow for analysis.
        """
        # Log kernel hyperparameters
        lengthscales = self.covar_module.base_kernel.lengthscale.detach().cpu().numpy()[0]
        self.log('lengthscale_lat', lengthscales[0])
        self.log('lengthscale_lon', lengthscales[1])
        self.log('lengthscale_time', lengthscales[2])
        self.log('lengthscale_depth', lengthscales[3])
        self.log('outputscale', self.covar_module.outputscale.detach().cpu().item())
        self.log('noise', self.likelihood.noise.detach().cpu().item())
        
        print(f"\n=== Learned Kernel Hyperparameters ===")
        print(f"Lengthscales: lat={lengthscales[0]:.4f}, lon={lengthscales[1]:.4f}, "
              f"time={lengthscales[2]:.4f}, depth={lengthscales[3]:.4f}")
        print(f"Output scale: {self.covar_module.outputscale.item():.4f}")
        print(f"Noise: {self.likelihood.noise.item():.4f}")


@hydra.main(config_path="conf", config_name="config_gp", version_base="1.3")
def main(cfg: DictConfig):
    """Main training function.
    
    Orchestrates data loading, model training, and MLflow logging using
    Hydra configuration.
    
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
    print(f"Experiment: {experiment_name}, Run: {run_name}")
    
    # Initialize data module
    datamodule = GliderDataModule(
        csv_path=cfg.data.csv_path,
        batch_size=cfg.data.batch_size,
        val_split=cfg.data.val_split,
        num_workers=cfg.data.num_workers
    )
    
    # Setup data to get feature dimensions
    datamodule.setup()
    
    # Get a sample batch to determine inducing point locations
    train_loader = datamodule.train_dataloader()
    sample_batch = next(iter(train_loader))
    X_sample = sample_batch[0]
    
    # Initialize inducing points: random subset from training data
    num_inducing = cfg.model.num_inducing
    random_indices = torch.randperm(X_sample.size(0))[:num_inducing]
    inducing_points = X_sample[random_indices]
    
    print(f"\nInitializing GP with {num_inducing} inducing points")
    
    # Initialize model
    model = GPSpatioTemporalModel(
        inducing_points=inducing_points,
        learning_rate=cfg.model.learning_rate
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=mlf_logger,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )
    
    print("\n=== Starting Training ===")
    trainer.fit(model, datamodule=datamodule)
    
    # Prediction demo
    print("\n=== Prediction Demo ===")
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    
    predictions = trainer.predict(model, dataloaders=[val_loader])
    if predictions:
        mean = predictions[0]['mean']
        std = predictions[0]['variance'].sqrt()
        
        print(f"Sample predictions (mean ± std) in °C:")
        for i in range(min(5, len(mean))):
            print(f"  Prediction {i+1}: {mean[i]:.2f} ± {std[i]:.2f} °C")
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
