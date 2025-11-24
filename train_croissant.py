
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl
from torchmetrics import MeanSquaredError

import mlcroissant as mlc
from mlcroissant import Dataset

import mlflow
import mlflow.pytorch
import pandas as pd


# ---------------------------------------------------
# 1. LightningModule for a Simple Regression Model
# ---------------------------------------------------
class RegressionModel(pl.LightningModule):
    def __init__(self, input_dim: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        # StandardScaler should be applied in the DataModule, not here.
        # self.scaler = StandardScaler()

        self.layer_1 = nn.Linear(input_dim, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 1)
        self.val_mse = MeanSquaredError()

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.val_mse.update(y_hat, y)
        self.log("val_mse", self.val_mse, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ---------------------------------------------------
# 2. DataModule for a Croissant Dataset
# ---------------------------------------------------
class CroissantDataModule(pl.LightningDataModule):
    def __init__(self, file_path: str, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # Define features and target for the regression task
        self.feature_columns = ["dataRecordSet/GLIDER_DEPTH", "dataRecordSet/PSAL", "dataRecordSet/MOLAR_DOXY"]
        self.target_column = "dataRecordSet/TEMP"

    def prepare_data(self):
        # This will download the metadata and prepare for data loading
        print(f"Loading Croissant metadata from: {self.file_path}")
        Dataset(self.file_path)

    def setup(self, stage=None):
        ta_factory = mlc.torch.LoaderFactory(jsonld="Bella_626_R_867f_70c9_33ec.jsonld")
        specification = {
            "dataRecordSet/GLIDER_DEPTH": mlc.torch.LoaderSpecificationDataType.INFER,
            "dataRecordSet/PSAL": mlc.torch.LoaderSpecificationDataType.INFER,
            "dataRecordSet/MOLAR_DOXY": mlc.torch.LoaderSpecificationDataType.INFER,
        }
        train_data_pipe = ta_factory.as_datapipe(
            record_set="dataRecordSet",
            specification=specification,
        )

        train_dataset = train_data_pipe       
        # # Load the actual data records into a pandas DataFrame
        # croissant_dataset = Dataset(self.file_path)
        # records = croissant_dataset.records("dataRecordSet")
        # # Convert records to a list of dictionaries, then to a DataFrame
        # data_list = []
        # for record in records:
        #     data_list.append(record)
        # df = pd.DataFrame(data_list)
        # print("DataFrame columns:", df.columns)

        # # --- Data Cleaning and Preparation ---
        # # Select only the columns we need
        # all_columns = self.feature_columns + [self.target_column]
        # df2 = df[all_columns].copy()

        # # Drop rows with missing values
        # df2.dropna(inplace=True)

        # # Split into training and validation sets (80/20 split)
        # n_train = int(len(df2) * 0.8)
        # n_val = len(df)2 - n_train
        # X_train, X_val = df2[self.feature_columns][:n_train], df2[self.feature_columns][n_train:]
        # y_train, y_val = df2[[self.target_column]][:n_train], df2[[self.target_column]][n_train:]

        # # --- Scaling ---
        # # Fit scalers on the training data ONLY to avoid data leakage
        # self.feature_scaler.fit(X_train)
        # self.target_scaler.fit(y_train)

        # # Apply scaling to both training and validation data
        # X_train_scaled = self.feature_scaler.transform(X_train)
        # X_val_scaled = self.feature_scaler.transform(X_val)
        # y_train_scaled = self.target_scaler.transform(y_train)
        # y_val_scaled = self.target_scaler.transform(y_val)

        # --- Create TensorDatasets ---
        self.train_dataset = train_dataset # TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32))
        self.val_dataset = train_dataset #(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val_scaled, dtype=torch.float32))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


# -----------------------------
# 3. Main training with MLflow
# -----------------------------
if __name__ == "__main__":
    pl.seed_everything(42)

    CROISSANT_FILE_PATH = "Bella_626_R_867f_70c9_33ec.jsonld" # C:\\work\\gemini\\frame_project\\

    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Croissant-Regression-Demo")

    # Instantiate DataModule and Model
    datamodule = CroissantDataModule(file_path=CROISSANT_FILE_PATH, batch_size=64)
    # The input_dim must match the number of feature columns
    model = RegressionModel(input_dim=len(datamodule.feature_columns), lr=1e-4)

    # MLflow autologging
    mlflow.pytorch.autolog(log_models=True)

    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10
        )
        trainer.fit(model, datamodule=datamodule)

    print("Training complete. View results in the MLflow UI.")
