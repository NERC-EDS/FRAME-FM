# Import necessary libraries
import os  # Used for operating system dependent functionality, like creating directories
import tempfile  # Used to create temporary files and directories
import hydra  # Hydra for configuration management
from omegaconf import DictConfig  # Type hint for Hydra's config object
import pytorch_lightning as pl  # The main PyTorch Lightning library
from pytorch_lightning.loggers import MLFlowLogger  # Logger for MLflow integration

# Import components from the torchx lightning example
from torchx.examples.apps.lightning.data import (
    create_random_data,  # Function to create a dummy dataset
    TinyImageNetDataModule,  # The LightningDataModule for our dataset
)
from torchx.examples.apps.lightning.model import (
    TinyImageNetModel,  # The LightningModule we will train
)

# --- Hydra Configuration ---
# The @hydra.main decorator turns this function into a Hydra-configurable application.
# config_path: Points to the 'conf' directory where 'config.yaml' is located.
# config_name: Specifies the name of the configuration file (without the .yaml extension).
# version_base: Sets the Hydra version base to avoid breaking changes.
@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main function to run the training and tracking workflow, configured by Hydra.
    'cfg' is a DictConfig object automatically populated by Hydra from the YAML file.
    """
    # --- 1. Set up MLflow Tracking using Hydra Config ---
    # Access configuration values using dot notation (e.g., cfg.mlflow.tracking_uri).
    tracking_uri = cfg.mlflow.tracking_uri
    experiment_name = cfg.mlflow.experiment_name
    run_name = cfg.mlflow.run_name

    # Create an MLFlowLogger instance.
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
    )
    print(f"MLflow tracking enabled. View results at: {tracking_uri}")
    print(f"Hydra config: \n{cfg}")

    # --- 2. Prepare the Data using Hydra Config ---
    with tempfile.TemporaryDirectory() as data_dir:
        print(f"Generating random image data in: {data_dir}")
        # Use parameters from the 'data' section of the config file.
        create_random_data(data_dir, num_images=cfg.data.num_images)

        # Instantiate the LightningDataModule with config values.
        data_module = TinyImageNetDataModule(
            data_dir=data_dir,
            batch_size=cfg.data.batch_size,
            num_samples=cfg.data.num_samples,
        )

        # --- 3. Initialize the Model using Hydra Config ---
        # Instantiate the LightningModule with the learning rate from the config.
        model = TinyImageNetModel(lr=cfg.model.learning_rate)
        print("TinyImageNetModel initialized.")

        # --- 4. Configure and Run the Trainer using Hydra Config ---
        # Initialize the PyTorch Lightning Trainer with values from the 'trainer' section.
        trainer = pl.Trainer(
            max_epochs=cfg.trainer.max_epochs,
            logger=mlf_logger,
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
        )
        print("PyTorch Lightning Trainer initialized. Starting training...")

        # Start the training process.
        trainer.fit(model, datamodule=data_module)
        print("Training complete.")

        # --- 5. Log the Final Model ---
        model_path = os.path.join(data_dir, "final_model.ckpt")
        trainer.save_checkpoint(model_path)
        print(f"Model checkpoint saved to: {model_path}")

        # Log the saved model checkpoint as an artifact in the MLflow run.
        mlf_logger.experiment.log_artifact(
            run_id=mlf_logger.run_id,
            local_path=model_path,
            artifact_path="checkpoints",
        )
        print("Model logged as an artifact to MLflow.")

if __name__ == "__main__":
    # This block ensures the main function is called only when the script is executed directly.
    main()