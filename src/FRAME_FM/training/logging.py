# src/FRAME_FM/training/logger.py

from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import DictConfig


def create_mlflow_logger(cfg: DictConfig) -> MLFlowLogger:
    """
    Create an MLFlowLogger from Hydra config.

    Example config block:

    logging:
      _target_: FRAME_FM.training.logger.create_mlflow_logger
      experiment_name: "frame-fm-eurosat-ae"
      tracking_uri: ${env:MLFLOW_TRACKING_URI}
      run_name: "experiment-1"
      tags:
        project: "FRAME-FM"
        dataset: "EuroSAT"
    """
    return MLFlowLogger(
        experiment_name=cfg.experiment_name,
        tracking_uri=cfg.tracking_uri,
        run_name=cfg.get("run_name", None),
        tags=dict(cfg.get("tags", {})),
    )
