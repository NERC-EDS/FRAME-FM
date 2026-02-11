# src/FRAME_FM/training/logger.py

from pytorch_lightning.loggers import MLFlowLogger
from typing import Any, Optional

def create_mlflow_logger(
    experiment_name: str,
    tracking_uri: str,
    run_name: Optional[str] = None,
    tags: Optional[dict[str, Any]] = None,
) -> MLFlowLogger:
    return MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags=tags or {},
    )