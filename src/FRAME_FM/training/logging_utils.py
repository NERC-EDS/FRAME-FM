# SPDX-FileCopyrightText: 2026 FRAME-FM Contributors
#
# SPDX-License-Identifier: Apache-2.0

# src/FRAME_FM/training/logger.py

from pytorch_lightning.loggers import CSVLogger, MLFlowLogger
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



def create_csv_logger(
    save_dir: str = ".",
    name: str = "csv_logs",
    version: Optional[str] = None,
) -> CSVLogger:
    return CSVLogger(
        save_dir=save_dir,
        name=name,
        version=version,
    )