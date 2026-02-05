# src/FRAME_FM/training/logger.py

#AK: had to rename this from logging as it was creating a circular import issue
#with the standard logging module. Eror code below for reference:
# Exception has occurred: AttributeError
# partially initialized module 'logging' has no attribute 'getLogger' (most likely due to a circular import)
#   File "/gws/ssde/j25b/eds_ai/frame-fm/users/akadobgs/FRAME-FM/src/FRAME_FM/training/logging.py", line 3, in <module>
#     from pytorch_lightning.loggers import MLFlowLogger
#   File "/gws/ssde/j25b/eds_ai/frame-fm/users/akadobgs/FRAME-FM/src/FRAME_FM/training/train.py", line 4, in <module>
#     import pytorch_lightning as pl
# AttributeError: partially initialized module 'logging' has no attribute 'getLogger' (most likely due to a circular import)


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
