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