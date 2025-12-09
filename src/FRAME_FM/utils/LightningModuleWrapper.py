# src/FRAME_FM/utils/LightningModuleWrapper.py
from typing import Any, Dict, Optional
import pytorch_lightning as pl


class BaseModule(pl.LightningModule):
    """
    A thin wrapper around PyTorch Lightning's LightningModule to allow for future extensions
    and customizations specific to FRAME-FM project needs.

    Subclasses should implement `training_step_body` and `validation_step_body` methods
    instead of `training_step` and `validation_step` directly.

    Enforces consistent logging patterns across training and validation steps.
    Enforces logging of loss by default.
    """

    def __init__(self, model: pl.LightningModule):
        super().__init__()
        self.save_hyperparameters()
        self.model = model

    # ------ OVERWRITES -----

    def log_metrics(
        self, name: str, value: Any, on_step: bool = True, on_epoch: bool = True
    ):
        """Wrapper around self.log to enforce consistent logging defaults."""
        self.log(
            name, value, on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True
        )

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """Default behaviour: call a user-overridable hook and log loss."""
        loss, logs = self.training_step_body(batch, batch_idx)

        # standard logging pattern
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        for k, v in logs.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        """Default behaviour: call a user-overridable hook and log loss."""
        loss, logs = self.validation_step_body(batch, batch_idx)

        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        for k, v in logs.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, logger=True)

        return loss

    # ---- Hooks model developers are expected to override ----
    def training_step_body(
        self, batch: Any, batch_idx: int
    ) -> tuple[Any, Dict[str, Any]]:
        """
        Subclasses implement this instead of training_step.
        Should return (loss, logs_dict).
        """
        raise NotImplementedError

    def validation_step_body(
        self, batch: Any, batch_idx: int
    ) -> tuple[Any, Dict[str, Any]]:
        raise NotImplementedError
