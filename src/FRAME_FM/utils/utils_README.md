# utils/

Shared utilities used across the project, such as I/O helpers, metrics, and transforms.  
Generic code that supports models, dataloaders, and training logic.

Initially this will contain thin wrapper classes for pytorch lightning to ensure consistency through model development, but retain the flexibility afforded by configs.

### Lightning Wrappers

- `BaseModule`: A wrapper around PyTorch Lightning's LightningModule to enforce consistent logging and provide a clear structure for model training and validation steps.

#### Example Usage

In the model code, you can extend the `BaseModule` to create your own model:

```python
import torch
....
from FRAME_FM.utils.LightningModuleWrapper import BaseModule
class MyModel(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        # Define your model architecture here
        self.net = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    #### OVERRIDE *_body METHODS #### NOT THE MAIN STEP METHODS (validation/training_step)
    def training_step_body(self, batch: Any, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        logs: Dict[str, Any] = {
            "mse": loss,
        }
        return loss, logs

    def validation_step_body(self, batch: Any, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        logs = {
            "mse": loss,
        }
        return loss, logs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)