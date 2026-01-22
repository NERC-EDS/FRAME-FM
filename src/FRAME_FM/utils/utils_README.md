# utils/

Shared utilities used across the project, such as I/O helpers, metrics, and transforms.  
Generic code that supports models, dataloaders, and training logic.

Initially this will contain thin wrapper classes for pytorch lightning to ensure consistency through model development, but retain the flexibility afforded by configs.

### Lightning Wrappers

- `BaseModule`: A wrapper around PyTorch Lightning's LightningModule to enforce consistent logging and provide a clear structure for model training and validation steps.
- `BaseDataModule`: A wrapper around PyTorch Lightning's LightningDataModule to standardise dataset and dataloader creation.

#### Example Usage

##### 1. Using `BaseModule` for your model
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
```
##### 2. Using `BaseDataModule` for your data
In the dataloader code, you can extend the `BaseDataModule` to create your own data module:
```python
from __future__ import annotations
...
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from .base_datamodule import BaseDataModule


class TabularDataset(TensorDataset):
    """Very simple tabular dataset: X numeric features, y target."""

    # Inherit TensorDataset behaviour; you might extend this later if needed.
    ...


class TabularDataModule(BaseDataModule):
    """
    Example DataModule for tabular data stored as CSV or Parquet.

    Expects:
        - A file at data_root / filename
        - Columns split into features / target via config
    """

    def __init__(
        self,
        data_root: str,
        filename: str,
        feature_cols: list[str],
        target_col: str,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(data_root=data_root, **base_kwargs)
        self.filename = filename
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

    def _load_raw_data(self) -> pd.DataFrame:
        path = Path(self.data_root) / self.filename
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")
        return df

    def _create_datasets(self, stage: Optional[str] = None) -> None:
        df: pd.DataFrame = self._raw_data

        X = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)
        y = torch.tensor(df[self.target_col].values, dtype=torch.float32).unsqueeze(-1)

        full_ds = TabularDataset(X, y)

        n = len(full_ds)
        n_test = int(n * self.test_fraction)
        n_val = int(n * self.val_fraction)
        n_train = n - n_val - n_test

        # simple random split; for more control, use indices + samplers
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds
```

----
To be able to get the most out of these wrappers, you can configure them via Hydra configs. See the [Hydra Configuration System](./configs/configs_README.md) documentation for more details. Here is a brief overview: for the specific example provided above, you would create config files like:

```yaml
# configs/data/tabular.yml
_target_: FRAME_FM.dataloaders.tabular_datamodule.TabularDataModule

data_root: ${env:DATA_ROOT}
filename: "my_table.parquet"

feature_cols:
  - feature_1
  - feature_2
  - feature_3

target_col: "target"

batch_size: 64
num_workers: 4
pin_memory: true
persistent_workers: true

val_fraction: 0.1
test_fraction: 0.1
```