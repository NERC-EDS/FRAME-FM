# src/FRAME_FM/dataloaders/gridded_dataloader.py
from __future__ import annotations
from typing import Optional, Any, Sequence
from FRAME_FM.utils.LightningDataModuleWrapper import BaseDataModule
from FRAME_FM.datasets.ImageLabel_Dataset import TransformedDataset
from torch.utils.data import Dataset, TensorDataset
import torch
from pathlib import Path
import polars as pl

class TabularDataset(TensorDataset):
    """Very simple tabular dataset: X numeric features, y target."""

    # Inherit TensorDataset behaviour; you might extend this later if needed.
    ...

class GriddedDataModule(BaseDataModule):
    """
    Loads regularly gridded datasets, inherits from the FRAME-FM base class BaseDataModule


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

    def _load_raw_data(self) -> pl.DataFrame:
        path = Path(self.data_root) / self.filename

        if path.suffix == ".shp":
            ... # TODO
            xarr = 1
        elif path.suffix in {".parquet", ".pq"}:
            ... # TODO
            xarr = 1
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")
        return xarr

    def _create_datasets(self, stage: Optional[str] = None) -> None:
        xarr: pl.DataFrame = self._raw_data # TODO

        # TODO
        full_ds = 1

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


def main():
    GriddedDataModule

    pause = 1

if __name__ == "__main__":
    main()