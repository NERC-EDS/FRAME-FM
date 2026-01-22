# src/FRAME_FM/utils/LightningDataModuleWrapper.py
from __future__ import annotations

from typing import Optional, Any, Sequence
from abc import ABC, abstractmethod
import warnings
from torch.utils.data import Subset
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Sampler


class BaseDataModule(pl.LightningDataModule, ABC):
    """
    Base class for all DataModules in FRAME-FM.

    - Standardises common arguments (data_root, batch_size, num_workers, etc.).
    - Provides consistent DataLoader construction.
    - Leaves actual dataset creation to subclasses so they can handle
      arbitrary data formats (shapefiles, tabular, NetCDF, etc.).
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        # Optional split controls
        split_strategy: str = "fraction",  # "indices" | "fraction" | "none" | "custom"
        train_split: float = 0.8,
        val_split: float = 0.2,
        test_split: float = 0.0,
        train_indices: Optional[Sequence[int]] = None,
        val_indices: Optional[Sequence[int]] = None,
        test_indices: Optional[Sequence[int]] = None,
        # Optional samplers
        train_sampler: Optional[Sampler[Any]] = None,
        val_sampler: Optional[Sampler[Any]] = None,
        test_sampler: Optional[Sampler[Any]] = None,
        # Optional transforms
        train_transforms: Optional[callable] = None,
        val_transforms: Optional[callable] = None,
        test_transforms: Optional[callable] = None,
    ) -> None:
        super().__init__()

        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        # Split COntrols
        self.split_strategy = split_strategy
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self._train_indices = train_indices
        self._val_indices = val_indices
        self._test_indices = test_indices

        self._train_sampler = train_sampler
        self._val_sampler = val_sampler
        self._test_sampler = test_sampler

        # Will be filled by subclasses in setup()
        self.train_dataset: Optional[Dataset[Any]] = None
        self.val_dataset: Optional[Dataset[Any]] = None
        self.test_dataset: Optional[Dataset[Any]] = None

        # Optional: hold raw data representation (e.g. GeoDataFrame, DataFrame)
        self._raw_data: Any = None

        # Optional: PyTorch Transform objects
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def _split_dataset(
        self, full_ds: Dataset[Any]
    ) -> tuple[Dataset[Any], Dataset[Any], Optional[Dataset[Any]]]:
        n_total = len(full_ds)

        # 1) explicit indices
        if self.split_strategy == "indices":
            if self._train_indices is None or self._val_indices is None:
                raise RuntimeError(
                    "split_strategy='indices' but train/val indices missing."
                )
            train_ds = Subset(full_ds, self._train_indices)
            val_ds = Subset(full_ds, self._val_indices)
            test_ds = (
                Subset(full_ds, self._test_indices)
                if self._test_indices is not None
                else None
            )
            return train_ds, val_ds, test_ds

        # 2) fraction splits
        if self.split_strategy == "fraction":
            n_train = int(n_total * self.train_split)
            n_val = int(n_total * self.val_split)
            if n_train + n_val > n_total:
                raise ValueError("Train + val fraction exceed dataset size.")

            g = torch.Generator().manual_seed(42)
            indices = torch.randperm(n_total, generator=g).tolist()

            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val :]

            train_ds = Subset(full_ds, train_idx)
            val_ds = Subset(full_ds, val_idx)
            test_ds = Subset(full_ds, test_idx) if test_idx else None
            return train_ds, val_ds, test_ds

        # 3) no split
        if self.split_strategy == "none":
            return full_ds, full_ds, None

        raise ValueError(f"Unknown split_strategy={self.split_strategy!r}")

    # --- Subclasses must implement this to create datasets ----
    @abstractmethod
    def _create_datasets(self, stage: Optional[str] = None) -> None:
        """
        Subclasses must implement this method to create and assign
        self.train_dataset, self.val_dataset, and self.test_dataset.

        Makes use of self._train_indices, self._val_indices, self._test_indices
        if provided.

        Can contain custom logic to read data from self.data_root or self._raw_data and process it.
        E.g
        df = pd.read_csv(path)

        z = zarr.open(path)

        or splits

        train_ds = full_ds[gdf.intersects(training_geometry)]
        val_ds   = full_ds[gdf.intersects(validation_geometry)]

        """
        raise NotImplementedError

    def _load_raw_data(self) -> None:
        """
        Optional hook for subclasses to load raw data representation
        (e.g. GeoDataFrame, DataFrame, etc.) before dataset creation.
        """
        return None

    # --- PyTorch Lightning hooks ----
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Called by Lightning at the beginning of training/validation/testing.

        Use to:
        - Optionally load raw data (once)
        - Delegate to _create_datasets to build train/val/test datasets
        """
        if self._raw_data is None:
            self._raw_data = self._load_raw_data()

        self._create_datasets(stage=stage)

        # Basic sanity checks
        if self.train_dataset is None:
            raise RuntimeError(
                "train_dataset has not been created in _create_datasets()"
            )
        if self.val_dataset is None:
            warnings.warn(
                "val_dataset has not been created in _create_datasets(), :: copying train_dataset",
                UserWarning,
                stacklevel=2,
            )
            self.val_dataset = self.train_dataset

    # --- DataLoader methods ----
    def _make_dloader(
        self,
        dataset: Dataset[Any],
        sampler: Optional[Sampler[Any]] = None,
        shuffle: bool = False,
    ) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    # --- Public PyTorch Lightning Hooks ----
    def train_dataloader(self) -> DataLoader[Any]:
        if self.train_dataset is None:
            raise RuntimeError(
                "train_dataset is not set up yet. Did you call setup()? "
            )
        return self._make_dloader(
            self.train_dataset, sampler=self._train_sampler, shuffle=True
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not set up yet. Did you call setup()?")
        return self._make_dloader(
            self.val_dataset, sampler=self._val_sampler, shuffle=False
        )

    def test_dataloader(self) -> DataLoader[Any]:
        if self.test_dataset is None:
            warnings.warn(
                "BaseDataModule: `test_dataset` is None. "
                "Falling back to `val_dataset` or `train_dataset`.",
                UserWarning,
                stacklevel=2,
            )
            self.test_dataset = self.val_dataset or self.train_dataset

        if self.test_dataset is None:
            raise RuntimeError(
                "BaseDataModule: `test_dataset` is None and neither "
                "`val_dataset` nor `train_dataset` are available to fall back to. "
                "Ensure datasets are created in `_create_datasets()` Did you call setup()?"
            )
        return self._make_dloader(
            self.test_dataset, sampler=self._test_sampler, shuffle=False
        )
