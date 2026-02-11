# src/FRAME_FM/dataloaders/demo_dataloader.py

from __future__ import annotations

from typing import Optional, Any

from torchvision.datasets import EuroSAT

from FRAME_FM.utils.LightningDataModuleWrapper import BaseDataModule
from FRAME_FM.datasets.ImageLabel_Dataset import TransformedDataset

class EuroSATDataModule(BaseDataModule):
    """
    FRAME-FM DataModule for EuroSAT.

    - Uses BaseDataModule for split logic (`split_strategy`, indices/fractions).
    - Uses Hydra-provided transforms (`train_transforms`, `val_transforms`, `test_transforms`).
    - Wraps split datasets in `TransformedDataset` so each split can have its own transform.
    """

    def __init__(
        self,
        data_root: str = "data",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **kwargs,
        )

    def prepare_data(self) -> None:
        """Download EuroSAT once."""
        EuroSAT(root=self.data_root, download=True)

    def _load_raw_data(self) -> Any:
        """
        Load the full EuroSAT dataset once, with no transform.
        Split and per-split transforms are handled later.
        """
        return EuroSAT(root=self.data_root, download=False, transform=None)

    def _create_datasets(self, stage: Optional[str] = None) -> None:
        """
        - Take the full dataset (`self._raw_data`).
        - Use BaseDataModule._split_dataset(...) to create train/val/test splits
          based on `split_strategy` + indices/fractions from config.
        - Wrap each split in TransformedDataset with the appropriate transform
          provided via Hydra (train_transforms, val_transforms, test_transforms).
        """
        full_ds = self._raw_data
        train_base, val_base, test_base = self._split_dataset(full_ds)

        self.train_dataset = TransformedDataset(
            train_base,
            transform=self.train_transforms,
        )
        self.val_dataset = TransformedDataset(
            val_base,
            transform=self.val_transforms,
        )
        # test_base may be None if no test split configured
        self.test_dataset = (
            TransformedDataset(test_base, transform=self.test_transforms)
            if test_base is not None
            else None
        )