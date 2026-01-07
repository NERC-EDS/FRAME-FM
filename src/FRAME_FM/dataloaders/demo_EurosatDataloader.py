# src/FRAME_FM/dataloaders/eurosat_datamodule.py

from __future__ import annotations

from typing import Optional, Any

import torch
from torchgeo.datasets import EuroSAT100 as EuroSAT

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
        EuroSAT(self.data_root, download=True)

    def _load_raw_data(self) -> Any:
        """
        Load the full EuroSAT dataset once, with no transform.
        Split and per-split transforms are handled later.
        """
        return EuroSAT(self.data_root, download=False, transforms=None)

    def _create_datasets(self, stage: Optional[str] = None) -> None:
        """
        TorchGeo datasets return dict samples (e.g. {"image": ..., "label": ...}).
        So transforms should be applied to the whole sample dict, not just the image.

        We split the raw dataset, then apply transforms per split via a small wrapper.
        """
        full_ds = self._raw_data
        train_base, val_base, test_base = self._split_dataset(full_ds)

        def apply_transforms(ds, tfm):
            if ds is None or tfm is None:
                return ds

            class _Wrapped(torch.utils.data.Dataset):
                def __init__(self, base, transform):
                    self.base = base
                    self.transform = transform

                def __len__(self):
                    return len(self.base)

                def __getitem__(self, idx):
                    sample = self.base[idx]
                    if self.transform is not None:
                        sample["image"] = self.transform(sample["image"])
                    return sample


            return _Wrapped(ds, tfm)

        self.train_dataset = apply_transforms(train_base, self.train_transforms)
        self.val_dataset = apply_transforms(val_base, self.val_transforms)
        self.test_dataset = apply_transforms(test_base, self.test_transforms) if test_base is not None else None
