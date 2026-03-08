from pathlib import Path

import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

from FRAME_FM.utils.data_utils import load_data_from_uri, unify_transforms
from FRAME_FM.transforms import resolve_transform, apply_preprocessors


class BaseDataset(Dataset):
    _transforms = []

    def __init__(self, 
                 data_uri: str | Path | list | tuple,
                 preprocessors: list | None = None,
                 transforms: list | None = None,
                 chunks: dict | None = None,
                 override_transforms: bool = False
                 ):
        self.data_uri = data_uri
        self.preprocessors = preprocessors or []
        self.transforms = unify_transforms(transforms, self._transforms, override_transforms)
        self.chunks = chunks

        # Either of the following may be overriden in child classes.
        self._setup_dataset()
        self._apply_preprocessors()

    def _setup_dataset(self):
        # Load the dataset ready for training
        self.data = load_data_from_uri(self.data_uri, chunks=self.chunks)

    def _apply_preprocessors(self):
        # Apply preprocessing steps
        self.data = apply_preprocessors(self.data, self.preprocessors)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return the data sample at the specified index
        sample = self.data[idx]

        # Apply runtime transforms if any
        for transform in self.transforms:
            sample = resolve_transform(transform)(sample)

        return sample  # type: ignore