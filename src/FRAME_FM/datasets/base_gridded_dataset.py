from pathlib import Path

import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

from FRAME_FM.utils.data_utils import load_data_from_uri
from FRAME_FM.transforms import resolve_transform


class BaseGriddedTimeSeriesDataset(Dataset):
    # Define transforms that are always appended to the end of the transforms list in any child class. 
    # This ensures that the data is always converted to a tensor and has a "variable" dimension for 
    # the model to work with, even if the user doesn't specify these transforms themselves.
    _transforms = [
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable"},
        {"type": "to_tensor"}
    ]

    def __init__(self, 
                    data_uri: str | Path,
                    transforms: list | None = None,
                    time_range: tuple | None = None,
                    time_stride: int = 1,
                    chunks: dict | None = None,
                    override_transforms: bool = False
                 ):
        self.data_uri = data_uri
        self.transforms = self._unify_transforms(transforms, override_transforms)
        self.time_range = time_range
        self.time_stride = time_stride   # Used for temporal subsetting if needed.
        self.chunks = chunks or {"time": 64}  # Default chunking strategy to ensure Dask is used

        # Load the dataset ready for training
        subset_selection = {"time": time_range} if time_range else {}
        self.data = load_data_from_uri(self.data_uri, chunks=self.chunks, subset_selection=subset_selection)   # type: ignore

    def _unify_transforms(self, transforms: list | None, override_transforms: bool) -> list:
        transforms = transforms or []
        if override_transforms:
            return transforms
        else:
            consolidated_transforms = []
            for transform in transforms + self._transforms:
                if transform["type"] not in [t["type"] for t in consolidated_transforms]:
                    consolidated_transforms.append(transform)
            return consolidated_transforms

    def __len__(self) -> int:
        return len(self.data.time) // self.time_stride

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return the data sample at the specified index
        time_slice = slice(idx * self.time_stride, (idx + 1) * self.time_stride)
        sample = self.data.isel(time=time_slice)

        # Apply runtime transforms if any
        for transform in self.transforms:
            sample = resolve_transform(transform)(sample)

        return sample  # type: ignore
    

if __name__ == "__main__":

    data_uri="tests/fixtures/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json.zip"
    transforms = [
        {"type": "subset", "time": ("2000-01-01", "2000-01-10"), "latitude": (60, -30), "longitude": (40, 100)},
        {"type": "vars_to_dimension", "variables": ["d2m", "d2m", "d2m"], "new_dim": "variable"},
        {"type": "to_tensor"}
    ]

    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=data_uri,
        transforms=transforms,
        time_stride=8
    )

    print(f"Transform list on dataset: {dataset.transforms}")

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]

    print(f"Sample shape: {sample.shape}")
    next_sample = next(iter(dataset))
    print(f"Next sample shape: {next_sample.shape}")

    # ---

    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=data_uri,
        transforms=[],
        time_stride=8
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]