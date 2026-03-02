from pathlib import Path

from polars import time_range
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

from FRAME_FM.utils.data_utils import load_data_from_uri, unify_transforms
from FRAME_FM.transforms import resolve_transform


class BaseGriddedDataset(Dataset):
    _transforms = [
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable"},
        {"type": "to_tensor"}
    ]

    def __init__(self, 
                 data_uri: str | Path | list | tuple,
                 transforms: list | None = None,
                 chunks: dict | None = None,
                 override_transforms: bool = False
                 ):
        self.data_uri = data_uri
        self.transforms = unify_transforms(transforms, self._transforms, override_transforms)
        self.chunks = chunks

        # Load the dataset ready for training
        self.data = load_data_from_uri(self.data_uri, chunks=self.chunks)

    def __len__(self) -> int:
        return len(self.data["band"])

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return the data sample at the specified index
        sample = self.data.isel(band=idx)

        # Apply runtime transforms if any
        for transform in self.transforms:
            sample = resolve_transform(transform)(sample)

        return sample  # type: ignore


class BaseGeoTIFFDataset(Dataset):
    _transforms = [
        {"type": "vars_to_dimension", "variables": ["band_data"], "new_dim": "variable"},
        {"type": "to_tensor"}
    ]

    def __init__(self, 
                 data_uri: str | Path | list | tuple,
                 transforms: list | None = None,
                 chunks: dict | None = None,
                 override_transforms: bool = False
                 ):
        self.data_uri = data_uri
        self.transforms = unify_transforms(transforms, self._transforms, override_transforms)
        self.chunks = chunks

        # Load the dataset ready for training
        self.data = load_data_from_uri(self.data_uri, chunks=self.chunks)

    def __len__(self) -> int:
        return len(self.data["band_data"])

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return the data sample at the specified index
        sample = self.data.isel(band=idx)

        # Apply runtime transforms if any
        for transform in self.transforms:
            sample = resolve_transform(transform)(sample)

        return sample  # type: ignore


# class BaseASCIIGridDataset(Dataset):
#     _transforms = [
#         {"type": "vars_to_dimension", "variables": ["band_data"], "new_dim": "variable"},
#         {"type": "to_tensor"}
#     ]

#     def __init__(self, 
#                  data_uri: str | Path | list | tuple,
#                  transforms: list | None = None,
#                  chunks: dict | None = None,
#                  override_transforms: bool = False
#                  ):
#         self.data_uri = data_uri
#         self.transforms = unify_transforms(transforms, self._transforms, override_transforms)
#         self.chunks = chunks

#         # Load the dataset ready for training: xarray.DataAraray
#         self.data = load_data_from_uri(self.data_uri, chunks=self.chunks)

#     def __len__(self) -> int:
#         return len(self.data.band_data)

#     def __getitem__(self, idx: int) -> torch.Tensor:
#         # Return the data sample at the specified index
#         sample = self.data.isel(band=idx)

#         # Apply runtime transforms if any
#         for transform in self.transforms:
#             sample = resolve_transform(transform)(sample)

#         return sample  # type: ignore



class BaseGriddedTimeSeriesDataset(Dataset):
    # Define transforms that are always appended to the end of the transforms list in any child class. 
    # This ensures that the data is always converted to a tensor and has a "variable" dimension for 
    # the model to work with, even if the user doesn't specify these transforms themselves.
    _transforms = [
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable"},
        {"type": "to_tensor"}
    ]

    def __init__(self, 
                 data_uri: str | Path | list | tuple,
                 transforms: list | None = None,
                 time_range: tuple | None = None,
                 time_stride: int = 16,
                 chunks: dict | None = None,
                 override_transforms: bool = False
                 ):
        self.data_uri = data_uri
        self.transforms = unify_transforms(transforms, self._transforms, override_transforms)
        self.time_range = time_range
        self.time_stride = time_stride   # Used for temporal subsetting if needed.
        self.chunks = chunks or {"time": 64}  # Default chunking strategy to ensure Dask is used

        # Load the dataset ready for training
        subset_selection = {"time": time_range} if time_range else {}
        self.data = load_data_from_uri(self.data_uri, chunks=self.chunks, subset_selection=subset_selection)

    def __len__(self) -> int:
        return len(self.data["time"]) // self.time_stride

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return the data sample at the specified index
        time_slice = slice(idx * self.time_stride, (idx + 1) * self.time_stride)
        sample = self.data.isel(time=time_slice)

        # Apply runtime transforms if any
        for transform in self.transforms:
            sample = resolve_transform(transform)(sample)

        return sample  # type: ignore
    

if __name__ == "__main__":

    # Try: BaseGriddedDataset with a single GeoTIFF file
    data_uri = "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/land_cover_map_2015/data/LCM2015_GB_1km_percent_cover_aggregate_class.tif"

    # Set transforms for dataset
    dataset = BaseGriddedDataset(
        data_uri=data_uri,
        transforms=[
            {"type": "vars_to_dimension", "variables": ["band_data"], "new_dim": "variable"},
            {"type": "to_tensor"}
        ]
    )       

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")

    print("Trying with BaseGeoTIFFDataset")
    dataset = BaseGeoTIFFDataset(
        data_uri=data_uri
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")

    print("---------------------------------------\n")

    data_uri="tests/transforms/fixtures/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json.zip"
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

    print("Trying with a simple NC file")
    ncpath = "tests/datasets/precip_2010.nc"
    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=ncpath,
        transforms=[],
        time_stride=1
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")