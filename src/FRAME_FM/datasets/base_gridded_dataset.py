from pathlib import Path

from polars import time_range
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


class BaseGriddedDataset(BaseDataset):
    _transforms = [
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable"},
        {"type": "to_tensor"}
    ]


class BaseGeoTIFFDataset(BaseDataset):
    _transforms = [
        {"type": "vars_to_dimension", "variables": ["band_data"], "new_dim": "variable"},
        {"type": "to_tensor"}
    ]

    def __len__(self) -> int:
        return len(self.data["band_data"])

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return the data sample at the specified index
        sample = self.data.isel(band=idx)

        # Apply runtime transforms if any
        for transform in self.transforms:
            sample = resolve_transform(transform)(sample)

        return sample  # type: ignore


class BaseASCIIGridDataset(BaseGeoTIFFDataset):
    pass


class BaseGriddedTimeSeriesDataset(BaseDataset):
    # Define transforms that are always appended to the end of the transforms list in any child class. 
    # This ensures that the data is always converted to a tensor and has a "variable" dimension for 
    # the model to work with, even if the user doesn't specify these transforms themselves.
    _transforms = [
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable"},
        {"type": "to_tensor"}
    ]
    DEFAULT_CHUNKS = {"time": 64}  # Default chunking strategy to ensure Dask is used for time series data

    def __init__(self, 
                 data_uri: str | Path | list | tuple,
                 preprocessors: list | None = None,
                 transforms: list | None = None,
                 time_range: tuple | None = None,
                 time_stride: int = 16,
                 chunks: dict | None = None,
                 override_transforms: bool = False
                 ):
        # Set instance variables specific to time series datasets
        self.time_range = time_range
        self.time_stride = time_stride
        self.chunks = chunks or self.DEFAULT_CHUNKS

        # Call super init to set up transforms and preprocessors
        super().__init__(
            data_uri=data_uri,
            preprocessors=preprocessors,
            transforms=transforms,
            chunks=chunks,
            override_transforms=override_transforms
        )

    def _setup_dataset(self):
        # Apply the time selection at the start, to allow any subsequent processing to focus within
        # the selected time range (if specified).
        # Load the dataset ready for training
        # Get the time axis using cf-xarray conventions, and apply the time range selection if specified.
        subset_selection = {"time": self.time_range} if self.time_range else {}
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
    ascii_grid_uri = "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/ihdtm_elevation/data/DTMGEN_HGHT_GB_HGHT_0_0_0_700_1250.asc"

    # Define interval for resampled data in x and y
    stride = 10

    # Set transforms for dataset
    dataset = BaseGriddedDataset(
        data_uri=data_uri,
        preprocessors = [
            {"type": "subset", "band": (0, 1)},
            {"type": "to_dataarray", "var_id": "band_data"},
            {"type": "squeeze"},
            {"type": "resample", "dim": "x", "freq": stride, "method": "mean"},
            {"type": "resample", "dim": "y", "freq": stride, "method": "mean"}
        ],
        transforms=[
#            {"type": "vars_to_dimension", "variables": ["band_data"], "new_dim": "variable"},
            {"type": "to_tensor"}
        ],
        override_transforms=True
    )       

    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == 130, f"Expected dataset length to be 130 after resampling, but got {len(dataset)}"
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
    print("Trying with BaseASCIIGridDataset")
    dataset = BaseASCIIGridDataset(
        data_uri=ascii_grid_uri,
        transforms=[],
        override_transforms=True
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]

    print(f"Sample shape: {sample['band_data'].shape}")   # type: ignore

    # Subsample to every 10th point in x and y dimensions, squeeze and then plot result
    subsampled = sample["band_data"].isel(x=slice(None, None, 10), y=slice(None, None, 10)).squeeze()   # type: ignore
    print(f"Subsampled shape: {subsampled.shape}")
    import matplotlib.pyplot as plt
    subsampled.plot()
    plt.title("Subsampled ASCII Grid Data")
    plt.show()
    img = "subsampled_ascii_grid.png"
    plt.savefig(img)
    print(f"Saved subsampled ASCII grid plot to {img}")

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
    ncpath = "tests/datasets/fixtures/precip_2010.nc"
    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=ncpath,
        transforms=[],
        time_stride=1
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")