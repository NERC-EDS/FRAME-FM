import torch
from pathlib import Path

from FRAME_FM.datasets.base_dataset import BaseDataset
from FRAME_FM.utils.data_utils import load_data_from_uri, unify_transforms, get_main_vars
from FRAME_FM.transforms import resolve_transform, apply_preprocessors


class BaseGriddedDataset(BaseDataset):
    _transforms = [
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable", "only_vars_with_time": False},
        {"type": "to_tensor"}
    ]

    def _setup_dataset(self):
        self.data = load_data_from_uri(self.data_uri, chunks=self.chunks)
        self.main_var = get_main_vars(self.data)[0]
        self.first_coord = list(self.data[self.main_var].coords.keys())[0]

    def __len__(self) -> int:
        return len(self.data[self.main_var])

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return the data sample at the specified index
        sample = self.data.isel(**{self.first_coord: idx})

        # Apply runtime transforms if any
        for transform in self.transforms:
            sample = resolve_transform(transform)(sample)

        return sample  # type: ignore


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
    
