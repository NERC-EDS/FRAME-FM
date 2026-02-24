from pathlib import Path

from torch.utils.data import Dataset
import xarray as xr
import numpy as np

from FRAME_FM.utils.data_utils import load_data_from_uri
from FRAME_FM.transforms import resolve_transform


class BaseGriddedTimeSeriesDataset(Dataset):
    def __init__(self, 
                    data_uri: str | Path,
                    transforms: list | None = None,
                    time_range: tuple | None = None,
                    time_stride: int = 1,
                    chunks: dict | None = None
                 ):
        self.data_uri = data_uri
        self.transforms = transforms or []
        self.time_range = time_range
        self.time_stride = time_stride   # Used for temporal subsetting if needed.
        self.chunks = chunks or {"time": 64}  # Default chunking strategy to ensure Dask is used

        # Load the dataset ready for training
        subset_selection = {"time": time_range} if time_range else {}
        self.data = load_data_from_uri(self.data_uri, chunks=self.chunks, subset_selection=subset_selection)   # type: ignore

    def __len__(self):
        return len(self.data.time) // self.time_stride

    def __getitem__(self, idx):
        # Return the data sample at the specified index
        time_slice = slice(idx * self.time_stride, (idx + 1) * self.time_stride)
        sample = self.data.isel(time=time_slice)

        # Apply runtime transforms if any
        for transform in self.transforms:
            sample = resolve_transform(transform)(sample)

        return sample
    

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

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]

    print(f"Sample shape: {sample.shape}")
    next_sample = next(iter(dataset))
    print(f"Next sample shape: {next_sample.shape}")

    # ---
