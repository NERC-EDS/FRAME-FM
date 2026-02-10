from typing import List, Dict, Any
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
import xarray as xr
import numpy as np

from .settings import DatasetSettings as SETTINGS
from .zarr_writer import cache_data_to_zarr
from .utils import open_zarrs
from .transforms import resolve_transform


class BigGeoDataset(Dataset):
    def __init__(self, 
                    selectors: list[dict],
                    cache_dir: Path | str = SETTINGS.cache_dir,
                    pre_transforms: list | None = None,
                    runtime_transforms: list | None = None,
                    generate_stats: bool = True
                 ):
        """
        The `BigGeoDataset` will have the following constructor arguments:
        """
        self.selectors = selectors
        self.cache_dir = cache_dir
        self.pre_transforms = pre_transforms or []
        self.runtime_transforms = runtime_transforms or []  # PyTorch transforms
        self.generate_stats = generate_stats  # Should we accumulate/cache stats (e.g. min, max) during the precaching step?
        self.is_precached = False

    def precache_data(self):
        var_zarr_dict = cache_data_to_zarr(selectors=self.selectors, cache_dir=self.cache_dir, 
                           generate_stats=self.generate_stats,
                            pre_transforms=self.pre_transforms)
        self.is_precached = True

        # Load the cached Zarr files into memory
        self.data = open_zarrs(var_zarr_dict)

    def __len__(self):
        if not self.is_precached:
            raise RuntimeError("Data must be precached before accessing length.")
        
        # Return longest time dimension across all datasets as the length of the dataset
        return max(len(uri_dict["xr_dset"].time) for uri_dict in self.data.values())

    def __getitem__(self, idx):
        # Return the data sample at the specified index
        # In this example, the data is one time slice for all variables
        if not self.is_precached:
            raise RuntimeError("Data must be precached before accessing length.")
        
        # idx refers to each time step across all datasets, stack all arrays along the variable dimension
        # and convert to tensor in the runtime transforms
        sample_slices = []
        for uri, uri_dict in self.data.items():
            ds = uri_dict["xr_dset"]
            # Extract the time slice for the current index
            variables = uri_dict["variables"]
            for var in variables:
                sample_slices.append(ds[var].isel(time=idx).values)  # Shape: (height, width)

        sample = np.stack(sample_slices, axis=0)  # Shape: (num_variables, height, width)

        # Apply runtime transforms if any
        for transform in self.runtime_transforms:
            sample = resolve_transform(transform)(sample)

        return sample
    

