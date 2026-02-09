from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset
import xarray as xr

from .settings import DatasetSettings as SETTINGS
from .zarr_writer import cache_data_to_zarr


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
        self.data = [1, 2, 3]  # Placeholder for actual data loading logic

    def precache_data(self):
        var_zarr_dict = cache_data_to_zarr(selectors=self.selectors, cache_dir=self.cache_dir, 
                           generate_stats=self.generate_stats,
                            pre_transforms=self.pre_transforms)
        self.is_precached = True
#        self.data = open_zarrs(var_zarr_dict)  # Placeholder for actual logic to open the cached Zarr files and load them into memory)

    def __len__(self):
        if not self.is_precached:
            raise RuntimeError("Data must be precached before accessing length.")
        
        return len(self.data)

    def __getitem__(self, idx):
        # Return the data sample at the specified index
        # In this example, the data is one time slice for all variables
        if not self.is_precached:
            raise RuntimeError("Data must be precached before accessing length.")
        


        return self.data[idx]
    

