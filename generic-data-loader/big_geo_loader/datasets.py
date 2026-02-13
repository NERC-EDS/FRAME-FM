from typing import List, Dict, Any
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
import xarray as xr
import numpy as np

from .settings import DatasetSettings as SETTINGS
from .zarr_writer import cache_data_to_zarr
from .utils import hash_selector, open_cached_zarrs, create_cache_path
from .transforms import resolve_transform


class BigGeoDataset(Dataset):
    def __init__(self, 
                    selectors: list[dict],
                    cache_dir: Path | str = SETTINGS.cache_dir,
                    pre_transforms: list | None = None,
                    runtime_transforms: list | None = None,
                    generate_stats: bool = True,
                    force_recache: bool = False
                 ):
        """
        The `BigGeoDataset` will have the following constructor arguments:
        """
        self.selectors = selectors
        self.cache_dir = cache_dir
        self.pre_transforms = pre_transforms or []
        self.runtime_transforms = runtime_transforms or []  # PyTorch transforms
        self.generate_stats = generate_stats  # Should we accumulate/cache stats (e.g. min, max) during the precaching step?
        self.force_recache = force_recache    # If True, will force regeneration of cache (even if cache exists)
        self.is_cached = False

    def _detect_existing_cache(self):
        # Check if the cache directory exists and contains Zarr files for all selectors
        if not Path(self.cache_dir).exists():
            print(f"No cache directory found at {self.cache_dir}.")
            return False

        for selector in self.selectors:
            zarr_path = create_cache_path(selector["uri"], self.cache_dir)
            if not zarr_path.exists():
                print(f"Cache not found for URI: {selector['uri']}")
                return False

            # Check if the Zarr file contains the expected cache hash
            if "_selection_cache_hash" not in xr.open_zarr(zarr_path).attrs:
                print(f"Cache hash not found for URI: {selector['uri']}")
                return False

            # Check if the cache hash matches the current selector
            zarr_hash = xr.open_zarr(zarr_path).attrs["_selection_cache_hash"]
            if zarr_hash != hash_selector(selector):
                print(f"Cache hash mismatch for URI: {selector['uri']}. Expected: {hash_selector(selector)}, Found: {zarr_hash}")
                return False

        print(f"Cache detected for all selectors in {self.cache_dir}.\nNOT REGENERATING CACHE.")
        return True

    def precache_data(self):
        if not self.force_recache and self._detect_existing_cache():  # If cache exists and force_recache is False, we can skip the caching step
            self.data = open_cached_zarrs(uris=[selector["uri"] for selector in self.selectors], cache_dir=self.cache_dir)
        else:
            self.data = cache_data_to_zarr(selectors=self.selectors, cache_dir=self.cache_dir, 
                            generate_stats=self.generate_stats,
                            pre_transforms=self.pre_transforms)

        self.is_cached = True

    def __len__(self):
        if not self.is_cached:
            raise RuntimeError("Data must be precached before accessing length.")
        
        # Return longest time dimension across all datasets as the length of the dataset
        return max(len(uri_dict["xr_dset"].time) for uri_dict in self.data.values())

    def __getitem__(self, idx):
        # Return the data sample at the specified index
        # In this example, the data is one time slice for all variables
        if not self.is_cached:
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
    

