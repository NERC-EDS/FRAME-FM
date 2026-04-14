# SPDX-FileCopyrightText: 2026 FRAME-FM Contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

from FRAME_FM.utils.data_utils import (
    load_data_from_uri, unify_transforms, 
    create_cache_path, hash_preprocessors,
    cache_data_to_zarr, preprocessor_hash_key
)

from FRAME_FM.transforms import resolve_transform, apply_preprocessors


class BaseDataset(Dataset):
    _transforms = []

    def __init__(self, 
                 data_uri: str | Path | list | tuple,
                 preprocessors: list | None = None,
                 transforms: list | None = None,
                 chunks: dict | None = None,
                 override_transforms: bool = False,
                 cache_dir: None | Path | str = None,
                 generate_stats: bool = True,
                 force_recache: bool = False
                 ):
        self.data_uri = data_uri
        self.preprocessors = preprocessors or []
        self.transforms = unify_transforms(transforms, self._transforms, override_transforms)
        self.chunks = chunks
        self.cache_dir = cache_dir
        self.generate_stats = generate_stats
        self.force_recache = force_recache

        # If cache_dir is provided, we will attempt to cache the data to Zarr format 
        # (if not already cached) and load from cache for faster subsequent loading.
        if self.cache_dir is not None:
            self.cache_path = create_cache_path(self.data_uri, self.cache_dir)
            self.precache_data()
        else:
            print(f"No cache directory provided, loading data from source.")
            # Either of the following may be overriden in child classes.
            self._setup_dataset()
            self._apply_preprocessors()

    def _setup_dataset(self):
        # Load the dataset ready for training
        self.data = load_data_from_uri(self.data_uri, chunks=self.chunks)

    def _apply_preprocessors(self):
        # Apply preprocessing steps
        self.data = apply_preprocessors(self.data, self.preprocessors)

    def _detect_existing_cache(self):
        # Check if the cache directory exists and contains Zarr files for all selectors
        if not Path(self.cache_dir).exists():
            print(f"No cache directory found at {self.cache_dir}.")
            return False

        if not self.cache_path.exists():
            print(f"Cache not found for URI: {self.data_uri}")
            return False

        # Check if the Zarr file contains the expected cache hash
        _ds = xr.open_zarr(self.cache_path)
        if preprocessor_hash_key not in _ds.attrs:
            print(f"Cache hash not found for URI: {self.data_uri}")
            return False

        # Check if the cache hash matches the current preprocessor list
        zarr_hash = _ds.attrs[preprocessor_hash_key]
        if zarr_hash != hash_preprocessors(self.preprocessors):
            print(f"Cache hash mismatch for URI: {self.data_uri}. Expected: {hash_preprocessors(self.preprocessors)}, Found: {zarr_hash}")
            return False

        print(f"Cache detected for all data in {self.cache_dir}.\nNOT REGENERATING CACHE!")
        return True

    def precache_data(self):
        if not self.force_recache and self._detect_existing_cache():  # If cache exists and force_recache is False, we can skip the caching step
            self.data = load_data_from_uri(
                uri=self.cache_path
            )
        else:
            self.data = cache_data_to_zarr(
                self.data_uri,
                preprocessors=self.preprocessors,
                chunks=self.chunks,
                cache_path=self.cache_path,
                generate_stats=self.generate_stats
            )

        self.is_cached = True

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return the data sample at the specified index
        sample = self.data[idx]

        # Apply runtime transforms if any
        for transform in self.transforms:
            sample = resolve_transform(transform)(sample)

        return sample  # type: ignore
