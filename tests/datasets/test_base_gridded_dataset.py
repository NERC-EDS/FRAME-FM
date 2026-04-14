# SPDX-FileCopyrightText: 2026 FRAME-FM Contributors
#
# SPDX-License-Identifier: Apache-2.0


import os
import json
from pathlib import Path

import torch
import pytest

from .common import (
    GEOTIFF_URI,
    TIMESERIES_URI,
    ASCII_GRID_URI,
    NC_URI,
    ERA5_URI,
)

from FRAME_FM.utils.data_utils import (
    hash_preprocessors,
    safely_remove_dir, 
    preprocessor_hash_key
)

from FRAME_FM.datasets.base_gridded_dataset import (
    BaseGriddedDataset,
    BaseGeoTIFFDataset,
    BaseASCIIGridDataset,
    BaseGriddedTimeSeriesDataset,
)



# ------------------------------------------------
# Check they can read data
# ------------------------------------------------
@pytest.mark.parametrize(
    "dataset_cls,uri",
    [
        (BaseGriddedDataset, GEOTIFF_URI),
        (BaseGeoTIFFDataset, GEOTIFF_URI),
        (BaseASCIIGridDataset, ASCII_GRID_URI),
        (BaseGriddedTimeSeriesDataset, TIMESERIES_URI),
    ],
)
def test_base_datasets_load(dataset_cls, uri):
    dataset = dataset_cls(data_uri=uri)

    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.ndim >= 2


def test_base_gridded_dataset():
    # Define interval for resampled data in x and y
    stride = 10

    # Set transforms for dataset
    dataset = BaseGriddedDataset(
        data_uri=GEOTIFF_URI,
        preprocessors = [
            {"type": "resample", "dim": "x", "freq": stride, "method": "mean"},
            {"type": "resample", "dim": "y", "freq": stride, "method": "mean"}
        ],
        transforms=[
            {"type": "vars_to_dimension", "variables": ["band_data"], "new_dim": "variable"},
            {"type": "to_tensor"}
        ],
        override_transforms=True
    )       

    assert len(dataset) == 10, f"Expected dataset length to be 10, but got {len(dataset)}"
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (1, 130, 70), f"Expected sample shape to be (1, 130, 70) after resampling and squeezing, but got {sample.shape}"



def test_base_geotiff_dataset():
    dataset = BaseGeoTIFFDataset(
        data_uri=GEOTIFF_URI,
        # transforms=[],
        # override_transforms=True
    )

    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.ndim >= 2


def test_base_ascii_grid_dataset():
    dataset = BaseASCIIGridDataset(
        data_uri=ASCII_GRID_URI,
        # transforms=[],
        # override_transforms=True
    )

    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.ndim >= 2






def test_base_timeseries_dataset_with_transforms():
    transforms = [
        {
            "type": "subset",
            "time": ("2000-01-01", "2000-01-10"),
            "latitude": (60, -30),
            "longitude": (40, 100),
        },
        {"type": "vars_to_dimension", "variables": ["d2m", "d2m", "d2m"], "new_dim": "variable"},
        {"type": "to_tensor"},
    ]

    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=TIMESERIES_URI,
        transforms=transforms,
        time_stride=8,
    )

    assert len(dataset) > 0, f"Expected dataset length to be greater than 0 after applying time range and stride, but got {len(dataset)}"   
    sample = dataset[0]
    next_sample = next(iter(dataset))
    assert isinstance(sample, torch.Tensor), f"Expected sample to be a torch.Tensor after transforms, but got {type(sample)}"
    assert isinstance(next_sample, torch.Tensor), f"Expected next_sample to be a torch.Tensor after transforms, but got {type(next_sample)}"


def test_base_timeseries_dataset_nc_file():
    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=NC_URI,
        time_stride=8
    )
    sample = dataset[0]

    assert isinstance(sample, torch.Tensor), f"Expected sample to be a torch.Tensor after transforms, but got {type(sample)}"
    assert sample.ndim >= 2, f"Expected sample to have at least 2 dimensions after transforms, but got {sample.ndim}"


def test_base_timeseries_dataset_with_cache():

    cache_dir = "./test_cache"

    preprocessors = [
        {"type": "subset", "time": ("2000-01-01", "2000-01-10"), "latitude": (60, -30), "longitude": (40, 100)},
    ]
    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=TIMESERIES_URI,
        preprocessors=preprocessors,
        time_stride=8,
        cache_dir=cache_dir,
        chunks="auto"
    )

    assert len(dataset) > 0, f"Expected dataset length to be greater than 0 with cache enabled, but got {len(dataset)}"
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor), f"Expected sample to be a torch.Tensor after transforms, but got {type(sample)}"
    assert sample.ndim >= 2, f"Expected sample to have at least 2 dimensions after transforms, but got {sample.ndim}"

    # Create a version without cache and compare the results - initially the dataset after construction
    dataset_no_cache = BaseGriddedTimeSeriesDataset(
        data_uri=TIMESERIES_URI,
        preprocessors=preprocessors,
        time_stride=8,
        cache_dir=None,
        chunks="auto"
    )

    # Compare the datasets 
    assert dataset.data.equals(dataset_no_cache.data), "Expected cached dataset to have the same data as non-cached dataset after preprocessing, but they differ"
    assert (dataset.data["d2m"].values == dataset_no_cache.data["d2m"].values).all(), "Expected cached dataset variable values to match non-cached dataset variable values after preprocessing, but they differ"

    # Assert that the hash of the preprocessors is stored in the cached dataset attributes and matches the hash of the preprocessors used
    cached_hash = dataset.data.attrs.get(preprocessor_hash_key, None)
    expected_hash = hash_preprocessors(preprocessors)
    assert cached_hash is not None, f"Expected cached dataset to have the hash attribute {preprocessor_hash_key}, but it was not found"
    assert cached_hash == expected_hash, f"Expected cached dataset hash {cached_hash} to match expected hash {expected_hash} based on the preprocessors, but they differ"

    # Assert that the non-cached file does not include the hash attribute in its attributes, since it should not have been modified by the caching process
    non_cached_hash = dataset_no_cache.data.attrs.get(preprocessor_hash_key, None)
    assert non_cached_hash is None, f"Expected non-cached dataset to not have the hash attribute {preprocessor_hash_key}, but it was found with value {non_cached_hash}"


@pytest.mark.parametrize("caching_backend", ["basic", "series", "dask_distributed", "slurm"])
def test_base_timeseries_dataset_caching_backends(caching_backend):

    os.environ["FRAME_CACHING_BACKEND"] = caching_backend
    cache_dir = "./cache-zarrs"
    # Remove cache directory if it already exists to ensure we are testing the caching process from scratch
    safely_remove_dir(cache_dir)

    data_uri = ERA5_URI.replace("*", "2d")

    subset = {"type": "subset", "time": ("2000-01-01 00:00:00", "2000-01-31 23:00:00"), "latitude": (60, -30), "longitude": (40, 100)}
    chunks = {"time": 12, "latitude": 361, "longitude": 720}

    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=data_uri,
        preprocessors=[subset.copy()],
        time_stride=8,
        cache_dir=cache_dir,
        chunks=chunks
    )

    assert len(dataset) > 0, f"Expected dataset length to be greater than 0 with cache enabled, but got {len(dataset)}"
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor), f"Expected sample to be a torch.Tensor after transforms, but got {type(sample)}"
    assert sample.ndim >= 2, f"Expected sample to have at least 2 dimensions after transforms, but got {sample.ndim}"

    # Create a version without cache and compare the results - initially the dataset after construction
    dataset_no_cache = BaseGriddedTimeSeriesDataset(
        data_uri=data_uri,
        preprocessors=[subset.copy()],
        time_stride=8,
        cache_dir=None,
        chunks=chunks
    )

    # Assert that the dataset has the hash key in its attributes, and that it matches the expected hash based on the preprocessors used
    # Assert that the hash of the preprocessors is stored in the cached dataset attributes and matches the hash of the preprocessors used
    cached_hash = dataset.data.attrs.get(preprocessor_hash_key, None)
    expected_hash = hash_preprocessors([subset.copy()])
    assert cached_hash is not None, f"Expected cached dataset to have the hash attribute {preprocessor_hash_key}, but it was not found"
    assert cached_hash == expected_hash, f"Expected cached dataset hash {cached_hash} to match expected hash {expected_hash} based on the preprocessors, but they differ"

    # Assert that the non-cached file does not include the hash attribute in its attributes, since it should not have been modified by the caching process
    non_cached_hash = dataset_no_cache.data.attrs.get(preprocessor_hash_key, None)
    assert non_cached_hash is None, f"Expected non-cached dataset to not have the hash attribute {preprocessor_hash_key}, but it was found with value {non_cached_hash}"

    # Now remove the hash attribute from the cached dataset to simulate the case where we are comparing the datasets without considering the cache metadata, and compare the datasets
    dataset.data.attrs.pop(preprocessor_hash_key, None)

    ds1 = dataset.data.isel(time=0, latitude=0, longitude=slice(-3, None)) # Select a small subset of the data for comparison to speed up the process
    ds2 = dataset_no_cache.data.isel(time=0, latitude=0, longitude=slice(-3, None)) # Select the same small subset of the data for comparison

    # Compare the datasets - they should be the same after preprocessing, since the cache should not modify the data, only store it for future use. We compare both the dataset as a whole and the values of a specific variable to ensure they match.
    assert ds1.equals(ds2), "Expected cached dataset to have the same data as non-cached dataset after preprocessing, but they differ"
    assert (ds1["d2m"].values == ds2["d2m"].values).all(), "Expected cached dataset variable values to match non-cached dataset variable values after preprocessing, but they differ"

    # Check that a croissant file was created for the cached dataset, and that it contains the expected variables and values
    croissant_path = str(dataset.cache_path).replace(".zarr", "_croissant.json")
    assert Path(croissant_path).exists(), f"Expected croissant file {croissant_path} to be created for cached dataset, but it was not found"
    # Load croissant file and check it has a key called "@context"
    with open(croissant_path, "r") as f:
        croissant_data = json.load(f)
    assert "@context" in croissant_data, f"Expected croissant file {croissant_path} to contain a key called '@context', but it was not found"

    # Clean up cache directory after test
    safely_remove_dir(cache_dir)