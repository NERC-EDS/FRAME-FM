import torch
import pytest

from .common import (
    NC_URI,
)

from FRAME_FM.utils.settings import DatasetSettings, DefaultSettings
from FRAME_FM.utils.data_utils import load_data_from_uri, hash_preprocessors, create_cache_path
from FRAME_FM.datasets.base_dataset import BaseDataset


# ------------------------------------------------
# Base datasets
# ------------------------------------------------
def test_base_dataset():
    dataset = BaseDataset(
        data_uri=NC_URI,
        preprocessors=[
            {"type": "subset", "variables": ["pre"], "time": ("2010-04-16", "2010-05-16"),
             "latitude": (45, 60), "longitude": (7.5, 45)},
            {"type": "to_dataarray", "var_id": "pre"},
        ],
        transforms=[
            {"type": "to_tensor"},
        ],
    )

    assert len(dataset) == 2, "Dataset length should match the number of samples in the data"
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (13, 21), f"Expected sample shape to be (13, 21) after transforms, but got {sample.shape}"

    # Compare min and max of original data versus sample to ensure transforms are working correctly
    original_data = dataset.data.isel(time=0).values
    assert torch.isclose(sample.min(), torch.tensor(original_data.min()), atol=1e-5), f"Sample min value {sample.min()} does not match original data min {original_data.min()}"
    assert torch.isclose(sample.max(), torch.tensor(original_data.max()), atol=1e-5), f"Sample max value {sample.max()} does not match original data max {original_data.max()}"


def test_dataset_caching():
    cache_dir = "./test_cache"

    dataset = BaseDataset(
        data_uri=NC_URI,
        preprocessors=[
            {"type": "subset", "variables": ["pre"], "time": ("2010-04-16", "2010-05-16"),
             "latitude": (45, 60), "longitude": (7.5, 45)},
        ],
        transforms=[
            {"type": "to_tensor"},
        ],
        cache_dir=cache_dir,
    )

    # Test cache directory exists
    assert dataset.cache_path is not None, "Cache path should be set after precaching"
    assert dataset.cache_path.exists(), f"Cache directory {dataset.cache_path} should exist after precaching"

    # Test the correct zarr cache file is created
    zarr_path = create_cache_path(NC_URI, cache_dir)
    assert zarr_path.exists(), f"Zarr cache file {zarr_path} should exist after precaching"

    # Assert that the cached data has the hash attribute and that it matches the hash of the preprocessors
    cached_ds = load_data_from_uri(zarr_path, zarr_format=2)
    assert DatasetSettings.preprocessor_hash_key in cached_ds.attrs, f"Cached dataset should have the hash attribute {DatasetSettings.preprocessor_hash_key}"
    cached_hash = cached_ds.attrs[DatasetSettings.preprocessor_hash_key]
    expected_hash = hash_preprocessors(dataset.preprocessors)
    assert cached_hash == expected_hash, f"Cached dataset hash {cached_hash} should match expected hash {expected_hash} based on the preprocessors"

    # Now try comparing the data from the cache with an equivalent dataset without caching to ensure they are the same
    dataset_no_cache = BaseDataset(
        data_uri=NC_URI,
        preprocessors=[
            {"type": "subset", "variables": ["pre"], "time": ("2010-04-16", "2010-05-16"),
             "latitude": (45, 60), "longitude": (7.5, 45)},
        ],
        transforms=[
            {"type": "to_tensor"},
        ],
        cache_dir=None,  # No caching for this dataset
    )

    # Compare the data from the cached dataset with the non-cached dataset
    # First assert that the hash is NOT in the non-cached dataset
    assert DatasetSettings.preprocessor_hash_key not in dataset_no_cache.data.attrs, f"Non-cached dataset should not have the hash attribute {DatasetSettings.preprocessor_hash_key}"

    # Now add it in to the non-cached dataset to ensure that the presence of the hash attribute does not affect the data
    dataset_no_cache.data.attrs[DatasetSettings.preprocessor_hash_key] = expected_hash
    assert dataset.data.equals(dataset_no_cache.data), "Cached and non-cached datasets should be equal"