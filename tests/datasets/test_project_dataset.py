"""
Tests for specific dataset loaders, using the base dataset classes.

"""

import glob

# NOTE: potential fix for unit tests: single threading dask???
# import dask
# dask.config.set(scheduler='single-threaded')

import numpy as np
import torch
import pytest

from .common import (
    CHESS_URI,
    ERA5_URI,
    LAND_COVER_URI,
    SOIL_WATER_INDEX_URI as SOIL_WATER_INDEX_GLOB_URI,
    COSMOSUK_DATA_URI,
)

from FRAME_FM.utils.common_utils import get_main_vars
from FRAME_FM.datasets.base_gridded_dataset import (
    BaseGriddedDataset,
    BaseGriddedTimeSeriesDataset
)

from FRAME_FM.datasets.cosmosuk_dataset import CosmosUKDataset


# Override Glob Pattern for Soil Water Index dataset to use a smaller subset of files for testing
SOIL_WATER_INDEX_FILE_URI = [fpath for fpath in glob.glob(SOIL_WATER_INDEX_GLOB_URI) if "2020" in fpath][0]

# ------------------------------------------------
# Dataset wrappers
# ------------------------------------------------
@pytest.mark.parametrize(
    "dataset_cls,uri",
    [
        (BaseGriddedTimeSeriesDataset, CHESS_URI),
        (BaseGriddedTimeSeriesDataset, ERA5_URI),
        (BaseGriddedDataset, LAND_COVER_URI),
        # (BaseGriddedTimeSeriesDataset, SOIL_WATER_INDEX_FILE_URI),
    ],
)
def test_dataset_wrappers_basic(dataset_cls, uri):
    dataset = dataset_cls(data_uri=uri)

    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.ndim >= 2


# ------------------------------------------------
# CHESSMet specific checks
# ------------------------------------------------
def test_chessmet_dataset_with_transforms():
    preprocessors = [
        {
            "type": "subset",
            "y": (400500., 405500.),
            "x": (400500., 405500.),
            "time": ("1961-01-01T00:00:00", "1961-01-02T00:00:00"),
        }
    ]
    transforms = [
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable"},
        {"type": "to_tensor"},
    ]

    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=CHESS_URI,
        preprocessors=preprocessors,
        transforms=transforms,
        time_stride=1,
        chunks={"time": 2},
    )

    ds = dataset.data
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)


def test_chessmet_dataset_retains_2d_coordinate_variables():
    preprocessors = [
        {
            "type": "subset",
            "y": (400500., 405500.),
            "x": (400500., 405500.),
            "time": ("1961-01-01T00:00:00", "1961-01-02T00:00:00"),
        }
    ]
    transforms = [
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable"},
        {"type": "to_tensor"},
    ]

    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=CHESS_URI,
        preprocessors=preprocessors,
        transforms=transforms,
        time_stride=1,
        chunks={"time": 2},
    )

    # Check that 2D coordinate variables are retained in the dataset
    ds = dataset.data

    assert "lat" in ds and "lon" in ds, "Expected 'lat' and 'lon' to be ancinllary 2d coordinate variables in the dataset, but they were not found in the coordinates"
    assert ds["lat"].ndim == 2 and ds["lon"].ndim == 2, f"Expected 'lat' and 'lon' to be 2D coordinate variables, but got dimensions {ds['lat'].dims} and {ds['lon'].dims} respectively"
    assert list(ds["lat"].coords.keys()) == ["x", "y"] and list(ds["lon"].coords.keys()) == ["x", "y"], f"Expected 'lat' and 'lon' to have dimensions ['x', 'y'], but got: {ds['lat'].coords.keys()} and {ds['lon'].coords.keys()}"

    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)

    # Assert that the subset has some valid data points (i.e. not all NaN) - this is to check that the spatial subset is working correctly and not resulting in an empty dataset
    assert not np.isnan(sample.numpy()).all(), "Expected subsetted dataset to contain some valid data points, but all values were NaN - this may indicate an issue with the spatial subset transform resulting in an empty dataset"


# ------------------------------------------------
# ERA5 specific checks
# ------------------------------------------------
def test_era5_dataset_structure():
    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=ERA5_URI,
        time_stride=4,
    )

    ds = dataset.data
    assert "time" in ds.coords
    required_vars = {"u10", "v10", "t2m", "d2m"}
    assert required_vars.issubset(set(ds.data_vars)), "Dataset must contain the required variables" 


def test_era5_dataset_sampling():
    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=ERA5_URI,
        time_stride=4,
    )

    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.ndim >= 2


# ------------------------------------------------
# Land Cover Map specific checks
# ------------------------------------------------
def test_land_cover_map_dataset_structure():
    dataset = BaseGriddedDataset(
        data_uri=LAND_COVER_URI
    )

    ds = dataset.data
    assert "band_data" in ds.data_vars, "Expected dataset to have 'band_data' variable after preprocessing, but it was not found in the data variables"
    assert ds["band_data"].ndim == 3, f"Expected 'band_data' to be a 3D variable, but got dimensions {ds['band_data'].dims}"


def test_land_cover_map_with_transforms():
    # Parent class will avoid applying the same transforms twice, so we can apply some additional ones here if we want to test them out.
    transforms = [
        {"type": "subset", "variables": ["band_data"], "y": (10_000, 500), "x": (0, 10_000)},
        {"type": "vars_to_dimension", "variables": ["band_data"], "new_dim": "variable"},
        {"type": "to_tensor"},
    ]

    dataset = BaseGriddedDataset(
        data_uri=LAND_COVER_URI,
        transforms=transforms,
        override_transforms=True
    )

    ds = dataset.data

    # Assert original shape
    assert ds["band_data"].shape == (10, 1300, 700), f"Expected original 'band_data' shape to be (1, 1300, 700), but got {ds['band_data'].shape}"

    sample = dataset[0]
    assert isinstance(sample, torch.Tensor), f"Expected sample to be a torch.Tensor after applying transforms, but got {type(sample)}"
    assert sample.ndim >= 2, f"Expected sample to have at least 2 dimensions after applying transforms, but got {sample.ndim}"
    assert sample.shape == (1, 10, 10), f"Expected sample shape to be (1, 1500, 4500) after applying subset transform, but got {sample.shape}"


#-------------------------------------------------
# Soil Water Index specific checks
#-------------------------------------------------
def test_soil_water_index_dataset_structure():
    
    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=SOIL_WATER_INDEX_FILE_URI,
        time_stride=1,
        chunks="auto"
    )

    ds = dataset.data
    assert "time" in ds.coords
    assert "lat" in ds.coords
    assert "lon" in ds.coords
    required_vars = set(['QFLAG_002', 'QFLAG_005', 'QFLAG_010', 'QFLAG_015', 'QFLAG_020', 'QFLAG_040', 'QFLAG_060', 'QFLAG_100', 'SSF', 'SWI_002', 'SWI_005', 'SWI_010', 'SWI_015', 'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100', 'crs'])
    assert required_vars == set(ds.data_vars), "Dataset must contain the required variables"


def test_cosmosuk_dataset():
    dataset = CosmosUKDataset(
        data_uri=COSMOSUK_DATA_URI,
        qc_bitmask=0b0000000001,  # Mask missing data
        drop_qc_flags=["M", "U", "I", "E"],
    )

    data = dataset.data
    assert type(data) == list, f"Expected dataset.data to be a list of xarray Datasets (one per site), but got {type(data)}"
    # Test data has one site, so assert length of dataset is 1
    assert len(dataset) == 1, f"Expected dataset length to be 1 since there is only one site"
    
    sample = dataset[0]
    assert len(sample) == 8
    assert set(sample.data_vars.keys()) == {'TDT1_VWC', 'TDT2_VWC', 'TDT3_VWC', 'TDT4_VWC', 'TDT5_VWC', 'TDT6_VWC', 'TDT7_VWC', 'TDT8_VWC'}, f"Expected sample to contain the 8 TDT_VWC variables, but got {sample.data_vars.keys()}"
    assert "time" in sample.coords, f"Expected sample to have 'time' coordinate, but it was not found in the coordinates"
    assert sample.TDT1_VWC.max().item() == 323.6, f"Expected max value of TDT1_VWC to be 323.6 after applying QC mask, but got {sample.TDT1_VWC.max().item()}"
