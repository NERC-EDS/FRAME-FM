import glob

# NOTE: potential fix for unit tests: single threading dask???
# import dask
# dask.config.set(scheduler='single-threaded')

import torch
import pytest

from .common import (
    CHESS_URI,
    ERA5_URI,
    LAND_COVER_URI,
    SOIL_WATER_INDEX_URI as SOIL_WATER_INDEX_GLOB_URI
)

from FRAME_FM.utils.data_utils import get_main_vars

from FRAME_FM.datasets.chessmet_dataset import CHESSMetGriddedTimeSeriesDataset
from FRAME_FM.datasets.era5_dataset import ERA5GriddedTimeSeriesDataset
from FRAME_FM.datasets.land_cover_map_dataset import LandCoverMapGriddedDataset
from FRAME_FM.datasets.soil_water_index_dataset import SoilWaterIndexGriddedTimeSeriesDataset


# Override Glob Pattern for Soil Water Index dataset to use a smaller subset of files for testing
SOIL_WATER_INDEX_FILE_URI = [fpath for fpath in glob.glob(SOIL_WATER_INDEX_GLOB_URI) if "2020" in fpath][0]

# ------------------------------------------------
# Dataset wrappers
# ------------------------------------------------
@pytest.mark.parametrize(
    "dataset_cls,uri",
    [
        (CHESSMetGriddedTimeSeriesDataset, CHESS_URI),
        (ERA5GriddedTimeSeriesDataset, ERA5_URI),
        (LandCoverMapGriddedDataset, LAND_COVER_URI),
#        (SoilWaterIndexGriddedTimeSeriesDataset, SOIL_WATER_INDEX_FILE_URI),
    ],
)
def test_dataset_wrappers_basic(dataset_cls, uri):
    dataset = dataset_cls(data_uri=uri)

    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.ndim >= 2


def test_chessmet_dataset_with_transforms():
    transforms = [
        {
            "type": "subset",
            "variables": ["precip"],
            "y": (100500.0, 257500.0),
            "x": (200500.0, 156500.0),
            "time": ("2016-01-27", "2016-01-02"),
        },
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable"},
        {"type": "to_tensor"},
    ]

    dataset = CHESSMetGriddedTimeSeriesDataset(
        data_uri=CHESS_URI,
        transforms=transforms,
        time_range=("2016-01-01", "2018-10-20"),
        time_stride=1,
        chunks={"time": 64},
    )

    # Preprocessing checks
    assert get_main_vars(dataset.data) == ["precip"], f"Expected dataset to have only 'precip' variable after subset transform, but got {dataset.data.data_vars}"

    # Sampling and transform checks
    assert len(dataset) > 0, "Expected dataset length to be greater than 0 after applying time range and stride, but got 0"
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.ndim >= 2


# ------------------------------------------------
# CHESSMet specific checks
# ------------------------------------------------
def test_chessmet_dataset_retains_2d_coordinate_variables():
    transforms = [
        {
            "type": "subset",
            "variables": ["precip"],
            "y": (100500.0, 257500.0),
            "x": (200500.0, 156500.0),
            "time": ("2016-01-27", "2016-01-02"),
        },
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable"},
        {"type": "to_tensor"},
    ]

    dataset = CHESSMetGriddedTimeSeriesDataset(
        data_uri=CHESS_URI,
        transforms=transforms,
        time_range=("2016-01-01", "2018-10-20"),
        time_stride=1,
        chunks={"time": 64},
    )

    # Check that 2D coordinate variables are retained in the dataset
    ds = dataset.data
    assert "lat" in ds and "lon" in ds, "Expected 'lat' and 'lon' to be ancillary 2d coordinate variables in the dataset, but they were not found in the coordinates"
    assert ds["lat"].ndim == 2 and ds["lon"].ndim == 2, f"Expected 'lat' and 'lon' to be 2D coordinate variables, but got dimensions {ds['lat'].dims} and {ds['lon'].dims} respectively"
    assert list(ds["lat"].coords.keys()) == ["x", "y"] and list(ds["lon"].coords.keys()) == ["x", "y"], f"Expected 'lat' and 'lon' to have dimensions ['x', 'y'], but got: {ds['lat'].coords.keys()} and {ds['lon'].coords.keys()}"

# ------------------------------------------------
# ERA5 specific checks
# ------------------------------------------------
def test_era5_dataset_structure():
    dataset = ERA5GriddedTimeSeriesDataset(
        data_uri=ERA5_URI,
        time_range=("2010-05-01", "2012-10-05"),
        time_stride=4,
    )

    ds = dataset.data
    assert "time" in ds.coords
    required_vars = {"u10", "v10", "t2m", "d2m"}
    assert required_vars.issubset(set(ds.data_vars)), "Dataset must contain the required variables" 


def test_era5_dataset_sampling():
    dataset = ERA5GriddedTimeSeriesDataset(
        data_uri=ERA5_URI,
        time_range=("2010-05-01", "2012-10-05"),
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
    dataset = LandCoverMapGriddedDataset(
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

    dataset = LandCoverMapGriddedDataset(
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
@pytest.mark.xfail(reason="Problem with NCA file parsing/loading - need to investigate further")
def test_soil_water_index_dataset_structure():
    
    dataset = SoilWaterIndexGriddedTimeSeriesDataset(
        data_uri=SOIL_WATER_INDEX_FILE_URI,
        time_range=("2020-02-01", "2020-02-05"),
        time_stride=1,
        chunks={"time": 1}
    )

    ds = dataset.data
    assert "time" in ds.coords
    assert "lat" in ds.coords
    assert "lon" in ds.coords
    required_vars = {"swvl1", "swvl2", "swvl3", "swvl4"}
    assert required_vars.issubset(set(ds.data_vars)), "Dataset must contain the required variables"

# @pytest.mark.xfail(reason="Problem with NCA file parsing/loading - need to investigate further")
# def test_soil_water_index_dataset_sampling():
#     dataset = SoilWaterIndexGriddedTimeSeriesDataset(
#         data_uri=SOIL_WATER_INDEX_FILE_URI,
#         time_range=("2020-02-01", "2020-02-05"),
#         time_stride=1,
#         chunks={"time": 1}
#     )

#     assert len(dataset) > 0
#     sample = dataset[0]
#     assert isinstance(sample, torch.Tensor)
#     assert sample.ndim >= 2


