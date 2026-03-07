import torch
import pytest

from .common import (
    GEOTIFF_URI,
    TIMESERIES_URI,
    ASCII_GRID_URI,
    NC_URI,
    CHESS_URI,
    ERA5_URI,
)

from FRAME_FM.datasets.base_gridded_dataset import (
    BaseGriddedDataset,
    BaseGeoTIFFDataset,
    BaseASCIIGridDataset,
    BaseGriddedTimeSeriesDataset,
)

from FRAME_FM.datasets.chessmet_dataset import CHESSMetGriddedTimeSeriesDataset
from FRAME_FM.datasets.era5_dataset import ERA5GriddedTimeSeriesDataset
from FRAME_FM.datasets.land_cover_map_dataset import LandCoverMapGriddedDataset





# ------------------------------------------------
# Base datasets
# ------------------------------------------------
def test_base_gridded_dataset_geotiff():
    dataset = BaseGriddedDataset(
        data_uri=GEOTIFF_URI,
        transforms=[
            {"type": "vars_to_dimension", "variables": ["band_data"], "new_dim": "variable"},
            {"type": "to_tensor"},
        ],
    )

    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.ndim >= 2


def test_base_geotiff_dataset():
    dataset = BaseGeoTIFFDataset(data_uri=GEOTIFF_URI)

    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.ndim >= 2


# ------------------------------------------------
# Base time series datasets
# ------------------------------------------------

@pytest.mark.parametrize(
    "uri",
    [
        TIMESERIES_URI,
        NC_URI,
    ],
)
def test_base_timeseries_dataset(uri):
    dataset = BaseGriddedTimeSeriesDataset(
        data_uri=uri,
        transforms=[],
        time_stride=1,
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

    assert len(dataset) > 0
    sample = dataset[0]
    next_sample = next(iter(dataset))
    assert isinstance(sample, torch.Tensor)
    assert isinstance(next_sample, torch.Tensor)


# ------------------------------------------------
# Dataset wrappers
# ------------------------------------------------
@pytest.mark.parametrize(
    "dataset_cls,uri",
    [
        (LandCoverMapGriddedDataset, GEOTIFF_URI),
        (CHESSMetGriddedTimeSeriesDataset, CHESS_URI),
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

    sample = dataset[0]

    assert isinstance(sample, torch.Tensor)
    assert sample.ndim >= 2


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
    assert required_vars.issubset(set(ds.data_vars))


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