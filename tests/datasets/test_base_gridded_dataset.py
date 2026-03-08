import torch
import pytest

from .common import (
    GEOTIFF_URI,
    TIMESERIES_URI,
    ASCII_GRID_URI,
    NC_URI
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
