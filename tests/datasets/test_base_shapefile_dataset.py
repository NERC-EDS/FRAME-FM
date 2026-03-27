import torch
import pytest

from .common import (
    SHPFILE_CFG_URI,
)

from FRAME_FM.datasets.base_shapefile_dataset import BaseShapefileDataset


# Specify the path to the config file.
#cfg_path = "/home/users/colinsau/FRAME-FM/configs/data/config_FRAME_shpfiles.yaml"

# Set up the class and build the dataset.
ds = BaseShapefileDataset(data_uri=SHPFILE_CFG_URI)

def test_dimensions_and_attributes():
    """
    Test the dimensions and attributes of the dataset are as expected.
    """
    assert ds.dataset_out.sizes["x"] == 604, (
        f"Expected width of dataset is 604, got {r.dataset_out.sizes['x']}"
    )
    assert ds.dataset_out.sizes["y"] == 1212, (
        f"Expected width of dataset is 1212, got {r.dataset_out.sizes['y']}"
    )
    assert ds.resolution == 1000, (
        f"Expected resolution is 1000, got {r.dataset_out.resolution}"
    )
    assert ds.target_crs == "EPSG:27700", f"Expect CRS is ESPG:27700, got {r.target_crs}"
