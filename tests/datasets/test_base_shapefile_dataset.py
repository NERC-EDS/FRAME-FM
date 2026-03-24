import torch
import pytest

from FRAME_FM.datasets.base_shapefile_dataset import BaseShapefileDataset


def test_dimensions_and_attributes():
    """
    Test the dimensions and attributes of the dataset are as expected.
    """
    # Specify the path to the config file.
    cfg_path = "/home/users/colinsau/FRAME-FM/configs/data/config_FRAME_shpfiles.yaml"

    # Set up the class and build the dataset.
    r = BaseShapefileDataset(data_uri=cfg_path)

    assert r.dataset_out.sizes["x"] == 604, (
        f"Expected width of dataset is 604, got {r.dataset_out.sizes['x']}"
    )
    assert r.dataset_out.sizes["y"] == 1212, (
        f"Expected width of dataset is 1212, got {r.dataset_out.sizes['y']}"
    )
    assert r.resolution == 1000, (
        f"Expected resolution is 1000, got {r.dataset_out.resolution}"
    )
    assert r.target_crs == "EPSG:27700", f"Expect CRS is ESPG:27700, got {r.target_crs}"
