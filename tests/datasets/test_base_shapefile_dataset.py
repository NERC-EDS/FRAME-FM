from .common import (
    SHPFILE_CFG_URI,
)

from FRAME_FM.datasets.base_shapefile_dataset import BaseShapefileDataset

SHAPEFILE_TEST_DATASET = None

# Specify the path to the config file.
#cfg_path = "/home/users/colinsau/FRAME-FM/configs/data/config_FRAME_shpfiles.yaml"

def _create_dataset():
    """
    Helper function to create the dataset for testing.
    """
    # Set up the class and build the dataset.
    global SHAPEFILE_TEST_DATASET
    if SHAPEFILE_TEST_DATASET is None:
        SHAPEFILE_TEST_DATASET = BaseShapefileDataset(data_uri=SHPFILE_CFG_URI)

    return SHAPEFILE_TEST_DATASET


def test_dimensions_and_attributes():
    """
    Test the dimensions and attributes of the dataset are as expected.
    """
    ds = _create_dataset()
    
    assert ds.dataset_out.sizes["x"] == 604, (
        f"Expected width of dataset is 604, got {ds.dataset_out.sizes['x']}"
    )
    assert ds.dataset_out.sizes["y"] == 1212, (
        f"Expected width of dataset is 1212, got {ds.dataset_out.sizes['y']}"
    )
    assert ds.resolution == 1000, (
        f"Expected resolution is 1000, got {ds.dataset_out.resolution}"
    )
    assert ds.target_crs == "EPSG:27700", f"Expect CRS is ESPG:27700, got {ds.target_crs}"
