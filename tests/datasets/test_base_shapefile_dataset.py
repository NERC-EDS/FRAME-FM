import torch
import pytest

from FRAME_FM.datasets.base_shapefile_dataset import BaseShapefileDataset

def test_dimensions():
    # Specify the path to the config file. 
    cfg_path = '/home/users/colinsau/FRAME-FM/configs/data/config_FRAME_shpfiles.yaml'

    # Set up the class and build the dataset.
    r = BaseShapefileDataset(data_uri = cfg_path)
    
    assert r.dataset_out.sizes['x'] == 604
    assert r.dataset_out.sizes['y'] == 1212