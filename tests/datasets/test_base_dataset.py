import torch
import pytest

from .common import (
    NC_URI,
)

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