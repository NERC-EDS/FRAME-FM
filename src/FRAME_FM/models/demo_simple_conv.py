from typing import Sequence
from FRAME_FM.utils.embedders import BaseEmbedder, PatchEmbed
from FRAME_FM.utils.LightningModuleWrapper import BaseModule


class MyModel(BaseModule): pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialCollapse(nn.Module):
    """
    Input:  [B, V, H, W]
    Output: [B]
    """

    def __init__(self, in_variables, hidden_channels=32, dropout=0.2):
        super().__init__()

        # --- Conv stack ---
        self.conv1 = nn.Conv2d(in_variables, hidden_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(hidden_channels * 2)

        self.conv3 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(hidden_channels * 4)

        self.dropout = nn.Dropout2d(dropout)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Linear(hidden_channels * 4, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, V, H, W]

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pool(x)         # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]

        x = self.head(x)         # [B, 1]

        return x.squeeze(-1)     # [B]


from pathlib import Path
from io import BytesIO
import zipfile
import xarray as xr
import json
import pandas as pd


KERCHUNK_ZIP = Path("tests/fixtures/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json.zip")
KERCHUNK_FILE = BytesIO()
pdt = pd.to_datetime

# Unzip the file into an in-memory BytesIO object
with KERCHUNK_ZIP.open("rb") as f:
    with zipfile.ZipFile(f) as z:
        with z.open(z.namelist()[0]) as kerchunk_file:
            KERCHUNK_FILE.write(kerchunk_file.read())


def _load_kerchunk_dataset(kerchunk_file: str | Path | BytesIO = KERCHUNK_FILE,
                           response_type: str = "Dataset") -> xr.Dataset | xr.DataArray:
    # Reset the file pointer to the beginning
    kerchunk_file.seek(0)             # type: ignore
    refs = json.load(kerchunk_file)   # type: ignore
    ds = xr.open_dataset(refs, engine="kerchunk")
    if response_type == "DataArray":
        return ds["d2m"].isel(time=slice(0, 2))
    return ds

from FRAME_FM.transforms import transform_mapping, ToTensorTransform

if __name__ == "__main__":
    model = SpatialCollapse(in_variables=5)
    x = torch.randn(16, 5, 128, 128)
    y = model(x)
    print(y.shape)


    # Read in this zipped Kerchunk file and modify it and then do a basic Pytorch
    # training loop with it.
    ds = _load_kerchunk_dataset()
    transforms = [
        {"type": "reverse_axis", "dim": "latitude"},
        {"type": "subset", "variables": "d2m", "time": ("2000-01-01", "2000-01-02"), "latitude": (60, -30), "longitude": (40, 160)},
    ]

    for transform in reversed(transforms):
        if transform["type"] not in transform_mapping:
            raise ValueError(f"Unsupported transform type: {transform['type']}")
        transform_class = transform_mapping[transform["type"]]

        transform = transform_class(**{k: v for k, v in transform.items() if k != "type"})
        ds = transform(ds)

    # Duplicate the variable dimension 5 times to create a fake "variable" dimension for the model
    n_variables = 5
    array =  []
    for i in range(n_variables):
        array.append(ds["d2m"].values[None, ...])  # [1, T, lat, lon]
    
    data = np.concatenate(array, axis=0)  # [V=2, T, lat, lon]

    # Reshape to [T, V, lat, lon]
    data = data.transpose(1, 0, 2, 3)

    data = ToTensorTransform()(data)  # Convert to PyTorch tensor
    # Final dataset is now a PyTorch tensor, ready for training
    print(data.shape)  # Should be [V, T, lat, lon] after subsetting

    model = SpatialCollapse(in_variables=n_variables)
    x = data[:16]  # Take a batch of 16 time steps
    y = model(x)
    print(y.shape)

    # Now create a 5 epoch training loop with this data and model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), "spatial_collapse_model.pth")

    