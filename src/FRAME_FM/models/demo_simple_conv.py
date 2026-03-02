from typing import Sequence
from FRAME_FM.utils.embedders import BaseEmbedder, PatchEmbed
from FRAME_FM.utils.LightningModuleWrapper import BaseModule


class MyModel(BaseModule): pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.multiprocessing as mp
# mp.set_start_method("spawn", force=True)


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
import xarray as xr
import pandas as pd


KERCHUNK_ZIP = "tests/transforms/fixtures/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json.zip"


from FRAME_FM.utils.data_utils import load_data_from_uri
from FRAME_FM.transforms import transform_mapping, ToTensorTransform




if __name__ == "__main__":

    n_variables = 5
    model = SpatialCollapse(in_variables=n_variables)
    x = torch.randn(16, n_variables, 128, 128)
    y = model(x)
    print(y.shape)


    # Read in this zipped Kerchunk file and modify it and then do a basic Pytorch
    # training loop with it.
    ds = load_data_from_uri(KERCHUNK_ZIP)
    transforms = [
        {"type": "reverse_axis", "dim": "latitude"},
        {"type": "subset", "variables": "d2m", "time": ("2001-01-01", "2001-01-01T00:05:00"), "latitude": (60, -30), "longitude": (40, 160)},
    ]

    for transform in reversed(transforms):
        if transform["type"] not in transform_mapping:
            raise ValueError(f"Unsupported transform type: {transform['type']}")
        transform_class = transform_mapping[transform["type"]]

        transform = transform_class(**{k: v for k, v in transform.items() if k != "type"})
        ds = transform(ds)

    # Calculate the mean and std for each time step
    mean = ds["d2m"].mean(dim=("latitude", "longitude"))
    std = ds["d2m"].std(dim=("latitude", "longitude"))

    # Write them to a csv file via pandas
    csv_file = "d2m_mean_std.csv"
    pd.DataFrame({"time": ds.time.values, "mean": mean.values, "std": std.values}).to_csv(csv_file, index=False)
    print(f"Saved mean and std to {csv_file}")

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

    