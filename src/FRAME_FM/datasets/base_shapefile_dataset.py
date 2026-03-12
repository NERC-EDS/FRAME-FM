from pathlib import Path

import torch
from torch.utils.data import Dataset

from FRAME_FM.utils.data_utils import unify_transforms, load_shapefile_from_uri
from FRAME_FM.transforms import resolve_transform


class BaseShapefileDataset(Dataset):
    _transforms = [
        {"type": "to_tensor"}
    ]

    def __init__(self, 
                 data_uri: str | Path | list | tuple,
                 transforms: list | None = None,
                 override_transforms: bool = False
                 ):
        self.data_uri = data_uri
        self.transforms = unify_transforms(transforms, self._transforms, override_transforms)

        # Load the dataset ready for training
        self.data = load_shapefile_from_uri(self.data_uri)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return the data sample at the specified index
        sample = self.data.isel(band=idx)

        # Apply runtime transforms if any
        for transform in self.transforms:
            sample = resolve_transform(transform)(sample)

        return sample  # type: ignore
    

if __name__ == "__main__":

    # Try: BaseShapefileDataset with a single GeoTIFF file
    data_uri = "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/model_estimates_of_topsoil_carbon/data"

    # Set transforms for dataset
    dataset = BaseShapefileDataset(
        data_uri=data_uri,
    )       

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
