from FRAME_FM.transforms.transforms import resolve_transform
from FRAME_FM.utils.data_utils import load_data_from_uri
import torch

from FRAME_FM.datasets.base_dataset import BaseDataset


class COSMOSUKSiteTimeSeriesDataset(BaseDataset):
    _transforms = [
        {"type": "to_tensor"}
    ]

    def _setup_dataset(self):
        self.data = load_data_from_uri(self.data_uri, chunks=self.chunks)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return the data sample at the specified index
        sample = self.data.sel(site=idx)

        # Apply runtime transforms if any
        for transform in self.transforms:
            sample = resolve_transform(transform)(sample)

        return sample  # type: ignore
