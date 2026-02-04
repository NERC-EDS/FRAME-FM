# src/FRAME_FM/datasets/InputOnly_Dataset.py
"""
Lightweight Dataset wrapper that takes inputs only"""
from typing import Optional, Any
from torch.utils.data import Dataset


class TransformedInputDataset(Dataset):
    """
    This class applies to input only style datasets that are useful for visual
    autoencoders. The method currently uses a scaling coefficient to scale the
    input, this will change in the future when the decision about transform
    settings are finalized.

    """

    def __init__(self, base: Dataset, scaling_coefficient: float = 0.01, transform: Optional[Any] = None) -> None:
        self.base = base
        self.scaling_coefficient = scaling_coefficient
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        tile = self.base[idx] # expected dimensions are (C x H x W) or (T x C x H x W)
        tile = tile * self.scaling_coefficient
        if self.transform is not None:
            tile = self.transform(tile)
        return tile
