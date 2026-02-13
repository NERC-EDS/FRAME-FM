# src/FRAME_FM/datasets/ImageLabel_Dataset.py
"""
Lightweight Dataset wrapper that applies transforms to images only,
preserving the (image, label) structure of torchvision datasets."""
from typing import Any, Optional
from torch.utils.data import Dataset


class SplitTransformDataset(Dataset):
    """
    Adapter that:
    - Normalises sample format to {"image": ..., "label": ...}
    - Applies per-split transform to the image
    """

    def __init__(self, base: Dataset, transform=None):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        # Normalize sample format
        if isinstance(sample, dict):
            image = sample["image"]
            label = sample["label"]
        elif isinstance(sample, tuple):
            image, label = sample
        else:
            raise TypeError(f"Unsupported sample type: {type(sample)}")

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
        }
