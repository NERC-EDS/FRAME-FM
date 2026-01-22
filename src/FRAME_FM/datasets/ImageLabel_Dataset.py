# src/FRAME_FM/datasets/ImageLabel_Dataset.py
"""
Lightweight Dataset wrapper that applies transforms to images only,
preserving the (image, label) structure of torchvision datasets."""
from torch.utils.data import Dataset


class TransformedDataset(Dataset):
    """
    PyTorch Dataset wrapper that applies transforms to images only,
    """

    def __init__(self, base: Dataset, transform: Optional[Any] = None) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, target = self.base[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target
