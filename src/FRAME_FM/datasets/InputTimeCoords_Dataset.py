from typing import Optional, Any
from torch.utils.data import Dataset


class TransformedInputTimeCoordsDataset(Dataset):
    """
    Wrap a dataset that yields (tile, time, coords) and apply transforms to tile only.
    """
    def __init__(self, base: Dataset, transform: Optional[Any] = None) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        tile, time, coordinates = self.base[idx]
        if self.transform is not None:
            tile = self.transform(tile)
        return tile, time, coordinates


class MMMAEInputFromTimeCoordsDataset(Dataset):
    """
    Wrap a dataset that yields (tile, time, coords) and emit MMMAE-ready inputs.

    Output per sample is a single-input modality list:
        [(tile, coords)]
    """

    def __init__(self, base: Dataset, transform: Optional[Any] = None) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        tile, _time, coordinates = self.base[idx]
        if self.transform is not None:
            tile = self.transform(tile)
        return [(tile, coordinates)]