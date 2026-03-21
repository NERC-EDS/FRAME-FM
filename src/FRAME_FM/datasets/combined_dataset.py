from torch.utils.data import Dataset
from typing import Any
from FRAME_FM.datasets.base_dataset import BaseDataset


class ZipDataset(Dataset):
    def __init__(self, *datasets: BaseDataset):
        self.datasets = datasets
        self.length = min(len(dataset) for dataset in datasets)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Any:
        return [dataset[index] for dataset in self.datasets]
