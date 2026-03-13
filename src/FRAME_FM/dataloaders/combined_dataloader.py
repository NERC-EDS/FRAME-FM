from torch.utils.data import StackDataset
from typing import Callable

from ..utils.LightningDataModuleWrapper import BaseDataModule


class CombinedDataModule(BaseDataModule):
    '''
    A DataModule for combining datasets from multiple DataModules.
    # TODO: Implement post-combination transforms
    '''
    def __init__(self,
                 datamodules: list[BaseDataModule],
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 persistent_workers: bool = False,
                 split_strategy: str = "fraction",
                 train_split: float = 0.85,
                 val_split: float = 0.15,
                 test_split: float = 0.0,
                 train_transforms: Callable | None = None,
                 val_transforms: Callable | None = None,
                 test_transforms: Callable | None = None,
                 ) -> None:
        super().__init__(
            data_root="",
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
        )
        for dm in datamodules:
            dm.split_strategy = split_strategy
            dm.train_split = train_split
            dm.val_split = val_split
            dm.test_split = test_split
        self.datamodules = datamodules

    def _load_raw_data(self):
        for dm in self.datamodules:
            dm._raw_data = dm._load_raw_data()
        return None

    def _create_datasets(self, stage: str | None = None) -> None:
        for dm in self.datamodules:
            dm._create_datasets(stage)
        self.train_dataset = StackDataset(*[
            dm.train_dataset for dm in self.datamodules if dm.train_dataset is not None
            ])
        self.val_dataset = StackDataset(*[
            dm.val_dataset for dm in self.datamodules if dm.val_dataset is not None
            ])
        self.test_dataset = StackDataset(*[
            dm.test_dataset for dm in self.datamodules if dm.test_dataset is not None
            ])
