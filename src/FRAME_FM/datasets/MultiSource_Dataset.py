# src/FRAME_FM/datasets/MultiSource_Dataset.py
"""
Lightweight Dataset wrapper that transforms the dataset composed form multiple
sources. 
"""
from typing import Optional, Any
from torch.utils.data import Dataset
import numpy as np


class TransformedInputIndexMappingDataset(Dataset):
    """
    This class applies to input only datasets that may be static or dynamic.
    Besides standard dataset arguments, it expects a mappting of indeces from
    the lagest dataset in the collection to all the datasets in the form:
    idx_mapping = {idx_global: stacked_tile_id} where idx_global is the 
    index of the tile in the largest dataset, and stacked_tile_id is the tile id
    in the dataset that is being indexed. This is required to be able to call a
    speicific tile from both static and ST datasets. The selected tiles are then
    transformed by the provided transform. Coordinates are read and passed as is.

    """

    def __init__(self, base: Dataset, index_mapping: dict, transform: Optional[Any] = None) -> None:
        self.base = base
        self.index_mapping = index_mapping # has to come fro the outside as it will be source specific
        self.transform = transform
    
    def _getindeces_(self, idx: int):
        tile_id = self.index_mapping[idx]
        return tile_id    
    
    def __len__(self) -> int:
        return len(self.index_mapping)

    def __getitem__(self, idx: int):
        tile_id = self._getindeces_(idx)
        tile_stack, *coords = self.base[tile_id]
        if type(tile_stack) != np.ndarray:
            tile_stack = np.asarray(tile_stack) # load item into memory if it is lazy
        if self.transform is not None:
            tile_stack = self.transform(tile_stack)
        return (tile_stack, *coords)
    