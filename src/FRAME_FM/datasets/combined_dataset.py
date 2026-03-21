from torch.utils.data import Dataset
from typing import Any

from FRAME_FM.datasets.base_dataset import BaseDataset
from FRAME_FM.transforms.transforms import TiledIndexMapper
from FRAME_FM.utils.transform_utils import CRS_conversion_spec, CRS_convertor


class ZipDataset(Dataset):
    def __init__(self, *datasets: BaseDataset):
        self.datasets = datasets
        self.length = min(len(dataset) for dataset in datasets)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Any:
        return [dataset[index] for dataset in self.datasets]


class CorrespondingTilesDataset(Dataset):
    def __init__(self,
                 datasets: list[BaseDataset],
                 crs_specs: list[CRS_conversion_spec | tuple | list | None] | None = None
                 ) -> None:
        if crs_specs is None:
            crs_specs = [None for _ in range(len(datasets))]
        if len(datasets) != len(crs_specs):
            raise ValueError(
                f"Length of datasets ({len(datasets)}) must equal that of crs_specs"
                f" ({len(crs_specs)})."
                )
        self.datasets = datasets
        self.value_shapes = [dataset.data[0].shape for dataset in datasets]
        # TODO: Refactor to TiledDataset(BaseDataset) class with tile_index method, and enforce
        self.index_mappers = [
            TiledIndexMapper.from_tiled_array(dataset.data) for dataset in datasets
            ]
        self.crs_convertors = [
            None if crs_conversion_spec is None else CRS_convertor(crs_conversion_spec)
            for crs_conversion_spec in crs_specs
            ]

    def __len__(self):
        return len(self.datasets[0].data)

    def __getitem__(self, index) -> list:
        ref_location = {
            dim: self.datasets[0].data[index][dim].mean().values
            for dim in self.datasets[0].data[index].dims
            }
        if self.crs_convertors[0] is not None:
            ref_location = self.crs_convertors[0].transform(ref_location)  # type: ignore
            # (DataArray.dims may in theory be Hashable, but in practice are str)
        corresponding_tiles = [self.datasets[0][index]]
        ds_iterable = zip(self.datasets[1:], self.crs_convertors[1:], self.index_mappers[1:])
        for dataset, crs_convertor, index_mapper in ds_iterable:
            if crs_convertor is None:
                search_coords = ref_location
            else:
                search_coords = crs_convertor.transform(ref_location, inverse=True)  # type: ignore
            # (DataArray.dims may in theory be Hashable, but in practice are str)
            search_coords = {
                dim: coord.item() for dim, coord in search_coords.items()
                if dim in index_mapper.tile_sizes.keys()
                }
            tile_id = index_mapper.tile_id_from_coordinates(search_coords)  # type: ignore
            # (DataArray.dims may in theory be Hashable, but in practice are str)
            corresponding_tiles.append(dataset[tile_id])
        return corresponding_tiles
