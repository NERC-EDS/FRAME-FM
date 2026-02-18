import matplotlib.pyplot as plt
from pathlib import Path
import pyproj
import rioxarray as rxr
import torch
from torch.utils.data import TensorDataset
from typing import Callable
import xarray

from ..utils.LightningDataModuleWrapper import BaseDataModule
from ..datasets.InputOnly_Dataset import TransformedInputCoordsDataset


def convert_to_long_lat(x, y, src_crs, dst_crs="EPSG:4326"):
    # convert to lat/lon if the dataset has a CRS (Coordinate Reference System) defined
    if src_crs is not None:
        transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        longitude, latitude = transformer.transform(x, y)
        return list(zip(longitude, latitude))
    else:
        raise ValueError("CRS not defined in dataset attributes, cannot convert to lat/lon")


class XarrayStaticDataModule(BaseDataModule):
    '''
    A simple DataModule for loading static data from a geotiff file using xarray.
    '''
    train_dataset: TransformedInputCoordsDataset
    val_dataset: TransformedInputCoordsDataset
    test_dataset: TransformedInputCoordsDataset | None

    def __init__(self,
                 data_root: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 persistent_workers: bool = False,
                 train_split: float = 0.85,
                 val_split: float = 0.15,
                 test_split: float = 0.0,
                 split_strategy: str = "fraction",
                 train_transforms: Callable | None = None,
                 val_transforms: Callable | None = None,
                 test_transforms: Callable | None = None,
                 tile_size: int = 256,
                 ) -> None:
        super().__init__(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            split_strategy=split_strategy,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
        )
        self.tile_size = tile_size

    # def __len__(self):
    #     return len(self.ar[self.batch_dim])

    # def __getitem__(self, idx):
    #     return self.ar[{self.batch_dim: idx}].values

    def _load_raw_data(self):
        # currently reading a single file
        ds = rxr.open_rasterio(self.data_root, parse_coordinates=True)
        return ds

    def tile_array(self, array):
        # sanity check
        _, nY, nX = array.shape
        if nY < self.tile_size or nX < self.tile_size:
            raise ValueError(
                "DataArray is smaller than minimal tile size required for encoding: "
                f"{nY}x{nX} < {self.tile_size}x{self.tile_size}"
                )
        # tile
        tiles = array.coarsen(
            x=self.tile_size, y=self.tile_size, boundary='pad'
            ).construct(
            x=("tile_xid", "x"), y=("tile_yid", "y")
            ).stack(
            batch_dim=("tile_xid", "tile_yid")
            ).transpose("batch_dim", "band", "y", "x")
        # THIS WILL CHANGE
        # replace nans with zeros (required for PCA for latent space visualization)
        tiles = tiles.fillna(0)
        return tiles

    def _create_datasets(self, stage: str | None = None) -> None:
        """
        Reads the DataArray from the attributes, tiles it into patches along x
        and y axis, and outputs stacked tiles. This dataset contains only
        inputs to b

        AK: the tiling in this way does not preserve relative positions of
        tiles, so for the exrension to multiple layers the bands need to be
        stacked first, then tiled.
        """
        tiles = self.tile_array(self._raw_data)
        # to tensor dataset
        dataset = TensorDataset(torch.tensor(tiles.values, dtype=torch.float32))
        # split into subsets
        train_base, val_base, test_base = self._split_dataset(dataset)
        # transform datasets
        self.train_dataset = TransformedInputCoordsDataset(train_base, self.train_transforms)
        self.val_dataset = TransformedInputCoordsDataset(val_base, self.val_transforms)
        if test_base is None:
            self.test_dataset = None
        else:
            self.test_dataset = TransformedInputCoordsDataset(test_base, self.test_transforms)


class GeotiffSpatialDataModule(XarrayStaticDataModule):

    def extract_position_tensor(self, array: xarray.DataArray) -> torch.Tensor:
        y, x = xarray.broadcast(array.y, array.x)
        positions = torch.stack([
            torch.tensor(y.values, dtype=torch.float32),
            torch.tensor(x.values, dtype=torch.float32),
            ], dim=1
            )
        return positions

    def _create_datasets(self, stage: str | None = None) -> None:
        """
        Reads the DataArray from the attributes, tiles it into patches along x
        and y axis, and outputs stacked tiles. This dataset contains only
        inputs to b

        AK: the tiling in this way does not preserve relative positions of
        tiles, so for the exrension to multiple layers the bands need to be
        stacked first, then tiled.
        """
        tiles = self.tile_array(self._raw_data)
        # to tensor dataset
        spatial_dataset = TensorDataset(
            torch.tensor(tiles.values, dtype=torch.float32),
            self.extract_position_tensor(tiles),
            )
        # split into subsets
        train_base, val_base, test_base = self._split_dataset(spatial_dataset)
        # transform datasets
        self.train_dataset = TransformedInputCoordsDataset(train_base, self.train_transforms)
        self.val_dataset = TransformedInputCoordsDataset(val_base, self.val_transforms)
        if test_base is None:
            self.test_dataset = None
        else:
            self.test_dataset = TransformedInputCoordsDataset(test_base, self.test_transforms)


class GeotiffBoundedDataModule(GeotiffSpatialDataModule):
    def extract_position_tensor(self, tiles: xarray.DataArray) -> torch.Tensor:
        # get bounds for each tile
        dx = (tiles.x[:, 1] - tiles.x[:, 0]) / 2
        x_min, x_max = tiles.x.min(axis=1) - dx, tiles.x.max(axis=1) + dx
        x_bounds = torch.stack([torch.tensor(x_min.values), torch.tensor(x_max.values)], dim=1)
        dy = (tiles.y[:, 1] - tiles.y[:, 0]) / 2
        y_min, y_max = tiles.y.min(axis=1) - dy, tiles.y.max(axis=1) + dy
        y_bounds = torch.stack([torch.tensor(y_min.values), torch.tensor(y_max.values)], dim=1)
        return torch.stack([x_bounds, y_bounds], dim=1)


def main():
    """
    Try creating the data module and plotting examples. Currently not using hydra.
    """
    PLOTTING = True
    DEBUG = True
    # example usage
    geotiff_path = Path(
        "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/land_cover_map_2015/data/"
        "LCM2015_GB_1km_percent_cover_aggregate_class.tif"
        )
    # try initializing the dataloader
    tile_size = 128
    data_module = GeotiffSpatialDataModule(data_root=geotiff_path.as_posix(), tile_size=tile_size)
    data_module.setup()
    for tile_id in range(len(data_module.train_dataset)):
        tile_values, tile_positions = data_module.train_dataset[tile_id]
        if (tile_values != 0).any():
            break
    bounded_data_module = GeotiffBoundedDataModule(data_root=geotiff_path.as_posix(), tile_size=tile_size)
    bounded_data_module.setup()
    _, tile_bounds = bounded_data_module.train_dataset[tile_id]
    tile_bounds = tile_bounds.tolist()

    if DEBUG:
        print("GeotiffSpatialDataModule:")
        print("_________________")
        print(f"Train dataset length: {len(data_module.train_dataset)}")
        print(f"Validation dataset length: {len(data_module.val_dataset)}")
        if data_module.test_dataset is not None:
            print(f"Test dataset length: {len(data_module.test_dataset)}")
        print(f"First non-zero tile #{tile_id}")
        print(f"Values shape (should be nBands, {tile_size}, {tile_size}): {tile_values.shape}")
        print(f"Positions shape (should be 2, {tile_size}, {tile_size}): {tile_positions.shape}")
        print()
        print("GeotiffBoundedDataModule:")
        print("_________________")
        print(f"Bounds: {tile_bounds}")

    if PLOTTING:
        fig_path = Path("./experiments/figures/")
        fig_path.mkdir(parents=True, exist_ok=True)
        # see what the values are...
        nBands, _, _ = tile_values.shape
        vmin, vmax = tile_values.min(), tile_values.max()
        fig, axs = plt.subplots(2, nBands // 2, figsize=(15, 6))
        for band, ax in enumerate(axs.flatten()):
            sm = ax.pcolormesh(
                tile_positions[1], tile_positions[0], tile_values[band],
                cmap="viridis", vmin=vmin, vmax=vmax, shading='nearest'
                )
            ax.set_title(f"Band {band}")
        fig.colorbar(sm, ax=axs, label="% coverage")
        fig.suptitle(f"Land cover in OSGB bounds {tile_bounds[1]}, {tile_bounds[0]}")
        fig.savefig(fig_path / "geotiff_bands.png")
        # ^ this is meant to be in percentages of 10 aggregate classes

    return None


if __name__ == "__main__":
    main()
