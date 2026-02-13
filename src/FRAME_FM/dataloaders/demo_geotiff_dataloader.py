import matplotlib
from sqlalchemy import exists
matplotlib.use("Agg") #Ensure a non-interactive Matplotlib backend
import matplotlib.pyplot as plt
import os
import torch
import rioxarray as rxr
from typing import Optional, Any
import pyproj


from FRAME_FM.utils.LightningDataModuleWrapper import BaseDataModule
from FRAME_FM.datasets.InputOnly_Dataset import TransformedInputDataset, TransformedInputCoordsDataset

class XarrayStaticDataModule(BaseDataModule):
    '''
    A simple DataModule for loading static data from geotiff files using xarray.
    '''
    def __init__(
        self,
        data_root: str ,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        train_split: float = 0.85,
        val_split: float = 0.15,
        test_split: float = 0.0,
        split_strategy: str = "fraction",
        train_transforms: Optional[callable] = None,
        val_transforms: Optional[callable] = None,
        test_transforms: Optional[callable] = None,
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
        ds = rxr.open_rasterio(self.data_root)
        return ds

    
    def _create_datasets(self, stage: Optional[str] = None) -> None:
        """
        Reads the DataArray from the attributes, tiles it into patches along x
        and y axis, and outputs stacked tiles. This dataset contains only
        inputs to b
        
        AK: the tiling in this way does not preserve relative positions of
        tiles, so for the exrension to multiple layers the bands need to be
        stacked first, then tiled.
        """
        array = self._raw_data
        _, nY, nX = array.shape
        # sanity check
        if nY < self.tile_size or nX < self.tile_size:
            raise ValueError(f"DataArray is smaller than minimal tile size required for encoding: {self.tile_size}x{self.tile_size}")
        # tile
        tiles = array.coarsen(x=self.tile_size, y=self.tile_size, boundary='pad').construct(x=("x_coarse", "x_fine"), y=("y_coarse", "y_fine"))
        # stack
        stacked_tiles = tiles.stack(batch_dim=("x_coarse", "y_coarse"))
        ## THIS WILL CHANGE
        # replace nans with zeros (if any - this is required for latent space visualization, otherwise the PCA will fail)
        stacked_tiles = stacked_tiles.fillna(0)
        # transpose to have batch dim first
        batch_ready = stacked_tiles.transpose("batch_dim", "band", "y_fine", "x_fine")
        # to tensor
        batch_ready = torch.tensor(batch_ready.values, dtype=torch.float32)
        # split into subsets
        split_datasets = self._split_dataset(batch_ready)
        # transform datasets
        train_base, val_base, test_base = split_datasets
        self.train_dataset = TransformedInputDataset(
            train_base,
            transform=self.train_transforms,
        )
        self.val_dataset = TransformedInputDataset(
            val_base,
            transform=self.val_transforms,
        )
        # test_base may be None if no test split configured
        self.test_dataset = (
            TransformedInputDataset(
                test_base, 
                transform=self.test_transforms)
            if test_base is not None
            else None
        )


class XarrayStaticWCoordsData(BaseDataModule):
    def __init__(
            self,
            data_root: str ,
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = True,
            persistent_workers: bool = False,
            train_split: float = 0.85,
            val_split: float = 0.15,
            test_split: float = 0.0,
            split_strategy: str = "fraction",
            train_transforms: Optional[callable] = None,
            val_transforms: Optional[callable] = None,
            test_transforms: Optional[callable] = None,
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
    
    def _load_raw_data(self):
        # currently reading a single file
        ds = rxr.open_rasterio(self.data_root)
        return ds
    
    def _create_datasets(self, stage: Optional[str] = None) -> None:
        
        """
        Reads the DataArray from attributes, tiles it, gets tile centroids,
        converts to lat/lon if CRS defined, and outputs a dataset of (tile, coordinates) tuples.
        """
        array = self._raw_data
        _, nY, nX = array.shape
        # sanity check
        if nY < self.tile_size or nX < self.tile_size:
            raise ValueError(f"DataArray is smaller than minimal tile size required for encoding: {self.tile_size}x{self.tile_size}")
        # tile
        tiles = array.coarsen(x=self.tile_size, y=self.tile_size, boundary='pad').construct(x=("x_coarse", "x_fine"), y=("y_coarse", "y_fine"))
        # stack tiles
        stacked_tiles = tiles.stack(batch_dim=("x_coarse", "y_coarse"))
        tile_values = stacked_tiles.transpose("batch_dim", "band", "y_fine", "x_fine")
        # get coordinates for each tile
        tile_coords = self._get_tile_coords(tile_values)
        # to tensor
        tile_values = torch.tensor(tile_values.values, dtype=torch.float32)
        tile_coords = torch.tensor(tile_coords, dtype=torch.float32)
        batch_ready = list(zip(tile_values, tile_coords))
        # split into subsets
        split_datasets = self._split_dataset(batch_ready)
        # transform datasets
        train_base, val_base, test_base = split_datasets
        self.train_dataset = TransformedInputCoordsDataset(
            train_base,
            transform=self.train_transforms,
        )
        self.val_dataset = TransformedInputCoordsDataset(
            val_base,
            transform=self.val_transforms,
        )
        # test_base may be None if no test split configured
        self.test_dataset = (
            TransformedInputCoordsDataset(
                test_base, 
                transform=self.test_transforms)
            if test_base is not None
            else None
        )
        
    def _get_tile_coords(self, tile_stack, target_crs="EPSG:4326"):
        # is this correct level of abstraction?
        # get centroid coordinates for each tile
        x_centroids = tile_stack.x_coarse + self.tile_size // 2
        y_centroids = tile_stack.y_coarse + self.tile_size // 2
        # convert to lat/lon if the dataset has a CRS (Coordinate Reference System) defined
        if tile_stack.rio.crs is not None:
            crs = tile_stack.rio.crs
            transformer = pyproj.Transformer.from_crs(crs, target_crs, always_xy=True)
            lon_centroids, lat_centroids = transformer.transform(x_centroids.values, y_centroids.values)
            return list(zip(lon_centroids, lat_centroids))
        else:
            raise ValueError("CRS not defined in dataset attributes, cannot convert to lat/lon")

def main():
    """ 
    PLot example batches and try creating the data module.
    Currently not using hydra.
    """
    Plotting = False
    Debug = True
    fig_path = "./experiments/figures/"
    os.makedirs(fig_path, exist_ok=True)
    # example usage
    geotiff_path = "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/land_cover_map_2015/data/LCM2015_GB_1km_percent_cover_aggregate_class.tif"
    geotiff_da = rxr.open_rasterio(geotiff_path)

    if Plotting:
        # see what the values are...
        nBands, nY, nX = geotiff_da.shape
        fig, axes = plt.subplots(2, nBands//2, figsize=(15, 6))
        axes = axes.flatten()
        for iBand in geotiff_da.band.values:
            ax = axes[iBand-1]
            temp = geotiff_da.sel(band=iBand)
            temp = temp[::-1, :] # flip updown
            temp.plot(ax=ax)
            ax.set_title(f"Band {iBand}")
            ax.axis('off')
            ax.collections[0].colorbar.remove()
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        norm = plt.Normalize(vmin=geotiff_da.min(), vmax=geotiff_da.max())
        # dummy mappable for colorbar
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Cover Percentage")
        # fig.tight_layout()
        fig.savefig(os.path.join(fig_path, "geotiff_bands.png"))
        # ^ this is meant to be in percentages of 10 aggregate classes

    
    #tile the dataarray into patches using coarsen (from https://docs.xarray.dev/en/stable/user-guide/reshaping.html)
    tiles = geotiff_da.coarsen(x=70, y=130, boundary='pad').construct(x=("x_coarse", "x_fine"), y=("y_coarse", "y_fine")) 
    if Debug:
        print("Tiled array shape:", tiles.shape)
        # expected shape of the above is (nBands, y_coarse, y_tile_dim, x_coarse, x_tile_dim)
    if Plotting:
        iBand = 1
        # plot for the first band only
        tiles.isel(band=iBand).plot(x="x_fine", y="y_fine", col="x_coarse", row="y_coarse", yincrease=False)
        plt.savefig(os.path.join(fig_path, f"tiling_results_on_band_{iBand}.png"))
    
    # Stack tiles with fine dimenstion
    stacked_tiles = tiles.stack(batch_dim=("x_coarse", "y_coarse"))
    if Debug:
        print("Stacked tiles shape:", stacked_tiles.shape)
         # expected shape of the above is (nBands,  y_tile_dim, x_tile_dim, y_coarse * x_coarse)
  
    
    # try initializing the dataloader
    tile_size = 128
    data_module = XarrayStaticWCoordsData(data_root=geotiff_path, tile_size=tile_size)
    data_module.setup() 
    if Debug:
        print(f"Train dataset length: {len(data_module.train_dataset)}. Data type: {type(data_module.train_dataset)}")
        print(f"Validation dataset length: {len(data_module.val_dataset)}. Data type: {type(data_module.val_dataset)}")
        first_item = data_module.train_dataset[0]
        print(first_item) # if input only it will be a tensor, if it inclues coordinates it will be a tuple of (tensor, coordinates)
        print(f"First item shape (should be nBands, tile_size, tile_size): {first_item[0].shape}")
        if data_module.test_dataset is not None:
            print(f"Test dataset length: {len(data_module.test_dataset)}. Data type: {type(data_module.test_dataset)}")
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    return None      
    
if __name__ == "__main__":
    main()