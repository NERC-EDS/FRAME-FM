import matplotlib
matplotlib.use("Agg") #Ensure a non-interactive Matplotlib backend
import matplotlib.pyplot as plt
import os
import torch
import rioxarray as rxr
import xarray as xr
from typing import Optional, Any, List, Tuple
import pyproj
import warnings
from torch.utils.data import Dataset, DataLoader, Sampler
import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
import pandas as pd
import numpy as np

# add src to python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
print("Python path:", sys.path)

from FRAME_FM.utils.LightningDataModuleWrapper import BaseDataModule
from FRAME_FM.datasets.MultiSource_Dataset import TransformedInputIndexMappingDataset

class XarrayMultiSourceDataModule(BaseDataModule):
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
        stochastic_transform: Optional[callable] = None,
        list_of_variables_to_extract: Optional[List[str]] = None,
        list_of_source_crs: Optional[List[str]] = None,
        tile_size_x: int = 256,
        tile_size_y: int = 256,
        time_slice_size: int = 1,
        time_extent: Optional[Tuple[str, str]] = None,
        chunks: Optional[dict] = None
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
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.time_slice_size = time_slice_size
        self.stochastic_transform = stochastic_transform
        self.list_of_variables_to_extract = list_of_variables_to_extract
        self.list_of_source_crs = list_of_source_crs
        self.time_extent = time_extent
        self.chunks = chunks

    def _load_raw_data(self) -> List[xr.DataArray]:
        """
        Iteratevely goes through subdirectories of self.data_root, reads all
        files with supported formats (geotiff, netcdf, zarr, etc.) 
        into xarray DataArrays, and returns a tuple of DataArrays. 
        The order of the DataArrays in the tuple is determined by the sorted
        order of the file paths.
        
        """
        # AK we should be able to specify a list of specific direcroties, so
        # cannot use os.wal here. Just should we treat data_root as a list of strings?
        das = []
        for root_subdir in self.data_root:
            for dirpath, _, filenames in os.walk(root_subdir):
                for filename in sorted(filenames):
                    file_path = os.path.join(dirpath, filename)
                    variable_name = None
                    if self.list_of_variables_to_extract is not None:
                        for var in self.list_of_variables_to_extract:
                            if var in filename:
                                variable_name = var
                                break
                        if variable_name is None:
                            print(f"No variable from the list {self.list_of_variables_to_extract} found in file name: {file_path}. Skipping this file.")
                            continue
                    da = self._load_from_single_file(file_path, variable_name)
                    if da is not None:
                        das.append(da)
        # no need to send it to self, as it assigns to self._raw_data in the
        # BaseDataModule.setup()
        return das
    
    def _load_from_single_file(self, file_path: str, variable_name: str=None) -> xr.DataArray:
        """
        Based on the extension of the file (tif/tiff, nc/ncs, zarr, etc.) use the appropriate library to read it and return an xarray DataArray.
        This is meant to be used in the case of multiple input files with
        different formats that 

        Args:
            file_path (str): path to the specific file withind self.data_root
            directory.
            variable_name (str): the name of the variable to extract from the
            file.
            This is either to extract a specific variable from xr dataset, or
            assign to a xr dataarray if the name is missing
        """
        # AK: where should subsetting happen???
        ext = os.path.splitext(file_path)[-1].lower()
        if ext in [".tif", ".tiff"]:
            da = rxr.open_rasterio(file_path)
            if variable_name is not None and da.name is None:
                da.name = variable_name 
        elif ext in [".nc", ".nc4"]:
            ds = xr.open_dataset(file_path, engine="netcdf4")  
            if variable_name is not None:
                da = ds[variable_name]
            else:
                da = ds.to_array()
        elif ext in [".zarr"]:
            ds = xr.open_zarr(file_path)
            if variable_name is not None:
                da = ds[variable_name]
            else:                
                da = ds.to_array()
        elif ext in [".nca"]:
            ds = xr.open_dataset(file_path, engine="CFA", chunks=self.chunks) # must pass chunks with CFA
            # if subsetting is happening, it should only apply to the time
            # dimension - it does not feel right to have this sitting here
            if self.time_extent is not None:
                ds = ds.sel(time=slice(self.time_extent[0], self.time_extent[1]))
            if variable_name is not None:
                da = ds[variable_name]
            else:
                da = ds.to_array()
        elif ext in [".asc"]:
            raise NotImplementedError("Reading ASCII grid files is not yet implemented.")
        elif ext in [".shp", ".geojson"]:
            raise NotImplementedError("Reading vector data formats is not yet implemented.")
        elif ext in [".csv"]:
            raise NotImplementedError("Reading tabular data formats is not yet implemented.")
        else:
            return None # there will be files in aggregations dir that are not geospatial data
        # should returnt the data in the DataArray format
        if 'band' not in da.dims:
            da = da.expand_dims('band')
        return da
    
    def _correct_slice(self, coord, vmin, vmax):
        # make sure slice bounds are aligned with coordinate order for an xarray coords
        return slice(vmin, vmax) if coord[0] < coord[-1] else slice(vmax, vmin)
    
    def _align_dataarrays(self, das: List[xr.DataArray]) -> List[xr.DataArray]:
        """
        This function takes a list of xarray DataArrays, finds the common
        spatial and temporal extent, assuming that the datasets have dims
        time (optiopnal), x and y, and crops all DataArrays to that common extent.
        If any of the arrays has no time dimension, it is not cropped. 
        All datasets are then reindexed to have coordinates in increasing order.
        
        THIS METHOD WILL BE REPLACED BY CALLING TRANSFORMS
        
        Args:
            das (List[xr.DataArray]): a list of xarray DataArrays to be aligned.
        Returns:
            List[xr.DataArray]: a list of xarray DataArrays cropped to the
            common spatial and temporal extent.
        """
        # find the smallest common extent in space and crop all datasets to it
        min_x =[da.x.min().item() for da in das]
        max_x = [da.x.max().item() for da in das]
        min_y = [da.y.min().item() for da in das]
        max_y = [da.y.max().item() for da in das]
        common_min_x = max(min_x)
        common_max_x = min(max_x)
        common_min_y = max(min_y)
        common_max_y = min(max_y)
        for i, da in enumerate(das):
            das[i] = da.sel(x=self._correct_slice(da.x, common_min_x, common_max_x), 
                            y=self._correct_slice(da.y, common_min_y, common_max_y))
        
        
        # find the smallest common extent in time, crop all ST datasets to it
        min_times = [da.time.min().values for da in das if 'time' in da.dims]
        max_times = [da.time.max().values for da in das if 'time' in da.dims]
        common_min_time = max(min_times)
        common_max_time = min(max_times)
        # this needs to be modified to cope with datsets that are on different
        # time scale (e.g. daily, weekly, hourly)
        for i, da in enumerate(das):
            if 'time' in da.dims:
                das[i] = da.sel(time=self._correct_slice(da.time, 
                                                         common_min_time, 
                                                         common_max_time))
                
        # reindex all datasets with coordinates in increasing order
        for i, da in enumerate(das):
            if 'x' in da.dims and da.x[0] > da.x[-1]:
                das[i] = da.reindex(x=da.x[::-1])
            if 'y' in da.dims and da.y[0] > da.y[-1]:
                das[i] = da.reindex(y=da.y[::-1])
            if 'time' in da.dims and da.time[0] > da.time[-1]:
                das[i] = da.reindex(time=da.time[::-1])
        return das
    
    def _tile_and_stack_dataarrays(self,
                        das: List[xr.DataArray]
                        ) -> Tuple[dict, dict]:
        """
        This function tiles all the pre-aligned data araays, extracts their
        termporal and spatial coordinates, and packages them into an iterable
        with the first dimension along which the data will be batched.
        
        Args:
            das (List[xr.DataArray]): list of aligned dataarrays to be tiled and stacked.

        Returns:
            Tuple[dict, dict]: a tuple of two dictionaries.
            the first dictionary contains data values and coordss in 
            {var_name: (tile_stack, time_coord, spatial_coords)} format. 
            The second dictionary {var_name: index mapping} 
            contains index mapping for aligning datasets when sampling them.
        """
        das_tiled = {}  
        index_mappings = {} 
        # we need to process ST dimensions first to work out the length of
        # index mapping for static data - potentially need to find the londgest
        # ST dataset
        idx_of_first_ST_da = next(i for i, da in enumerate(das) if 'time' in da.dims)
        first_da = das[idx_of_first_ST_da]
        first_da_name = self.list_of_variables_to_extract[idx_of_first_ST_da]
        # drop them from the list of das to loop over, and add them at the end, 
        # to ensure that the index mapping is created based on the ST dataset.
        das = [first_da] + das[:idx_of_first_ST_da] + das[idx_of_first_ST_da+1:]
        da_names = [first_da_name] + self.list_of_variables_to_extract[:idx_of_first_ST_da] + self.list_of_variables_to_extract[idx_of_first_ST_da+1:]
        
        for da, var_name in zip(das, da_names):
            if 'time' in da.dims:
                # coarse-construct reshaping over time allows for user-defined time slice size. 
                tiles = da.coarsen(
                    time=self.time_slice_size, 
                    x=self.tile_size_x, 
                    y=self.tile_size_y, 
                    boundary='pad'
                    ).construct(
                        time=("time_coarse", "time_inner"), 
                        x=("x_coarse", "x_fine"), 
                        y=("y_coarse", "y_fine")
                        )
                ST_data_time_length = len(tiles.time_coarse) # need to know how much to extend the index mapping for static
                stacked_tiles = tiles.stack(
                    batch_dim=("time_coarse", "y_coarse", "x_coarse")
                    ).transpose("batch_dim", "band", "time_inner", "y_fine", "x_fine")
                time_of_tiles = stacked_tiles.time.values.astype(np.int64)
                # index mapping is the idx: idx for the ST dataset
                index_mapping = {idx: idx for idx in range(stacked_tiles.sizes['batch_dim'])}
            else:
                # for  static data, stack tiles
                tiles = da.coarsen(
                    x=self.tile_size_x, 
                    y=self.tile_size_y, 
                    boundary='pad'
                    ).construct(
                        x=("x_coarse", "x_fine"), 
                        y=("y_coarse", "y_fine")
                        )
                stacked_tiles = tiles.stack(
                    batch_dim=("y_coarse", "x_coarse")
                    ).transpose("batch_dim", "band", "y_fine", "x_fine")
                time_of_tiles = [np.datetime64('NaT').astype(np.int64) ] * stacked_tiles.sizes['batch_dim']# assign NaT as time coordinate for static data
                # for static data, the index mapping is of length ST_data_length * batch_dim_length, 
                # and maps each time slice of the ST dataset to the same spatial tile in the static dataset
                index_mapping = {idx: idx % stacked_tiles.sizes['batch_dim'] for idx in range(ST_data_time_length * stacked_tiles.sizes['batch_dim'])}
            # spatial coordinates are extracted in the same manner
            coords_of_tiles = self._get_tile_coords(stacked_tiles, source_crs=self.list_of_source_crs[self.list_of_variables_to_extract.index(var_name)])
            das_tiled[var_name] = list(zip(stacked_tiles.data, time_of_tiles, coords_of_tiles)) # keepinng it lazy
            index_mappings[var_name] = index_mapping
        return das_tiled, index_mappings

    def _get_tile_coords(self, 
                         tile_stack: xr.DataArray, 
                         source_crs: str=None, 
                         target_crs: str="EPSG:4326"
                         ) -> List[Tuple[float, float]]:
        """
        This function finds centroids of spatial tiles in the dataset

        Args:
            tile_stack (xr.DataArray): a tile stack build for a single data source.
            source_crs (str, optional): crs of orignial datas in EPSG format. Defaults to None.
            target_crs (str, optional): target crs. Defaults to "EPSG:4326".

        Returns:
            list: a list of (lon, lat) tuples for each tile centroid
        """
        # get centroid coordinates for each tile
        x_centroids = tile_stack.x_coarse.values * self.tile_size_x + tile_stack.x_fine.values[0] + self.tile_size_x // 2
        y_centroids = tile_stack.y_coarse.values * self.tile_size_y + tile_stack.y_fine.values[0] + self.tile_size_y // 2
        # convert to lat/lon if the dataset has a CRS (Coordinate Reference
        # System) defined
        if source_crs is not None:
            transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
            lon_centroids, lat_centroids = transformer.transform(x_centroids, y_centroids)
            return list(zip(lon_centroids, lat_centroids))
        elif tile_stack.rio.crs is not None:
            crs = tile_stack.rio.crs
            transformer = pyproj.Transformer.from_crs(crs, target_crs, always_xy=True)
            lon_centroids, lat_centroids = transformer.transform(x_centroids, y_centroids)
            return list(zip(lat_centroids, lon_centroids))
        else:
            raise ValueError("Source CRS not provided, cannot convert to lat/lon")

    def _create_datasets(self, stage: Optional[str] = None) -> None:
        """
        Reads the raw data from self._raw_data, applies tiling and transformation, and creates the train/val/test datasets.
        """
        das = self._raw_data
        aligned_das = self._align_dataarrays(das)
        tiled_and_stacked_das, index_mappings = self._tile_and_stack_dataarrays(aligned_das)
        # for now just createa datset per source without splitting
        self.train_datasets = {}
        for var_name in tiled_and_stacked_das.keys():
            self.train_datasets[var_name] = TransformedInputIndexMappingDataset(
                base=tiled_and_stacked_das[var_name], 
                index_mapping=index_mappings[var_name], 
                transform=self.stochastic_transform
                )
            
    # --- PyTorch Lightning hooks ----
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Ths method has to be overridden from source one because
        """
        if self._raw_data is None:
            self._raw_data = self._load_raw_data()

        self._create_datasets(stage=stage)

        # Basic sanity checks
        if self.train_datasets is None:
            raise RuntimeError(
                "train_datasets have not been created in _create_datasets()"
            )
    
    # create combined loader
    def _make_combined_dloader(
        self,
        datasets: dict[Dataset[Any]],
        sampler: Optional[Sampler[Any]] = None,
        shuffle: bool = False,
    ) -> DataLoader[Any]:
        # create iterable - to keep things aligned in dataloader, have to have
        # shuffle=false. If aligning tiles is not important, can be changed
        iterables = {}
        for var_name, dataset in datasets.items():
            iterables[var_name] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=self.num_workers,
            )
        return CombinedLoader(iterables=iterables, mode='max_size_cycle')

    def combined_train_dataloader(self) -> DataLoader[Any]:
        if self.train_datasets is None:
            raise RuntimeError(
                "train_datasets is not set up yet. Did you call setup()? "
            )
        return self._make_combined_dloader(
            self.train_datasets, sampler=self._train_sampler, shuffle=True
        )
       


def main():
    """ 
    PLot example batches and try creating the data module.
    Currently not using hydra.
    """
    plotting = True
    debug = True
    # large tiles for debug!
    if debug:
        tile_size_x = 330    
        tile_size_y = 550
        time_slice_size = 2  
        data_vars = ['LCM','dtr', 'huss']
        time_extent = pd.to_datetime(["2000-01-01", "2000-02-02"])# convert to pd.Datetime64    
    else:
        tile_size_x = 128    
        tile_size_y = 256
        time_slice_size = 1
        # data variable have to be sent to the dataloader as a list 
        # data_vars = ['land_cover','dtr', 'huss', 'psurf',
        # 'precip','rlds','rsds', 'sfcWind', 'tas']
        time_extent = pd.to_datetime(["2000-01-01", "2000-02-02"])# convert to pd.Datetime64
    
    fig_path = "./experiments/figures/"
    os.makedirs(fig_path, exist_ok=True)
    
    # only need to convert to tensor for now
    from torchvision.transforms.v2 import ToTensor
    basic_transform = ToTensor()
    
    chunks = {'time': 64}
    
    # example usage
    data_paths = ["/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/land_cover_map_2015/data/", "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/chess-met/aggregations"]
    data_crs = ['EPSG:27700'] * len(data_vars)
    
    dm = XarrayMultiSourceDataModule(
        data_root=data_paths,
        batch_size=4,
        list_of_variables_to_extract=data_vars,
        list_of_source_crs=data_crs,
        tile_size_x=tile_size_x,
        tile_size_y=tile_size_y,
        time_slice_size=time_slice_size,
        time_extent=time_extent,
        train_transforms=basic_transform,
        chunks=chunks   
    )
    # load the data and create datasets
    dm.setup() 
    if debug:
        print("Data module setup complete. Train datasets created:", dm.train_datasets.keys())

    # create combined loader
    dloader = dm.combined_train_dataloader()
    for batch, batch_idx, dataloader_idx in dloader:
        # uh oh, dataloader does not have its own shuffling
        print(f"Batch idx: {batch_idx}, dataloader idx: {dataloader_idx}")
        for var_name, data in batch.items():
            tile_stack, time_coords, geo_coords = data
            print(f"Variable: {var_name}")
            print(f"Tile stack shape: {tile_stack.shape}")
            print(f"Time coords: {time_coords}")
            print(f"Geo coords: {geo_coords}")
            if plotting:
                # plot the first tile in the stack
                fig, ax = plt.subplots(figsize=(6, 8))
                if len(tile_stack.shape) == 5: 
                    ax.imshow(tile_stack[0, 0, 0, :, :], cmap='viridis')
                    ts = pd.Timestamp(int(time_coords[0][0]))
                else:
                    ax.imshow(tile_stack[0, 0, :, :], cmap='viridis')
                    ts = pd.Timestamp(int(time_coords[0]))
                ax.set_title(f"{var_name} \n centroid coords: {float(geo_coords[0][0]):.3f}, {float(geo_coords[1][0]):.3f} \n time: {ts}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                fig.savefig(os.path.join(fig_path, f"{var_name}_tile_0.png"))
                plt.close(fig)
        break  

if __name__ == "__main__":
    main()