"""
AK: This is a minimal example of loading dataarrays from multiple files using
CFA engine, and tiling them in space and time, to work out the logic of tiling
and index mapping.
"""

import matplotlib
matplotlib.use("Agg") #Ensure a non-interactive Matplotlib backend
import matplotlib.pyplot as plt
import os
import rioxarray as rxr
import xarray as xr
from typing import Optional, Any, List, Tuple


def tiling_with_concat(das: List[xr.DataArray], 
                       data_vars: List[str], 
                       tile_size: int, 
                       time_slice_size: int, 
                       debug: bool = False, 
                       plotting: bool = False,
                       fig_path: str = None) -> dict:
    das_tiled = {}
    for da, var_name in zip(das, data_vars):
        tiles = da.coarsen(x=tile_size, y=tile_size, boundary='pad').construct(x=("x_coarse", "x_fine"), y=("y_coarse", "y_fine"))
        if 'time' in da.dims:
            if da.sizes['time'] < time_slice_size:
                time_slices = [slice(0, da.sizes['time'])]
            else:
                # for ST data, the batches are split into  time slices
                time_slices = [slice(i, i + time_slice_size) for i in range(0, da.sizes['time'], time_slice_size)]
            tile_values = []
            for time_slice in time_slices:
                time_sliced_tiles = tiles.isel(time=time_slice)
                stacked_time_sliced_tiles = time_sliced_tiles.stack(batch_dim=("x_coarse", "y_coarse"))
                tile_values.append(stacked_time_sliced_tiles.transpose("batch_dim", "time", "band", "y_fine", "x_fine"))
            tile_values = xr.concat(tile_values, dim="batch_dim") # this actually aligns time so the time dimension does not collapse
            # alternative solutions would be to stack batch_dim with time, or to
            # create a new time index for each time slice and stack on that.
            # Actual time must be preserved per tile for thw dataloader, similar
            # to geographical coordinates.
        else:
            # for  static data, stack tiles
            stacked_tiles = tiles.stack(batch_dim=("x_coarse", "y_coarse"))
            tile_values = stacked_tiles.transpose("batch_dim", "band", "y_fine", "x_fine")
    das_tiled[var_name] = tile_values
        
    if plotting:
        # see what the values are for the for the chess-met data after tiling.
        batches_to_plot = [20, 74]
        var_to_plot = 'dtr'
        for iBatch in batches_to_plot:
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            for iax, ax in enumerate(axes):
                tile = das_tiled[var_to_plot].isel(batch_dim=iBatch, time=iax, band=0) 
                tile.plot(ax=ax)
                ax.axis('off')
                ax.set_title(f"Batch {iBatch}, time {iax}")
                # ax.collections[0].colorbar.remove()
            fig.savefig(os.path.join(fig_path, f"tiling_results_on_variable_{var_to_plot}_{iBatch}.png"))
    return das_tiled

def tiling_and_stacking(das: List[xr.DataArray],
                       data_vars: List[str],
                       tile_size_x: int,
                       tile_size_y: int,
                       time_slice_size: int,
                       debug: bool = False,
                       plotting: bool = False,
                       fig_path: str = None) -> Tuple[dict, dict]:
    das_tiled = {}  
    index_mappings = {} 
    # we need to process ST dimensions first to work out the length of
    # index mapping for static data
    idx_of_first_ST_da = next(i for i, da in enumerate(das) if 'time' in da.dims)
    first_da = das[idx_of_first_ST_da]
    first_da_name = data_vars[idx_of_first_ST_da]
    # drop them from the list of das to loop over, and add them at the end, to ensure that the index mapping is created based on the ST dataset, and that the same spatial tiles are sampled for static and ST datasets.
    das = [first_da] + das[:idx_of_first_ST_da] + das[idx_of_first_ST_da+1:]
    data_vars = [first_da_name] + data_vars[:idx_of_first_ST_da] + data_vars[idx_of_first_ST_da+1:]
    
    for da, var_name in zip(das, data_vars):
        if 'time' in da.dims:
            # coarse-construct reshaping over time allows for user-defined time slice size. 
            tiles = da.coarsen(
                time=time_slice_size, 
                x=tile_size_x, 
                y=tile_size_y, 
                boundary='pad'
                ).construct(
                    time=("time_coarse", "time_inner"), 
                    x=("x_coarse", "x_fine"), 
                    y=("y_coarse", "y_fine")
                    )
            if debug:
                # check how fine time looks like:
                print(tiles.dims)
                print(tiles.shape)
                print(tiles.time)
            ST_data_time_length = len(tiles.time_coarse) # need to know how much to extend the index mapping for static
            stacked_tiles = tiles.stack(
                batch_dim=("time_coarse", "y_coarse", "x_coarse")
                ).transpose("batch_dim", "band", "time_inner", "y_fine", "x_fine")
            if debug:
                print(f" Dims: {stacked_tiles.dims}, shape: {stacked_tiles.sizes}")
                # where does time go as as a coordinate after stacking?
                print(stacked_tiles.coords)
                print(stacked_tiles.time)
            time_of_tiles = stacked_tiles.time.values 
            # index mapping is the idx: idx for the ST dataset
            index_mapping = {idx: idx for idx in range(stacked_tiles.sizes['batch_dim'])}
        else:
            # for  static data, stack tiles
            tiles = da.coarsen(
                x=tile_size_x, 
                y=tile_size_y, 
                boundary='pad'
                ).construct(
                    x=("x_coarse", "x_fine"), 
                    y=("y_coarse", "y_fine")
                    )
            stacked_tiles = tiles.stack(
                batch_dim=("y_coarse", "x_coarse")
                ).transpose("batch_dim", "band", "y_fine", "x_fine")
            time_of_tiles = [None] * stacked_tiles.sizes['batch_dim'] # assign None as time coordinate for static data
            # for static data, the index mapping is of length ST_data_length
            # * batch_dim_length, and maps each time slice of the ST dataset to the same spatial tile in the static dataset
            index_mapping = {idx: idx % stacked_tiles.sizes['batch_dim'] for idx in range(ST_data_time_length * stacked_tiles.sizes['batch_dim'])}

        das_tiled[var_name] = list(zip(stacked_tiles.values, time_of_tiles))
        index_mappings[var_name] = index_mapping
        return das_tiled, index_mappings

def main():
    plotting = False
    debug = True
    fig_path = "./experiments/figures/"
    os.makedirs(fig_path, exist_ok=True)
    # example usage
    data_path = "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/chess-met/aggregations"
    # data variable have to be sent to the dataloader as a list - this is also a
    # way to have user-determined variable order
    data_vars = ['dtr', 'huss', 'psurf']
    
    # data_vars = ['dtr']
    
    
    # operating with DataArrays directly
    das = []            
    for subdir, _, files in os.walk(data_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()   
            nc_path = None 
            if ext in [".nc", ".nc4"]:
                nc_path = os.path.join(subdir, file)
                print("Found netcdf file:", nc_path)
            elif ext in [".nca"]:
                nc_path = os.path.join(subdir, file)
                print("Found aggregations file:", nc_path)
            if nc_path is not None:
                # I want to read individual variable from the dataset as an
                # array. We may wish use individual variables as "modalities"
                var_to_extract = None
                for var in data_vars:
                    if var in file:
                        var_to_extract = var
                        print(f"Found variable {var} in file name: {nc_path}")
                        break
                if var_to_extract is None:
                    print(f"No variable from the list {data_vars} found in file name: {nc_path}. Skipping this file.")
                    continue
                else:                    
                    ds = xr.open_dataset(nc_path, engine="CFA", chunks={"time": 64})
                    ds = ds.isel(time=range(10)) #grab first 10 time points for testing
                    da = ds[var_to_extract]
                    # # do we need a band dimenstion here?
                    if 'band' not in da.dims:
                        da = da.expand_dims(dim={"band": 1}, axis=1).assign_coords(band=[1]) # add a band dimension with a single band, and name it 1
                    das.append(da)
    
    # check if the chunks are intact
    move_on = True
    try:
        for da in das:
            print(f"Chunks for {da.name}: {da.data.chunks}")
    except Exception as e:
        print("Error accessing chunks:", e)
        move_on = False
   
    # tile the datasets in space and time. This will be required for global
    # index mapping idx: (time, tile_xy) to be passed to the dataset, to ensure
    # that the same spatial tile is sampled from spatial and spatio-temporal
    # datasets.
    if move_on:
        
        # get times
        min_times = [da.time.min() for da in das if 'time' in da.dims]
        max_times = [da.time.max() for da in das if 'time' in da.dims]
        common_min_time = max(min_times)
        common_max_time = min(max_times)
        if debug:
            print(f"Min time: {min_times} \n Max time: {max_times}")
            print(f"Smallest common time range: \n Min time: {common_min_time.values} \n Max time: {common_max_time.values}")
        
        # crop datasets to the smallest common time range
        for i, da in enumerate(das):
            if 'time' in da.dims:
                das[i] = da.sel(time=slice(common_min_time.values, common_max_time.values))
        if debug:
            min_times = [da.time.min().values for da in das if 'time' in da.dims]
            max_times = [da.time.max().values for da in das if 'time' in da.dims]
            print(f"After cropping to the smallest common time range: \n Min time: {min_times} \n Max time: {max_times}")
            

        tile_size_x = 128    
        tile_size_y = 128
        time_slice_size = 2      
        ## Option 1: concatenate
        # das_tiled_1 = tiling_with_concat(das, data_vars, tile_size,
        # time_slice_size, debug=debug, plotting=plotting, fig_path=fig_path)
        
        ## Option 2: stack time and batch dims
        das_tiled_2, index_mappings = tiling_and_stacking(das, data_vars, 
                                                          tile_size_x, tile_size_y, 
                                                          time_slice_size, debug=debug, 
                                                          plotting=plotting, fig_path=fig_path)
    
        print("Index mapping for variable dtr:", index_mappings['dtr'])
        
       
if __name__ == "__main__":
    main()