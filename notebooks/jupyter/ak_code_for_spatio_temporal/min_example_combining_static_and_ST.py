"""
AK: This is a mini example to test the tiling and indexing approach for combining static and ST data.
"""
import matplotlib
matplotlib.use("Agg") #Ensure a non-interactive Matplotlib backend
import matplotlib.pyplot as plt
import os
import rioxarray as rxr
import xarray as xr
from typing import Optional, Any, List, Tuple
import numpy as np

def tile_and_stack(das: List[xr.DataArray], data_vars: List[str], tile_size_x: int, tile_size_y: int, time_slice_size: int, debug: bool=False) -> Tuple[dict, dict]:
    # check that index mapping works for static
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
            # need to know how much to extend the index mapping for static
            ST_data_time_length = len(tiles.time_coarse) 
            stacked_tiles = tiles.stack(
                batch_dim=("time_coarse", "y_coarse", "x_coarse")
                ).transpose("batch_dim", "band", "time_inner", "y_fine", "x_fine")
            if debug:
                print(f" Dims: {stacked_tiles.dims}, shape: {stacked_tiles.sizes}")
                # where does time go as as a coordinate after stacking?
                print(stacked_tiles.coords)
                print(stacked_tiles.time)
            time_of_tiles = stacked_tiles.time.values 
            x_coarse_of_tiles = stacked_tiles.x_coarse.values
            y_coarse_of_tiles = stacked_tiles.y_coarse.values
            # x_fine_of_tiles = stacked_tiles.x_fine.values
            # y_fine_of_tiles = stacked_tiles.y_fine.values
            # index mapping is the idx: idx for the ST dataset
            index_mapping = {idx: idx for idx in range(stacked_tiles.sizes['batch_dim'])}
        else:
            # for static data, stack tiles
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
            x_coarse_of_tiles = stacked_tiles.x_coarse.values
            y_coarse_of_tiles = stacked_tiles.y_coarse.values
            # x_fine_of_tiles = stacked_tiles.x_fine.values
            # y_fine_of_tiles = stacked_tiles.y_fine.values
            # for static data, the index mapping is of length ST_data_length * batch_dim_length, 
            # and maps each time slice of the ST dataset to the same spatial tile in the static dataset
            index_mapping = {idx: idx % stacked_tiles.sizes['batch_dim'] for idx in range(ST_data_time_length * stacked_tiles.sizes['batch_dim'])}

        das_tiled[var_name] = list(zip(stacked_tiles.data, time_of_tiles, zip(x_coarse_of_tiles, y_coarse_of_tiles))) # keeping it lazy
        index_mappings[var_name] = index_mapping
    return das_tiled, index_mappings

def correct_slice(coord, vmin, vmax):
    # make sure slice bounds are aligned with coordinate order
    return slice(vmin, vmax) if coord[0] < coord[-1] else slice(vmax, vmin)

def main():
    """ 
    PLot example batches and try creating the data module.
    Currently not using hydra.
    """
    plotting = True
    debug = False
    # large tiles for debug!
    if debug:
        data_vars = ['land_cover','dtr', 'huss']
        tile_size_x = 330    
        tile_size_y = 550
        time_slice_size = 1    
        idx_to_plot = 0   
    else:
        data_vars = ['land_cover','dtr', 'huss', 'psurf', 'precip','rlds','rsds', 'sfcWind', 'tas']
        tile_size_x = 128    
        tile_size_y = 256
        time_slice_size = 1
        idx_to_plot = 20 
    
    
    fig_path = "./experiments/figures/"
    os.makedirs(fig_path, exist_ok=True)
    # example usage
    data_paths = ["/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/land_cover_map_2015/data/", "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/chess-met/aggregations"]

    # load stuff
    das = []
    for subdir, _, files in os.walk(data_paths[0]):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in [".tif", ".tiff"]:
                geotiff_path = os.path.join(subdir, file)
                print("Found tif file:", geotiff_path)
                da = rxr.open_rasterio(geotiff_path)
                #name the data array
                da.name = data_vars[0]
                das.append(da)
                
    for subdir, _, files in os.walk(data_paths[1]):
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
                # find which varible name is contained in the file
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
                    ds1 = ds.isel(time=range(10)) #grab first 10 time points for testing
                    da = ds1[var_to_extract]
                    # # do we need a band dimenstion here?
                    if 'band' not in da.dims:
                        da = da.expand_dims(dim={"band": 1}, axis=1).assign_coords(band=[1]) # add a band dimension with a single band, and name it 1
                    das.append(da)
    
    # get min min and max x coordinate and min and max y coordinate across all
    # datasets, to check the alignment
    min_x =[da.x.min().item() for da in das]
    max_x = [da.x.max().item() for da in das]
    min_y = [da.y.min().item() for da in das]
    max_y = [da.y.max().item() for da in das]
    common_min_x = max(min_x)
    common_max_x = min(max_x)
    common_min_y = max(min_y)
    common_max_y = min(max_y)
    if debug:
        print(f"Min x: {min_x} \n Max x: {max_x} \n Min y: {min_y} \n Max y: {max_y}")
        print('Smallest bounding box:')
        print(f"Min x: {common_min_x} \n Max x: {common_max_x} \n Min y: {common_min_y} \n Max y: {common_max_y}")
    
    # plot land cover and temoerature on top of each other to check the alignment
    if plotting:
        fig, ax = plt.subplots(figsize=(16, 20))
        das[1].isel(time=0, band=0).plot(ax=ax, yincrease=False, alpha=0.9, cmap=plt.cm.Blues, add_colorbar=False)
        das[0].isel(band=3).plot(ax=ax, yincrease=False, alpha=0.3, cmap=plt.cm.Wistia, add_colorbar=False)
        # check bounding boxes
        ax.plot([min_x[1], max_x[1], max_x[1], min_x[1], min_x[1]], [min_y[1], min_y[1], max_y[1], max_y[1], min_y[1]], color='blue', linestyle='-', label='CHESS-MET bbox') 
        ax.plot([min_x[0], max_x[0], max_x[0], min_x[0], min_x[0]], [min_y[0], min_y[0], max_y[0], max_y[0], min_y[0]], color='magenta', linestyle='--', label='Land cover bbox')
        ax.invert_yaxis()
        ax.set_axis_off()
        ax.legend(loc='upper left' , fontsize=24)
        ax.set_title(f"Overlay of {das[1].name} and land cover band 3", fontsize=30)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_path, f"overlay_of_{das[1].name}_and_land_cover_original.png"))
        

    # crop datasets to the smallest bounding box
    for i, da in enumerate(das):
        das[i] = da.sel(x=correct_slice(da.x, common_min_x, common_max_x), y=correct_slice(da.y, common_min_y, common_max_y))
    if debug:
        min_x = [da.x.min().item() for da in das]
        max_x = [da.x.max().item() for da in das]
        min_y = [da.y.min().item() for da in das]
        max_y = [da.y.max().item() for da in das]
        print(f"After cropping to the smallest bounding box: \n Min x: {min_x} \n Max x: {max_x} \n Min y: {min_y} \n Max y: {max_y}")
        
    # make sure that all x and y are increasing for all datasets, if not,
    # reverse the order of x and y and the values
    for i, da in enumerate(das):
        if da.x[0] > da.x[-1]:
            das[i] = da.reindex(x=da.x[::-1])
        if da.y[0] > da.y[-1]:
            das[i] = da.reindex(y=da.y[::-1])
    
    # plot land cover and temoerature on top of each other to check the alignment
    if plotting:
        fig, ax = plt.subplots(figsize=(16, 20))
        das[1].isel(time=0, band=0).plot(ax=ax, yincrease=False, alpha=0.9, cmap=plt.cm.Blues, add_colorbar=False)
        das[0].isel(band=3).plot(ax=ax, yincrease=False, alpha=0.3, cmap=plt.cm.Wistia, add_colorbar=False)
        if debug:
            ax.plot([min_x[1], max_x[1], max_x[1], min_x[1], min_x[1]], [min_y[1], min_y[1], max_y[1], max_y[1], min_y[1]], color='blue', linestyle='-', label='CHESS-MET bbox') 
            ax.plot([min_x[0], max_x[0], max_x[0], min_x[0], min_x[0]], [min_y[0], min_y[0], max_y[0], max_y[0], min_y[0]], color='magenta', linestyle='--', label='Land cover bbox')
        ax.invert_yaxis()
        ax.set_axis_off()
        ax.legend(loc='upper left' , fontsize=24)
        ax.set_title(f"Overlay of {das[1].name} and land cover band 3", fontsize=30)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_path, f"overlay_of_{das[1].name}_and_land_cover_after_cropping.png"))

    # get the longest time dimension across datasets, to check the alignment in time
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
    
    
    # tile and stack datasets; create index mapping
    das_tiled, index_mappings = tile_and_stack(das, data_vars, tile_size_x, tile_size_y, time_slice_size, debug=debug)
    first_da_name = das_tiled.keys().__iter__().__next__()
    # get time length of the first dataset, to check the alignment in time
    #show ids_mapping for land cover
    print(index_mappings['land_cover'])
    print(f"Length of ds for land cover: {len(das_tiled['land_cover'])},\n length of index mapping: {len(index_mappings['land_cover'])}")
    # get all the keys of dataset where item 0
    keys_w_zero = [key for key, value in index_mappings['land_cover'].items() if value == 0]
    difference_in_length = keys_w_zero[1] - keys_w_zero[0]
    
    
    if plotting:
        # plot tiles using index mapping to check alignment
        idx_to_plot_1 = idx_to_plot 
        idx_to_plot_2 = idx_to_plot + 2*difference_in_length # plot 2 time steps ahead
        idx_in_st_1 = index_mappings[first_da_name][idx_to_plot_1]
        idx_in_st_2 = index_mappings[first_da_name][idx_to_plot_2]
        idx_in_static_1 = index_mappings['land_cover'][idx_to_plot_1]
        idx_in_static_2 = index_mappings['land_cover'][idx_to_plot_2]
        fig, axes = plt.subplots(2, 2, figsize=(12, 16))
        axes = axes.ravel() # flatten the axes for easier indexing
        axes[0].imshow(np.asarray(das_tiled[first_da_name][idx_in_st_1][0][0][0])) # first tile, first band, first time slice
        axes[0].set_title(f"{first_da_name}: idx: {idx_to_plot_1}, x_coarse: {das_tiled[first_da_name][idx_in_st_1][2][0]}, y_coarse: {das_tiled[first_da_name][idx_in_st_1][2][1]}", fontsize=16)
        axes[1].imshow(np.asarray(das_tiled['land_cover'][idx_in_static_1][0][3])) # first tile, first band
        axes[1].set_title(f"Land cover: idx: {idx_to_plot_1}, x_coarse: {das_tiled['land_cover'][idx_in_static_1][2][0]}, y_coarse: {das_tiled['land_cover'][idx_in_static_1][2][1]}", fontsize=16)
        axes[2].imshow(np.asarray(das_tiled[first_da_name][idx_in_st_2][0][0][0])) # first tile, first band, second time slice
        axes[2].set_title(f"{first_da_name}: idx: {idx_to_plot_2}, x_coarse: {das_tiled[first_da_name][idx_in_st_2][2][0]}, y_coarse: {das_tiled[first_da_name][idx_in_st_2][2][1]}", fontsize=16)
        axes[3].imshow(np.asarray(das_tiled['land_cover'][idx_in_static_2][0][3])) # first tile, first band - should be the same as for time slice 0
        axes[3].set_title(f"Land cover: idx: {idx_to_plot_2}, x_coarse: {das_tiled['land_cover'][idx_in_static_2][2][0]}, y_coarse: {das_tiled['land_cover'][idx_in_static_2][2][1]}", fontsize=16)
        for ax in axes:
            ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_path, f"check_alignment_of_tiles.png"))
        if debug:
            # plot the first four tiles in thefirst dataset to check the tiling
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            axes = axes.ravel() # flatten the axes for easier indexing
            for i in range(4):
                axes[i].imshow(das_tiled[first_da_name][i][0][0].transpose(1,2,0)) # first tile, first band, first time slices
                axes[i].set_title(f"{first_da_name}: idx: {i}, x_coarse: {das_tiled[first_da_name][i][2][0]}, y_coarse: {das_tiled[first_da_name][i][2][1]}", fontsize=16)
                # axes[i].invert_yaxis()
            fig.tight_layout()
            fig.savefig(os.path.join(fig_path, f"check_tiling_of_{first_da_name}.png"))
            # and the same for land cover
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            axes = axes.ravel() # flatten the axes for easier indexing
            for i in range(4):  
                axes[i].imshow(das_tiled['land_cover'][i][0][3]) # first tile, first band
                axes[i].set_title(f"Land cover: idx: {i}, x_coarse: {das_tiled['land_cover'][i][2][0]}, y_coarse: {das_tiled['land_cover'][i][2][1]}", fontsize=16)
                # axes[i].invert_yaxis()
            fig.tight_layout()
            fig.savefig(os.path.join(fig_path, f"check_tiling_of_land_cover.png"))
    
if __name__ == "__main__":
    main()