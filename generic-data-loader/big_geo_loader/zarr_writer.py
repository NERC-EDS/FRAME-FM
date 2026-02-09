from .utils import load_data_from_selector, safely_remove_dir
from .transforms import *
from .settings import DefaultSettings as DEFAULTS

from pathlib import Path

import dask
import xarray as xr


def create_zarr_name(data_uri: str) -> str:
    """
    Create a Zarr file name based on the data URI.
    Args:
        data_uri (str): The URI of the data source.
    Returns:
        str: A string representing the Zarr file name.
    """
    # Extract the base name from the data URI to create a unique Zarr file name
    base_name = Path(data_uri).stem
    zarr_name = f"{base_name}.zarr"
    return zarr_name


def cache_data_to_zarr(selectors: list[dict], 
                       cache_dir: str | Path, 
                       generate_stats: bool = True, 
                       pre_transforms: list | None = None) -> dict:
    """
    Cache data to Zarr format based on the provided selectors and cache directory.

    Args:
        selectors (list[dict]): A list of dictionaries defining selections for data.
        cache_dir (str): The directory where cached Zarr files will be stored.
        generate_stats (bool): Whether to generate statistics during caching.
        pre_transforms (list | None): A list of pre-transform class instances to apply before caching.
    """
    # Ensure pre_transformas are list
    pre_transforms = pre_transforms or []

    # Set up a dictionary to keep track of the mapping from (uri, var_id) to Zarr file paths
    var_zarr_dict = {}

    # Create the cache directory if it doesn't exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    for selector in selectors:
        # Load the data from the URI
        ds = load_data_from_selector(selector)
        variables = selector.get("variables", {})
        common_pre_transforms = selector.get("common", {}).get("pre_transforms", [])
        transform_rule = selector.get("common", {}).get("pre_transform_rule", "append")

        if transform_rule == "override":
            transforms = common_pre_transforms
        elif transform_rule == "append":
            transforms = pre_transforms + common_pre_transforms

        # Apply dataset-specific transforms if specified in var_settings
        for var_id in variables:
            var_pre_transforms = selector["variables"][var_id].get("pre_transforms", [])
            for transform in var_pre_transforms:
                ds[var_id] = resolve_transform(transform)(ds[var_id])

        # Apply common pre-transforms if any
        for transform in transforms:
            ds = resolve_transform(transform)(ds)

        # Cache the dataset to Zarr
        data_uri = selector["uri"]
        zarr_name = create_zarr_name(data_uri)
        cache_path = Path(cache_dir) / zarr_name

        # Clear any existing cache files before caching new data (note that it is a directory)
        safely_remove_dir(cache_path)

        print(f"\nCaching data to {cache_path} from {data_uri}")

        USE_CHUNKED_METHOD = True  # Set to True to use chunked writing method, False for direct writing
        if USE_CHUNKED_METHOD:
            # Use output_utils to write in chunks
            print("Using chunked writing method...")
            write_zarr(ds, cache_path, chunks=selector.get("common", {}).get("chunks", DEFAULTS.chunks))
        else:
            ds.compute()  # Ensure the dataset is computed before writing to Zarr
            _ = ds.to_zarr(cache_path, mode="w", zarr_version=2)
            print(f"Finished caching data to {cache_path}")

        if generate_stats:
            # Generate and save statistics for the cached data
            print("\nHandle Stats here... (placeholder)")

        # Update the response dictionary
        var_zarr_dict[data_uri] = {"zarr_path": cache_path, "variables": variables}

    print("\nFinished processing all selectors.")
    return var_zarr_dict


def write_zarr(ds: xr.Dataset,  output_path: Path | str, chunks: dict[str, int] | None = None) -> Path | str:
    """
    Return output after applying chunking and determining the output format and chunking.
    """
    print(f"Writing output to {output_path} with chunking: {chunks}")
    chunked_ds = ds.chunk(chunks) if chunks else ds

    # TODO: writing output works currently only in sync mode, see:
    #  - https://github.com/roocs/rook/issues/55
    #  - https://docs.dask.org/en/latest/scheduling.html
    with dask.config.set(scheduler="synchronous"):
        delayed_obj = chunked_ds.to_zarr(output_path, zarr_version=2, compute=False)
        delayed_obj.compute()

    print(f"Wrote output file: {output_path}")
    return output_path

