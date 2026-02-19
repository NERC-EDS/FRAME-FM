from .utils import load_data_from_selector, open_cached_zarrs, safely_remove_dir, create_cache_path, hash_selector
from .transforms import *
from .settings import DEBUG, DefaultSettings as DEFAULTS

from pathlib import Path

import dask
import xarray as xr


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

    # Create the cache directory if it doesn't exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Collect data_uris for later
    data_uris = []

    for selector in selectors:
        # Load the data from the URI, without applying any subset or transforms
        ds = load_data_from_selector(selector)

        # Set the rules
        variables = selector.get("variables", {})
        common_pre_transforms = selector.get("common", {}).get("pre_transforms", [])
        transform_rule = selector.get("common", {}).get("pre_transform_rule", "append")

        if transform_rule == "override":
            transforms = common_pre_transforms
        else:  # If not set, or set to "append"
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
        cache_path = create_cache_path(selector["uri"], cache_dir)

        # Clear any existing cache files before caching new data (note that it is a directory)
        safely_remove_dir(cache_path)

        data_uri = selector["uri"]
        data_uris.append(data_uri)
        print(f"\nCaching data to {cache_path} from {data_uri}")

        # Compute a hash of the selector for caching purposes
        selector_hash = hash_selector(selector)
        print(f"Computed hash for selector: {selector_hash}")
        ds.attrs["_selection_cache_hash"] = selector_hash  # Store the hash in the dataset attributes for reference

        USE_CHUNKED_METHOD = True  # Set to True to use chunked writing method, False for direct writing
        if USE_CHUNKED_METHOD:
            # Use output_utils to write in chunks
            print("Using chunked writing method...")
            write_zarr(ds, cache_path, chunks=selector.get("common", {}).get("chunks", DEFAULTS.chunks))
        else:
            ds.compute()  # Ensure the dataset is computed before writing to Zarr
            _ = ds.to_zarr(cache_path, mode="w", zarr_format=2)
            print(f"Finished caching data to {cache_path}")

        if generate_stats:
            # Generate and save statistics for the cached data
            print("\nHandle Stats here... (placeholder)")

    print("\nFinished processing all selectors.")

    # Now load the cached Zarr files into memory and add to the response dictionary
    var_zarr_dict = open_cached_zarrs(uris=data_uris, cache_dir=cache_dir)
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
        delayed_obj = chunked_ds.to_zarr(output_path, zarr_format=DEFAULTS.zarr_format, compute=False)
        delayed_obj.compute()

    print(f"Wrote output file: {output_path}")
    return output_path

