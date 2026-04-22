# SPDX-FileCopyrightText: 2026 FRAME-FM Contributors
#
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Union
from collections.abc import Callable
import yaml

import dask
import xarray as xr

from FRAME_FM.utils.common_utils import convert_subset_selectors_to_slices
from FRAME_FM.transforms import apply_preprocessors
from FRAME_FM.utils.croissant_utils.croissant_bakery import write_croissant_file

from zarr_parallel import ZarrParallelAssembler
from zarr_parallel.utils import set_verbose as zp_set_verbose

preprocessor_hash_key = "_preprocessor_cache_hash"
default_zarr_format = os.getenv("FRAME_ZARR_FORMAT", 2)  # Use Zarr v2 format for better compatibility with xarray and Dask
default_chunks = yaml.safe_load(os.getenv("FRAME_DEFAULT_CHUNKS", "{time: 64}"))  # Default chunking strategy, can be overridden by environment variable
DEBUG = True


def safely_remove_dir(path: Path | str):
    """
    Safely remove a directory and its contents if it exists.
    Args:
        path (Path | str): The path to the directory to be removed.
    """
    path = Path(path)
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                safely_remove_dir(item)
        path.rmdir()

    if DEBUG: print(f"Removed directory at: {path}")


def _infer_extension(uri: Union[str, Path, list, tuple]) -> str:
    """
    Infer the file extension from the URI, handling cases where the URI might be a list 
    or tuple of paths.
    
    Args:
        - uri (str, Path, list, or tuple): The URI of the data source, which can be a string, a Path object, or a list/tuple of URIs.
    
    Returns:
        - str: The inferred file extension from the URI.
    """
    if isinstance(uri, (list, tuple)):
        uri = uri[0]  # Take the first URI if it's a list/tuple

    # Cast to string for ease of processing
    uri = str(uri)

    # Apply a special rule for files that might end in ".gz", ".zip" or ".Z"
    if uri.endswith((".gz", ".zip", ".Z")):
        uri, compressed_ext = os.path.splitext(uri)
    else:
        compressed_ext = ""
    return "." + os.path.basename(uri).split(".")[-1].lower() + compressed_ext.lower()


def get_xr_kwargs(uri: Union[str, Path, list, tuple]) -> dict:
    """
    Determine the appropriate xarray loading engine and any additional kwargs 
    based on the URI format or file extension.
    
    Args:
        - uri (str, Path, list, or tuple): The URI of the data source, which can be a string, a Path object, or a list/tuple of URIs.
        
    Returns:
        - dict: A dictionary of kwargs to pass to xarray loading functions, including the 'engine' key.
    """
    # Map the URI to the appropriate engine for loading data
    # This function should determine the correct engine based on the URI format or file extension
    ext = _infer_extension(uri)
    kwargs = {}

    if ext == ".zarr":
        kwargs["engine"] = "zarr"
    elif ext == ".nc":
        kwargs["engine"] = "netcdf4"
    elif ext == ".json" or ext == ".json.zip":
        kwargs["engine"] = "kerchunk"
    elif ext == ".nca":
        kwargs["engine"] = "CFA"
    elif ext in [".tif", ".tiff", ".geotiff", ".asc", ".txt", ".asciigrid"]:
        kwargs["engine"] = "rasterio"
        kwargs["masked"] = True  # Ensure that rasterio engine returns masked arrays for nodata values
    else:
        raise ValueError(f"Unsupported data URI format: {uri}")
    
    return kwargs


def handle_special_uri_case(uri: Union[str, Path, list, tuple], engine: str) -> Union[str, Path, list, tuple, BytesIO]:
    """
    Handle special cases for certain URI formats and engines, such as loading refs for kerchunk.
    Args:
        uri (str): The original URI of the data source.
        engine (str): The engine determined for loading the data.
    Returns:
        str: The modified URI if special handling was applied, otherwise the original URI.
    """
    if isinstance(uri, str) and uri.endswith(".json.zip"):
        bytestream = BytesIO()
        # For zipped kerchunk files, we need to extract the JSON file from the zip and 
        # pass it to xarray as an in-memory BytesIO object.
        with Path(uri).open("rb") as f:
            with zipfile.ZipFile(f) as z:
                with z.open(z.namelist()[0]) as kerchunk_file:
                    bytestream.write(kerchunk_file.read())

        # Rewind pointer to start of in-memory file
        bytestream.seek(0)
        resource = json.load(bytestream)
    else:
        resource = uri                
    
    return resource


def _get_xr_loader(uri: Union[str, Path, list, tuple]) -> Callable:
    # Simple heuristic to detect if the URI is a glob pattern (e.g., contains wildcards like '*' or '?')
    if isinstance(uri, list) or isinstance(uri, tuple) or any(char in str(uri) for char in ["*", "?", "[", "]"]):
        return xr.open_mfdataset
    else:
        return xr.open_dataset


def load_data_from_uri(uri: Union[str, Path, list, tuple], 
                       chunks: dict | None = None, 
                       subset_selection: dict | None = None,
                       **kwargs
                       ) -> xr.Dataset | xr.DataArray:
    """
    Load data from a URI with optional subset selection.
    Args:
        uri (str): The URI of the data source, or a glob pattern, or a list of URIs.
        chunks (dict | None): Optional dictionary specifying chunking strategy for Dask.
        subset_selection (dict | None): A dictionary specifying the subset selection criteria.
        **kwargs: Additional keyword arguments to pass to the xarray loading function.
    Returns:
        xr.Dataset: The loaded dataset with applied subset selection.
    """
    # Cast the URI to a string if it's a Path object for easier processing
    if isinstance(uri, Path):
        uri = str(uri)

    # Set a default chunking strategy if not provided to ensure Dask is used for larger datasets
    chunks = chunks or default_chunks

    # Load dataset from URI
    subset_selection = convert_subset_selectors_to_slices(subset_selection) if subset_selection else {}
    extra_args = get_xr_kwargs(uri)

    # Get Xarray loader function depending on the URI type (single file vs glob pattern/list)
    xr_loader = _get_xr_loader(uri)
    print(f"Using xarray loader: {xr_loader.__name__} for URI: {uri}")

    # Apply special handling if necessary based on the engine type (e.g., for zipped kerchunk we 
    # might need to load the refs first)
    resource = handle_special_uri_case(uri, extra_args.get("engine"))

    # Merge kwargs and extra_args, giving precedence to kwargs for any overlapping keys
    # Make sure kwargs take precedence over extra_args, so that users can override any automatically determined settings if needed
    kwargs = {**kwargs, **extra_args}

    # Can return either a Dataset or a DataArray depending on the engine and URI.
    data = xr_loader(resource, chunks=chunks, **kwargs)  # type: ignore

    # Apply subset selection if specified
    data = data.sel(**subset_selection)
    return data


def unify_transforms(transforms: list | None, class_transforms: list, override_transforms: bool) -> list:
    """
    Unify the list of transforms by combining user-specified transforms with the default (class) transforms.
    If override_transforms is True, only the user-specified transforms will be used. 
    If False, the user-specified transforms will be combined with the default transforms, ensuring that 
    there are no duplicates based on the "type" key of each transform.
    """
    transforms = transforms or []

    if override_transforms:
        return transforms
    else:
        consolidated_transforms = []
        for transform in transforms + class_transforms:
            if transform["type"] not in [tr["type"] for tr in consolidated_transforms]:
                consolidated_transforms.append(transform)
        return consolidated_transforms


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

def create_cache_path(data_uri: str, cache_dir: Path | str) -> Path:
    "Create cache path from URI."
    zarr_name  = create_zarr_name(data_uri)
    cache_path = Path(cache_dir) / zarr_name
    return cache_path

def hash_preprocessors(preprocessors: list | None) -> str:
    # Create a hash of the preprocessor list to use for caching
    # This function should generate a unique hash based on the contents of the preprocessor list
    preprocessor_str = str(preprocessors).encode("utf-8")
    return hashlib.md5(preprocessor_str).hexdigest()


def cache_data_to_zarr(data_uri: str | Path, 
                       preprocessors: list | None,
                       chunks: dict | None,
                       cache_path: str | Path,
                       generate_stats: bool = True,
                       variables: list | None = None) -> xr.Dataset:
    """
    Cache data to Zarr format based on the provided preprocessors and cache directory.

    Args:
    - data_uri (str | Path): The URI of the data source to be cached.
    - preprocessors (list | None): A list of preprocessors (used for generating a hash only).
    - chunks (dict | None): A dictionary specifying chunking strategy for Dask.
    - cache_dir (str | Path): The directory where cached Zarr files will be stored.
    - generate_stats (bool): Whether to generate statistics during caching.

    Returns:
    - xr.Dataset: The cached dataset loaded from the Zarr file.
    """
    # Create the cache directory if it doesn't exist
    cache_dir = Path(cache_path).parent
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Copy preprocessors because ZarrParallel mutates them.
    preprocs = [prep.copy() for prep in preprocessors] if preprocessors else None

    # Compute a hash of the preprocessors for caching purposes
    preprocessor_hash = hash_preprocessors(preprocessors)

    # Get caching backend from environment variable or settings, default to "basic"
    caching_backend = os.getenv("FRAME_CACHING_BACKEND", "basic")

    # If backend is "basic", we will use the simple caching implementation that loads the data
    # into memory, applies preprocessors, and writes to Zarr format. If the backend is "series" 
    # or "dask_distributed", we will use the ZarrParallelAssembler to handle the caching in a more distributed manner.
    if caching_backend == "basic":
        ds = load_data_from_uri(
            uri=data_uri,
            chunks=chunks)
        ds.attrs[preprocessor_hash_key] = preprocessor_hash
        ds = apply_preprocessors(ds, preprocs) if preprocs else ds
        write_zarr(ds, cache_path, chunks=chunks)
    
    else:
        print(f"Caching data from {data_uri} to {cache_path} using ZarrParallelAssembler with preprocessors: {preprocessors} and chunks: {chunks}")
        zp = ZarrParallelAssembler(
            data_uri=data_uri,
            preprocessors=preprocessors,
            add_attrs={preprocessor_hash_key: preprocessor_hash},
            chunks=chunks,
            engine=get_xr_kwargs(data_uri)['engine']
        )
        zp_set_verbose(1)  # Enable verbose logging for debugging purposes
        
        # Assume the cache path includes the actual zarr store name
        zp.cache(
            str(cache_path),
            generate_stats=generate_stats, 
            await_completion=True,
            simultaneous_worker_limit=4, # Number of workers
            deploy_mode=caching_backend # "series"  - Deploy mode for Dask distributed cluster
            # Memory limit if less than 2GB per worker?
            # Worker timeout if not 30 minutes
            # Deploy mode if specifying between dask/slurm
            )

    # Write Croissant record for the cached Zarr file
    # This should include metadata about the original data source, the preprocessors applied, and the
    # location of the cached Zarr file. This will allow us to trace back from the Croissant record to the cached data.
    write_croissant_file(
        data_uri=cache_path,
        preprocessors=preprocs,
    )

    # Now load the cached Zarr files into memory and add to the response dictionary
    return load_data_from_uri(
                uri=cache_path,
                zarr_format=default_zarr_format
            )


def write_zarr(ds: xr.Dataset,  
               output_path: Path | str, 
               chunks: dict[str, int] | None = None) -> Path | str:
    """
    Return output after applying chunking and determining the output format and chunking.
    """
    print(f"Writing output to {output_path} with chunking: {chunks}")
    chunked_ds = ds.chunk(chunks) if chunks else ds

    # TODO: writing output works currently only in sync mode, see:
    #  - https://github.com/roocs/rook/issues/55
    #  - https://docs.dask.org/en/latest/scheduling.html
    with dask.config.set(scheduler="synchronous"):
        delayed_obj = chunked_ds.to_zarr(output_path, zarr_format=default_zarr_format, compute=False)
        delayed_obj.compute()

    print(f"Wrote output file: {output_path}")
    return output_path
