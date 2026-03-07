from io import BytesIO
import json
import os
import zipfile
from pathlib import Path
from typing import Union
from collections.abc import Callable

import xarray as xr


def get_main_vars(dset: xr.Dataset) -> list:
    """
    Get the main variable names from an xarray Dataset, excluding coordinate variables.
    Match only variables that have the maximum size (i.e., the main data variables) to 
    avoid including ancillary variables that may be present in the dataset.
    
    Args:
        - dset (xr.Dataset): The xarray Dataset from which to extract variable names.
    
    Returns:
        - list: A list of variable names that are not coordinates.
    """
    max_var_size = max([variable.size for variable in dset.data_vars.values()])
    return [var_id for var_id, variable in dset.data_vars.items() 
            if var_id not in dset.coords and variable.size == max_var_size]


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


def convert_subset_selectors_to_slices(selector: dict) -> dict:
    """
    Convert a dictionary of subset selectors with (low, high) tuples to a dictionary of slice objects.

    Args:
        - selector (dict): A dictionary where keys are dimension names and values are tuples of (low, high) bounds.
    Returns:
        - dict: A new dictionary where the values are slice objects created from the (low, high) tuples.
    """
    new_selector = {key: slice(low, high) for key, (low, high) in selector.items()}
    return new_selector


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
                       subset_selection: dict | None = None
                       ) -> xr.Dataset | xr.DataArray:
    """
    Load data from a URI with optional subset selection.
    Args:
        uri (str): The URI of the data source, or a glob pattern, or a list of URIs.
        chunks (dict | None): Optional dictionary specifying chunking strategy for Dask.
        subset_selection (dict | None): A dictionary specifying the subset selection criteria.
    Returns:
        xr.Dataset: The loaded dataset with applied subset selection.
    """
    # Cast the URI to a string if it's a Path object for easier processing
    if isinstance(uri, Path):
        uri = str(uri)

    # Set a default chunking strategy if not provided to ensure Dask is used for larger datasets
    chunks = chunks or {"time": 64}

    # Load dataset from URI
    subset_selection = convert_subset_selectors_to_slices(subset_selection) if subset_selection else {}
    kwargs = get_xr_kwargs(uri)

    # Get Xarray loader function depending on the URI type (single file vs glob pattern/list)
    xr_loader = _get_xr_loader(uri)
    print(f"Using xarray loader: {xr_loader.__name__} for URI: {uri}")

    # Apply special handling if necessary based on the engine type (e.g., for zipped kerchunk we 
    # might need to load the refs first)
    resource = handle_special_uri_case(uri, kwargs.get("engine"))

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
