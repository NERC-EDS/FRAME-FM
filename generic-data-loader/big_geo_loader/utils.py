import xarray as xr
from pathlib import Path
from typing import Hashable, Type
from collections import defaultdict
import hashlib
import yaml

from .settings import DEBUG, DefaultSettings as DEFAULTS


def _map_uri_to_engine(uri: Path | str):
    # Map the URI to the appropriate engine for loading data
    # This function should determine the correct engine based on the URI format or file extension
    uri = str(uri)

    if uri.endswith(".zarr"):
        return "zarr"
    elif uri.endswith(".nc"):
        return "netcdf4"
    elif uri.endswith(".json"):
        return "kerchunk"
    elif uri.endswith(".nca"):
        return "CFA"
    else:
        raise ValueError(f"Unsupported data URI format: {uri}")


def load_data_from_selector(selector: dict) -> xr.Dataset:
    """
    Load data from a selector dictionary.
    Args:
        selector (dict): A dictionary containing variable settings, including selection criteria and other options.
    Returns:
        xr.Dataset: The loaded dataset with applied selections.
    """
    # Initialize an empty Dataset to store the loaded data
    dset = xr.Dataset()

    # Get the URI
    uri = selector["uri"]

    # Load dataset from URI
    engine = _map_uri_to_engine(uri)
    ds = xr.open_dataset(uri, engine=engine, chunks="auto")
    return ds
    
    # # Copy global metadata from the original dataset if needed (optional, depending on use case)
    # dset.attrs.update(ds.attrs)

    # # Get common subset selection criteria if provided
    # common_subset = selector.get("common", {}).get("subset", {})
    # variables = selector.get("variables", {})

    # # Load data from the specified URI using the selection criteria
    # # This function should handle the logic for loading data, applying selections, and returning the data
    # print(f"Loading data from URI: {uri} using engine: {engine}")

    # for var_id in variables:
    #     # Use common subset selectors unless overridden by variable-specific selectors
    #     subset_selectors = variables[var_id].get("subset", common_subset).copy()
    #     subset_selectors = convert_subset_selectors_to_slices(subset_selectors)

    #     if subset_selectors:
    #         print(f"Subsetting variable: {var_id} with selectors: {subset_selectors}")
    #         dset[var_id] = ds[var_id].sel(**subset_selectors)
    #         print(f"Var size difference after subsetting: {dset[var_id].size/(2**20):<.5f}MB / {ds[var_id].size/(2**20):<.5f}MB")
    #     else:
    #         print(f"No selection criteria provided for variable: {var_id}. Loading full variable.")
    #         dset[var_id] = ds[var_id]

    return dset


def convert_subset_selectors_to_slices(selector: dict) -> dict:
    new_selector = {key: slice(low, high) for key, (low, high) in selector.items()}
    return new_selector


def hash_selector(selector: dict) -> str:
    # Create a hash of the selector dictionary to use for caching
    # This function should generate a unique hash based on the contents of the selector
    selector_str = str(selector).encode("utf-8")
    return hashlib.md5(selector_str).hexdigest()


def check_object_type(obj: object, allowed_types: object | tuple[object, ...]) -> object:
    # Check if allowed_types is a single type, if so convert it to a tuple
    if isinstance(allowed_types, type):
        allowed_types = (allowed_types,)

    for t in allowed_types:   # type: ignore
        if isinstance(t, type):
            return obj

    raise TypeError(f"Expected an object of type: {allowed_types}, but received {type(obj)}.")


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


def dump_selectors_to_yaml(selectors: list[dict], yaml_path: str):
    with open(yaml_path, "w") as yaml_file:
        yaml.dump({"selectors": selectors}, yaml_file)

    print(f"Dumped selectors to YAML file at: {yaml_path}")


def load_selectors_from_yaml(yaml_path: str) -> list[dict]:

    with open(yaml_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)
        selectors = data.get("selectors", [])  # type: ignore

    print(f"Loaded selectors from YAML file at: {yaml_path}")
    return selectors


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


def get_variables(ds: xr.Dataset) -> list[Hashable]: 
    """
    Returns a list of variable IDs from an xarray Dataset, excluding coordinate variables.
    """
    return [v for v in ds.variables if v not in ds.coords]


def open_cached_zarrs(uris: list[str], cache_dir: Path | str) -> dict:
    var_zarr_dict = defaultdict(dict)

    for uri in uris:
        zarr_path = create_cache_path(uri, cache_dir)
        print(f"Opening Zarr file for URI '{uri}' at path: {zarr_path}")
        ds = xr.open_zarr(zarr_path, zarr_format=DEFAULTS.zarr_format)
        var_zarr_dict[uri]["xr_dset"] = ds
        var_zarr_dict[uri]["variables"] = get_variables(ds)
        print(f"Dataset for URI '{uri}' has variables: {var_zarr_dict[uri]['variables']}")

    return var_zarr_dict


def load_data_from_uri(uri: str, subset_selection: dict|None = None) -> xr.Dataset:
    # Load dataset from URI
    subset_selection = subset_selection or {}
    engine = _map_uri_to_engine(uri)
    ds = xr.open_dataset(uri, engine=engine, chunks="auto")
    ds = ds.sel(**subset_selection)
    return ds