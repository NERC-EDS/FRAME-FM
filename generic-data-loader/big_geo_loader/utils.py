import xarray as xr
from pathlib import Path
from typing import Type
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
    
    # Copy global metadata from the original dataset if needed (optional, depending on use case)
    dset.attrs.update(ds.attrs)

    # Get common subset selection criteria if provided
    common_subset = selector.get("common", {}).get("subset", {})
    variables = selector.get("variables", {})

    # Load data from the specified URI using the selection criteria
    # This function should handle the logic for loading data, applying selections, and returning the data
    print(f"Loading data from URI: {uri} using engine: {engine}")

    for var_id in variables:
        # Use common subset selectors unless overridden by variable-specific selectors
        subset_selectors = variables[var_id].get("subset", common_subset).copy()
        subset_selectors = convert_subset_selectors_to_slices(subset_selectors)

        if subset_selectors:
            print(f"Subsetting variable: {var_id} with selectors: {subset_selectors}")
            dset[var_id] = ds[var_id].sel(**subset_selectors)
            print(f"Var size difference after subsetting: {dset[var_id].size/(2**20):<.5f}MB / {ds[var_id].size/(2**20):<.5f}MB")
        else:
            print(f"No selection criteria provided for variable: {var_id}. Loading full variable.")
            dset[var_id] = ds[var_id]

    return dset


def convert_subset_selectors_to_slices(selector: dict) -> dict:
    new_selector = {key: slice(low, high) for key, (low, high) in selector.items()}
    return new_selector


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


def open_zarrs(data_dict: dict) -> dict:
    ds_map = {}

    for uri, content in data_dict.items():
        zarr_path = content.get("zarr_path")
        print(f"Opening Zarr file for URI '{uri}' at path: {zarr_path}")
        ds = xr.open_zarr(zarr_path, zarr_version=DEFAULTS.zarr_version)
        ds_map[uri] = ds

    # Add datasets to the input dictionary
    for uri, ds in ds_map.items():
        print(f"Dataset for URI '{uri}' has variables: {list(ds.data_vars)} and dimensions: {list(ds.dims)}")
        data_dict[uri]["xr_dset"] = ds

    return data_dict


def load_data_from_uri(uri: str, subset_selection: dict) -> xr.Dataset:
    # Load dataset from URI
    engine = _map_uri_to_engine(uri)
    ds = xr.open_dataset(uri, engine=engine, chunks="auto")
    ds = ds.sel(**subset_selection)
    return ds