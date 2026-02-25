from io import BytesIO
import json
import zipfile
import xarray as xr
from pathlib import Path
from typing import Hashable


def _map_uri_to_engine(uri: Path | str):
    # Map the URI to the appropriate engine for loading data
    # This function should determine the correct engine based on the URI format or file extension
    uri = str(uri)

    if uri.endswith(".zarr"):
        return "zarr"
    elif uri.endswith(".nc"):
        return "netcdf4"
    elif uri.endswith(".json") or uri.endswith(".json.zip"):
        return "kerchunk"
    elif uri.endswith(".nca"):
        return "CFA"
    else:
        raise ValueError(f"Unsupported data URI format: {uri}")


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


def handle_special_uri_case(uri: str, engine: str) -> str | BytesIO:
    """
    Handle special cases for certain URI formats and engines, such as loading refs for kerchunk.
    Args:
        uri (str): The original URI of the data source.
        engine (str): The engine determined for loading the data.
    Returns:
        str: The modified URI if special handling was applied, otherwise the original URI.
    """
    if uri.endswith(".json.zip"):
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


def load_data_from_uri(uri: str, chunks: dict | None = None, subset_selection: dict | None = None) -> xr.Dataset:
    """
    Load data from a URI with optional subset selection.
    Args:
        uri (str): The URI of the data source.
        chunks (dict | None): Optional dictionary specifying chunking strategy for Dask.
        subset_selection (dict | None): A dictionary specifying the subset selection criteria.
    Returns:
        xr.Dataset: The loaded dataset with applied subset selection.
    """
    # Set a default chunking strategy if not provided to ensure Dask is used for larger datasets
    chunks = chunks or {"time": 64}

    # Load dataset from URI
    subset_selection = convert_subset_selectors_to_slices(subset_selection) if subset_selection else {}
    engine = _map_uri_to_engine(uri)

    # Apply special handling if necessary based on the engine type (e.g., for zipped kerchunk we 
    # might need to load the refs first)
    resource = handle_special_uri_case(uri, engine)

    ds = xr.open_dataset(resource, engine=engine, chunks=chunks)
    ds = ds.sel(**subset_selection)
    return ds

