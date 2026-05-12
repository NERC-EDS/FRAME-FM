# SPDX-FileCopyrightText: 2026 FRAME-FM Contributors
#
# SPDX-License-Identifier: Apache-2.0

import xarray as xr


def convert_subset_selectors_to_slices(selector: dict) -> dict:
    """
    Convert a dictionary of subset selectors with (first, last) tuples to a dictionary of slice objects.

    Args:
        - selector (dict): A dictionary where keys are dimension names and values are tuples of (first, last) bounds.
    Returns:
        - dict: A new dictionary where the values are slice objects created from the (first, last) tuples.
    """
    return {key: slice(first, last) for key, (first, last) in selector.items()}


def check_object_type(obj: object, allowed_types: object | tuple[object, ...], caller: str) -> object:
    """
    Check if the object is an instance of the allowed types, and raise a TypeError if not.

    Args:
        - obj (object): The object to check.
        - allowed_types (object or tuple of objects): The type or types that the object is allowed to be.
        - caller (str): The name of the calling function or transform, used for error messages.
    Returns:
        - object: The original object if it is of an allowed type.
    Raises:
        - TypeError: If the object is not an instance of any of the allowed types."""
    # Check if allowed_types is a single type, if so convert it to a tuple
    if isinstance(allowed_types, type):
        allowed_types = (allowed_types,)

    for t in allowed_types:   # type: ignore
        if isinstance(obj, t):
            return obj

    raise TypeError(f"Expected an object of type: {allowed_types} when calling `{caller}`, but received: {type(obj)}.")


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

