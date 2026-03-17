"""
Public API for geometry_utils.

The two entry points — ``get_centroids`` and ``get_bounds`` — accept any
Xarray object and dispatch to the correct geometry class automatically.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr

from .geometry import from_xarray

_TIME_LETTER = "t"
_BY_VALUES = ("tile", "pixel")


def _ns_to_datetime64(value: float) -> np.datetime64:
    return np.datetime64(int(value), "ns")


def _apply_time_as_datetime_dict(result: dict[str, Any]) -> dict[str, Any]:
    inner = result.get("centroid") or result.get("bounds")
    key = "centroid" if "centroid" in result else "bounds"
    if inner is None or _TIME_LETTER not in inner:
        return result
    converted = dict(inner)
    val = converted[_TIME_LETTER]
    if isinstance(val, tuple):
        converted[_TIME_LETTER] = (_ns_to_datetime64(val[0]), _ns_to_datetime64(val[1]))
    else:
        converted[_TIME_LETTER] = _ns_to_datetime64(val)
    return {**result, key: converted}


def _apply_time_as_datetime_dataset(ds: xr.Dataset) -> xr.Dataset:
    updates: dict[str, xr.DataArray] = {}
    for var in ds.data_vars:
        if var in ("centroid_t", "bounds_t_min", "bounds_t_max"):
            updates[var] = xr.DataArray(
                ds[var].values.astype("int64").astype("datetime64[ns]"),
                coords=ds[var].coords, dims=ds[var].dims, attrs=ds[var].attrs,
            )
    return ds.assign(updates) if updates else ds


def _validate_by(by: str) -> None:
    if by not in _BY_VALUES:
        raise ValueError(f"`by` must be one of {_BY_VALUES!r}, got {by!r}.")


def get_centroids(
    obj: xr.DataArray | xr.Dataset,
    *,
    expected: str | list[str] | None = None,
    chosen: str | list[str] | None = None,
    crs: str | None = None,
    target_crs: str | None = None,
    by: Literal["tile", "pixel"] = "pixel",
    time_as_datetime: bool = False,
) -> dict[str, Any] | xr.Dataset:
    """
    Return centroids of an Xarray object.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
    expected : str or list[str], optional
        Dimension letters that must exist as tiled coordinates.
    chosen : str or list[str], optional
        Subset of expected dimensions to include.
    crs : str, optional
        Source CRS.  Auto-detected if not given.
    target_crs : str, optional
        Output CRS for reprojection (requires pyproj).
    by : {"pixel", "tile"}, default "pixel"
        * ``"pixel"`` — one centroid per pixel in the fine grid.
          Returns an ``xr.Dataset`` with fine-dim coordinates, shaped
          ``(<fine_dim>,)`` for a tile or ``(tile, <fine_dim>)`` for a
          tiled array.
        * ``"tile"`` — one centroid for the whole tile.  Returns a dict
          for a single tile/pixel, or an ``xr.Dataset`` (one row per tile)
          for a tiled array.
    time_as_datetime : bool, default False
        Return ``t`` values as ``numpy.datetime64`` instead of float ns.
    """
    _validate_by(by)
    geom = from_xarray(obj, expected=expected, crs=crs)
    result = (
        geom.pixel_centroid(chosen=chosen, target_crs=target_crs)
        if by == "pixel"
        else geom.centroid(chosen=chosen, target_crs=target_crs)
    )
    if time_as_datetime:
        result = (
            _apply_time_as_datetime_dict(result)
            if isinstance(result, dict)
            else _apply_time_as_datetime_dataset(result)
        )
    return result


def get_bounds(
    obj: xr.DataArray | xr.Dataset,
    *,
    expected: str | list[str] | None = None,
    chosen: str | list[str] | None = None,
    crs: str | None = None,
    target_crs: str | None = None,
    by: Literal["tile", "pixel"] = "pixel",
    time_as_datetime: bool = False,
) -> dict[str, Any] | xr.Dataset:
    """
    Return spatial bounds of an Xarray object.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
    expected : str or list[str], optional
    chosen : str or list[str], optional
    crs : str, optional
    target_crs : str, optional
    by : {"pixel", "tile"}, default "pixel"
        * ``"pixel"`` — ``bounds_<letter>_min`` / ``bounds_<letter>_max``
          per pixel, half-cell padded from coordinate spacing.
        * ``"tile"`` — one bounding box per tile (full spatial envelope).
    time_as_datetime : bool, default False
        Return ``t`` values as ``numpy.datetime64`` instead of float ns.
    """
    _validate_by(by)
    geom = from_xarray(obj, expected=expected, crs=crs)
    result = (
        geom.pixel_bounds(chosen=chosen, target_crs=target_crs)
        if by == "pixel"
        else geom.bounds(chosen=chosen, target_crs=target_crs)
    )
    if time_as_datetime:
        result = (
            _apply_time_as_datetime_dict(result)
            if isinstance(result, dict)
            else _apply_time_as_datetime_dataset(result)
        )
    return result
