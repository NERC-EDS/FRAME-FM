"""
CRS resolution utilities.

Priority order:
  1. Explicitly passed crs= argument
  2. obj.attrs["crs"] / obj.attrs["grid_mapping"] / other known attr names
  3. rioxarray .rio.crs if available
  4. Coordinate name heuristic (lat/lon names → EPSG:4326)
  5. Raise CRSUnresolvableError
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from .constants import CRS_ATTR_NAMES, LAT_ALIASES, LON_ALIASES, LATLON_CRS
from .exceptions import CRSUnresolvableError

if TYPE_CHECKING:
    pass


def resolve_crs(obj: xr.DataArray | xr.Dataset, explicit_crs: str | None) -> str:
    """
    Resolve the CRS of *obj*, returning a string understood by pyproj.

    Parameters
    ----------
    obj:
        The Xarray object whose CRS we want to determine.
    explicit_crs:
        A CRS string passed directly by the caller (highest priority).
        If *None*, the function falls through to auto-detection.

    Returns
    -------
    str
        A CRS string (e.g. ``"EPSG:4326"``, a PROJ string, or a WKT).

    Raises
    ------
    CRSUnresolvableError
        When the CRS cannot be determined by any method.
    """
    # 1. Explicit argument wins immediately.
    if explicit_crs is not None:
        return explicit_crs

    # 2. Check object / coordinate attributes.
    crs = _crs_from_attrs(obj)
    if crs is not None:
        return crs

    # 3. Try rioxarray accessor (optional dependency).
    crs = _crs_from_rio(obj)
    if crs is not None:
        return crs

    # 4. Heuristic: lat/lon coordinate names → EPSG:4326.
    if _looks_like_latlon(obj):
        return LATLON_CRS

    raise CRSUnresolvableError(
        "Cannot determine CRS for the provided Xarray object.\n"
        "Please pass `crs=` explicitly, e.g. crs='EPSG:4326' or crs='EPSG:32633'.\n"
        "Alternatively, set obj.attrs['crs'] = '<your crs string>'."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _crs_from_attrs(obj: xr.DataArray | xr.Dataset) -> str | None:
    """Search known CRS attribute names on the object and its coordinates."""
    sources = [obj.attrs]
    # Also inspect coordinate attributes — rasterio often writes to a
    # 'spatial_ref' coordinate rather than the top-level attrs.
    for coord in obj.coords.values():
        sources.append(coord.attrs)

    for attrs in sources:
        for key in CRS_ATTR_NAMES:
            if key in attrs:
                value = attrs[key]
                if isinstance(value, (str, int)):
                    return str(value) if isinstance(value, int) else value
    return None


def _crs_from_rio(obj: xr.DataArray | xr.Dataset) -> str | None:
    """Try to read CRS via rioxarray if it is installed and attached."""
    try:
        import rioxarray  # noqa: F401 — side-effect import attaches .rio
        rio_crs = obj.rio.crs
        if rio_crs is not None:
            return str(rio_crs)
    except (ImportError, AttributeError, Exception):
        pass
    return None


def _looks_like_latlon(obj: xr.DataArray | xr.Dataset) -> bool:
    """
    Return True when the object's coordinate names suggest geographic lat/lon.

    Requires *both* a lat-like and a lon-like coordinate name to be present
    (to avoid false positives for purely latitude or longitude 1-D objects).
    """
    coord_names = {c.lower() for c in obj.coords}
    dim_names = {d.lower() for d in obj.dims}
    all_names = coord_names | dim_names

    has_lat = bool(all_names & LAT_ALIASES)
    has_lon = bool(all_names & LON_ALIASES)
    return has_lat and has_lon
