"""
Bounds inference from coordinate arrays.

All public functions work on raw numpy / array-like coordinate values and
return ``(min_val, max_val)`` pairs representing the *spatial extent* of
a tile or pixel in one dimension.

Strategy (as agreed)
---------------------
* Centroid  = midpoint of the inferred bounds.
* Bounds    = (min_val, max_val) inferred from coordinate spacing.

For a dimension with both ``_coarse`` and ``_fine`` coordinates:
  - ``_fine`` values give the intra-tile extent.
  - ``_coarse`` gives the tile's origin offset.

For a dimension with only one of coarse/fine (e.g. a pixel), we infer
bounds from the coordinate value itself and its neighbours (if any), or
treat a single scalar as a zero-width point.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def infer_bounds_from_values(values: np.ndarray) -> tuple[float, float]:
    """
    Infer spatial bounds from a 1-D array of coordinate values.

    For a single value, the bound is the value itself (point geometry).
    For multiple values, the bounds are extrapolated outward by half a
    cell from the outermost coordinate values using the edge spacing.

    Parameters
    ----------
    values:
        1-D coordinate values (already sorted or assumed monotonic).

    Returns
    -------
    (min_val, max_val)
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("Cannot infer bounds from an empty coordinate array.")

    if values.size == 1:
        return float(values[0]), float(values[0])

    # Sort so min/max logic is reliable
    sorted_vals = np.sort(values)
    half_left = (sorted_vals[1] - sorted_vals[0]) / 2.0
    half_right = (sorted_vals[-1] - sorted_vals[-2]) / 2.0

    return float(sorted_vals[0] - half_left), float(sorted_vals[-1] + half_right)


def infer_bounds_coarse_fine(
    coarse_values: np.ndarray,
    fine_values: np.ndarray,
) -> tuple[float, float]:
    """
    Infer the *global* spatial bounds for a single tile given its coarse
    origin coordinate and fine intra-tile coordinates.

    The coarse coordinate locates the tile; the fine coordinates describe
    its internal grid.  Together they define the tile's full extent:

        global_coord = coarse_origin + fine_offset

    For the single-tile case (one coarse value), we add the coarse origin
    to the fine extent.  The fine bounds are inferred via
    :func:`infer_bounds_from_values`.

    Parameters
    ----------
    coarse_values:
        Coarse coordinate values for *this tile* (often a scalar or 1-element
        array representing the tile origin).
    fine_values:
        Fine coordinate values spanning the tile interior.

    Returns
    -------
    (min_val, max_val)
    """
    coarse_values = np.asarray(coarse_values, dtype=float).ravel()
    fine_values = np.asarray(fine_values, dtype=float).ravel()

    fine_min, fine_max = infer_bounds_from_values(fine_values)

    if coarse_values.size == 1:
        origin = float(coarse_values[0])
        return origin + fine_min, origin + fine_max

    # Multiple coarse values — this is a tiled array slice; return the
    # global extent across all tiles.
    coarse_min, coarse_max = infer_bounds_from_values(coarse_values)
    # Total extent = coarse span + fine span on each edge
    fine_span_half = (fine_max - fine_min) / 2.0
    return coarse_min + fine_min - fine_span_half, coarse_max + fine_max + fine_span_half


def infer_pixel_bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer per-pixel spatial bounds from a 1-D array of coordinate values.

    Each pixel's bounds extend halfway to its neighbours.  Edge pixels use
    the same half-step as their single neighbour.

    Parameters
    ----------
    values:
        1-D coordinate values in native order (need not be sorted).

    Returns
    -------
    (mins, maxs)
        Two float64 arrays of the same length as *values*.
    """
    vals = np.asarray(values, dtype=float).ravel()
    n = vals.size
    if n == 0:
        raise ValueError("Cannot infer pixel bounds from an empty coordinate array.")

    mins = np.empty(n, dtype=float)
    maxs = np.empty(n, dtype=float)

    if n == 1:
        mins[0] = maxs[0] = float(vals[0])
        return mins, maxs

    # Half-step to each neighbour; edges extrapolate from the single neighbour.
    half = np.diff(vals) / 2.0          # length n-1
    left_half = np.empty(n, dtype=float)
    right_half = np.empty(n, dtype=float)

    left_half[0]    = half[0]           # extrapolate at left edge
    left_half[1:]   = half              # step back from each interior/right point
    right_half[:-1] = half              # step forward from each interior/left point
    right_half[-1]  = half[-1]          # extrapolate at right edge

    mins = vals - left_half
    maxs = vals + right_half
    return mins, maxs



    """Return True if *values* has a datetime64 or timedelta64 dtype."""
    return np.issubdtype(values.dtype, np.datetime64) or np.issubdtype(
        values.dtype, np.timedelta64
    )


def to_float_ns(values: np.ndarray) -> np.ndarray:
    """
    Cast coord values to float64 for arithmetic, preserving datetime64 as
    nanoseconds-since-epoch.

    Handles three cases:
    - numpy datetime64 arrays (from xr.DataArray.values)
    - object arrays whose elements are pandas Timestamps (from MultiIndex iteration)
    - plain numeric arrays
    """
    arr = np.asarray(values)

    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype(np.float64)

    # Object array — elements may be pd.Timestamp or np.datetime64 scalars
    if arr.dtype == object and arr.size > 0:
        import pandas as pd
        first = arr.ravel()[0]
        if isinstance(first, (pd.Timestamp, np.datetime64)):
            return np.array(
                [pd.Timestamp(v).value for v in arr.ravel()], dtype=np.float64
            )

    return np.asarray(arr, dtype=float)


def from_float_ns(value: float, ref_dtype: np.dtype) -> Any:
    """
    Convert a float nanoseconds value back to the original dtype.

    For datetime64 dtypes this returns a ``numpy.datetime64``; for numeric
    dtypes it returns the plain float.
    """
    if np.issubdtype(ref_dtype, np.datetime64):
        return np.datetime64(int(value), "ns")
    return value


def bounds_to_native(
    mn: float, mx: float, ref_dtype: np.dtype
) -> tuple[Any, Any]:
    """Return (min, max) in the original coord dtype."""
    return from_float_ns(mn, ref_dtype), from_float_ns(mx, ref_dtype)


def centroid_to_native(value: float, ref_dtype: np.dtype) -> Any:
    """Return the centroid value in the original coord dtype."""
    return from_float_ns(value, ref_dtype)


def midpoint(min_val: float, max_val: float) -> float:
    """Return the midpoint of a (min, max) bounds pair."""
    return (min_val + max_val) / 2.0


# ---------------------------------------------------------------------------
# CRS-aware reprojection of a single point or bounds
# ---------------------------------------------------------------------------

def reproject_point(
    x: float,
    y: float,
    src_crs: str,
    dst_crs: str,
) -> tuple[float, float]:
    """
    Reproject a single (x, y) point from *src_crs* to *dst_crs*.

    Returns
    -------
    (x_dst, y_dst)
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x_dst, y_dst = transformer.transform(x, y)
    return float(x_dst), float(y_dst)


def reproject_bounds(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    src_crs: str,
    dst_crs: str,
) -> tuple[float, float, float, float]:
    """
    Reproject a bounding box from *src_crs* to *dst_crs*.

    Reprojects all four corners and returns the enclosing axis-aligned box
    in the target CRS.  This is correct for moderate extents; for very large
    tiles crossing datum discontinuities callers should densify the edges.

    Returns
    -------
    (min_x, min_y, max_x, max_y) in *dst_crs*
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    corners_src = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]
    xs, ys = transformer.transform(
        [c[0] for c in corners_src],
        [c[1] for c in corners_src],
    )
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
