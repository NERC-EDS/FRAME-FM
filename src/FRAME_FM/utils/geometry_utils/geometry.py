"""
Geometry classes.

Three classes cover the object-type spectrum:

  PixelGeometry      – single pixel (no coarse/fine coords)
  TileGeometry       – single tile  (coarse size == 1)
  TiledArrayGeometry – array of tiles (coarse size > 1, or MultiIndex)

All three share the same public interface:
  .centroid(chosen, target_crs) → dict  or  xr.Dataset
  .bounds(chosen, target_crs)   → dict  or  xr.Dataset
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from .constants import DIM_ALIASES, LATLON_CRS
from .crs import resolve_crs
from .dims import ResolvedDim, detect_object_type, has_tiling_structure, parse_dim_string, resolve_dims, validate_expected
from .exceptions import DimNotFoundError, ExpectedDimsMismatchError
from .inference import (
    infer_bounds_coarse_fine,
    infer_bounds_from_values,
    midpoint,
    reproject_bounds,
    reproject_point,
    to_float_ns,
)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _centroid_dict(coords: dict[str, float], crs: str) -> dict[str, Any]:
    """Package centroid coordinates into a standard result dict."""
    return {"centroid": coords, "crs": crs}


def _bounds_dict(bounds: dict[str, tuple[float, float]], crs: str) -> dict[str, Any]:
    """Package bounds into a standard result dict."""
    return {"bounds": bounds, "crs": crs}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseGeometry:
    """
    Internal base holding the Xarray object, resolved dims, and source CRS.
    Not part of the public API.
    """

    def __init__(
        self,
        obj: xr.DataArray | xr.Dataset,
        resolved: dict[str, ResolvedDim],
        src_crs: str,
    ) -> None:
        self._obj = obj
        self._resolved = resolved
        self._src_crs = src_crs

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _chosen_dims(self, chosen: str | list[str] | None) -> dict[str, ResolvedDim]:
        """
        Return the subset of resolved dims matching *chosen*.

        If *chosen* is None, return all resolved dims.
        """
        letters = parse_dim_string(chosen)
        if not letters:
            return self._resolved

        missing = [l for l in letters if l not in self._resolved]
        if missing:
            raise DimNotFoundError(
                f"`chosen` dims {missing} are not in the resolved dims "
                f"{list(self._resolved.keys())}. "
                "Make sure they appear in `expected` too."
            )
        return {l: self._resolved[l] for l in letters}

    def _maybe_reproject_point(
        self,
        coords: dict[str, float],
        target_crs: str | None,
    ) -> tuple[dict[str, float], str]:
        """Optionally reproject a centroid point; return (coords, effective_crs)."""
        if target_crs is None or target_crs == self._src_crs:
            return coords, self._src_crs

        # We need both x and y to reproject — check we have them.
        xy_letters = [l for l in coords if l in ("x", "y")]
        if len(xy_letters) < 2:
            # Not enough spatial axes to reproject — return as-is with a note.
            return coords, self._src_crs

        x_val = coords["x"]
        y_val = coords["y"]
        x_new, y_new = reproject_point(x_val, y_val, self._src_crs, target_crs)
        reprojected = {**coords, "x": x_new, "y": y_new}
        return reprojected, target_crs

    @staticmethod
    def _infer_ref_dtype(raw_coarse: np.ndarray, raw_fine: np.ndarray) -> np.dtype:
        """Kept for internal use only — detects whether a coord is temporal."""
        import pandas as pd
        if np.issubdtype(raw_fine.dtype, np.datetime64):
            return raw_fine.dtype
        if np.issubdtype(raw_coarse.dtype, np.datetime64):
            return raw_coarse.dtype
        if raw_coarse.dtype == object and raw_coarse.size > 0:
            first = raw_coarse.ravel()[0]
            if isinstance(first, (pd.Timestamp, np.datetime64)):
                return np.dtype("datetime64[ns]")
        return raw_coarse.dtype

    def _resolve_coord_values(
        self, rd: ResolvedDim
    ) -> tuple[np.ndarray, bool]:
        """
        Return coordinate values as float64 and whether they are absolute.

        Returns
        -------
        (float_values, is_absolute)
            ``float_values`` – 1-D float64 array (ns for datetimes).
            ``is_absolute``  – True if values already include the coarse
                               origin; False if coarse must be added.
        """
        raw_coarse = np.asarray(self._obj.coords[rd.coarse_name].values)
        raw_fine = np.asarray(self._obj.coords[rd.fine_name].values)

        if rd.base_name in self._obj.coords:
            raw_vals = np.asarray(self._obj.coords[rd.base_name].values).ravel()
            return to_float_ns(raw_vals), True
        else:
            return to_float_ns(raw_fine.ravel()), False

    def _maybe_reproject_bounds(
        self,
        bounds: dict[str, tuple[float, float]],
        target_crs: str | None,
    ) -> tuple[dict[str, tuple[float, float]], str]:
        """Optionally reproject bounds; return (bounds, effective_crs)."""
        if target_crs is None or target_crs == self._src_crs:
            return bounds, self._src_crs

        if "x" not in bounds or "y" not in bounds:
            return bounds, self._src_crs

        min_x, max_x = bounds["x"]
        min_y, max_y = bounds["y"]
        min_x_new, min_y_new, max_x_new, max_y_new = reproject_bounds(
            min_x, min_y, max_x, max_y, self._src_crs, target_crs
        )
        reprojected = {
            **bounds,
            "x": (min_x_new, max_x_new),
            "y": (min_y_new, max_y_new),
        }
        return reprojected, target_crs


# ---------------------------------------------------------------------------
# PixelGeometry
# ---------------------------------------------------------------------------

class PixelGeometry(_BaseGeometry):
    """
    Geometry for a single pixel (no ``_coarse`` / ``_fine`` tiling structure).

    The centroid is the coordinate value itself; the bounds are inferred from
    the coordinate value and its neighbours (if any), collapsing to a point
    for a truly scalar coordinate.
    """

    def centroid(
        self,
        chosen: str | list[str] | None = None,
        target_crs: str | None = None,
    ) -> dict[str, Any]:
        """
        Return the centroid of this pixel.

        Parameters
        ----------
        chosen:
            Dim letters to include in the result (subset of resolved dims).
        target_crs:
            Optional target CRS for reprojection of x/y coordinates.

        Returns
        -------
        dict with keys ``"centroid"`` (dict of letter → value) and ``"crs"``.
        """
        chosen_dims = self._chosen_dims(chosen)
        coords: dict[str, float] = {}

        for letter, rd in chosen_dims.items():
            # For a pixel, use the raw coordinate value (no coarse/fine).
            base = rd.base_name
            if base in self._obj.coords:
                vals = np.asarray(self._obj.coords[base].values, dtype=float).ravel()
                mn, mx = infer_bounds_from_values(vals)
                coords[letter] = midpoint(mn, mx)
            else:
                raise DimNotFoundError(
                    f"Coordinate {base!r} not found in object for pixel centroid."
                )

        coords, effective_crs = self._maybe_reproject_point(coords, target_crs)
        return _centroid_dict(coords, effective_crs)

    def bounds(
        self,
        chosen: str | list[str] | None = None,
        target_crs: str | None = None,
    ) -> dict[str, Any]:
        """
        Return the bounds of this pixel.

        Parameters
        ----------
        chosen:
            Dim letters to include.
        target_crs:
            Optional target CRS.

        Returns
        -------
        dict with keys ``"bounds"`` (dict of letter → (min, max)) and ``"crs"``.
        """
        chosen_dims = self._chosen_dims(chosen)
        bounds: dict[str, tuple[float, float]] = {}

        for letter, rd in chosen_dims.items():
            base = rd.base_name
            if base in self._obj.coords:
                vals = np.asarray(self._obj.coords[base].values, dtype=float).ravel()
                bounds[letter] = infer_bounds_from_values(vals)
            else:
                raise DimNotFoundError(
                    f"Coordinate {base!r} not found in object for pixel bounds."
                )

        bounds, effective_crs = self._maybe_reproject_bounds(bounds, target_crs)
        return _bounds_dict(bounds, effective_crs)

    def pixel_centroid(
        self,
        chosen: str | list[str] | None = None,
        target_crs: str | None = None,
    ) -> dict[str, Any]:
        """A pixel is already a single point — delegates to ``centroid``."""
        return self.centroid(chosen=chosen, target_crs=target_crs)

    def pixel_bounds(
        self,
        chosen: str | list[str] | None = None,
        target_crs: str | None = None,
    ) -> dict[str, Any]:
        """A pixel is already a single point — delegates to ``bounds``."""
        return self.bounds(chosen=chosen, target_crs=target_crs)

class TileGeometry(_BaseGeometry):
    """
    Geometry for a single tile (coarse coords present but size == 1).
    """

    def centroid(
        self,
        chosen: str | list[str] | None = None,
        target_crs: str | None = None,
    ) -> dict[str, Any]:
        """Return the centroid of this tile."""
        chosen_dims = self._chosen_dims(chosen)
        coords: dict[str, Any] = {}

        for letter, rd in chosen_dims.items():
            vals, is_absolute = self._resolve_coord_values(rd)
            mn, mx = infer_bounds_from_values(vals)
            if not is_absolute:
                origin = to_float_ns(
                    np.asarray(self._obj.coords[rd.coarse_name].values).ravel()
                )[0]
                mn, mx = origin + mn, origin + mx
            coords[letter] = midpoint(mn, mx)

        coords, effective_crs = self._maybe_reproject_point(coords, target_crs)
        return _centroid_dict(coords, effective_crs)

    def bounds(
        self,
        chosen: str | list[str] | None = None,
        target_crs: str | None = None,
    ) -> dict[str, Any]:
        """Return the bounds of this tile."""
        chosen_dims = self._chosen_dims(chosen)
        bounds: dict[str, tuple[Any, Any]] = {}

        for letter, rd in chosen_dims.items():
            vals, is_absolute = self._resolve_coord_values(rd)
            mn, mx = infer_bounds_from_values(vals)
            if not is_absolute:
                origin = to_float_ns(
                    np.asarray(self._obj.coords[rd.coarse_name].values).ravel()
                )[0]
                mn, mx = origin + mn, origin + mx
            bounds[letter] = (mn, mx)

        bounds, effective_crs = self._maybe_reproject_bounds(bounds, target_crs)
        return _bounds_dict(bounds, effective_crs)


    def pixel_centroid(
        self,
        chosen: str | list[str] | None = None,
        target_crs: str | None = None,
    ) -> xr.Dataset:
        """
        Return one centroid per pixel in this tile as an ``xr.Dataset``.

        The output has the fine coordinate dimension(s) as its index, with
        one ``centroid_<letter>`` variable per chosen dim.
        """
        from .inference import infer_pixel_bounds
        chosen_dims = self._chosen_dims(chosen)
        data_vars: dict[str, xr.DataArray] = {}
        fine_coords: dict[str, xr.DataArray] = {}

        for letter, rd in chosen_dims.items():
            abs_vals, is_absolute = self._resolve_coord_values(rd)
            if not is_absolute:
                origin = to_float_ns(
                    np.asarray(self._obj.coords[rd.coarse_name].values).ravel()
                )[0]
                abs_vals = origin + abs_vals

            fine_dim = rd.fine_name
            fine_index = np.asarray(self._obj.coords[rd.fine_name].values).ravel()
            if fine_dim not in fine_coords:
                fine_coords[fine_dim] = xr.DataArray(fine_index, dims=[fine_dim])
            data_vars[f"centroid_{letter}"] = xr.DataArray(
                abs_vals, coords={fine_dim: fine_index}, dims=[fine_dim]
            )

        # Reproject x/y if needed
        if target_crs is not None and target_crs != self._src_crs:
            data_vars = self._reproject_pixel_centroids(data_vars, target_crs)

        effective_crs = target_crs if target_crs else self._src_crs
        ds = xr.Dataset(data_vars)
        ds.attrs["crs"] = effective_crs
        return ds

    def pixel_bounds(
        self,
        chosen: str | list[str] | None = None,
        target_crs: str | None = None,
    ) -> xr.Dataset:
        """
        Return per-pixel bounds in this tile as an ``xr.Dataset``.

        Variables are named ``bounds_<letter>_min`` / ``bounds_<letter>_max``.
        """
        from .inference import infer_pixel_bounds
        chosen_dims = self._chosen_dims(chosen)
        data_vars: dict[str, xr.DataArray] = {}

        for letter, rd in chosen_dims.items():
            abs_vals, is_absolute = self._resolve_coord_values(rd)
            if not is_absolute:
                origin = to_float_ns(
                    np.asarray(self._obj.coords[rd.coarse_name].values).ravel()
                )[0]
                abs_vals = origin + abs_vals

            fine_dim = rd.fine_name
            fine_index = np.asarray(self._obj.coords[rd.fine_name].values).ravel()
            mins, maxs = infer_pixel_bounds(abs_vals)
            data_vars[f"bounds_{letter}_min"] = xr.DataArray(
                mins, coords={fine_dim: fine_index}, dims=[fine_dim]
            )
            data_vars[f"bounds_{letter}_max"] = xr.DataArray(
                maxs, coords={fine_dim: fine_index}, dims=[fine_dim]
            )

        if target_crs is not None and target_crs != self._src_crs:
            data_vars = self._reproject_pixel_bounds(data_vars, target_crs)

        effective_crs = target_crs if target_crs else self._src_crs
        ds = xr.Dataset(data_vars)
        ds.attrs["crs"] = effective_crs
        return ds

    def _reproject_pixel_centroids(
        self,
        data_vars: dict[str, xr.DataArray],
        target_crs: str,
    ) -> dict[str, xr.DataArray]:
        """Reproject centroid_x / centroid_y pixel arrays."""
        if "centroid_x" not in data_vars or "centroid_y" not in data_vars:
            return data_vars
        from pyproj import Transformer
        transformer = Transformer.from_crs(self._src_crs, target_crs, always_xy=True)
        xs_new, ys_new = transformer.transform(
            data_vars["centroid_x"].values, data_vars["centroid_y"].values
        )
        updated = dict(data_vars)
        updated["centroid_x"] = data_vars["centroid_x"].copy(data=xs_new)
        updated["centroid_y"] = data_vars["centroid_y"].copy(data=ys_new)
        return updated

    def _reproject_pixel_bounds(
        self,
        data_vars: dict[str, xr.DataArray],
        target_crs: str,
    ) -> dict[str, xr.DataArray]:
        """Reproject bounds_x_* / bounds_y_* pixel arrays."""
        keys = ("bounds_x_min", "bounds_x_max", "bounds_y_min", "bounds_y_max")
        if not all(k in data_vars for k in keys):
            return data_vars
        from pyproj import Transformer
        transformer = Transformer.from_crs(self._src_crs, target_crs, always_xy=True)
        x_min = data_vars["bounds_x_min"].values
        x_max = data_vars["bounds_x_max"].values
        y_min = data_vars["bounds_y_min"].values
        y_max = data_vars["bounds_y_max"].values
        corners_x = np.stack([x_min, x_max, x_max, x_min])
        corners_y = np.stack([y_min, y_min, y_max, y_max])
        cx_new, cy_new = transformer.transform(corners_x.ravel(), corners_y.ravel())
        n = x_min.size
        cx_new = cx_new.reshape(4, n)
        cy_new = cy_new.reshape(4, n)
        updated = dict(data_vars)
        updated["bounds_x_min"] = data_vars["bounds_x_min"].copy(data=cx_new.min(axis=0))
        updated["bounds_x_max"] = data_vars["bounds_x_max"].copy(data=cx_new.max(axis=0))
        updated["bounds_y_min"] = data_vars["bounds_y_min"].copy(data=cy_new.min(axis=0))
        updated["bounds_y_max"] = data_vars["bounds_y_max"].copy(data=cy_new.max(axis=0))
        return updated

class TiledArrayGeometry(_BaseGeometry):
    """
    Geometry for a DataArray / Dataset containing many tiles.

    Results are returned as an ``xr.Dataset`` preserving the tile indexing
    structure.  All coarse coordinate dimensions are kept as index coordinates
    regardless of ``chosen``, so tiles remain uniquely addressable.
    """

    def centroid(
        self,
        chosen: str | list[str] | None = None,
        target_crs: str | None = None,
    ) -> xr.Dataset:
        """
        Return one centroid per tile as an ``xr.Dataset``.

        Data variables are named ``centroid_<letter>`` for each letter in
        *chosen*.  The tile MultiIndex (or coarse coords) are retained as
        index coordinates.

        Parameters
        ----------
        chosen:
            Dim letters to compute centroids for.
        target_crs:
            Optional CRS for reprojecting x/y results.
        """
        chosen_dims = self._chosen_dims(chosen)
        tile_index, tile_coords = self._build_tile_index()

        data_vars: dict[str, xr.DataArray] = {}

        for letter, rd in chosen_dims.items():
            centroids = self._compute_per_tile_centroids(rd, tile_index)
            data_vars[f"centroid_{letter}"] = xr.DataArray(
                centroids, coords=tile_coords, dims=["tile"]
            )

        if target_crs is not None and target_crs != self._src_crs:
            data_vars = self._reproject_dataset_centroids(data_vars, tile_coords, target_crs)

        effective_crs = target_crs if target_crs else self._src_crs
        ds = xr.Dataset(data_vars, coords=tile_coords)
        ds.attrs["crs"] = effective_crs
        return ds

    def bounds(
        self,
        chosen: str | list[str] | None = None,
        target_crs: str | None = None,
    ) -> xr.Dataset:
        """
        Return bounds per tile as an ``xr.Dataset``.

        Data variables are named ``bounds_<letter>_min`` and
        ``bounds_<letter>_max`` for each letter in *chosen*.
        """
        chosen_dims = self._chosen_dims(chosen)
        tile_index, tile_coords = self._build_tile_index()

        data_vars: dict[str, xr.DataArray] = {}

        for letter, rd in chosen_dims.items():
            mins, maxs = self._compute_per_tile_bounds(rd, tile_index)
            data_vars[f"bounds_{letter}_min"] = xr.DataArray(
                mins, coords=tile_coords, dims=["tile"]
            )
            data_vars[f"bounds_{letter}_max"] = xr.DataArray(
                maxs, coords=tile_coords, dims=["tile"]
            )

        if target_crs is not None and target_crs != self._src_crs:
            data_vars = self._reproject_dataset_bounds(data_vars, tile_coords, target_crs)

        effective_crs = target_crs if target_crs else self._src_crs
        ds = xr.Dataset(data_vars, coords=tile_coords)
        ds.attrs["crs"] = effective_crs
        return ds

    def pixel_centroid(
        self,
        chosen: str | list[str] | None = None,
        target_crs: str | None = None,
    ) -> xr.Dataset:
        """
        Return one centroid per pixel across all tiles as an ``xr.Dataset``.

        Output dims are ``(tile, <fine_dim>)`` for each chosen dim.
        Variables are named ``centroid_<letter>``.
        """
        from .inference import infer_pixel_bounds
        chosen_dims = self._chosen_dims(chosen)
        tile_index, tile_coords = self._build_tile_index()
        n_tiles = len(tile_index)
        data_vars: dict[str, xr.DataArray] = {}
        all_coords = dict(tile_coords)

        for letter, rd in chosen_dims.items():
            fine_offsets = self._get_fine_offsets(rd)
            fine_dim = rd.fine_name
            fine_index = np.asarray(self._obj.coords[rd.fine_name].values).ravel()
            n_fine = len(fine_index)
            if fine_dim not in all_coords:
                all_coords[fine_dim] = xr.DataArray(fine_index, dims=[fine_dim])

            out = np.empty((n_tiles, n_fine), dtype=float)
            for i, tile in enumerate(tile_index):
                origin = to_float_ns(np.asarray([tile.get(rd.coarse_name, 0.0)]))[0]
                out[i] = origin + fine_offsets

            data_vars[f"centroid_{letter}"] = xr.DataArray(
                out, coords={**tile_coords, fine_dim: fine_index},
                dims=["tile", fine_dim],
            )

        if target_crs is not None and target_crs != self._src_crs:
            data_vars = self._reproject_pixel_centroids_tiled(
                data_vars, tile_coords, target_crs
            )

        effective_crs = target_crs if target_crs else self._src_crs
        ds = xr.Dataset(data_vars, coords=all_coords)
        ds.attrs["crs"] = effective_crs
        return ds

    def pixel_bounds(
        self,
        chosen: str | list[str] | None = None,
        target_crs: str | None = None,
    ) -> xr.Dataset:
        """
        Return per-pixel bounds across all tiles as an ``xr.Dataset``.

        Output dims are ``(tile, <fine_dim>)`` for each chosen dim.
        Variables are named ``bounds_<letter>_min`` / ``bounds_<letter>_max``.
        """
        from .inference import infer_pixel_bounds
        chosen_dims = self._chosen_dims(chosen)
        tile_index, tile_coords = self._build_tile_index()
        n_tiles = len(tile_index)
        data_vars: dict[str, xr.DataArray] = {}
        all_coords = dict(tile_coords)

        for letter, rd in chosen_dims.items():
            fine_offsets = self._get_fine_offsets(rd)
            fine_dim = rd.fine_name
            fine_index = np.asarray(self._obj.coords[rd.fine_name].values).ravel()
            n_fine = len(fine_index)
            if fine_dim not in all_coords:
                all_coords[fine_dim] = xr.DataArray(fine_index, dims=[fine_dim])

            # Per-pixel half-steps are the same for every tile (uniform fine grid)
            _, fine_maxs = infer_pixel_bounds(fine_offsets)
            fine_mins_arr, _ = infer_pixel_bounds(fine_offsets)
            pixel_mins = np.empty((n_tiles, n_fine), dtype=float)
            pixel_maxs = np.empty((n_tiles, n_fine), dtype=float)

            for i, tile in enumerate(tile_index):
                origin = to_float_ns(np.asarray([tile.get(rd.coarse_name, 0.0)]))[0]
                pixel_mins[i] = origin + fine_mins_arr
                pixel_maxs[i] = origin + fine_maxs

            data_vars[f"bounds_{letter}_min"] = xr.DataArray(
                pixel_mins, coords={**tile_coords, fine_dim: fine_index},
                dims=["tile", fine_dim],
            )
            data_vars[f"bounds_{letter}_max"] = xr.DataArray(
                pixel_maxs, coords={**tile_coords, fine_dim: fine_index},
                dims=["tile", fine_dim],
            )

        effective_crs = target_crs if target_crs else self._src_crs
        ds = xr.Dataset(data_vars, coords=all_coords)
        ds.attrs["crs"] = effective_crs
        return ds

    def _reproject_pixel_centroids_tiled(
        self,
        data_vars: dict[str, xr.DataArray],
        tile_coords: dict[str, xr.DataArray],
        target_crs: str,
    ) -> dict[str, xr.DataArray]:
        if "centroid_x" not in data_vars or "centroid_y" not in data_vars:
            return data_vars
        from pyproj import Transformer
        transformer = Transformer.from_crs(self._src_crs, target_crs, always_xy=True)
        xs = data_vars["centroid_x"].values.ravel()
        ys = data_vars["centroid_y"].values.ravel()
        xs_new, ys_new = transformer.transform(xs, ys)
        updated = dict(data_vars)
        updated["centroid_x"] = data_vars["centroid_x"].copy(
            data=xs_new.reshape(data_vars["centroid_x"].shape)
        )
        updated["centroid_y"] = data_vars["centroid_y"].copy(
            data=ys_new.reshape(data_vars["centroid_y"].shape)
        )
        return updated

    # ------------------------------------------------------------------
    # Private: tile indexing
    # ------------------------------------------------------------------

    def _build_tile_index(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, xr.DataArray]]:
        """
        Build a flat list of per-tile coordinate dicts and a matching
        set of Dataset coords.

        Returns
        -------
        tile_index : list[dict]
            Each entry maps coarse coord name → scalar value for one tile.
        tile_coords : dict[str, xr.DataArray]
            Ready to pass as ``coords=`` to ``xr.Dataset``.
        """
        coarse_names = [rd.coarse_name for rd in self._resolved.values()]

        # Try to use a MultiIndex dimension if one exists.
        multi_dim = self._find_multi_dim(coarse_names)

        if multi_dim is not None:
            return self._index_from_multiindex(multi_dim, coarse_names)
        else:
            return self._index_from_coarse_coords(coarse_names)

    def _find_multi_dim(self, coarse_names: list[str]) -> str | None:
        """Return the name of a MultiIndex dimension covering coarse coords, or None."""
        for dim in self._obj.dims:
            idx = self._obj.indexes.get(dim)
            if idx is not None and hasattr(idx, "levels"):
                if set(idx.names) & set(coarse_names):
                    return dim
        return None

    def _index_from_multiindex(
        self, multi_dim: str, coarse_names: list[str]
    ) -> tuple[list[dict[str, Any]], dict[str, xr.DataArray]]:
        idx = self._obj.indexes[multi_dim]
        tile_index = [
            {name: val for name, val in zip(idx.names, key)}
            for key in idx
        ]
        tile_coords: dict[str, xr.DataArray] = {
            name: xr.DataArray(
                [t[name] for t in tile_index], dims=["tile"]
            )
            for name in idx.names
        }
        return tile_index, tile_coords

    def _index_from_coarse_coords(
        self, coarse_names: list[str]
    ) -> tuple[list[dict[str, Any]], dict[str, xr.DataArray]]:
        """Build tile index from broadcasting coarse coordinate dims."""
        import itertools

        coarse_vals: dict[str, np.ndarray] = {}
        for name in coarse_names:
            if name in self._obj.coords:
                coarse_vals[name] = np.asarray(
                    self._obj.coords[name].values
                ).ravel()  # preserve original dtype (datetime safe)

        combinations = list(itertools.product(*coarse_vals.values()))
        tile_index = [
            dict(zip(coarse_vals.keys(), combo)) for combo in combinations
        ]
        tile_coords = {
            name: xr.DataArray(
                [t[name] for t in tile_index], dims=["tile"]
            )
            for name in coarse_vals
        }
        return tile_index, tile_coords

    # ------------------------------------------------------------------
    # Private: per-tile computation
    # ------------------------------------------------------------------

    def _select_tile(self, tile: dict[str, Any]) -> xr.DataArray | xr.Dataset:
        """Select a single tile from the object using its coarse coord values."""
        obj = self._obj
        for coarse_name, val in tile.items():
            if coarse_name in obj.coords and coarse_name in obj.dims:
                obj = obj.sel({coarse_name: val})
            elif coarse_name in obj.coords:
                obj = obj.where(obj.coords[coarse_name] == val, drop=True)
        return obj

    def _get_fine_offsets(self, rd: ResolvedDim) -> np.ndarray:
        """
        Return within-tile coordinate offsets from the tile origin as float64.

        After ``coarsen() → construct()``, ``_fine`` holds integer positional
        indices (0, 1, 2, …).  Step size is derived from:

        1. ``base_name`` dim coord — use first n_fine values, zero-based.
        2. ``_fine`` is already datetime64 — zero-base as ns offsets.
        3. Otherwise infer step from coarse spacing divided by n_fine.
        """
        raw_fine_idx = np.asarray(self._obj.coords[rd.fine_name].values).ravel()
        n_fine = len(raw_fine_idx)

        if rd.base_name in self._obj.coords:
            base_vals = to_float_ns(
                np.asarray(self._obj.coords[rd.base_name].values).ravel()
            )
            tile_vals = base_vals[:n_fine]
            return tile_vals - tile_vals[0]
        elif np.issubdtype(raw_fine_idx.dtype, np.datetime64):
            fine_float = to_float_ns(raw_fine_idx)
            return fine_float - fine_float[0]
        else:
            raw_coarse = self._obj.coords[rd.coarse_name].values
            coarse_float = to_float_ns(np.asarray(raw_coarse).ravel())
            step = (coarse_float[1] - coarse_float[0]) / n_fine if coarse_float.size >= 2 else 1.0
            return raw_fine_idx.astype(float) * step

    def _compute_per_tile_centroids(
        self, rd: ResolvedDim, tile_index: list[dict[str, Any]]
    ) -> np.ndarray:
        fine_offsets = self._get_fine_offsets(rd)
        out = np.empty(len(tile_index), dtype=float)
        fine_min, fine_max = infer_bounds_from_values(fine_offsets)
        for i, tile in enumerate(tile_index):
            origin = to_float_ns(np.asarray([tile.get(rd.coarse_name, 0.0)]))[0]
            out[i] = midpoint(origin + fine_min, origin + fine_max)
        return out

    def _compute_per_tile_bounds(
        self, rd: ResolvedDim, tile_index: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        fine_offsets = self._get_fine_offsets(rd)
        mins = np.empty(len(tile_index), dtype=float)
        maxs = np.empty(len(tile_index), dtype=float)
        fine_min, fine_max = infer_bounds_from_values(fine_offsets)
        for i, tile in enumerate(tile_index):
            origin = to_float_ns(np.asarray([tile.get(rd.coarse_name, 0.0)]))[0]
            mins[i] = origin + fine_min
            maxs[i] = origin + fine_max
        return mins, maxs

    # ------------------------------------------------------------------
    # Private: dataset-level reprojection
    # ------------------------------------------------------------------

    def _reproject_dataset_centroids(
        self,
        data_vars: dict[str, xr.DataArray],
        tile_coords: dict[str, xr.DataArray],
        target_crs: str,
    ) -> dict[str, xr.DataArray]:
        if "centroid_x" not in data_vars or "centroid_y" not in data_vars:
            return data_vars

        from pyproj import Transformer

        transformer = Transformer.from_crs(self._src_crs, target_crs, always_xy=True)
        xs = data_vars["centroid_x"].values
        ys = data_vars["centroid_y"].values
        xs_new, ys_new = transformer.transform(xs, ys)

        updated = dict(data_vars)
        updated["centroid_x"] = xr.DataArray(xs_new, coords=tile_coords, dims=["tile"])
        updated["centroid_y"] = xr.DataArray(ys_new, coords=tile_coords, dims=["tile"])
        return updated

    def _reproject_dataset_bounds(
        self,
        data_vars: dict[str, xr.DataArray],
        tile_coords: dict[str, xr.DataArray],
        target_crs: str,
    ) -> dict[str, xr.DataArray]:
        x_min_key = "bounds_x_min"
        x_max_key = "bounds_x_max"
        y_min_key = "bounds_y_min"
        y_max_key = "bounds_y_max"

        if not all(k in data_vars for k in (x_min_key, x_max_key, y_min_key, y_max_key)):
            return data_vars

        from pyproj import Transformer

        transformer = Transformer.from_crs(self._src_crs, target_crs, always_xy=True)

        x_mins = data_vars[x_min_key].values
        x_maxs = data_vars[x_max_key].values
        y_mins = data_vars[y_min_key].values
        y_maxs = data_vars[y_max_key].values

        # Reproject all four corners per tile
        corners_x = np.concatenate([x_mins, x_maxs, x_maxs, x_mins])
        corners_y = np.concatenate([y_mins, y_mins, y_maxs, y_maxs])
        cx_new, cy_new = transformer.transform(corners_x, corners_y)
        n = len(x_mins)
        cx_new = cx_new.reshape(4, n)
        cy_new = cy_new.reshape(4, n)

        updated = dict(data_vars)
        updated[x_min_key] = xr.DataArray(cx_new.min(axis=0), coords=tile_coords, dims=["tile"])
        updated[x_max_key] = xr.DataArray(cx_new.max(axis=0), coords=tile_coords, dims=["tile"])
        updated[y_min_key] = xr.DataArray(cy_new.min(axis=0), coords=tile_coords, dims=["tile"])
        updated[y_max_key] = xr.DataArray(cy_new.max(axis=0), coords=tile_coords, dims=["tile"])
        return updated


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def from_xarray(
    obj: xr.DataArray | xr.Dataset,
    expected: str | list[str] | None,
    crs: str | None,
) -> _BaseGeometry:
    """
    Inspect *obj* and return the appropriate geometry class instance.

    This is the central dispatcher used by the public API functions.
    """
    src_crs = resolve_crs(obj, crs)

    # If the object has no _coarse/_fine structure at all, it is a pixel.
    # In that case validate_expected is skipped entirely (there is nothing
    # to validate against), and we resolve raw coord aliases instead.
    # Note: if the user passed expected= on a non-tiled object, we still
    # honour it below via _resolve_pixel_dims so chosen/expected work.
    if not has_tiling_structure(obj):
        # No _coarse/_fine structure → pixel. expected= on a pixel is fine:
        # it tells _resolve_pixel_dims which raw coordinate letters to look up.
        # We do NOT raise here — passing expected="yx" on a lat/lon array is
        # a normal, supported usage that selects which coords to return.
        pixel_resolved = _resolve_pixel_dims(obj, expected)
        return PixelGeometry(obj, pixel_resolved, src_crs)

    # Object has tiling structure — validate expected and dispatch.
    resolved = validate_expected(obj, expected)
    obj_type = detect_object_type(obj, resolved)

    if obj_type == "pixel":
        # Has tiling coords but none matched expected — shouldn't normally
        # reach here after validate_expected, but handle gracefully.
        pixel_resolved = _resolve_pixel_dims(obj, expected)
        return PixelGeometry(obj, pixel_resolved, src_crs)
    elif obj_type == "tile":
        return TileGeometry(obj, resolved, src_crs)
    else:
        return TiledArrayGeometry(obj, resolved, src_crs)


def _resolve_pixel_dims(
    obj: xr.DataArray | xr.Dataset,
    expected: str | list[str] | None,
) -> dict[str, ResolvedDim]:
    """
    For pixel objects (no ``_coarse`` / ``_fine`` structure), build a
    ``ResolvedDim`` map by matching alias names directly against the
    object's raw coordinates.

    The ``coarse_name`` / ``fine_name`` fields are set to the raw coord
    name (since there is no tiling), so ``PixelGeometry`` can use
    ``rd.base_name`` to look up values.

    When *expected* is ``None``, all known dimension aliases are searched
    and every match is included, so that ``get_centroids(obj)`` without
    an ``expected`` argument still returns useful coordinates.
    """
    letters = parse_dim_string(expected)
    # If no expected specified, attempt to auto-discover all known dim types.
    if not letters:
        letters = list(DIM_ALIASES.keys())  # ["t", "z", "y", "x"]

    all_names: set[str] = set(obj.coords) | set(obj.dims)
    resolved: dict[str, ResolvedDim] = {}

    for letter in letters:
        aliases = DIM_ALIASES.get(letter, [])
        for alias in aliases:
            if alias in all_names:
                resolved[letter] = ResolvedDim(
                    letter=letter,
                    base_name=alias,
                    coarse_name=alias,
                    fine_name=alias,
                )
                break

    return resolved
