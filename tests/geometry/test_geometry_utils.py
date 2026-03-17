"""
Tests for xarray_geometry.

Covers:
  - PixelGeometry         (single pixel, lat/lon and projected)
  - TileGeometry          (single tile, various coarse offsets and fine grids)
  - TiledArrayGeometry    (YX, TYX, ZYX, TZYX combinations; MultiIndex and non-MultiIndex)
  - expected / chosen     (validation contract, subset selection, error paths)
  - CRS resolution        (explicit > attrs > rioxarray > heuristic > error)
  - Reprojection          (point, bounds, dataset-level; identity transform)
  - Inference helpers     (bounds, midpoint, coarse+fine math)
  - Dim parsing           (string, list, case, whitespace, unknown letter)
  - Object-type detection (pixel/tile/tiled_array discriminator)
  - xr.Dataset input      (not just DataArray)
  - Edge cases            (1×1 tile grid, negative coordinates, large step sizes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from FRAME_FM.utils.geometry_utils import get_bounds, get_centroids
from FRAME_FM.utils.geometry_utils.dims import (
    detect_object_type,
    parse_dim_string,
    resolve_dims,
    validate_expected,
)
from FRAME_FM.utils.geometry_utils.exceptions import (
    CRSUnresolvableError,
    DimNotFoundError,
    ExpectedDimsMismatchError,
)
from FRAME_FM.utils.geometry_utils.inference import (
    infer_bounds_coarse_fine,
    infer_bounds_from_values,
    midpoint,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def make_pixel(lat: float = 51.5, lon: float = -0.1, crs: str | None = "EPSG:4326") -> xr.DataArray:
    """Single-pixel DataArray with lat/lon coords."""
    da = xr.DataArray(
        [[1.0]],
        dims=["latitude", "longitude"],
        coords={"latitude": [lat], "longitude": [lon]},
    )
    if crs:
        da.attrs["crs"] = crs
    return da


def make_single_tile(
    y_coarse: float = 0.0,
    x_coarse: float = 0.0,
    fine_steps: int = 4,
    crs: str = "EPSG:4326",
) -> xr.DataArray:
    """
    DataArray representing a single tile using the _coarse/_fine convention.
    Fine values span [0, fine_steps - 1]; coarse locates the tile origin.
    """
    fine_y = np.arange(fine_steps, dtype=float)
    fine_x = np.arange(fine_steps, dtype=float)
    data = np.ones((fine_steps, fine_steps))
    da = xr.DataArray(
        data,
        dims=["y_fine", "x_fine"],
        coords={
            "y_fine": fine_y,
            "x_fine": fine_x,
            "y_coarse": y_coarse,
            "x_coarse": x_coarse,
        },
    )
    da.attrs["crs"] = crs
    return da


def make_tiled_array(
    n_y: int = 3,
    n_x: int = 3,
    fine_steps: int = 4,
    step_size: float = 10.0,
    crs: str = "EPSG:4326",
) -> xr.DataArray:
    """
    Simulate the coarsen() -> construct() -> stack() -> transpose() output.

    Creates an (n_y * n_x) tiled DataArray with a MultiIndex 'tile' dim
    built from y_coarse and x_coarse.
    """
    fine_y = np.arange(fine_steps, dtype=float)
    fine_x = np.arange(fine_steps, dtype=float)
    coarse_y = np.arange(n_y, dtype=float) * step_size
    coarse_x = np.arange(n_x, dtype=float) * step_size

    n_tiles = n_y * n_x
    data = np.random.rand(n_tiles, fine_steps, fine_steps)

    # Build MultiIndex
    tile_index = pd.MultiIndex.from_product(
        [coarse_y, coarse_x], names=["y_coarse", "x_coarse"]
    )
    tile_coord = xr.Coordinates.from_pandas_multiindex(tile_index, "tile")

    da = xr.DataArray(
        data,
        dims=["tile", "y_fine", "x_fine"],
        coords={
            **tile_coord,
            "y_fine": fine_y,
            "x_fine": fine_x,
        },
    )
    da.attrs["crs"] = crs
    return da


def make_tiled_array_zyx(
    n_z: int = 2,
    n_y: int = 2,
    n_x: int = 2,
    fine_steps: int = 3,
    step_size: float = 5.0,
    crs: str = "EPSG:4326",
) -> xr.DataArray:
    """Tiled array with Z, Y, X coarse dims (MultiIndex).

    z_fine and the spatial fine coords are added as dimension coordinates.
    The data array has dims (tile, z_fine, y_fine, x_fine).
    """
    fine_vals = np.arange(fine_steps, dtype=float)
    coarse_z = np.arange(n_z, dtype=float) * step_size
    coarse_y = np.arange(n_y, dtype=float) * step_size
    coarse_x = np.arange(n_x, dtype=float) * step_size

    n_tiles = n_z * n_y * n_x
    data = np.random.rand(n_tiles, fine_steps, fine_steps, fine_steps)

    tile_index = pd.MultiIndex.from_product(
        [coarse_z, coarse_y, coarse_x], names=["z_coarse", "y_coarse", "x_coarse"]
    )
    tile_coord = xr.Coordinates.from_pandas_multiindex(tile_index, "tile")

    da = xr.DataArray(
        data,
        dims=["tile", "z_fine", "y_fine", "x_fine"],
        coords={
            **tile_coord,
            "z_fine": fine_vals,
            "y_fine": fine_vals,
            "x_fine": fine_vals,
        },
    )
    da.attrs["crs"] = crs
    return da


def make_tiled_array_tzyx(
    n_t: int = 2,
    n_z: int = 2,
    n_y: int = 2,
    n_x: int = 2,
    fine_steps: int = 3,
    step_size: float = 5.0,
    crs: str = "EPSG:4326",
) -> xr.DataArray:
    """Tiled array with T, Z, Y, X coarse dims (MultiIndex) — full TZYX case.

    All four fine dimensions (t_fine, z_fine, y_fine, x_fine) are present.
    """
    fine_vals = np.arange(fine_steps, dtype=float)
    coarse_t = np.arange(n_t, dtype=float)
    coarse_z = np.arange(n_z, dtype=float) * step_size
    coarse_y = np.arange(n_y, dtype=float) * step_size
    coarse_x = np.arange(n_x, dtype=float) * step_size

    n_tiles = n_t * n_z * n_y * n_x
    # data: (tile, t_fine, z_fine, y_fine, x_fine) — use 2 fine dims for brevity
    data = np.random.rand(n_tiles, fine_steps, fine_steps, fine_steps, fine_steps)

    tile_index = pd.MultiIndex.from_product(
        [coarse_t, coarse_z, coarse_y, coarse_x],
        names=["t_coarse", "z_coarse", "y_coarse", "x_coarse"],
    )
    tile_coord = xr.Coordinates.from_pandas_multiindex(tile_index, "tile")

    da = xr.DataArray(
        data,
        dims=["tile", "t_fine", "z_fine", "y_fine", "x_fine"],
        coords={
            **tile_coord,
            "t_fine": fine_vals,
            "z_fine": fine_vals,
            "y_fine": fine_vals,
            "x_fine": fine_vals,
        },
    )
    da.attrs["crs"] = crs
    return da


def make_single_tile_latlon(
    lat_coarse: float = 51.0,
    lon_coarse: float = -1.0,
    fine_steps: int = 4,
    crs: str = "EPSG:4326",
) -> xr.DataArray:
    """Single tile using lat/lon alias names instead of y/x."""
    fine_lat = np.arange(fine_steps, dtype=float) * 0.1
    fine_lon = np.arange(fine_steps, dtype=float) * 0.1
    data = np.ones((fine_steps, fine_steps))
    da = xr.DataArray(
        data,
        dims=["lat_fine", "lon_fine"],
        coords={
            "lat_fine": fine_lat,
            "lon_fine": fine_lon,
            "lat_coarse": lat_coarse,
            "lon_coarse": lon_coarse,
        },
    )
    da.attrs["crs"] = crs
    return da


def make_single_tile_negative_coords(crs: str = "EPSG:4326") -> xr.DataArray:
    """Single tile with negative coarse coordinates (southern/western hemisphere)."""
    fine_y = np.arange(4, dtype=float)
    fine_x = np.arange(4, dtype=float)
    data = np.ones((4, 4))
    da = xr.DataArray(
        data,
        dims=["y_fine", "x_fine"],
        coords={
            "y_fine": fine_y,
            "x_fine": fine_x,
            "y_coarse": -40.0,
            "x_coarse": -75.0,
        },
    )
    da.attrs["crs"] = crs
    return da


def make_dataset_single_tile(crs: str = "EPSG:4326") -> xr.Dataset:
    """xr.Dataset (not DataArray) representing a single tile."""
    da = make_single_tile(crs=crs)
    return da.to_dataset(name="values")


def make_tiled_array_non_multiindex(
    n_y: int = 3,
    n_x: int = 3,
    fine_steps: int = 4,
    step_size: float = 10.0,
    crs: str = "EPSG:4326",
) -> xr.DataArray:
    """
    Tiled array WITHOUT a MultiIndex — uses separate coarse coordinate dims.
    Tests the fallback path in TiledArrayGeometry._index_from_coarse_coords().
    """
    fine_y = np.arange(fine_steps, dtype=float)
    fine_x = np.arange(fine_steps, dtype=float)
    coarse_y = np.arange(n_y, dtype=float) * step_size
    coarse_x = np.arange(n_x, dtype=float) * step_size
    data = np.random.rand(n_y, n_x, fine_steps, fine_steps)
    da = xr.DataArray(
        data,
        dims=["y_coarse", "x_coarse", "y_fine", "x_fine"],
        coords={
            "y_coarse": coarse_y,
            "x_coarse": coarse_x,
            "y_fine": fine_y,
            "x_fine": fine_x,
        },
    )
    da.attrs["crs"] = crs
    return da


# ===========================================================================
# Unit tests: inference helpers
# ===========================================================================

class TestInferBoundsFromValues:
    def test_single_value_returns_point(self):
        mn, mx = infer_bounds_from_values(np.array([5.0]))
        assert mn == mx == 5.0

    def test_two_values_symmetric(self):
        mn, mx = infer_bounds_from_values(np.array([0.0, 2.0]))
        assert mn == pytest.approx(-1.0)
        assert mx == pytest.approx(3.0)

    def test_uniform_grid(self):
        vals = np.arange(5, dtype=float)  # 0 1 2 3 4, spacing=1
        mn, mx = infer_bounds_from_values(vals)
        assert mn == pytest.approx(-0.5)
        assert mx == pytest.approx(4.5)

    def test_unsorted_input(self):
        mn, mx = infer_bounds_from_values(np.array([4.0, 0.0, 2.0]))
        assert mn < mx

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            infer_bounds_from_values(np.array([]))


class TestMidpoint:
    def test_basic(self):
        assert midpoint(0.0, 10.0) == pytest.approx(5.0)

    def test_negative(self):
        assert midpoint(-3.0, 3.0) == pytest.approx(0.0)

    def test_same_value(self):
        assert midpoint(7.0, 7.0) == pytest.approx(7.0)


class TestInferBoundsCoarseFine:
    def test_single_coarse_zero_offset(self):
        # Coarse=0, fine=[0,1,2,3] → bounds should be (-0.5, 3.5)
        mn, mx = infer_bounds_coarse_fine(
            np.array([0.0]), np.arange(4, dtype=float)
        )
        assert mn == pytest.approx(-0.5)
        assert mx == pytest.approx(3.5)

    def test_single_coarse_nonzero_offset(self):
        # Coarse=10, fine=[0,1,2,3] → bounds should be (9.5, 13.5)
        mn, mx = infer_bounds_coarse_fine(
            np.array([10.0]), np.arange(4, dtype=float)
        )
        assert mn == pytest.approx(9.5)
        assert mx == pytest.approx(13.5)

    def test_centroid_is_midpoint(self):
        mn, mx = infer_bounds_coarse_fine(
            np.array([0.0]), np.arange(4, dtype=float)
        )
        assert midpoint(mn, mx) == pytest.approx(1.5)


# ===========================================================================
# PixelGeometry
# ===========================================================================

class TestPixelGeometry:
    def test_centroid_returns_dict(self):
        da = make_pixel()
        result = get_centroids(da, expected="yx", crs="EPSG:4326", by="tile")
        assert isinstance(result, dict)
        assert "centroid" in result
        assert "y" in result["centroid"]
        assert "x" in result["centroid"]

    def test_centroid_values(self):
        da = make_pixel(lat=51.5, lon=-0.1)
        result = get_centroids(da, expected="yx", crs="EPSG:4326", by="tile")
        assert result["centroid"]["y"] == pytest.approx(51.5)
        assert result["centroid"]["x"] == pytest.approx(-0.1)

    def test_bounds_returns_dict(self):
        da = make_pixel()
        result = get_bounds(da, expected="yx", crs="EPSG:4326", by="tile")
        assert "bounds" in result
        assert "y" in result["bounds"]
        assert "x" in result["bounds"]

    def test_bounds_point_geometry(self):
        # Single pixel → bounds collapse to point
        da = make_pixel(lat=10.0, lon=20.0)
        result = get_bounds(da, expected="yx", crs="EPSG:4326", by="tile")
        y_min, y_max = result["bounds"]["y"]
        x_min, x_max = result["bounds"]["x"]
        assert y_min == y_max == pytest.approx(10.0)
        assert x_min == x_max == pytest.approx(20.0)

    def test_crs_in_result(self):
        da = make_pixel()
        result = get_centroids(da, expected="yx", crs="EPSG:4326", by="tile")
        assert result["crs"] == "EPSG:4326"

    def test_chosen_subset(self):
        da = make_pixel()
        result = get_centroids(da, expected="yx", chosen="y", crs="EPSG:4326", by="tile")
        assert "y" in result["centroid"]
        assert "x" not in result["centroid"]

    def test_chosen_x_only(self):
        da = make_pixel()
        result = get_centroids(da, expected="yx", chosen="x", crs="EPSG:4326", by="tile")
        assert "x" in result["centroid"]
        assert "y" not in result["centroid"]


# ===========================================================================
# TileGeometry
# ===========================================================================

class TestTileGeometry:
    def test_centroid_returns_dict(self):
        da = make_single_tile()
        result = get_centroids(da, expected="yx", by="tile")
        assert isinstance(result, dict)
        assert "centroid" in result

    def test_centroid_y_x_values(self):
        # y_coarse=0, x_coarse=0, fine=[0,1,2,3]
        # centroid_y = midpoint(-0.5, 3.5) = 1.5
        # centroid_x = midpoint(-0.5, 3.5) = 1.5
        da = make_single_tile(y_coarse=0.0, x_coarse=0.0, fine_steps=4)
        result = get_centroids(da, expected="yx", by="tile")
        assert result["centroid"]["y"] == pytest.approx(1.5)
        assert result["centroid"]["x"] == pytest.approx(1.5)

    def test_centroid_with_offset(self):
        # y_coarse=10, x_coarse=20, fine=[0,1,2,3]
        # centroid_y = midpoint(9.5, 13.5) = 11.5
        da = make_single_tile(y_coarse=10.0, x_coarse=20.0, fine_steps=4)
        result = get_centroids(da, expected="yx", by="tile")
        assert result["centroid"]["y"] == pytest.approx(11.5)
        assert result["centroid"]["x"] == pytest.approx(21.5)

    def test_bounds_values(self):
        da = make_single_tile(y_coarse=0.0, x_coarse=0.0, fine_steps=4)
        result = get_bounds(da, expected="yx", by="tile")
        y_min, y_max = result["bounds"]["y"]
        assert y_min == pytest.approx(-0.5)
        assert y_max == pytest.approx(3.5)

    def test_chosen_y_only(self):
        da = make_single_tile()
        result = get_centroids(da, expected="yx", chosen="y", by="tile")
        assert "y" in result["centroid"]
        assert "x" not in result["centroid"]

    def test_chosen_x_only_bounds(self):
        da = make_single_tile()
        result = get_bounds(da, expected="yx", chosen="x", by="tile")
        assert "x" in result["bounds"]
        assert "y" not in result["bounds"]

    def test_crs_from_attrs(self):
        da = make_single_tile(crs="EPSG:32633")
        result = get_centroids(da, expected="yx", by="tile")
        assert result["crs"] == "EPSG:32633"


# ===========================================================================
# TiledArrayGeometry
# ===========================================================================

class TestTiledArrayGeometry:
    def test_centroid_returns_dataset(self):
        da = make_tiled_array(n_y=3, n_x=3)
        result = get_centroids(da, expected="yx", by="tile")
        assert isinstance(result, xr.Dataset)

    def test_centroid_has_correct_variables(self):
        da = make_tiled_array(n_y=2, n_x=2)
        result = get_centroids(da, expected="yx", by="tile")
        assert "centroid_y" in result
        assert "centroid_x" in result

    def test_centroid_tile_count(self):
        da = make_tiled_array(n_y=3, n_x=4)
        result = get_centroids(da, expected="yx", by="tile")
        assert result.sizes["tile"] == 12

    def test_bounds_returns_dataset(self):
        da = make_tiled_array(n_y=2, n_x=2)
        result = get_bounds(da, expected="yx", by="tile")
        assert isinstance(result, xr.Dataset)

    def test_bounds_has_correct_variables(self):
        da = make_tiled_array(n_y=2, n_x=2)
        result = get_bounds(da, expected="yx", by="tile")
        assert "bounds_y_min" in result
        assert "bounds_y_max" in result
        assert "bounds_x_min" in result
        assert "bounds_x_max" in result

    def test_bounds_min_less_than_max(self):
        da = make_tiled_array(n_y=3, n_x=3, fine_steps=4)
        result = get_bounds(da, expected="yx", by="tile")
        assert (result["bounds_y_min"].values <= result["bounds_y_max"].values).all()
        assert (result["bounds_x_min"].values <= result["bounds_x_max"].values).all()

    def test_crs_in_dataset_attrs(self):
        da = make_tiled_array(crs="EPSG:4326")
        result = get_centroids(da, expected="yx", by="tile")
        assert result.attrs["crs"] == "EPSG:4326"

    def test_chosen_y_only(self):
        da = make_tiled_array(n_y=2, n_x=2)
        result = get_centroids(da, expected="yx", chosen="y", by="tile")
        assert "centroid_y" in result
        assert "centroid_x" not in result

    def test_chosen_x_only_bounds(self):
        da = make_tiled_array(n_y=2, n_x=2)
        result = get_bounds(da, expected="yx", chosen="x", by="tile")
        assert "bounds_x_min" in result
        assert "bounds_y_min" not in result

    def test_tyx_expected_chosen_yx(self):
        da = make_tiled_array_tzyx(n_t=2, n_z=1, n_y=2, n_x=2)
        result = get_centroids(da, expected="tzyx", chosen="yx", by="tile")
        assert "centroid_y" in result
        assert "centroid_x" in result
        assert "centroid_t" not in result

    def test_tyx_tile_count(self):
        da = make_tiled_array_tzyx(n_t=2, n_z=1, n_y=3, n_x=4)
        result = get_centroids(da, expected="tzyx", chosen="yx", by="tile")
        assert result.sizes["tile"] == 24

    def test_coarse_coords_retained_in_output(self):
        da = make_tiled_array(n_y=2, n_x=2)
        result = get_centroids(da, expected="yx", by="tile")
        # Coarse coords should be present to keep tiles addressable
        assert "y_coarse" in result.coords or "x_coarse" in result.coords

    def test_centroid_monotonic_with_coarse(self):
        """Centroids should increase monotonically as coarse indices increase."""
        da = make_tiled_array(n_y=4, n_x=1, fine_steps=4, step_size=10.0)
        result = get_centroids(da, expected="yx", chosen="y", by="tile")
        centroids = result["centroid_y"].values
        assert np.all(np.diff(centroids) > 0), "centroids should be strictly increasing"


# ===========================================================================
# expected / chosen validation
# ===========================================================================

class TestExpectedChosen:
    def test_expected_mismatch_raises(self):
        da = make_single_tile()  # has y_coarse/x_coarse, no z
        with pytest.raises(ExpectedDimsMismatchError):
            get_centroids(da, expected="zyx", by="tile")

    def test_chosen_not_in_expected_raises(self):
        da = make_single_tile()
        with pytest.raises(DimNotFoundError):
            get_centroids(da, expected="yx", chosen="t", by="tile")

    def test_expected_none_no_error(self):
        # With no expected, the dispatcher still works (falls through to pixel)
        da = make_pixel()
        # No exception; type detected as pixel with empty resolved dims
        result = get_centroids(da, crs="EPSG:4326", by="tile")
        assert isinstance(result, dict)

    def test_expected_list_form(self):
        da = make_single_tile()
        result = get_centroids(da, expected=["y", "x"], by="tile")
        assert "y" in result["centroid"]

    def test_chosen_list_form(self):
        da = make_single_tile()
        result = get_centroids(da, expected=["y", "x"], chosen=["y"], by="tile")
        assert "y" in result["centroid"]
        assert "x" not in result["centroid"]

    def test_expected_string_case_insensitive(self):
        da = make_single_tile()
        result = get_centroids(da, expected="YX", by="tile")
        assert "y" in result["centroid"]


# ===========================================================================
# CRS resolution
# ===========================================================================

class TestCRSResolution:
    def test_explicit_crs_wins(self):
        da = make_pixel()
        da.attrs["crs"] = "EPSG:32633"
        result = get_centroids(da, expected="yx", crs="EPSG:4326", by="tile")
        assert result["crs"] == "EPSG:4326"

    def test_attrs_crs_used(self):
        da = make_pixel()
        da.attrs["crs"] = "EPSG:32633"
        result = get_centroids(da, expected="yx", by="tile")
        assert result["crs"] == "EPSG:32633"

    def test_latlon_heuristic(self):
        da = make_pixel()
        del da.attrs["crs"]
        result = get_centroids(da, expected="yx", by="tile")
        assert result["crs"] == "EPSG:4326"

    def test_no_crs_raises(self):
        # Object with non-lat/lon names and no attrs
        da = xr.DataArray(
            [[1.0]],
            dims=["northing", "easting"],
            coords={"northing": [500000.0], "easting": [300000.0]},
        )
        with pytest.raises(CRSUnresolvableError):
            get_centroids(da, crs=None, by="tile")

    def test_grid_mapping_attr(self):
        da = make_pixel()
        del da.attrs["crs"]
        da.attrs["grid_mapping"] = "EPSG:3857"
        result = get_centroids(da, expected="yx", by="tile")
        assert result["crs"] == "EPSG:3857"


# ===========================================================================
# Reprojection (requires pyproj)
# ===========================================================================

class TestReprojection:
    pytest.importorskip("pyproj")

    def test_tile_centroid_reprojection(self):
        """Centroid in EPSG:4326 reprojected to EPSG:3857."""
        # y_coarse=51, fine=[0,1,2,3] → centroid_y ≈ 51+1.5 = 52.5° lat
        # x_coarse=0,  fine=[0,1,2,3] → centroid_x ≈ 1.5° lon → ~167 000 m in 3857
        da = make_single_tile(y_coarse=51.0, x_coarse=0.0, fine_steps=4)
        result = get_centroids(da, expected="yx", crs="EPSG:4326", target_crs="EPSG:3857", by="tile")
        # y should be well above 5 million (northern hemisphere)
        assert result["centroid"]["y"] > 5_000_000
        # x should be a small positive number (near prime meridian)
        assert 0 < result["centroid"]["x"] < 500_000
        assert result["crs"] == "EPSG:3857"

    def test_tiled_array_centroid_reprojection(self):
        da = make_tiled_array(n_y=2, n_x=2, fine_steps=4, step_size=1.0, crs="EPSG:4326")
        result = get_centroids(da, expected="yx", target_crs="EPSG:3857", by="tile")
        assert result.attrs["crs"] == "EPSG:3857"
        # Values should be in metre range for Web Mercator
        assert result["centroid_x"].values.max() < 20_000_000

    def test_tiled_array_bounds_reprojection(self):
        da = make_tiled_array(n_y=2, n_x=2, fine_steps=4, step_size=1.0, crs="EPSG:4326")
        result = get_bounds(da, expected="yx", target_crs="EPSG:3857", by="tile")
        assert result.attrs["crs"] == "EPSG:3857"
        assert (result["bounds_x_min"].values < result["bounds_x_max"].values).all()

    def test_same_crs_no_change(self):
        da = make_single_tile(y_coarse=0.0, x_coarse=0.0)
        r1 = get_centroids(da, expected="yx", crs="EPSG:4326", by="tile")
        r2 = get_centroids(da, expected="yx", crs="EPSG:4326", target_crs="EPSG:4326", by="tile")
        assert r1["centroid"]["y"] == pytest.approx(r2["centroid"]["y"])
        assert r1["centroid"]["x"] == pytest.approx(r2["centroid"]["x"])

    def test_pixel_centroid_reprojection(self):
        """Pixel centroid reprojects correctly — no expected needed for pixels."""
        da = make_pixel(lat=48.8566, lon=2.3522)  # Paris
        result = get_centroids(da, crs="EPSG:4326", target_crs="EPSG:3857", by="tile")
        # Paris in Web Mercator: x ≈ 261_848, y ≈ 6_218_434
        assert result["centroid"]["x"] == pytest.approx(261_848, rel=0.01)
        assert result["centroid"]["y"] == pytest.approx(6_218_434, rel=0.01)

    def test_pixel_bounds_reprojection(self):
        """Pixel bounds reproject correctly — no expected needed for pixels."""
        da = make_pixel(lat=51.5, lon=-0.1)
        result = get_bounds(da, crs="EPSG:4326", target_crs="EPSG:3857", by="tile")
        assert result["crs"] == "EPSG:3857"

    def test_reprojection_preserves_tile_count(self):
        da = make_tiled_array(n_y=3, n_x=3, crs="EPSG:4326")
        result = get_centroids(da, expected="yx", target_crs="EPSG:3857", by="tile")
        assert result.sizes["tile"] == 9


# ===========================================================================
# Dim string parsing
# ===========================================================================

class TestParseDimString:
    def test_string_tyx(self):
        assert parse_dim_string("tyx") == ["t", "y", "x"]

    def test_string_uppercase(self):
        assert parse_dim_string("TYX") == ["t", "y", "x"]

    def test_string_with_spaces(self):
        assert parse_dim_string("t y x") == ["t", "y", "x"]

    def test_list_input(self):
        assert parse_dim_string(["t", "y", "x"]) == ["t", "y", "x"]

    def test_list_uppercase(self):
        assert parse_dim_string(["T", "Y", "X"]) == ["t", "y", "x"]

    def test_none_returns_empty(self):
        assert parse_dim_string(None) == []

    def test_empty_string(self):
        assert parse_dim_string("") == []

    def test_single_letter(self):
        assert parse_dim_string("y") == ["y"]

    def test_all_four(self):
        assert parse_dim_string("tzyx") == ["t", "z", "y", "x"]


# ===========================================================================
# resolve_dims — alias matching
# ===========================================================================

class TestResolveDims:
    def test_resolves_y_via_lat_alias(self):
        da = make_single_tile_latlon()
        resolved = resolve_dims(da, ["y", "x"])
        assert "y" in resolved
        assert resolved["y"].base_name == "lat"
        assert resolved["y"].coarse_name == "lat_coarse"
        assert resolved["y"].fine_name == "lat_fine"

    def test_resolves_x_via_lon_alias(self):
        da = make_single_tile_latlon()
        resolved = resolve_dims(da, ["x"])
        assert resolved["x"].base_name == "lon"

    def test_resolves_y_via_y_name(self):
        da = make_single_tile()
        resolved = resolve_dims(da, ["y"])
        assert resolved["y"].base_name == "y"

    def test_unknown_letter_raises(self):
        da = make_single_tile()
        with pytest.raises(DimNotFoundError, match="Unknown dimension letter"):
            resolve_dims(da, ["q"])

    def test_missing_coarse_raises_strict(self):
        da = make_single_tile()  # no z_coarse/z_fine
        with pytest.raises(DimNotFoundError):
            resolve_dims(da, ["z"], strict=True)

    def test_missing_dim_skipped_non_strict(self):
        da = make_single_tile()
        resolved = resolve_dims(da, ["z", "y", "x"], strict=False)
        assert "z" not in resolved
        assert "y" in resolved
        assert "x" in resolved

    def test_resolved_dim_repr(self):
        da = make_single_tile()
        resolved = resolve_dims(da, ["y"])
        r = repr(resolved["y"])
        assert "letter='y'" in r
        assert "coarse='y_coarse'" in r


# ===========================================================================
# detect_object_type
# ===========================================================================

class TestDetectObjectType:
    def test_pixel_empty_resolved(self):
        da = make_pixel()
        assert detect_object_type(da, {}) == "pixel"

    def test_tile_detected(self):
        da = make_single_tile()
        resolved = resolve_dims(da, ["y", "x"])
        assert detect_object_type(da, resolved) == "tile"

    def test_tiled_array_via_multiindex(self):
        da = make_tiled_array(n_y=2, n_x=2)
        resolved = resolve_dims(da, ["y", "x"])
        assert detect_object_type(da, resolved) == "tiled_array"

    def test_tiled_array_via_coarse_size(self):
        da = make_tiled_array_non_multiindex(n_y=2, n_x=2)
        resolved = resolve_dims(da, ["y", "x"])
        assert detect_object_type(da, resolved) == "tiled_array"

    def test_tzyx_tiled_array(self):
        da = make_tiled_array_tzyx(n_t=2, n_z=2, n_y=2, n_x=2)
        resolved = resolve_dims(da, ["t", "z", "y", "x"])
        assert detect_object_type(da, resolved) == "tiled_array"


# ===========================================================================
# Lat/lon alias tiles (y → lat, x → lon)
# ===========================================================================

class TestLatLonAlias:
    def test_single_tile_latlon_centroid(self):
        da = make_single_tile_latlon(lat_coarse=51.0, lon_coarse=-1.0, fine_steps=4)
        result = get_centroids(da, expected="yx", by="tile")
        assert "y" in result["centroid"]
        assert "x" in result["centroid"]

    def test_single_tile_latlon_bounds(self):
        da = make_single_tile_latlon(lat_coarse=51.0, lon_coarse=-1.0, fine_steps=4)
        result = get_bounds(da, expected="yx", by="tile")
        assert "y" in result["bounds"]
        assert "x" in result["bounds"]

    def test_centroid_y_value_near_coarse(self):
        # fine=[0,0.1,0.2,0.3]; coarse=51 → centroid ≈ 51.15
        da = make_single_tile_latlon(lat_coarse=51.0, lon_coarse=0.0, fine_steps=4)
        result = get_centroids(da, expected="yx", by="tile")
        assert result["centroid"]["y"] == pytest.approx(51.15, rel=0.01)


# ===========================================================================
# Negative coordinates
# ===========================================================================

class TestNegativeCoordinates:
    def test_tile_centroid_negative(self):
        da = make_single_tile_negative_coords()
        result = get_centroids(da, expected="yx", by="tile")
        # y_coarse=-40, fine=[0,1,2,3] → centroid_y = midpoint(-40.5, -36.5) = -38.5
        assert result["centroid"]["y"] == pytest.approx(-38.5)
        # x_coarse=-75, fine=[0,1,2,3] → centroid_x = midpoint(-75.5, -71.5) = -73.5
        assert result["centroid"]["x"] == pytest.approx(-73.5)

    def test_tile_bounds_negative(self):
        da = make_single_tile_negative_coords()
        result = get_bounds(da, expected="yx", by="tile")
        y_min, y_max = result["bounds"]["y"]
        assert y_min == pytest.approx(-40.5)
        assert y_max == pytest.approx(-36.5)

    def test_bounds_min_less_than_max_negative(self):
        da = make_single_tile_negative_coords()
        result = get_bounds(da, expected="yx", by="tile")
        for letter, (mn, mx) in result["bounds"].items():
            assert mn <= mx, f"bounds for {letter}: min={mn} > max={mx}"


# ===========================================================================
# ZYX and TZYX tiling
# ===========================================================================

class TestZYXTiling:
    def test_zyx_centroid_has_z(self):
        da = make_tiled_array_zyx(n_z=2, n_y=2, n_x=2)
        result = get_centroids(da, expected="zyx", by="tile")
        assert "centroid_z" in result
        assert "centroid_y" in result
        assert "centroid_x" in result

    def test_zyx_chosen_yx_excludes_z(self):
        da = make_tiled_array_zyx(n_z=2, n_y=2, n_x=2)
        result = get_centroids(da, expected="zyx", chosen="yx", by="tile")
        assert "centroid_z" not in result
        assert "centroid_y" in result

    def test_zyx_tile_count(self):
        da = make_tiled_array_zyx(n_z=3, n_y=2, n_x=4)
        result = get_centroids(da, expected="zyx", chosen="yx", by="tile")
        assert result.sizes["tile"] == 24

    def test_tzyx_full_centroid(self):
        da = make_tiled_array_tzyx(n_t=2, n_z=2, n_y=2, n_x=2)
        result = get_centroids(da, expected="tzyx", by="tile")
        assert all(f"centroid_{l}" in result for l in ["t", "z", "y", "x"])

    def test_tzyx_tile_count(self):
        da = make_tiled_array_tzyx(n_t=2, n_z=3, n_y=2, n_x=2)
        result = get_centroids(da, expected="tzyx", chosen="yx", by="tile")
        assert result.sizes["tile"] == 24

    def test_tzyx_chosen_single_dim(self):
        da = make_tiled_array_tzyx(n_t=2, n_z=2, n_y=2, n_x=2)
        result = get_centroids(da, expected="tzyx", chosen="t", by="tile")
        assert "centroid_t" in result
        assert "centroid_y" not in result
        assert "centroid_x" not in result

    def test_tzyx_bounds_chosen_yx(self):
        da = make_tiled_array_tzyx(n_t=2, n_z=2, n_y=2, n_x=2)
        result = get_bounds(da, expected="tzyx", chosen="yx", by="tile")
        assert "bounds_y_min" in result
        assert "bounds_x_min" in result
        assert "bounds_t_min" not in result
        assert "bounds_z_min" not in result


# ===========================================================================
# xr.Dataset input
# ===========================================================================

class TestDatasetInput:
    def test_dataset_tile_centroid(self):
        ds = make_dataset_single_tile()
        result = get_centroids(ds, expected="yx", by="tile")
        assert isinstance(result, dict)
        assert "centroid" in result

    def test_dataset_tile_bounds(self):
        ds = make_dataset_single_tile()
        result = get_bounds(ds, expected="yx", by="tile")
        assert "bounds" in result
        assert "y" in result["bounds"]


# ===========================================================================
# Non-MultiIndex tiled array (fallback path)
# ===========================================================================

class TestNonMultiIndexTiledArray:
    def test_centroid_returns_dataset(self):
        da = make_tiled_array_non_multiindex(n_y=2, n_x=3)
        result = get_centroids(da, expected="yx", by="tile")
        assert isinstance(result, xr.Dataset)

    def test_tile_count(self):
        da = make_tiled_array_non_multiindex(n_y=2, n_x=3)
        result = get_centroids(da, expected="yx", by="tile")
        assert result.sizes["tile"] == 6

    def test_bounds_min_less_than_max(self):
        da = make_tiled_array_non_multiindex(n_y=2, n_x=2)
        result = get_bounds(da, expected="yx", by="tile")
        assert (result["bounds_y_min"].values <= result["bounds_y_max"].values).all()


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_1x1_tiled_array(self):
        """A single-cell tiled array (1 tile) should still work."""
        da = make_tiled_array(n_y=1, n_x=1)
        result = get_centroids(da, expected="yx", by="tile")
        assert result.sizes["tile"] == 1

    def test_large_step_size(self):
        """Coarse steps of 100 000 (projected CRS units)."""
        da = make_single_tile(y_coarse=500_000.0, x_coarse=300_000.0, fine_steps=10, crs="EPSG:27700")
        result = get_centroids(da, expected="yx", by="tile")
        assert result["centroid"]["y"] == pytest.approx(500_004.5)

    def test_fine_steps_of_one(self):
        """Tile with a single fine step — infer bounds as point geometry."""
        fine_y = np.array([0.0])
        fine_x = np.array([0.0])
        da = xr.DataArray(
            np.ones((1, 1)),
            dims=["y_fine", "x_fine"],
            coords={"y_fine": fine_y, "x_fine": fine_x, "y_coarse": 10.0, "x_coarse": 20.0},
        )
        da.attrs["crs"] = "EPSG:4326"
        result = get_centroids(da, expected="yx", by="tile")
        assert result["centroid"]["y"] == pytest.approx(10.0)
        assert result["centroid"]["x"] == pytest.approx(20.0)

    def test_chosen_equals_expected(self):
        """When chosen == expected, all dims are returned."""
        da = make_single_tile()
        result = get_centroids(da, expected="yx", chosen="yx", by="tile")
        assert "y" in result["centroid"]
        assert "x" in result["centroid"]

    def test_tiled_array_centroid_values_correct(self):
        """
        For a 2-tile grid (n_y=2) with step_size=10 and fine_steps=4:
          step per pixel = 10/4 = 2.5
          offsets = [0, 2.5, 5, 7.5] -> fine_min=-1.25, fine_max=8.75

          tile 0: coarse_y=0,  centroid_y = midpoint(-1.25, 8.75) = 3.75
          tile 1: coarse_y=10, centroid_y = midpoint(8.75, 18.75) = 13.75
        """
        da = make_tiled_array(n_y=2, n_x=1, fine_steps=4, step_size=10.0)
        result = get_centroids(da, expected="yx", chosen="y", by="tile")
        centroids = sorted(result["centroid_y"].values.tolist())
        assert centroids[0] == pytest.approx(3.75)
        assert centroids[1] == pytest.approx(13.75)

    def test_bounds_width_consistent_across_tiles(self):
        """All tiles in a uniform grid should have the same width."""
        da = make_tiled_array(n_y=3, n_x=3, fine_steps=4, step_size=10.0)
        result = get_bounds(da, expected="yx", by="tile")
        widths = result["bounds_y_max"].values - result["bounds_y_min"].values
        assert np.allclose(widths, widths[0]), "All tiles should have equal height"

    def test_centroid_is_midpoint_of_bounds(self):
        """Centroid must always equal midpoint(bounds_min, bounds_max)."""
        da = make_tiled_array(n_y=3, n_x=3, fine_steps=5, step_size=8.0)
        centroids = get_centroids(da, expected="yx", by="tile")
        bounds = get_bounds(da, expected="yx", by="tile")
        for letter in ["y", "x"]:
            c = centroids[f"centroid_{letter}"].values
            mn = bounds[f"bounds_{letter}_min"].values
            mx = bounds[f"bounds_{letter}_max"].values
            expected_c = (mn + mx) / 2.0
            np.testing.assert_allclose(c, expected_c, rtol=1e-10,
                err_msg=f"centroid_{letter} != midpoint of bounds")


# ===========================================================================
# Datetime coordinate handling (bug fix: t should return datetime64, not int)
# ===========================================================================

def make_tiled_array_with_datetime_t(
    n_t: int = 3,
    n_y: int = 2,
    n_x: int = 2,
    fine_steps: int = 4,
    crs: str = "EPSG:4326",
) -> xr.DataArray:
    """
    Tiled array where t_coarse and t_fine are numpy.datetime64 values,
    as would be produced from real ERA5 data.
    """
    base = np.datetime64("2020-01-01", "ns")
    day = np.timedelta64(1, "D").astype("timedelta64[ns]")

    coarse_t = np.array([base + i * day * fine_steps for i in range(n_t)])
    fine_t   = np.array([base + i * day for i in range(fine_steps)])

    coarse_y = np.arange(n_y, dtype=float) * 10.0
    coarse_x = np.arange(n_x, dtype=float) * 10.0
    fine_y   = np.arange(fine_steps, dtype=float)
    fine_x   = np.arange(fine_steps, dtype=float)

    n_tiles = n_t * n_y * n_x
    data = np.random.rand(n_tiles, fine_steps, fine_steps, fine_steps)

    tile_index = pd.MultiIndex.from_product(
        [coarse_t, coarse_y, coarse_x],
        names=["time_coarse", "y_coarse", "x_coarse"],
    )
    tile_coord = xr.Coordinates.from_pandas_multiindex(tile_index, "tile")

    da = xr.DataArray(
        data,
        dims=["tile", "time_fine", "y_fine", "x_fine"],
        coords={
            **tile_coord,
            "time_fine": fine_t,
            "y_fine": fine_y,
            "x_fine": fine_x,
        },
    )
    da.attrs["crs"] = crs
    return da


class TestDatetimeCoords:
    def test_tile_centroid_t_default_is_float(self):
        """t centroid defaults to float ns — for model consumption."""
        da = make_tiled_array_with_datetime_t(n_t=1, n_y=1, n_x=1, fine_steps=4)
        single = da.isel(tile=0)
        single = single.assign_coords(
            time_coarse=da.coords["time_coarse"].values[0],
            y_coarse=da.coords["y_coarse"].values[0],
            x_coarse=da.coords["x_coarse"].values[0],
        )
        result = get_centroids(single, expected="tyx", by="tile")
        t_val = result["centroid"]["t"]
        assert isinstance(t_val, float), f"Expected float, got {type(t_val)}"

    def test_tile_centroid_t_is_datetime_with_flag(self):
        """t centroid with time_as_datetime=True should be datetime64."""
        da = make_tiled_array_with_datetime_t(n_t=1, n_y=1, n_x=1, fine_steps=4)
        single = da.isel(tile=0)
        single = single.assign_coords(
            time_coarse=da.coords["time_coarse"].values[0],
            y_coarse=da.coords["y_coarse"].values[0],
            x_coarse=da.coords["x_coarse"].values[0],
        )
        result = get_centroids(single, expected="tyx", time_as_datetime=True, by="tile")
        t_val = result["centroid"]["t"]
        assert np.issubdtype(np.asarray(t_val).dtype, np.datetime64), (
            f"Expected datetime64, got {type(t_val)} / {np.asarray(t_val).dtype}"
        )

    def test_tiled_array_centroid_t_default_is_float(self):
        """t centroid in a Dataset defaults to float64."""
        da = make_tiled_array_with_datetime_t(n_t=3, n_y=2, n_x=2, fine_steps=4)
        result = get_centroids(da, expected="tyx", chosen="t", by="tile")
        assert "centroid_t" in result
        assert np.issubdtype(result["centroid_t"].values.dtype, np.floating), (
            f"Expected float dtype, got {result['centroid_t'].values.dtype}"
        )

    def test_tiled_array_centroid_t_is_datetime_with_flag(self):
        """t centroid in a Dataset with time_as_datetime=True → datetime64."""
        da = make_tiled_array_with_datetime_t(n_t=3, n_y=2, n_x=2, fine_steps=4)
        result = get_centroids(da, expected="tyx", chosen="t", time_as_datetime=True, by="tile")
        assert "centroid_t" in result
        assert np.issubdtype(result["centroid_t"].values.dtype, np.datetime64), (
            f"Expected datetime64 dtype, got {result['centroid_t'].values.dtype}"
        )

    def test_tiled_array_bounds_t_is_datetime_with_flag(self):
        """t bounds with time_as_datetime=True → datetime64 min/max."""
        da = make_tiled_array_with_datetime_t(n_t=3, n_y=2, n_x=2, fine_steps=4)
        result = get_bounds(da, expected="tyx", chosen="t", time_as_datetime=True, by="tile")
        assert "bounds_t_min" in result
        assert "bounds_t_max" in result
        assert np.issubdtype(result["bounds_t_min"].values.dtype, np.datetime64)
        assert np.issubdtype(result["bounds_t_max"].values.dtype, np.datetime64)

    def test_tiled_array_t_bounds_ordered(self):
        """t bounds min should always be <= max (works in float mode)."""
        da = make_tiled_array_with_datetime_t(n_t=3, n_y=2, n_x=2, fine_steps=4)
        result = get_bounds(da, expected="tyx", chosen="t", by="tile")
        assert (result["bounds_t_min"].values <= result["bounds_t_max"].values).all()

    def test_tiled_array_t_centroids_monotonic(self):
        """t centroids should increase as the t_coarse index increases."""
        da = make_tiled_array_with_datetime_t(n_t=4, n_y=1, n_x=1, fine_steps=4)
        result = get_centroids(da, expected="tyx", chosen="t", by="tile")
        vals = result["centroid_t"].values
        assert np.all(np.diff(vals) > 0), "t centroids should be strictly increasing"

    def test_spatial_coords_still_float_when_t_present(self):
        """y/x centroids remain float regardless of time_as_datetime."""
        da = make_tiled_array_with_datetime_t(n_t=2, n_y=2, n_x=2, fine_steps=4)
        result = get_centroids(da, expected="tyx", time_as_datetime=True, by="tile")
        assert np.issubdtype(result["centroid_y"].values.dtype, np.floating)
        assert np.issubdtype(result["centroid_x"].values.dtype, np.floating)


# ===========================================================================
# expected= raises on untiled / mismatched objects (bug fix)
# ===========================================================================

class TestExpectedStrictEnforcement:
    def test_tiled_object_missing_expected_dim_raises(self):
        """Tiled DataArray passed with expected= for a dim it lacks should raise."""
        da = make_tiled_array(n_y=2, n_x=2)  # has y_coarse/x_coarse, no t or z
        with pytest.raises(ExpectedDimsMismatchError):
            get_centroids(da, expected="tyx", by="tile")  # t not present → raise

    def test_tyx_object_with_tzyx_expected_raises(self):
        """tyx tiled array passed with expected='tzyx' should raise for missing z."""
        da = make_tiled_array(n_y=2, n_x=2)  # only y_coarse/x_coarse, no t or z
        with pytest.raises(ExpectedDimsMismatchError):
            get_centroids(da, expected="tzyx", by="tile")

    def test_yx_object_with_tyx_expected_raises(self):
        """yx tiled array passed with expected='tyx' should raise for missing t."""
        da = make_tiled_array(n_y=2, n_x=2)
        with pytest.raises(ExpectedDimsMismatchError):
            get_centroids(da, expected="tyx", by="tile")

    def test_tyx_object_with_tyx_expected_passes(self):
        """tyx tiled array with expected='tyx' should not raise."""
        da = make_tiled_array_tzyx(n_t=2, n_z=1, n_y=2, n_x=2)
        result = get_centroids(da, expected="tzyx", chosen="yx", by="tile")
        assert isinstance(result, xr.Dataset)

    def test_expected_none_on_untiled_object_does_not_raise(self):
        """No expected= on a pixel object should succeed (auto-discovery)."""
        da = make_pixel()
        del da.attrs["crs"]  # rely on lat/lon heuristic
        result = get_centroids(da, by="tile")
        assert isinstance(result, dict)

    def test_expected_partial_match_raises(self):
        """Object with y/x tiling but expected='zyx' should raise for missing z."""
        da = make_tiled_array(n_y=2, n_x=2)
        with pytest.raises(ExpectedDimsMismatchError):
            get_centroids(da, expected="zyx", by="tile")

# ===========================================================================
# by="pixel" — per-pixel output
# ===========================================================================

class TestByPixel:
    def test_tile_pixel_centroid_returns_dataset(self):
        """by='pixel' on a single tile returns an xr.Dataset."""
        da = make_single_tile(y_coarse=0.0, x_coarse=0.0, fine_steps=4)
        result = get_centroids(da, expected="yx", by="pixel")
        assert isinstance(result, xr.Dataset)

    def test_tile_pixel_centroid_has_fine_dim(self):
        """Per-pixel centroid Dataset is indexed by the fine dimension."""
        da = make_single_tile(y_coarse=0.0, x_coarse=0.0, fine_steps=4)
        result = get_centroids(da, expected="yx", by="pixel")
        assert "centroid_y" in result
        assert "centroid_x" in result
        # Indexed by y_fine / x_fine, not a 'tile' dim
        assert "y_fine" in result.dims or "x_fine" in result.dims

    def test_tile_pixel_centroid_length(self):
        """Per-pixel centroid has one entry per fine step."""
        fine_steps = 6
        da = make_single_tile(fine_steps=fine_steps)
        result = get_centroids(da, expected="yx", by="pixel")
        assert result["centroid_y"].size == fine_steps
        assert result["centroid_x"].size == fine_steps

    def test_tile_pixel_centroid_values_absolute(self):
        """
        Per-pixel centroids should be absolute coords, not indices.
        With y_coarse=10, fine=[0,1,2,3] -> pixel centroids = [10, 11, 12, 13]
        (fine indices treated as unit offsets from coarse origin).
        """
        da = make_single_tile(y_coarse=10.0, x_coarse=20.0, fine_steps=4)
        result = get_centroids(da, expected="yx", by="pixel")
        y_centroids = result["centroid_y"].values
        assert y_centroids[0] == pytest.approx(10.0)
        assert y_centroids[-1] == pytest.approx(13.0)

    def test_tile_pixel_bounds_returns_dataset(self):
        """by='pixel' bounds on a tile returns an xr.Dataset."""
        da = make_single_tile(fine_steps=4)
        result = get_bounds(da, expected="yx", by="pixel")
        assert isinstance(result, xr.Dataset)
        assert "bounds_y_min" in result
        assert "bounds_y_max" in result

    def test_tile_pixel_bounds_min_less_than_max(self):
        """Per-pixel bounds min should always be <= max."""
        da = make_single_tile(y_coarse=0.0, x_coarse=0.0, fine_steps=5)
        result = get_bounds(da, expected="yx", by="pixel")
        assert (result["bounds_y_min"].values <= result["bounds_y_max"].values).all()
        assert (result["bounds_x_min"].values <= result["bounds_x_max"].values).all()

    def test_tile_pixel_centroid_equals_midpoint_of_bounds(self):
        """Per-pixel centroid should equal midpoint of per-pixel bounds."""
        da = make_single_tile(y_coarse=5.0, x_coarse=3.0, fine_steps=6)
        centroids = get_centroids(da, expected="yx", by="pixel")
        bounds = get_bounds(da, expected="yx", by="pixel")
        for letter in ["y", "x"]:
            c = centroids[f"centroid_{letter}"].values
            mn = bounds[f"bounds_{letter}_min"].values
            mx = bounds[f"bounds_{letter}_max"].values
            np.testing.assert_allclose(c, (mn + mx) / 2.0, rtol=1e-10)

    def test_tiled_array_pixel_centroid_returns_dataset(self):
        """by='pixel' on a tiled array returns xr.Dataset."""
        da = make_tiled_array(n_y=2, n_x=2, fine_steps=4)
        result = get_centroids(da, expected="yx", by="pixel")
        assert isinstance(result, xr.Dataset)

    def test_tiled_array_pixel_centroid_dims(self):
        """Tiled array per-pixel centroid has (tile, fine_dim) shape."""
        n_y, n_x, fine_steps = 2, 3, 4
        da = make_tiled_array(n_y=n_y, n_x=n_x, fine_steps=fine_steps)
        result = get_centroids(da, expected="yx", by="pixel")
        assert result["centroid_y"].shape == (n_y * n_x, fine_steps)
        assert result["centroid_x"].shape == (n_y * n_x, fine_steps)

    def test_tiled_array_pixel_bounds_dims(self):
        """Tiled array per-pixel bounds have (tile, fine_dim) shape."""
        n_y, n_x, fine_steps = 2, 2, 5
        da = make_tiled_array(n_y=n_y, n_x=n_x, fine_steps=fine_steps)
        result = get_bounds(da, expected="yx", by="pixel")
        assert result["bounds_y_min"].shape == (n_y * n_x, fine_steps)
        assert result["bounds_x_max"].shape == (n_y * n_x, fine_steps)

    def test_tiled_array_pixel_centroids_monotonic_across_tiles(self):
        """Pixel centroids should increase across tiles as coarse index increases."""
        da = make_tiled_array(n_y=3, n_x=1, fine_steps=4, step_size=10.0)
        result = get_centroids(da, expected="yx", chosen="y", by="pixel")
        # First pixel of each successive tile should be larger than last of previous
        tile_first = result["centroid_y"].values[:, 0]
        assert np.all(np.diff(tile_first) > 0)

    def test_by_invalid_raises(self):
        """Invalid by= value should raise ValueError."""
        da = make_single_tile()
        with pytest.raises(ValueError, match="`by`"):
            get_centroids(da, expected="yx", by="banana")

    def test_pixel_object_by_pixel_same_as_by_tile(self):
        """For a plain pixel object, by='pixel' and by='tile' return the same thing."""
        da = make_pixel()
        del da.attrs["crs"]
        r_tile = get_centroids(da, by="tile")
        r_pixel = get_centroids(da, by="pixel")
        assert r_tile["centroid"] == r_pixel["centroid"]

    def test_chosen_respected_in_pixel_mode(self):
        """chosen= should still filter dims in by='pixel' mode."""
        da = make_single_tile(fine_steps=4)
        result = get_centroids(da, expected="yx", chosen="y", by="pixel")
        assert "centroid_y" in result
        assert "centroid_x" not in result
