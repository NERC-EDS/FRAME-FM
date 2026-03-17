# xarray_geometry

Centroid and bounds extraction for tiled Xarray objects.

Designed for ML data pipelines that produce tiled datasets via the
`coarsen() → construct() → stack() → transpose()` pattern, using
`<coord>_coarse` / `<coord>_fine` coordinate naming.

---

## Features

- **Auto-detects object type** — single pixel, single tile, or a DataArray of many tiles
- **Works with any tiling combination** — T, Z, Y, X or any subset
- **Coordinate alias resolution** — `y` resolves to `lat`, `latitude`, or `y`; likewise for `x`, `t`, `z`
- **Always infers bounds** — no need for explicit bounds coordinates; inferred from coordinate spacing
- **Optional CRS reprojection** — reproject results to any target CRS via pyproj
- **`expected` / `chosen` API** — validate the object's tiling structure and select which dims to return

---

## Installation

```bash
pip install xarray_geometry            # core (xarray + numpy)
pip install xarray_geometry[crs]       # adds pyproj for reprojection
pip install xarray_geometry[test]      # adds pytest + pyproj
```

---

## Quick start

```python
from xarray_geometry import get_centroids, get_bounds
```

### Single tile

```python
result = get_centroids(tile_da, expected="yx", crs="EPSG:4326")
# {'centroid': {'y': 51.5, 'x': -0.1}, 'crs': 'EPSG:4326'}

result = get_bounds(tile_da, expected="yx")
# {'bounds': {'y': (51.0, 52.0), 'x': (-0.5, 0.3)}, 'crs': 'EPSG:4326'}
```

### DataArray of tiles

```python
ds = get_centroids(tiled_da, expected="tyx", chosen="yx")
# <xarray.Dataset>
# Dimensions:     (tile: N)
# Coordinates:
#   * tile          (tile) MultiIndex
#   - t_coarse      (tile) int64 ...
#   - y_coarse      (tile) int64 ...
#   - x_coarse      (tile) int64 ...
# Data variables:
#     centroid_y    (tile) float64 ...
#     centroid_x    (tile) float64 ...
# Attributes:
#     crs:          EPSG:4326

ds = get_bounds(tiled_da, expected="yx", target_crs="EPSG:3857")
# Bounds reprojected into Web Mercator — min/max per tile
```

---

## API reference

### `get_centroids(obj, *, expected, chosen, crs, target_crs)`
### `get_bounds(obj, *, expected, chosen, crs, target_crs)`

Both functions share the same signature.

| Parameter | Type | Description |
|-----------|------|-------------|
| `obj` | `xr.DataArray` or `xr.Dataset` | The Xarray object to compute geometry for |
| `expected` | `str` or `list[str]`, optional | Dim letters that **must** exist as tiled coords — raises `ExpectedDimsMismatchError` if not. E.g. `"tyx"` or `["t","y","x"]` |
| `chosen` | `str` or `list[str]`, optional | Subset of `expected` dims to include in the result. Defaults to all of `expected` |
| `crs` | `str`, optional | Source CRS of the input data. Auto-detected if not given |
| `target_crs` | `str`, optional | Target CRS for output coordinates. Requires pyproj |

**Returns:**
- `dict` for a pixel or single tile: `{"centroid": {letter: value}, "crs": str}` / `{"bounds": {letter: (min, max)}, "crs": str}`
- `xr.Dataset` for a tiled array: one row per tile, with coarse coords retained as index coordinates

---

## Supported dimension letters

| Letter | Resolved aliases (in order) |
|--------|-----------------------------|
| `t` | `time`, `t` |
| `z` | `z`, `depth`, `level`, `band` |
| `y` | `y`, `lat`, `latitude` |
| `x` | `x`, `lon`, `longitude` |

For each letter, the library looks for `<alias>_coarse` and `<alias>_fine`
coordinates in the object. The first matching alias wins.

---

## Coordinate naming convention

Your tiled objects must use the `<coord>_coarse` / `<coord>_fine` naming
produced by the standard Xarray tiling pipeline:

```python
tiled = (
    da.coarsen(y=tile_size, x=tile_size, boundary="trim")
      .construct(y="y_coarse", x="x_coarse")   # → y_coarse, y as y_fine
      .stack(tile=("y_coarse", "x_coarse"))
      .transpose("tile", "y", "x")
)
# After rename:
# tiled = tiled.rename({"y": "y_fine", "x": "x_fine"})
```

The resulting object has:
- A `tile` MultiIndex dimension built from `y_coarse` / `x_coarse`
- `y_fine` / `x_fine` as the within-tile coordinate dims

---

## `expected` vs `chosen`

```python
# Validate that T, Y, X tiles exist — but only compute Y and X results
get_centroids(obj, expected="tyx", chosen="yx")

# Validate and return all dims
get_centroids(obj, expected="yx")

# No validation — object type is auto-detected, no tiling expected
get_centroids(obj, crs="EPSG:4326")
```

`expected` is a **validation contract** — it asserts the object was tiled in
these dimensions and raises loudly if not.

`chosen` is a **projection** — it selects which dims appear in the result.
It must be a subset of `expected`.

---

## CRS resolution

The source CRS is resolved in this priority order:

1. **Explicit `crs=` argument** — highest priority
2. **Object attributes** — `obj.attrs["crs"]`, `obj.attrs["grid_mapping"]`, `obj.attrs["spatial_ref"]`, etc.
3. **rioxarray** — `obj.rio.crs` if rioxarray is installed and attached
4. **Coordinate name heuristic** — if both a lat-like and a lon-like coordinate name are present, assume `EPSG:4326`
5. **Raise `CRSUnresolvableError`** — with a message explaining how to fix it

---

## Bounds inference

Bounds are always inferred from coordinate spacing — explicit bounds
coordinates are not required.

**For a pixel** (no `_coarse` / `_fine` structure):
- Single coordinate value → point geometry (min == max)
- Multiple values → half-cell padding extrapolated from edge spacing

**For a tile** (`_coarse` + `_fine` coordinates):
- `_coarse` provides the tile's origin offset
- `_fine` values give the within-tile positions
- Total bounds = `coarse_origin + inferred_fine_bounds`

**Centroid** is always the midpoint of the inferred bounds.

---

## Object type detection

The library inspects the Xarray object to classify it automatically:

| Type | How detected |
|------|-------------|
| `tiled_array` | Has a pandas MultiIndex dimension whose levels include a `_coarse` coordinate, OR any `_coarse` coordinate has size > 1 |
| `tile` | Has `_coarse` coordinates, all with size == 1 |
| `pixel` | No `_coarse` / `_fine` coordinates, or `expected=None` |

---

## Reprojection

Pass `target_crs` to reproject results. Only x/y coordinates are
reprojected; t and z are returned in their native units.

```python
# Single tile: centroid in Web Mercator
result = get_centroids(tile_da, expected="yx", crs="EPSG:4326", target_crs="EPSG:3857")

# Tiled array: bounds in UTM zone 30N
ds = get_bounds(tiled_da, expected="yx", target_crs="EPSG:32630")
```

For bounds, all four corners of each tile are reprojected and the enclosing
axis-aligned box in the target CRS is returned. This is correct for
moderate extents; for tiles spanning large areas or crossing the antimeridian,
consider densifying the edges before reprojection.

Reprojection requires [pyproj](https://pyproj4.github.io/pyproj/):

```bash
pip install pyproj
```

---

## Exceptions

| Exception | When raised |
|-----------|-------------|
| `ExpectedDimsMismatchError` | Object doesn't have the tiled coordinates asserted in `expected` |
| `DimNotFoundError` | Unknown dim letter, or `chosen` dim not in `expected` |
| `CRSUnresolvableError` | CRS cannot be determined by any method |

---

## Running the tests

```bash
pip install xarray_geometry[test]
pytest tests/ -v
```

The reprojection tests are automatically skipped if pyproj is not installed.

---

## Package structure

```
xarray_geometry/
├── api.py          # get_centroids() / get_bounds() — public entry points
├── geometry.py     # PixelGeometry, TileGeometry, TiledArrayGeometry, from_xarray() factory
├── dims.py         # Dim-letter parsing, alias resolution, object-type detection
├── inference.py    # Bounds inference math, midpoint, pyproj reprojection helpers
├── crs.py          # 4-step CRS resolution chain
├── constants.py    # DIM_ALIASES, suffixes, lat/lon alias sets
└── exceptions.py   # CRSUnresolvableError, DimNotFoundError, ExpectedDimsMismatchError
```
