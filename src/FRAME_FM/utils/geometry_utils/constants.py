"""
Canonical dimension alias map.

Each key is a single-letter dimension code used in `expected` / `chosen` strings.
Each value is an ordered list of coordinate name aliases to search for (case-insensitive).
The first match found in the Xarray object wins.
"""

DIM_ALIASES: dict[str, list[str]] = {
    "t": ["time", "t"],
    "z": ["z", "depth", "level", "band"],
    "y": ["y", "lat", "latitude"],
    "x": ["x", "lon", "longitude"],
}

# Suffixes produced by coarsen() -> construct() -> stack() -> transpose()
COARSE_SUFFIX = "_coarse"
FINE_SUFFIX = "_fine"

# Coordinate attributes / dataset attributes that may carry CRS information
CRS_ATTR_NAMES = ["crs", "grid_mapping", "spatial_ref", "proj4", "epsg"]

# Lat/lon alias sets — used to auto-detect EPSG:4326
LAT_ALIASES = {"lat", "latitude", "y"}  # 'y' only counts if paired with x=lon alias
LON_ALIASES = {"lon", "longitude", "x"}

# Fallback CRS when lat/lon names are detected
LATLON_CRS = "EPSG:4326"
