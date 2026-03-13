from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import xarray as xr

LonMode = Literal["[-180,180]", "[0,360)"]

@dataclass(frozen=True)

LonMode = Literal["[-180,180]", "[0,360)"]


@dataclass(frozen=True)
class BatchChecksConfig:
    lon_mode: LonMode = "[-180,180]"
    allow_nans_in_values: bool = False
    allow_nans_in_positions: bool = False
    # tolerance for "constant across an axis" checks
    axis_const_tol: float = 1e-10
    # tolerance for tile-vs-global comparisons
    tile_global_atol: float = 1e-5
    # min range to detect accidental normalization (0..1) / radians etc.
    min_lat_range_deg: float = 10.0
    min_lon_range_deg: float = 10.0


def _assert_finite(x: torch.Tensor, *, name: str) -> None:
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        raise AssertionError(f"{name} contains NaN/Inf (count={bad}).")


def assert_positions_not_normalized(
    pos: torch.Tensor,
    *,
    min_lat_range_deg: float = 10.0,
    min_lon_range_deg: float = 10.0,
) -> None:
    """
    Catches the common bug where someone normalized lat/lon to [0,1]
    or converted to radians, etc.
    pos: (B, 2, H, W)
    """
    if pos.ndim != 4 or pos.shape[1] != 2:
        raise AssertionError(f"pos must be (B,2,H,W), got {tuple(pos.shape)}")

    lat = pos[:, 0]
    lon = pos[:, 1]

    lat_range = (lat.max() - lat.min()).item()
    lon_range = (lon.max() - lon.min()).item()

    if lat_range < min_lat_range_deg:
        raise AssertionError(
            f"Latitude range too small ({lat_range:.6f} deg). "
            "Positions may have been normalized/scaled incorrectly."
        )
    if lon_range < min_lon_range_deg:
        raise AssertionError(
            f"Longitude range too small ({lon_range:.6f} deg). "
            "Positions may have been normalized/scaled incorrectly."
        )


def validate_batch(
    values: torch.Tensor,
    pos: Optional[torch.Tensor] = None,
    *,
    cfg: BatchChecksConfig = BatchChecksConfig(),
) -> None:
    """
    Validates post-transform batches coming out of the DataLoader.

    values: (B, C, H, W) float32
    pos:    (B, 2, H, W) float32 (only for pixels/bounds positioned modes)
    """
    if values.ndim != 4:
        raise AssertionError(f"values should be (B,C,H,W), got {tuple(values.shape)}")

    if values.dtype != torch.float32:
        raise AssertionError(f"values dtype should be float32, got {values.dtype}")

    if not cfg.allow_nans_in_values:
        _assert_finite(values, name="values")

    if values.abs().mean().item() == 0.0:
        raise AssertionError("values look all zeros (did fillna wipe data?)")

    if pos is None:
        return  # nothing else to check

    if pos.ndim != 4 or pos.shape[1] != 2:
        raise AssertionError(f"pos should be (B,2,H,W), got {tuple(pos.shape)}")

    if pos.dtype != torch.float32:
        raise AssertionError(f"pos dtype should be float32, got {pos.dtype}")

    if not cfg.allow_nans_in_positions:
        _assert_finite(pos, name="pos")

    B, C, H, W = values.shape
    if pos.shape != (B, 2, H, W):
        raise AssertionError(f"pos shape mismatch: expected {(B,2,H,W)}, got {tuple(pos.shape)}")

    lat = pos[:, 0]  # (B,H,W)
    lon = pos[:, 1]  # (B,H,W)

    # Range checks
    if not ((lat >= -90.0).all() and (lat <= 90.0).all()):
        raise AssertionError("lat out of range [-90,90]")

    if cfg.lon_mode == "[-180,180]":
        if not ((lon >= -180.0).all() and (lon <= 180.0).all()):
            raise AssertionError("lon out of range [-180,180]")
    elif cfg.lon_mode == "[0,360)":
        if not ((lon >= 0.0).all() and (lon < 360.0).all()):
            raise AssertionError("lon out of range [0,360)")
    else:
        raise AssertionError(f"Unknown lon_mode={cfg.lon_mode}")

    # Structural/broadcasting checks:
    # Latitude should be constant across columns within a row
    lat_col_var = lat.var(dim=2)  # variance across W, shape (B,H)
    if not (lat_col_var < cfg.axis_const_tol).all():
        raise AssertionError(
            "Latitude varies across columns -> likely wrong broadcasting or transform misalignment"
        )

    # Longitude should be constant across rows within a column
    lon_row_var = lon.var(dim=1)  # variance across H, shape (B,W)
    if not (lon_row_var < cfg.axis_const_tol).all():
        raise AssertionError(
            "Longitude varies across rows -> likely wrong broadcasting or transform misalignment"
        )

    # Monotonicity checks (expected after sortby longitude)
    # latitude monotonic down rows (increasing or decreasing OK)
    lat_1d = lat[:, :, 0]  # (B,H)
    dlat = lat_1d[:, 1:] - lat_1d[:, :-1]
    if not ((dlat >= -1e-6).all() or (dlat <= 1e-6).all()):
        raise AssertionError("Latitude not monotonic down rows")

    # longitude increasing across columns
    lon_1d = lon[:, 0, :]  # (B,W)
    dlon = lon_1d[:, 1:] - lon_1d[:, :-1]
    if not (dlon >= -1e-6).all():
        raise AssertionError("Longitude not increasing across columns (expected after sortby)")

    # Normalization/scaling guard
    assert_positions_not_normalized(
        pos,
        min_lat_range_deg=cfg.min_lat_range_deg,
        min_lon_range_deg=cfg.min_lon_range_deg,
    )

#Function to check - Our tiles come from stacking (time, tile_lat_id, tile_lon_id) into batch_dim. 
# That means we can pick a tile and reconstruct the exact global slice indices and compare to the original arr. 
def assert_tile_matches_global(
    *,
    raw_arr: xr.DataArray,
    tiles: xr.DataArray,
    tile_index: int,
    channel: int,
    tile_size_lat: int,
    tile_size_lon: int,
    atol: float = 1e-5,
) -> None:
    """
    Proves: tiles[tile_index, channel, :, :] == raw_arr[channel, time, lat_slice, lon_slice]
    after NaNs are treated as zeros (because tiles fillna(0)).

    raw_arr dims: (channel, time, latitude, longitude)
    tiles dims:    (batch_dim, channel, tile_lat, tile_lon)
    """
    if "batch_dim" not in tiles.dims:
        raise AssertionError(f"tiles missing batch_dim, got dims={tiles.dims}")

    bt = tiles["batch_dim"].values[tile_index]  # (time, tile_lat_id, tile_lon_id)
    t_val, tile_lat_id, tile_lon_id = bt
    tile_lat_id = int(tile_lat_id)
    tile_lon_id = int(tile_lon_id)

    H = tiles.sizes["tile_lat"]
    W = tiles.sizes["tile_lon"]
    lat_start = tile_lat_id * tile_size_lat
    lon_start = tile_lon_id * tile_size_lon

    global_patch = raw_arr.sel(time=t_val).isel(
        channel=channel,
        latitude=slice(lat_start, lat_start + H),
        longitude=slice(lon_start, lon_start + W),
    ).values

    tile_patch = tiles.isel(batch_dim=tile_index, channel=channel).values

    global_patch = np.nan_to_num(global_patch, nan=0.0)
    tile_patch = np.nan_to_num(tile_patch, nan=0.0)

    max_abs = float(np.max(np.abs(tile_patch - global_patch)))
    if max_abs > atol:
        raise AssertionError(
            f"Tile mismatch vs global slice (tile_index={tile_index}, channel={channel}). "
            f"max_abs={max_abs:.6e} > atol={atol:.6e}"
        )

#Prove positions match global coords
def assert_positions_match_coords(
    *,
    global_lat: np.ndarray,
    global_lon: np.ndarray,
    tiles: xr.DataArray,
    pos: torch.Tensor,
    tile_index: int,
    tile_size_lat: int,
    tile_size_lon: int,
    atol: float = 1e-6,
) -> None:
    """
    Proves that positions computed for a given tile match the stored global coord arrays.
    pos: (N,2,H,W)
    """
    bt = tiles["batch_dim"].values[tile_index]
    _, tile_lat_id, tile_lon_id = bt
    tile_lat_id = int(tile_lat_id)
    tile_lon_id = int(tile_lon_id)

    H = tiles.sizes["tile_lat"]
    W = tiles.sizes["tile_lon"]
    lat_start = tile_lat_id * tile_size_lat
    lon_start = tile_lon_id * tile_size_lon

    lat_slice = global_lat[lat_start : lat_start + H]
    lon_slice = global_lon[lon_start : lon_start + W]

    if len(lat_slice) == 0 or len(lon_slice) == 0:
        raise AssertionError("Empty lat/lon slice; check tile indices and coord arrays.")

    # expected broadcast
    lat_expected = torch.tensor(lat_slice, dtype=torch.float32).view(H, 1).repeat(1, W)
    lon_expected = torch.tensor(lon_slice, dtype=torch.float32).view(1, W).repeat(H, 1)

    lat_grid = pos[tile_index, 0]
    lon_grid = pos[tile_index, 1]

    # for padded edges, your code repeats last coordinate; compare only valid region
    H_valid = min(H, len(lat_slice))
    W_valid = min(W, len(lon_slice))

    lat_err = torch.max(torch.abs(lat_grid[:H_valid, :W_valid] - lat_expected[:H_valid, :W_valid])).item()
    lon_err = torch.max(torch.abs(lon_grid[:H_valid, :W_valid] - lon_expected[:H_valid, :W_valid])).item()

    if lat_err > atol:
        raise AssertionError(f"lat grid mismatch (tile_index={tile_index}) err={lat_err:.6e} > atol={atol:.6e}")
    if lon_err > atol:
        raise AssertionError(f"lon grid mismatch (tile_index={tile_index}) err={lon_err:.6e} > atol={atol:.6e}")